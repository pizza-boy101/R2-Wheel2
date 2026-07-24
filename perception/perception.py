#!/usr/bin/env python3
"""
perception.py — the robot's single always-on vision process.

Owns the camera and is the ONLY consumer of /dev/video. Replaces the old
vision_sidecar (YOLO) entirely. Each loop it:
  1. grabs a frame from the camera,
  2. runs MiDaS-small depth on the GPU (cv2.dnn CUDA, ~25 fps capable),
  3. reduces depth to a small nav-state (free space per column + optional target),
  4. publishes nav_state.json (every frame) and a throttled frame.jpg
     (so the realtime voice sidecar's look/ambient still have a fresh photo).

NO motor control here — this process only perceives and publishes. The nav loop
(separate process) reads nav_state.json and drives; the voice sidecar reads
frame.jpg + nav_state.json.

Output -> workspace/nav_state.json (atomic):
  {
    "ts", "fps",
    "near":  {"l","c","r"},   # inverse-depth 'nearness', HIGHER = CLOSER
    "clear": {"l","c","r"},   # 0..1 relative clearness within the frame (1 = clearest)
    "clearest": "l|c|r",
    "blocked": bool,          # center nearer than STOP_NEAR (advisory until calibrated)
    "loom":  float,           # depth-based: rise in center nearness over LOOM_WINDOW (+ = approaching)
    "flow":  float,           # flow-based: outward radial flow px/frame in center band (+ = approaching)
    "exp":   float,           # flow-based: per-frame fractional expansion (~1/TTC_frames)
    "ttc":   float,           # est. seconds-to-contact from expansion (99 = clear / not moving)
    "motion": float,          # mean abs pixel change vs prev frame (0 = view frozen -> stuck)
    "target": {"active","bearing","size","lost"}   # Stage 3 tracker; inert until a goal seed exists
  }
Also writes workspace/frame.jpg (throttled) for the voice sidecar.

Goal input (optional, Stage 3) -> workspace/nav_goal.json:
  {"seed": [x,y,w,h], "label": "..."}   # normalized bbox to seed the tracker; delete to clear

Env: PERCEPTION_FPS_CAP (12), STOP_NEAR (1000), CAM_INDEX (0), CAM_WIDTH (1280),
     CAM_HEIGHT (720), FRAME_WRITE_EVERY (0.25s), FRAME_QUALITY (80),
     PERCEPTION_LOG_EVERY (0.5s), PERCEPTION_BAND_TOP (0.45)
"""
import os
import json
import time
import signal
import cv2
import numpy as np

HOME = os.path.expanduser("~")


def env(n, d):
    v = os.environ.get(n)
    return v if v not in (None, "") else d


MODEL = os.path.join(HOME, "robot", "models", "model-small.onnx")
ENGINE = os.path.join(HOME, "robot", "models", "midas-small-fp16.trt")   # TensorRT FP16 (preferred)
WORKSPACE = env("ROBOT_WORKSPACE", os.path.join(HOME, "robot", "workspace"))
FRAME_OUT = os.path.join(WORKSPACE, "frame.jpg")      # we now PRODUCE this (was vision_sidecar's job)
FAST_OUT = os.path.join(WORKSPACE, "frame_fast.jpg")  # small, high-rate frame for the dashboard's smooth video feed
STATE = os.path.join(WORKSPACE, "nav_state.json")
GOAL = os.path.join(WORKSPACE, "nav_goal.json")

INSIZE = 256
MEAN = (123.675, 116.28, 103.53)          # MiDaS small (imagenet mean * 255)
FPS_CAP = float(env("PERCEPTION_FPS_CAP", "20"))   # raised from 12: FP16 gives the headroom, and
                                                   # fresher nav_state = lower reactive latency (old stack is off)
STOP_NEAR = float(env("STOP_NEAR", "1000"))
BOXED_NEAR = float(env("BOXED_NEAR", "600"))   # all three columns at/above this = boxed in (tight all around);
                                               # a spatial-awareness cue for the planner, not a motion threshold
CAM_INDEX = int(env("CAM_INDEX", "0"))
CAM_WIDTH = int(env("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(env("CAM_HEIGHT", "720"))
FRAME_WRITE_EVERY = float(env("FRAME_WRITE_EVERY", "0.25"))
FRAME_QUALITY = int(env("FRAME_QUALITY", "80"))
FAST_WRITE_EVERY = float(env("FAST_WRITE_EVERY", "0.0"))     # write the small teleop frame EVERY loop (~perception fps)
FAST_MAXDIM = int(env("FAST_MAXDIM", "480"))                 # small long edge: keep frames light so weak Wi-Fi can
FAST_QUALITY = int(env("FAST_QUALITY", "45"))                # actually deliver the full frame rate to the laptop
LOG_EVERY = float(env("PERCEPTION_LOG_EVERY", "0.5"))
BAND_TOP = float(env("PERCEPTION_BAND_TOP", "0.30"))   # look AHEAD (mid frame), not the floor at the wheels
BAND_BOT = float(env("PERCEPTION_BAND_BOT", "0.62"))
LOOM_WINDOW = float(env("PERCEPTION_LOOM_WINDOW", "0.35"))  # s; loom = rise in center nearness over this window
NEAR_EMA = float(env("PERCEPTION_NEAR_EMA", "0.6"))        # per-column smoothing weight on the new sample
                                                          # (1.0 = off; ~0.6 kills single-frame noise, ~1 frame lag)
FLOW_ON = env("PERCEPTION_FLOW", "1") == "1"              # optical-flow expansion (looming) detector; scale-free,
                                                          # complements MiDaS + catches transparent obstacles
FLOW_W = int(env("PERCEPTION_FLOW_W", "128"))            # flow works on a small gray (cheap, less noise than full res)
FLOW_H = int(env("PERCEPTION_FLOW_H", "72"))
FLOW_DEADZONE = float(env("PERCEPTION_FLOW_DEADZONE", "0.12"))   # drop the singular focus-of-expansion center
FLOW_WINDOW = float(env("PERCEPTION_FLOW_WINDOW", "0.6"))   # s; average expansion over this window. Per-frame flow
                                                           # is symmetric noise with a small +bias when approaching;
                                                           # only the windowed MEAN separates approach from noise.
FLOW_RATE_MIN = float(env("PERCEPTION_FLOW_RATE_MIN", "0.004"))  # windowed expansion below this = noise -> ttc clear

_run = True


def _stop(*_):
    global _run
    _run = False


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)           # keep only the latest frame -> frame.jpg tracks real
    except Exception:                                 # time instead of lagging a buffered queue (kills the
        pass                                          # "box shows where it WAS" afterimage during go_to)
    for _ in range(5):
        cap.read()                                    # let exposure settle
    return cap


def _blob(img):
    return cv2.dnn.blobFromImage(img, 1 / 255.0, (INSIZE, INSIZE), MEAN, swapRB=True, crop=False)


class Cv2Depth:
    """cv2.dnn CUDA backend (FP16 by default). Robust fallback — always available."""
    name = "cv2.dnn"

    def __init__(self):
        net = cv2.dnn.readNetFromONNX(MODEL)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        if env("PERCEPTION_FP32", "0") == "1":
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.name = "cv2.dnn FP32"
        else:
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            self.name = "cv2.dnn FP16"
        self.net = net

    def infer(self, img):
        self.net.setInput(_blob(img))
        return np.squeeze(self.net.forward())      # (256,256) inverse depth, higher = closer


class TRTDepth:
    """TensorRT FP16 engine backend (~6ms vs ~24ms for cv2.dnn FP16). Same preprocessing
    and same (256,256) output. Raises on any setup problem so caller can fall back."""
    name = "tensorrt FP16"

    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit            # noqa: F401  (creates the CUDA context)
        self._cuda = cuda
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("failed to deserialize engine")
        self.ctx = self.engine.create_execution_context()
        self.bindings = []
        self.inp = self.outp = None
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            if self.engine.binding_is_input(i):
                self.inp = (host, dev, shape)
            else:
                self.outp = (host, dev, shape)
        if self.inp is None or self.outp is None:
            raise RuntimeError("engine missing input/output binding")
        self.stream = cuda.Stream()

    def infer(self, img):
        cuda = self._cuda
        blob = _blob(img)                          # (1,3,256,256) float32, same as cv2 path
        np.copyto(self.inp[0], blob.ravel())
        cuda.memcpy_htod_async(self.inp[1], self.inp[0], self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outp[0], self.outp[1], self.stream)
        self.stream.synchronize()
        return np.squeeze(self.outp[0].reshape(self.outp[2]))   # (256,256)


def make_backend():
    """Prefer the TensorRT engine; fall back to cv2.dnn if it's absent or won't load.
    Set PERCEPTION_NO_TRT=1 to force the cv2.dnn path."""
    if env("PERCEPTION_NO_TRT", "0") != "1" and os.path.exists(ENGINE):
        try:
            b = TRTDepth(ENGINE)
            print("perception: depth backend = %s" % b.name, flush=True)
            return b
        except Exception as e:
            print("perception: TRT engine load failed (%s) -> cv2.dnn fallback" % e, flush=True)
    b = Cv2Depth()
    print("perception: depth backend = %s" % b.name, flush=True)
    return b


def free_space(depth):
    # a horizontal strip AHEAD of the car (excludes the near-floor at the wheels, which
    # otherwise dominates and hides obstacles). In this band, obstacle -> higher (closer).
    h, w = depth.shape
    band = depth[int(h * BAND_TOP):int(h * BAND_BOT), :]
    l, c, r = np.array_split(band, 3, axis=1)
    return [float(np.percentile(x, 90)) for x in (l, c, r)]


def clearness(near):
    lo, hi = min(near), max(near)
    span = (hi - lo) or 1.0
    return [round(1.0 - (n - lo) / span, 3) for n in near]


class FlowLoom:
    """Optical-flow expansion (looming) detector — a second, depth-independent way to sense
    "I'm about to hit something." As the car moves toward a surface the image expands, so
    dense flow radiates outward from the focus of expansion (roughly the heading). We measure
    the mean OUTWARD RADIAL flow in the center-ahead band:
      - pure sideways translation / yaw cancels (left pixels flow one way, right the other),
      - genuine approach shows up as net positive radial flow everywhere,
      - it needs no absolute scale, and it fires on TRANSPARENT obstacles because it tracks
        their visible contents/edges/label, not the (see-through) depth.

    Publishes each update():
      flow : median outward radial flow (px/frame) in the band; >0 = approaching, <0 = receding
      rate : per-frame fractional expansion (radial_flow / radial_distance) ~= 1/TTC_frames,
             which nav converts to a time-to-contact using the live fps.
    Only meaningful while the car is moving; ~0 when parked (no parallax)."""

    def __init__(self, w, h, band_top, band_bot, col_lo=0.20, col_hi=0.80, deadzone=0.12):
        self.w, self.h = w, h
        self.r0, self.r1 = int(h * band_top), int(h * band_bot)
        self.c0, self.c1 = int(w * col_lo), int(w * col_hi)
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.mgrid[self.r0:self.r1, self.c0:self.c1].astype(np.float32)
        dx, dy = xs - cx, ys - cy
        dist = np.sqrt(dx * dx + dy * dy)
        maxr = np.sqrt(cx * cx + cy * cy)
        self.mask = dist > (deadzone * maxr)          # exclude the singular center (FOE)
        safe = np.where(self.mask, dist, 1.0)
        self.ux, self.uy = dx / safe, dy / safe       # unit outward-radial directions
        self.rdist = dist[self.mask]                  # radial distance of the kept pixels
        self.prev = None

    def update(self, img_bgr):
        g = cv2.cvtColor(cv2.resize(img_bgr, (self.w, self.h)), cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = g
            return 0.0, 0.0
        flow = cv2.calcOpticalFlowFarneback(self.prev, g, None, 0.5, 2, 13, 2, 5, 1.1, 0)
        self.prev = g
        u = flow[self.r0:self.r1, self.c0:self.c1, 0]
        v = flow[self.r0:self.r1, self.c0:self.c1, 1]
        radial = (u * self.ux + v * self.uy)[self.mask]      # +ve = outward (expanding)
        if radial.size == 0:
            return 0.0, 0.0
        flow_px = float(np.median(radial))                    # px/frame outward
        exp_rate = float(np.median(radial / self.rdist))      # fractional expansion / frame
        return flow_px, exp_rate


def make_tracker():
    for ctor in ("legacy.TrackerCSRT_create", "TrackerCSRT_create",
                 "legacy.TrackerKCF_create", "TrackerKCF_create"):
        try:
            obj = cv2
            for part in ctor.split("."):
                obj = getattr(obj, part)
            return obj()
        except Exception:
            continue
    return None


def atomic_write(path, data, binary=False):
    tmp = path + ".tmp"
    with open(tmp, "wb" if binary else "w") as f:
        f.write(data)
    os.replace(tmp, path)


def read_goal():
    try:
        with open(GOAL) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())

    print("perception: opening camera %d @ %dx%d..." % (CAM_INDEX, CAM_WIDTH, CAM_HEIGHT), flush=True)
    cap = open_camera()
    ok, warm = cap.read()
    if not ok or warm is None:
        print("perception: FATAL — camera %d gave no frame (is another process holding it?)" % CAM_INDEX, flush=True)
        return

    print("perception: loading MiDaS-small depth backend...", flush=True)
    t0 = time.time()
    backend = make_backend()
    backend.infer(warm)                       # warm up (build graph / first launch)
    print("perception: ready in %.1fs, cap %.0f fps" % (time.time() - t0, FPS_CAP), flush=True)

    flow_det = None
    if FLOW_ON:
        flow_det = FlowLoom(FLOW_W, FLOW_H, BAND_TOP, BAND_BOT, deadzone=FLOW_DEADZONE)
        print("perception: optical-flow looming ON (%dx%d)" % (FLOW_W, FLOW_H), flush=True)

    tracker = None
    tracked_seed = None
    period = 1.0 / FPS_CAP if FPS_CAP > 0 else 0.0
    last_log = 0.0
    last_frame_write = 0.0
    last_fast_write = 0.0
    ema_fps = 0.0
    read_ema = 0.0                            # camera read() time (high => buffer lag / camera-bound)
    infer_ema = 0.0                           # depth inference time
    flow_ema = 0.0                            # optical-flow compute time
    exp_hist = []                             # (t, expansion_rate) over FLOW_WINDOW — windowed mean beats raw jitter
    miss = 0
    loom_hist = []                            # (t, center_near) for looming detection
    prev_small = None                         # downscaled gray of previous frame (scene-motion / stuck detection)
    near_ema = None                           # smoothed per-column nearness (spike-robust for nav + advance)

    while _run:
        loop_t = time.time()
        ok, img = cap.read()
        if not ok or img is None:
            miss += 1
            if miss > 30:
                print("perception: camera stalled, reopening...", flush=True)
                cap.release()
                cap = open_camera()
                miss = 0
            time.sleep(0.03)
            continue
        miss = 0
        read_ms = (time.time() - loop_t) * 1000.0
        read_ema = read_ms if read_ema == 0 else 0.9 * read_ema + 0.1 * read_ms
        H, W = img.shape[:2]

        # scene-motion: mean abs pixel change vs previous frame (downscaled gray).
        # ~0 when the view is frozen -> the car isn't actually moving (wedged on something
        # the depth band can't see). Consumed by nav.py for stuck detection.
        small = cv2.cvtColor(cv2.resize(img, (96, 54)), cv2.COLOR_BGR2GRAY)
        motion = float(np.mean(cv2.absdiff(small, prev_small))) if prev_small is not None else 0.0
        prev_small = small

        # optical-flow expansion (looming) — depth-independent approach detector (see FlowLoom)
        flow_px, exp_rate = 0.0, 0.0
        if flow_det is not None:
            t_fl = time.time()
            flow_px, exp_rate = flow_det.update(img)
            flow_ms = (time.time() - t_fl) * 1000.0
            flow_ema = flow_ms if flow_ema == 0 else 0.9 * flow_ema + 0.1 * flow_ms
            exp_hist.append((loop_t, exp_rate))
            while exp_hist and loop_t - exp_hist[0][0] > FLOW_WINDOW:
                exp_hist.pop(0)

        t_inf = time.time()
        depth = backend.infer(img)
        infer_ms = (time.time() - t_inf) * 1000.0
        infer_ema = infer_ms if infer_ema == 0 else 0.9 * infer_ema + 0.1 * infer_ms
        near = free_space(depth)
        # light per-column temporal smoothing: removes single-frame depth noise (so the
        # advance stop-threshold can't trip on one bad frame) without lagging real changes.
        if near_ema is None:
            near_ema = list(near)
        else:
            near_ema = [NEAR_EMA * n + (1.0 - NEAR_EMA) * p for n, p in zip(near, near_ema)]
        near = near_ema
        clr = clearness(near)
        clearest = "lcr"[int(np.argmax(clr))]
        blocked = near[1] >= STOP_NEAR
        # looming: rise in center-ahead nearness over LOOM_WINDOW (positive = something approaching)
        loom_hist.append((loop_t, near[1]))
        while loom_hist and loop_t - loom_hist[0][0] > LOOM_WINDOW:
            loom_hist.pop(0)
        loom = round(near[1] - loom_hist[0][1], 1) if loom_hist else 0.0

        # ---- Stage 3 scaffold: target tracking (inert unless a goal seed exists) ----
        target = {"active": False, "bearing": 0.0, "size": 0.0, "lost": False}
        goal = read_goal()
        seed = goal.get("seed") if goal else None
        if seed and seed != tracked_seed:
            tk = make_tracker()
            if tk is not None:
                x, y, w, h = seed
                try:
                    tk.init(img, (int(x * W), int(y * H), int(w * W), int(h * H)))
                    tracker, tracked_seed = tk, seed
                except Exception:
                    tracker = None
        elif not seed:
            tracker, tracked_seed = None, None

        if tracker is not None:
            tok, box = tracker.update(img)
            if tok:
                x, y, w, h = box
                cx = (x + w / 2.0) / W
                target = {"active": True, "bearing": round(cx - 0.5, 3),
                          "size": round((w * h) / float(W * H), 4), "lost": False,
                          "box": [round(x / W, 3), round(y / H, 3),   # normalized [x,y,w,h] for the dashboard overlay
                                  round(w / W, 3), round(h / H, 3)]}
            else:
                target = {"active": True, "bearing": 0.0, "size": 0.0, "lost": True}

        dt = time.time() - loop_t
        fps = 1.0 / dt if dt > 0 else 0.0
        ema_fps = fps if ema_fps == 0 else 0.9 * ema_fps + 0.1 * fps

        # time-to-contact from the WINDOWED expansion rate: 1/(rate*fps). Only trust it when the
        # scene is actually moving (parked -> no parallax) and the windowed mean clears the noise
        # floor (per-frame flow is symmetric noise; only a sustained +bias means real approach).
        rate = (sum(e for _, e in exp_hist) / len(exp_hist)) if exp_hist else 0.0
        ttc = 99.0
        if rate > FLOW_RATE_MIN and ema_fps > 0 and motion >= 1.0:
            ttc = round(min(99.0, 1.0 / (rate * ema_fps)), 2)

        atomic_write(STATE, json.dumps({
            "ts": round(loop_t, 3), "fps": round(ema_fps, 1),
            "near": {"l": round(near[0]), "c": round(near[1]), "r": round(near[2])},
            "clear": {"l": clr[0], "c": clr[1], "r": clr[2]},
            "clearest": clearest, "blocked": bool(blocked), "loom": loom,
            "flow": round(flow_px, 2), "exp": round(rate, 4), "ttc": ttc,
            "motion": round(motion, 2), "target": target,
            "boxed_in": bool(near[0] >= BOXED_NEAR and near[1] >= BOXED_NEAR and near[2] >= BOXED_NEAR),
        }))

        # throttled frame.jpg for the voice sidecar (look/ambient)
        if time.time() - last_frame_write >= FRAME_WRITE_EVERY:
            okj, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
            if okj:
                atomic_write(FRAME_OUT, buf.tobytes(), binary=True)
            last_frame_write = time.time()

        # small, high-rate frame for the dashboard's smooth video feed (teleop). Cheap encode.
        if time.time() - last_fast_write >= FAST_WRITE_EVERY:
            fh, fw = img.shape[:2]
            fscale = FAST_MAXDIM / float(max(fh, fw))
            fimg = cv2.resize(img, (int(fw * fscale), int(fh * fscale)),
                              interpolation=cv2.INTER_AREA) if fscale < 1.0 else img
            okf, fbuf = cv2.imencode(".jpg", fimg, [cv2.IMWRITE_JPEG_QUALITY, FAST_QUALITY])
            if okf:
                atomic_write(FAST_OUT, fbuf.tobytes(), binary=True)
            last_fast_write = time.time()

        if time.time() - last_log >= LOG_EVERY:
            last_log = time.time()
            tg = ""
            if target["active"]:
                tg = " | target %s bearing=%+.2f size=%.3f" % (
                    "LOST" if target["lost"] else "ok", target["bearing"], target["size"])
            print("%.1ffps read=%.0f infer=%.0f flow=%.0fms near L/C/R=%4d/%4d/%4d clearest=%s "
                  "motion=%.1f flow=%+.2f ttc=%.1f%s%s"
                  % (ema_fps, read_ema, infer_ema, flow_ema, near[0], near[1], near[2], clearest,
                     motion, flow_px, ttc, " [BLOCKED]" if blocked else "", tg), flush=True)

        if period:
            sleep = period - (time.time() - loop_t)
            if sleep > 0:
                time.sleep(sleep)

    cap.release()
    try:
        os.remove(STATE)
    except Exception:
        pass
    print("perception: stopped", flush=True)


if __name__ == "__main__":
    main()
