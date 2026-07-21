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
STATE = os.path.join(WORKSPACE, "nav_state.json")
GOAL = os.path.join(WORKSPACE, "nav_goal.json")

INSIZE = 256
MEAN = (123.675, 116.28, 103.53)          # MiDaS small (imagenet mean * 255)
FPS_CAP = float(env("PERCEPTION_FPS_CAP", "20"))   # raised from 12: FP16 gives the headroom, and
                                                   # fresher nav_state = lower reactive latency (old stack is off)
STOP_NEAR = float(env("STOP_NEAR", "1000"))
CAM_INDEX = int(env("CAM_INDEX", "0"))
CAM_WIDTH = int(env("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(env("CAM_HEIGHT", "720"))
FRAME_WRITE_EVERY = float(env("FRAME_WRITE_EVERY", "0.25"))
FRAME_QUALITY = int(env("FRAME_QUALITY", "80"))
LOG_EVERY = float(env("PERCEPTION_LOG_EVERY", "0.5"))
BAND_TOP = float(env("PERCEPTION_BAND_TOP", "0.30"))   # look AHEAD (mid frame), not the floor at the wheels
BAND_BOT = float(env("PERCEPTION_BAND_BOT", "0.62"))
LOOM_WINDOW = float(env("PERCEPTION_LOOM_WINDOW", "0.35"))  # s; loom = rise in center nearness over this window
NEAR_EMA = float(env("PERCEPTION_NEAR_EMA", "0.6"))        # per-column smoothing weight on the new sample
                                                          # (1.0 = off; ~0.6 kills single-frame noise, ~1 frame lag)

_run = True


def _stop(*_):
    global _run
    _run = False


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
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

    tracker = None
    tracked_seed = None
    period = 1.0 / FPS_CAP if FPS_CAP > 0 else 0.0
    last_log = 0.0
    last_frame_write = 0.0
    ema_fps = 0.0
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
        H, W = img.shape[:2]

        # scene-motion: mean abs pixel change vs previous frame (downscaled gray).
        # ~0 when the view is frozen -> the car isn't actually moving (wedged on something
        # the depth band can't see). Consumed by nav.py for stuck detection.
        small = cv2.cvtColor(cv2.resize(img, (96, 54)), cv2.COLOR_BGR2GRAY)
        motion = float(np.mean(cv2.absdiff(small, prev_small))) if prev_small is not None else 0.0
        prev_small = small

        depth = backend.infer(img)
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
                          "size": round((w * h) / float(W * H), 4), "lost": False}
            else:
                target = {"active": True, "bearing": 0.0, "size": 0.0, "lost": True}

        dt = time.time() - loop_t
        fps = 1.0 / dt if dt > 0 else 0.0
        ema_fps = fps if ema_fps == 0 else 0.9 * ema_fps + 0.1 * fps

        atomic_write(STATE, json.dumps({
            "ts": round(loop_t, 3), "fps": round(ema_fps, 1),
            "near": {"l": round(near[0]), "c": round(near[1]), "r": round(near[2])},
            "clear": {"l": clr[0], "c": clr[1], "r": clr[2]},
            "clearest": clearest, "blocked": bool(blocked), "loom": loom,
            "motion": round(motion, 2), "target": target,
        }))

        # throttled frame.jpg for the voice sidecar (look/ambient)
        if time.time() - last_frame_write >= FRAME_WRITE_EVERY:
            okj, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
            if okj:
                atomic_write(FRAME_OUT, buf.tobytes(), binary=True)
            last_frame_write = time.time()

        if time.time() - last_log >= LOG_EVERY:
            last_log = time.time()
            tg = ""
            if target["active"]:
                tg = " | target %s bearing=%+.2f size=%.3f" % (
                    "LOST" if target["lost"] else "ok", target["bearing"], target["size"])
            print("%.1ffps near L/C/R=%4d/%4d/%4d clearest=%s motion=%.1f%s%s"
                  % (ema_fps, near[0], near[1], near[2], clearest, motion,
                     " [BLOCKED]" if blocked else "", tg), flush=True)

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
