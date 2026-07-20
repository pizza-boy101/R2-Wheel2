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
WORKSPACE = env("ROBOT_WORKSPACE", os.path.join(HOME, "robot", "workspace"))
FRAME_OUT = os.path.join(WORKSPACE, "frame.jpg")      # we now PRODUCE this (was vision_sidecar's job)
STATE = os.path.join(WORKSPACE, "nav_state.json")
GOAL = os.path.join(WORKSPACE, "nav_goal.json")

INSIZE = 256
MEAN = (123.675, 116.28, 103.53)          # MiDaS small (imagenet mean * 255)
FPS_CAP = float(env("PERCEPTION_FPS_CAP", "12"))
STOP_NEAR = float(env("STOP_NEAR", "1000"))
CAM_INDEX = int(env("CAM_INDEX", "0"))
CAM_WIDTH = int(env("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(env("CAM_HEIGHT", "720"))
FRAME_WRITE_EVERY = float(env("FRAME_WRITE_EVERY", "0.25"))
FRAME_QUALITY = int(env("FRAME_QUALITY", "80"))
LOG_EVERY = float(env("PERCEPTION_LOG_EVERY", "0.5"))
BAND_TOP = float(env("PERCEPTION_BAND_TOP", "0.45"))

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


def load_net():
    net = cv2.dnn.readNetFromONNX(MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def infer_depth(net, img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (INSIZE, INSIZE), MEAN, swapRB=True, crop=False)
    net.setInput(blob)
    return np.squeeze(net.forward())          # (256,256) inverse depth, higher = closer


def free_space(depth):
    h, w = depth.shape
    band = depth[int(h * BAND_TOP):, :]
    l, c, r = np.array_split(band, 3, axis=1)
    return [float(np.percentile(x, 95)) for x in (l, c, r)]


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

    print("perception: loading MiDaS-small on CUDA...", flush=True)
    t0 = time.time()
    net = load_net()
    infer_depth(net, warm)                    # build CUDA graph (~6s one-time)
    print("perception: ready in %.1fs, cap %.0f fps" % (time.time() - t0, FPS_CAP), flush=True)

    tracker = None
    tracked_seed = None
    period = 1.0 / FPS_CAP if FPS_CAP > 0 else 0.0
    last_log = 0.0
    last_frame_write = 0.0
    ema_fps = 0.0
    miss = 0

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

        depth = infer_depth(net, img)
        near = free_space(depth)
        clr = clearness(near)
        clearest = "lcr"[int(np.argmax(clr))]
        blocked = near[1] >= STOP_NEAR

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
            "clearest": clearest, "blocked": bool(blocked), "target": target,
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
            print("%.1ffps near L/C/R=%4d/%4d/%4d clearest=%s%s%s"
                  % (ema_fps, near[0], near[1], near[2], clearest,
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
