#!/usr/bin/env python3
"""Benchmark MiDaS-small via cv2.dnn on the Orin GPU. Make-or-break for the
reactive nav layer: does it import, run on CUDA, and hit a usable frame rate?
Reads the live camera frame the vision sidecar writes, so it measures the real
pipeline (imread + preprocess + inference + a coarse free-space readout)."""
import os
import time
import cv2
import numpy as np

HOME = os.path.expanduser("~")
MODEL = os.path.join(HOME, "robot", "models", "model-small.onnx")
FRAME = os.path.join(HOME, "robot", "workspace", "frame.jpg")
INSIZE = 256
N = 60

# MiDaS small preprocessing (std omitted — blobFromImage can't divide per-channel;
# fine for relative depth + a speed benchmark).
MEAN = (123.675, 116.28, 103.53)


def load_net(use_cuda):
    net = cv2.dnn.readNetFromONNX(MODEL)
    if use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def infer(net, img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (INSIZE, INSIZE), MEAN, swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()          # (1,256,256) or (256,256)
    return np.squeeze(out)


def freespace(depth):
    """MiDaS output is inverse depth: larger = CLOSER. Report the max (nearest)
    value in the lower-center of each third — that's 'how close is the nearest
    thing ahead on the left/center/right'."""
    h, w = depth.shape
    band = depth[int(h * 0.45):, :]          # lower ~half of the frame = the floor/path ahead
    thirds = np.array_split(band, 3, axis=1)
    return [float(np.percentile(t, 95)) for t in thirds]   # 95th pct = nearest, noise-robust


def main():
    img = cv2.imread(FRAME)
    if img is None:
        print("no frame at", FRAME, "- is the vision sidecar running? using a gray test image")
        img = np.full((480, 640, 3), 128, np.uint8)
    print("input frame:", img.shape)

    for use_cuda in (True, False):
        tag = "CUDA" if use_cuda else "CPU"
        try:
            t0 = time.time()
            net = load_net(use_cuda)
            depth = infer(net, img)          # warmup (includes CUDA graph build)
            warm = time.time() - t0
            print("[%s] loaded+warmup %.2fs  out shape=%s" % (tag, warm, depth.shape))
            t0 = time.time()
            for _ in range(N):
                depth = infer(net, img)
            dt = time.time() - t0
            fps = N / dt
            dmin, dmax = float(depth.min()), float(depth.max())
            l, c, r = freespace(depth)
            print("[%s] %.1f fps (%.1f ms/frame)  depth[min=%.1f max=%.1f]  near L/C/R = %.0f / %.0f / %.0f"
                  % (tag, fps, 1000 / fps, dmin, dmax, l, c, r))
        except Exception as e:
            print("[%s] FAILED: %r" % (tag, e))


if __name__ == "__main__":
    main()
