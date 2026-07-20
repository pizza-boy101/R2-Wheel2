# R2-Wheel2 — a voice-driven, self-navigating mecanum robot car

A hobby autonomous ground robot built on an **NVIDIA Jetson Orin Nano**. You talk to
it and it drives; it sees the world through a single camera and a monocular-depth model
running on its GPU. It's a study in **splitting a robot's "brain" across two clocks** —
a cloud voice model for intent, and an on-board perception loop for reflexes.

> Personal project. Not affiliated with any employer. All code here is my own hobby work.

---

## What it does

- **Talk to it, and it moves.** A full-duplex voice loop (OpenAI Realtime API) listens,
  reasons, and speaks back with sub-second latency, driving the car through
  function-calls (`drive` / `stop` / `look`). Barge-in supported — interrupt it and it
  stops talking.
- **It sees.** A monocular depth model (MiDaS-small) runs on the Jetson GPU at ~22 fps,
  turning the camera feed into a live free-space estimate (how close is the nearest
  obstacle, left / center / right).
- **It won't drive blind.** Firmware enforces a 300 ms command watchdog, and a software
  kill switch (`.disarmed`) can block all motion instantly.

## The idea: a two-clock brain

Navigating through the cloud alone doesn't work — the ~1 s round-trip means the robot
decides to stop a meter after it should have, and streaming every frame up is slow and
expensive. So the brain is split:

```
          ┌─────────────────────────── cloud (intermittent, ~1 s) ───────────────────────────┐
  voice ─▶ │  OpenAI Realtime API: understands speech, sets intent, calls drive/stop/look     │
          └──────────────────────────────────────────┬───────────────────────────────────────┘
                                                      │ goals / commands
          ┌───────────────────────────────────────── ▼ ─── on-board (continuous, ~22 fps) ────┐
 camera ─▶ │  perception.py:  MiDaS depth ──▶ free-space + (optional) target tracker           │
          │  nav loop (WIP):  reactive obstacle avoidance + goal-directed servoing              │
          └──────────────────────────────────────────┬───────────────────────────────────────┘
                                                      │ serial (V/M/S @ 115200)
          ┌───────────────────────────────────────── ▼ ────────────────────────────────────────┐
   motors ◀ │  ATmega328P firmware:  mecanum inverse kinematics + 300 ms watchdog failsafe       │
          └────────────────────────────────────────────────────────────────────────────────────┘
```

The cloud sets *intent* ("head toward the doorway"); the on-board loop handles the
*reflexes* (don't hit the wall) at frame rate, with no network in the control path.

## Hardware

| Part | Component |
|------|-----------|
| Compute | NVIDIA Jetson Orin Nano (JetPack 5 / L4T R35), headless over Wi-Fi |
| Microcontroller | Keyestudio MAX (Arduino Uno-compatible, ATmega328P, CP2102 USB-UART) |
| Motor drivers | 2× L298N dual H-bridge |
| Drivetrain | 4× DC gear motors on mecanum wheels (X-roller layout) |
| Camera | USB webcam (Creative Live! Cam Sync 1080p V2) |
| Audio | Jabra Speak 510 USB speakerphone (mic + speaker) |
| Power | 12 V pack for motors; Jetson on its own supply (see note below) |

> **Power lesson learned the hard way:** never share one battery between the compute and
> the motors. Motor inrush current sags the shared rail and browns-out the Jetson mid-GPU-
> spike. Compute and motors get separate rails.

## Repository layout

```
firmware/mecanum_uno/   Arduino firmware: serial command interpreter + mecanum kinematics
motor/move              board-agnostic drive script (V/M/S protocol, kill switch)
motor/testwheel         per-wheel bring-up / calibration helper
perception/perception.py   the always-on vision loop: depth → free-space → nav_state.json
perception/depth_bench.py  GPU depth-model benchmark (cv2.dnn CUDA)
voice/realtime_sidecar.py  the voice+drive loop (OpenAI Realtime API)
```

## How it works

### Firmware (`firmware/mecanum_uno`)
An ATmega328P speaks a tiny line protocol over USB serial @ 115200:
- `V vx vy w` — body velocity (strafe, forward, yaw), each −1..1, run through mecanum
  inverse kinematics and scaled so no wheel clips.
- `M fl fr rl rr` — direct per-wheel drive (for calibration).
- `S` — stop.
Motors auto-stop if no valid command arrives within 300 ms, so a dropped link or a
crashed host can never leave the car driving away.

### Motor control (`motor/`)
`move forward 0.8 1.5` drives forward at 0.8 for 1.5 s (re-sending faster than the
watchdog so it keeps rolling), then stops. `move disarm` engages a kill switch that
blocks all motion until `move arm`. `testwheel` spins one wheel at a time to identify
wiring and direction.

### Perception (`perception/`)
`perception.py` owns the camera, runs MiDaS-small on the GPU via OpenCV's CUDA DNN
backend, and reduces each depth map to a compact `nav_state.json` (nearest obstacle per
column + which direction is most open). It also publishes a throttled `frame.jpg` for
the voice loop. `depth_bench.py` was the go/no-go test — it proved the model imports and
runs at ~26 fps on the Orin before any control logic was written.

### Voice (`voice/`)
`realtime_sidecar.py` opens one websocket to the OpenAI Realtime API, streams mic audio
up and voice audio down, and exposes `drive` / `stop` / `look` as function tools that
bridge to the same `move` script. Vision is *hybrid*: an on-demand `look` (crisp photo)
plus a silent, change-gated ambient frame push, so the model stays aware without
paying to stream every frame. All OpenCV work runs off the audio event loop so playback
never stutters.

## Running it

```sh
# firmware: flash with arduino-cli (FQBN arduino:avr:uno) to the ATmega328P board
arduino-cli upload --fqbn arduino:avr:uno -p /dev/ttyUSB0 firmware/mecanum_uno

# perception (needs a MiDaS-small ONNX in models/ — see isl-org/MiDaS v2_1)
python3 perception/perception.py

# voice (needs an OpenAI API key; provide it out-of-band, never commit it)
OPENAI_KEY_FILE=~/secrets/openai.key python3 voice/realtime_sidecar.py
```

Depends on: OpenCV (with CUDA), NumPy, `websockets`, PulseAudio (`parec`/`paplay`), and
the Jetson CUDA stack. The MiDaS ONNX model and all API keys are intentionally **not**
in the repo (see `.gitignore`).

## Build timeline

*(Commit history reconstructs the real development order; dates are approximate.)*

1. **Motor bring-up** — mecanum firmware + serial driver; first controlled motion.
2. **Calibration** — per-wheel identification and direction/inversion mapping.
3. **Voice control** — full-duplex voice→drive via the OpenAI Realtime API.
4. **Perception** — monocular depth on the GPU; the free-space foundation for navigation.

## Roadmap

- [ ] Reactive nav loop: creep + stop/steer on obstacles (threshold calibrated vs. a wall).
- [ ] Goal-directed navigation: cloud names a target, a local tracker servos toward it
      while the depth layer vetoes unsafe motion.
- [ ] Depth-camera upgrade (RealSense / OAK-D) for metric depth and eventual mapping.
