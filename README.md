# R2-Wheel2 🤖🛞

A little robot car I built that you can **talk to** and that **drives itself around** using
its camera. It runs on an NVIDIA Jetson Orin Nano, and the whole thing started because I
wanted an excuse to learn robotics and mess with the new realtime voice AI stuff at the
same time.

It's very much a work-in-progress hobby project — but it already listens, talks back, and
sees well enough to know when something's in front of it.

> Personal side project, built in my own time. Not affiliated with anyone I work for.

---

## What it can do right now

- 🎙️ **You talk, it drives.** I hooked it up to the OpenAI Realtime API, so you just say
  "move forward for a sec" or "stop" and it does it — replies out loud in about a second,
  and you can cut it off mid-sentence and it'll shut up and listen.
- 👀 **It sees.** There's a depth model running on the Jetson's GPU that figures out roughly
  how far away stuff is from a single camera, ~22 times a second. So it has a real sense of
  "wall on the left, open space ahead."
- 🛑 **It tries not to hurt itself (or me).** The motor firmware auto-stops if it stops
  getting commands, and there's a kill switch that halts everything instantly.

## The main idea I'm proud of

My first instinct was "just send the camera to the cloud AI and let it drive." That does
**not** work — there's about a 1-second round trip, so the car decides to stop about a meter
*after* it should have, and streaming video to the cloud is slow and pricey.

So I split the brain into two parts running at two different speeds:

```
  you talk ─▶  CLOUD (slow, ~1s):  the voice AI understands what you want
                                    and decides the goal → "go that way"
                    │
                    ▼
  camera ──▶  ON THE ROBOT (fast, ~22fps):  depth model + navigation loop
                                    handle the reflexes → "don't hit that"
                    │
                    ▼
  motors ◀── the Arduino runs the wheels + a safety timeout
```

The cloud handles *what to do*; the robot itself handles *not crashing*, at camera speed,
with nothing going over the network. That split is basically the whole trick.

## The parts

| Bit | What I used |
|-----|-------------|
| Brain | NVIDIA Jetson Orin Nano (runs headless over Wi-Fi) |
| Motor controller | Keyestudio MAX (an Arduino Uno-compatible board) |
| Motor drivers | 2× L298N H-bridges |
| Wheels | 4 DC motors + mecanum wheels (so it can strafe sideways, which is cool) |
| Eyes | a USB webcam |
| Ears + mouth | a Jabra USB speakerphone |
| Power | 12V battery for the motors, separate supply for the Jetson (learned this the hard way, see below) |

## How the code is laid out

```
firmware/mecanum_uno/   the Arduino code that actually turns the wheels
motor/move              a script to drive it ("./move forward 0.8 1.5")
motor/testwheel         spins one wheel at a time (for figuring out the wiring)
perception/perception.py   the "seeing" loop: camera → depth → free space
perception/depth_bench.py  a quick benchmark I wrote to check the GPU was fast enough
voice/realtime_sidecar.py  the talk-to-it voice loop
```

### The firmware
The Arduino listens for tiny text commands over USB (`V` = move like this, `M` = spin this
one wheel, `S` = stop) and does the mecanum wheel math. The important safety bit: if it
doesn't hear a command for 300ms it stops the motors on its own, so if the software crashes
or the cable pops out, the car doesn't just drive off into a wall.

### Seeing (perception)
`perception.py` grabs frames from the camera and runs **MiDaS** (a depth model) on the GPU
using OpenCV. That was the scary part — I wasn't sure the little Jetson could run it fast
enough to be useful. So before writing any driving logic I wrote `depth_bench.py` just to
check, and it hit ~26 fps, which was a huge relief. It turns each frame into a simple
"how close is the nearest thing on the left / middle / right" readout.

### Talking (voice)
`realtime_sidecar.py` opens one connection to the OpenAI Realtime API and streams the mic up
and the voice down. The AI can call `drive`, `stop`, and `look` (take a photo) as tools. One
thing that took me a while: I was doing the image processing on the same thread as the audio,
and it made the voice stutter — moving that to a background thread fixed it.

## Stuff that went wrong (the honest section)

- 💀 **I killed two ESP32 boards.** The first one's flash chip died (I think from mechanical
  stress when I boxed everything up), the second one shorted and overheated while I was
  rewiring it. I switched to the Arduino-compatible board after that.
- 🔌 **A "broken robot" that was just a bad cable.** Spent ages convinced a board was dead
  because it wouldn't show up over USB — turned out the USB cable was charge-only. Now that's
  the first thing I check.
- 🔋 **Brownouts.** The robot kept randomly rebooting under load. Turned out I was running the
  Jetson off the same tired 12V battery as the motors, and the current spikes from the GPU +
  motors sagged the voltage enough to reset the whole computer. **Lesson: give the computer
  and the motors their own power.** Fixed it and it's been rock solid since.

## Running it

```sh
# flash the Arduino firmware
arduino-cli upload --fqbn arduino:avr:uno -p /dev/ttyUSB0 firmware/mecanum_uno

# start the seeing loop (needs a MiDaS-small ONNX model in models/)
python3 perception/perception.py

# start the voice loop (needs an OpenAI API key — kept out of the repo!)
OPENAI_KEY_FILE=~/secrets/openai.key python3 voice/realtime_sidecar.py
```

You'll need OpenCV (built with CUDA), NumPy, `websockets`, and PulseAudio. The depth model
and any API keys are deliberately **not** in this repo (see `.gitignore`) — you bring your own.

## What I want to add next

- [ ] Actually let it drive itself around obstacles (right now it *sees* them, it doesn't
      steer around them yet).
- [ ] "Go to the doorway" — have the voice AI pick a target and let the robot navigate to it.
- [ ] Get a proper depth camera so the distances are real measurements, not estimates — and
      maybe try mapping a room.

Thanks for reading! This has been the most fun I've had learning something in a while. 🛠️
