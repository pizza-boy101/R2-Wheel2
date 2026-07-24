# Setting up R2-Wheel2

A step-by-step guide to getting the robot running from scratch. If you just want to know *what*
it is and *how* it works, read the [README](README.md) first — this doc is the *how do I run it*
tutorial.

The short version: an Arduino drives the wheels and reads a distance sensor; a Linux single-board
computer (a Jetson, or a Raspberry Pi) does the seeing and the talking. You'll flash the Arduino,
install some Python, drop in a depth model and two API keys, then start five small programs.

Budget about an afternoon the first time.

---

## 1. What you'll need

**Hardware**
- A Linux single-board computer with a USB port and Wi-Fi — an **NVIDIA Jetson** (what I used) or a
  **Raspberry Pi 5**. Anything that runs Python 3.8+ and OpenCV will do; a Jetson just makes the
  depth model fast (see [the Pi note](#10-running-on-a-raspberry-pi-instead) at the end).
- An **Arduino Uno-compatible board** (I used a Keyestudio MAX) + **two L298N motor drivers**.
- **4 DC motors with mecanum wheels** (regular wheels are fine too — you just lose sideways strafing).
- A **USB webcam**, a **USB speakerphone** (mic + speaker in one is easiest), and an
  **HC-SR04 ultrasonic distance sensor**.
- **Two power sources**: one for the motors (a 12V battery) and a **separate** one for the computer.
  Do not share them — see [Troubleshooting](#9-troubleshooting).

**Accounts / keys**
- An **OpenAI API key** with Realtime API access (the voice).
- An **Anthropic API key** (the "find things" and "think when stuck" smarts).

**On the computer**
- `python3` (3.8+), `pip`, `git`
- `arduino-cli` (to flash the firmware)
- **PulseAudio** (for the mic/speaker) — standard on most desktop Linux images

---

## 2. Get the code

```sh
git clone https://github.com/pizza-boy101/R2-Wheel2.git
cd R2-Wheel2
```

The programs read and write a few things from fixed folders under `~/robot/` by default
(a scratch `workspace/`, the depth `models/`, and your `secrets/`). Create them now:

```sh
mkdir -p ~/robot/workspace ~/robot/models ~/robot/secrets
```

(You can point them elsewhere with the `ROBOT_WORKSPACE`, and the model/key path env vars shown
later, but the defaults keep things simple.)

---

## 3. Wire it up

Follow the "parts" section of the [README](README.md) for the full picture. The essentials:

- Each L298N channel drives one motor: an **EN** pin (speed/PWM) plus **INA/INB** (direction). The
  pin mapping lives at the top of `firmware/mecanum_uno/mecanum_uno.ino` — match your wiring to it
  (or edit the arrays to match your wiring).
- The **HC-SR04** goes on the Arduino: **Trig → A0**, **Echo → A1**, plus **5V** and **GND**. It's
  wired to the Arduino (not the computer) because the Arduino is 5V-native and reads it directly.
- The webcam and speakerphone plug into the computer over USB.
- **Motors run off the 12V battery; the computer gets its own supply.**

> If your board has onboard demo buttons/LEDs/buzzer on the motor pins (the Keyestudio does),
> switch them **off** at the board's DIP switch before wiring, or they'll fight the motor signals.

---

## 4. Flash the Arduino firmware

Install `arduino-cli` and the AVR core, then upload:

```sh
arduino-cli core update-index
arduino-cli core install arduino:avr

# find the port (usually /dev/ttyUSB0 or /dev/ttyACM0)
arduino-cli board list

# compile + upload
arduino-cli compile --fqbn arduino:avr:uno firmware/mecanum_uno
arduino-cli upload  --fqbn arduino:avr:uno -p /dev/ttyUSB0 firmware/mecanum_uno
```

**Calibrate the wheels once** (the firmware's top comment walks through it): send test commands and
flip the `INVERT[]` entries until a positive command drives each wheel *forward*, then check that
"turn" and "strafe" go the right way. You can send commands the easy way once the motor daemon is
running (next section) with `motor/move` and `motor/testwheel`.

If the car drifts to one side when driving straight, nudge the `TRIM[]` values (1.0 = full power;
lower = slower) — slow down the wheels on the side that's outrunning the other, then re-flash.

---

## 5. Install the Python bits

```sh
# OpenCV (with contrib for the object tracker), plus the rest
pip3 install "opencv-contrib-python" numpy websockets

# audio (Debian/Ubuntu/Jetson)
sudo apt-get install -y pulseaudio pulseaudio-utils
```

On a **Jetson**, use the system OpenCV built with CUDA instead of the pip wheel if you have it —
that's what makes the depth model fast. On a **Pi**, the pip wheel is fine (depth just runs slower;
see the Pi note).

The debug dashboard needs **nothing** beyond the standard library.

---

## 6. Add the depth model

The seeing loop runs **MiDaS-small**. Put a copy in `~/robot/models/`:

- **CPU / simplest:** download the `model-small.onnx` MiDaS-small model and save it as
  `~/robot/models/model-small.onnx`. The code runs it on the CPU via OpenCV automatically.
- **Jetson / fast:** convert that ONNX to a TensorRT FP16 engine and save it as
  `~/robot/models/midas-small-fp16.trt`. The code prefers the engine when it's present.

Check it's working and fast enough before wiring up any driving:

```sh
python3 perception/depth_bench.py
```

You want comfortably into double-digit FPS. If it's a slideshow, you're on the CPU path — fine for a
Pi, but expect slower reactions.

---

## 7. Add your API keys

Save each key to its own file under `~/robot/secrets/` (these folders are git-ignored — never commit
keys):

```sh
printf '%s' 'sk-...your-openai-key...'      > ~/robot/secrets/openai-realtime.key
printf '%s' 'sk-ant-...your-anthropic-key...' > ~/robot/secrets/anthropic.key
chmod 600 ~/robot/secrets/*.key
```

(Paths are configurable via `OPENAI_KEY_FILE` and `ANTHROPIC_KEY_FILE` if you keep them elsewhere.)

---

## 8. Run it

Start these in separate terminals (or as services — see below). Order matters a little: motors and
seeing first, then the rest.

```sh
# 1) the ONLY program that talks to the motor board — everything else sends it commands
python3 motor/motor_daemon.py

# 2) the seeing loop: camera -> depth -> free space, and the camera frames the rest of the stack uses
python3 perception/perception.py

# 3) (optional) let it wander and avoid things on its own
python3 perception/nav.py

# 4) the voice: talk to it, and it drives. needs the OpenAI key (+ Anthropic for find/think)
python3 voice/realtime_sidecar.py

# 5) (optional but recommended) the debug dashboard
./debug/run.sh     # then open http://<robot-ip>:8099 in a browser on the same network
```

Sanity check with the dashboard open: you should see the camera view, the left/middle/right
closeness bars moving as you wave a hand, and the front distance in centimetres. The health strip
across the top shows which programs are alive.

**Drive it by hand first** (before trusting the voice): open the dashboard, hit **ARM**, and use the
on-screen d-pad or the arrow keys. Space stops; **Esc is the emergency stop**. Disarm when you're
done.

**Then talk to it:** hit **voice on** in the dashboard (it opens the mic + the paid Realtime API),
and try "move forward for a second", "come here and stop when you're close", or "find the red mug and
go to it". You can also type to it in the dashboard's chat box.

---

## 9. Troubleshooting

These are the ones that actually bit me:

- **The board won't show up over USB.** First suspect the cable — a lot of USB cables are
  charge-only with no data lines. Try another cable before assuming the board is dead.
- **Random reboots / brownouts under load.** The motors and the computer must have **separate**
  power. Sharing one battery lets the motor current spikes sag the voltage and reset the computer.
- **It drives into clear plastic or glass.** A single camera sees straight through it. That's what
  the ultrasonic sensor and the "how fast is the picture growing" cue are for — make sure the sonar
  is wired and reading (check the distance card on the dashboard).
- **Distance always reads "clear" / never changes.** Check the HC-SR04's 5V, GND, and that Trig/Echo
  are on A0/A1. It reports up the same USB serial line as the motors.
- **No voice / it can't hear you.** Confirm PulseAudio sees your speakerphone
  (`pactl list short sources` / `sinks`), and that the OpenAI key file exists and is valid.
- **It won't move at all.** Check it's **armed** (dashboard), and that the kill-switch file
  (`~/robot/workspace/.disarmed`) isn't present.

---

## 10. Running it as services (optional)

Once it all works, it's nicer to have the programs start on boot and restart if they crash. Any
service manager works; with **systemd user services** you'd create one unit per program
(`~/.config/systemd/user/`) that runs the matching command from section 8, with
`Restart=on-failure`, then `systemctl --user enable --now` each. Keep the voice one **manual** (you
probably don't want the paid mic running unattended) and start it from the dashboard's "voice on"
button.

---

## 11. Running on a Raspberry Pi instead

Almost everything here is plain Python and an Arduino that doesn't care what the brain is, so it
ports well. The one real difference: the depth model has no GPU to accelerate it, so it runs on the
CPU (single-digit FPS on a Pi). Lean on the **ultrasonic sensor** for forward collision (it's
independent of the brain and runs full speed), use a **Pi 5** for the best CPU, and consider a couple
more HC-SR04s for side coverage. Everything else — motors, voice, the tracker, the dashboard — runs
unchanged.

---

That's it. If something's unclear, the [README](README.md) explains the design and why each piece is
there. Have fun.
