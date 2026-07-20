# R2-Wheel2

A little robot car I built that you can talk to and that drives itself around using its
camera. It runs on an NVIDIA Jetson Orin Nano. I started it mostly as an excuse to learn
robotics and play with the new realtime voice AI stuff at the same time.

It's a work in progress, but it already listens, talks back, and sees well enough to know
when something is in front of it.

This is a personal side project I built in my own time. It isn't affiliated with anyone I
work for.

## What it can do right now

You talk and it drives. I connected it to the OpenAI Realtime API, so you can say something
like "move forward for a second" or "stop" and it just does it. It replies out loud in about
a second, and you can talk over it and it will stop and listen.

It sees. There's a depth model running on the Jetson's GPU that works out roughly how far
away things are from a single camera, about 22 times a second. So it has a real sense of
"wall on the left, open space ahead."

It tries not to hurt itself or me. The motor firmware stops on its own if it stops getting
commands, and there's a kill switch that halts everything instantly.

## The main idea

My first instinct was to just send the camera feed to the cloud AI and let it drive. That
does not work. There's about a one second round trip, so the car decides to stop roughly a
meter after it should have, and streaming video to the cloud is slow and expensive.

So I split the brain into two parts running at two different speeds. The cloud part is slow,
around one second, and it handles understanding what you want and deciding the goal. The
part running on the robot itself is fast, around 22 frames a second, and it handles the
reflexes, like not driving into things. The camera and the navigation never leave the robot.
The cloud decides what to do, and the robot handles not crashing. That split is basically the
whole trick.

## The parts

The brain is an NVIDIA Jetson Orin Nano, running headless over wifi. The motor controller is
a Keyestudio MAX, which is an Arduino Uno compatible board, driving two L298N H-bridges. The
wheels are four DC motors with mecanum wheels, so it can strafe sideways, which is fun. The
eyes are a plain USB webcam, and the ears and mouth are a Jabra USB speakerphone. For power,
the motors run off a 12V battery and the Jetson has its own separate supply, which I learned
the hard way (more on that below).

## How the code is laid out

- firmware/mecanum_uno: the Arduino code that actually turns the wheels
- motor/move: a script to drive it, like "./move forward 0.8 1.5"
- motor/testwheel: spins one wheel at a time, for figuring out the wiring
- perception/perception.py: the seeing loop, camera to depth to free space
- perception/depth_bench.py: a quick benchmark I wrote to check the GPU was fast enough
- voice/realtime_sidecar.py: the talk to it voice loop

## How it works

The firmware listens for tiny text commands over USB (V means move like this, M means spin
one specific wheel, S means stop) and does the mecanum wheel math. The safety bit that
matters most: if it doesn't hear a command for 300 milliseconds it stops the motors by
itself, so if the software crashes or the cable pops out, the car doesn't drive off into a
wall.

For seeing, perception.py grabs frames from the camera and runs a depth model called MiDaS on
the GPU using OpenCV. That was the scary part, because I wasn't sure the little Jetson could
run it fast enough to be useful. So before writing any driving logic I wrote depth_bench.py
just to check, and it hit about 26 frames a second, which was a big relief. It turns each
frame into a simple readout of how close the nearest thing is on the left, middle, and right.

For talking, realtime_sidecar.py opens one connection to the OpenAI Realtime API and streams
the microphone up and the voice down. The AI can call drive, stop, and look (take a photo) as
tools. One thing that took me a while to figure out: I was doing the image processing on the
same thread as the audio, and it made the voice stutter. Moving that work to a background
thread fixed it.

## Stuff that went wrong

I killed two ESP32 boards before switching to the Arduino compatible one. The first board's
flash chip died, I think from mechanical stress when I boxed everything up, and the second one
shorted and overheated while I was rewiring it.

I also spent ages convinced a board was dead because it wouldn't show up over USB, and it
turned out the USB cable was charge only. Now that's the first thing I check.

The one that really got me was random reboots under load. It turned out I was running the
Jetson off the same tired 12V battery as the motors, and the current spikes from the GPU and
the motors together sagged the voltage enough to reset the whole computer. The lesson was to
give the computer and the motors their own power. Once I did that it's been rock solid.

## Running it

```sh
# flash the Arduino firmware
arduino-cli upload --fqbn arduino:avr:uno -p /dev/ttyUSB0 firmware/mecanum_uno

# start the seeing loop (needs a MiDaS-small ONNX model in models/)
python3 perception/perception.py

# start the voice loop (needs an OpenAI API key, kept out of the repo)
OPENAI_KEY_FILE=~/secrets/openai.key python3 voice/realtime_sidecar.py
```

You'll need OpenCV built with CUDA, NumPy, websockets, and PulseAudio. The depth model and
any API keys are deliberately not in this repo (see .gitignore), so you bring your own.

## What I want to add next

- Actually let it drive around obstacles. Right now it sees them, it just doesn't steer
  around them yet.
- "Go to the doorway", where the voice AI picks a target and the robot navigates to it.
- Get a proper depth camera so the distances are real measurements instead of estimates, and
  maybe try mapping a room.

Thanks for reading. This has been the most fun I've had learning something in a while.
