# R2-Wheel2

A little robot car I built that you can talk to and that drives itself around using its
camera. It runs on an NVIDIA Jetson Orin Nano. I started it mostly as an excuse to learn
robotics and play with the new realtime voice AI stuff at the same time.

It's a work in progress, but it already listens, talks back, sees well enough to know when
something is in front of it, and drives itself around things instead of into them.

This is a personal side project I built in my own time. It isn't affiliated with anyone I
work for.

## What it can do right now

You talk and it drives. I connected it to the OpenAI Realtime API, so you can say something
like "move forward for a second" or "stop" and it just does it. If you say "come here and
stop when you get close" it rolls toward you and stops on its own; if you say "head over
there" it rolls off and steers around whatever's in the way. It replies out loud in about a
second, and you can talk over it and it will stop and listen.

It sees. There's a depth model running on the Jetson's GPU that works out roughly how far
away things are from a single camera, about 29 times a second. So it has a real sense of
"wall on the left, open space ahead." It also watches how fast the picture is swelling as it
moves, which is a second way to tell something is rushing up even when the distance guess is
unreliable.

It also feels. I added a little ultrasonic distance sensor pointing straight ahead, under the
camera. Unlike the camera it gives a real measurement in centimetres, and because it uses sound
instead of light it can "see" clear plastic and glass that the camera looks straight through.
So when it rolls up to something now it stops a hand's width short, on a real distance, instead
of a guess.

It drives itself around things. This is the part I'm happiest about. A small loop on the
robot reads the depth about twenty times a second and picks a move: go straight while the way
ahead is clear, and turn toward the more open side when something gets close. So you can point
it at a cluttered bit of floor and it threads through instead of stopping dead or crashing.

It can hunt for things and go to them. Ask it to find something and it turns in small, paced
steps, taking a photo at each one, until the thing is clearly and fully in view. Then it locks
on: a tracker on the robot follows that thing frame by frame, and the car drives to it on its
own — keeping it centred, steering around whatever's in the way, and stopping just short. If it
loses sight of it, it says so and looks again rather than wandering off blind.

It thinks harder when it's stuck. The voice is quick but not a great planner, so when it gets
genuinely stuck — a full spin turns up nothing, or it's boxed into a tight corner — it hands the
situation to a stronger model for a moment and gets back a short plan ("back out, then check both
sides, then head for the opening"), which it then carries out. It's the same split-brain idea one
layer up: something fast for talking, something slower and smarter for thinking.

It tries not to hurt itself or me. One program owns the link to the motors and keeps
re-sending the current command faster than the firmware's timeout, so motion stays smooth;
if that program dies or you hit the kill switch, everything stops within a fraction of a
second.

## The main idea

My first instinct was to just send the camera feed to the cloud AI and let it drive. That
does not work. There's about a one second round trip, so the car decides to stop roughly a
meter after it should have, and streaming video to the cloud is slow and expensive.

So I split the brain into two parts running at two different speeds. The cloud part is slow,
around one second, and it handles understanding what you want and deciding the goal. The
part running on the robot itself is fast, around twenty frames a second, and it handles the
reflexes: not driving into things, and steering around them. The camera and the navigation
never leave the robot. The cloud decides what to do, and the robot handles not crashing. That
split is basically the whole trick.

## The parts

The brain is an NVIDIA Jetson Orin Nano, running headless over wifi. The motor controller is
a Keyestudio MAX, which is an Arduino Uno compatible board, driving two L298N H-bridges. The
wheels are four DC motors with mecanum wheels, so it can strafe sideways, which is fun. The
eyes are a plain USB webcam, and the ears and mouth are a Jabra USB speakerphone. There's also
an HC-SR04 ultrasonic sensor on the front, wired to the Arduino (which is 5V, so it reads it
directly) and reported up to the Jetson over the same serial link the motors use. For power,
the motors run off a 12V battery and the Jetson has its own separate supply, which I learned
the hard way (more on that below).

## How the code is laid out

- firmware/mecanum_uno: the Arduino code that actually turns the wheels
- motor/move: a script to drive it, like "./move forward 0.8 1.5"
- motor/motor_daemon.py: the one program that owns the USB link to the motors; everything else
  sends it short commands instead of touching the port
- perception/perception.py: the seeing loop, camera to depth to free space
- perception/nav.py: the driving-itself loop, reads the depth and steers around things
- perception/avoid.py: the shared "which way do I go" logic, used by both the self-driving
  loop and the voice
- perception/depth_bench.py: a quick benchmark I wrote to check the GPU was fast enough
- voice/realtime_sidecar.py: the talk to it voice loop
- debug/debug_web.py: a little web dashboard I open in a browser to watch what it's doing
- debug/run.sh: starts that dashboard

## How it works

The firmware listens for tiny text commands over USB (V means move like this, M means spin
one specific wheel, S means stop) and does the mecanum wheel math. The safety bit that
matters most: if it doesn't hear a command for 300 milliseconds it stops the motors by
itself, so if the software crashes or the cable pops out, the car doesn't drive off into a
wall.

Everything that moves the car goes through one small program, motor_daemon.py, which is the
only thing allowed to talk to the motor board. It holds the connection open, re-sends the
current command about ten times a second so the wheels keep turning smoothly, and stops the
car if it stops hearing from whoever's driving. Having a single owner means the voice loop and
the self-driving loop can't talk over each other and garble a command down the wire.

For seeing, perception.py grabs frames from the camera and runs a depth model called MiDaS on
the GPU. That was the scary part, because I wasn't sure the little Jetson could run it fast
enough to be useful. So before writing any driving logic I wrote depth_bench.py just to check,
and it was fast enough, which was a big relief. It turns each frame into a simple readout of
how close the nearest thing is on the left, middle, and right. One thing a single camera is
bad at is see-through stuff: it looks straight through a clear plastic tub to whatever's
behind it, so the distance guess says "far" right up until you hit it. To cover that I added a
second cue that watches how fast the picture is expanding as the car moves. If things are
swelling quickly, something is close, even when the depth model was fooled.

The steering itself lives in avoid.py, deliberately in one place so the robot behaves the same
whether it's wandering on its own (nav.py) or being driven by voice. It's simple: if the
middle is clear, go forward; if something's close ahead or off to one side, turn toward
whichever side is more open, with a bit of hysteresis so it doesn't dither left-right-left. If
it ever gets physically wedged on something the camera couldn't see, it backs up and turns to
free itself.

For talking, realtime_sidecar.py opens one connection to the OpenAI Realtime API and streams
the microphone up and the voice down. The AI drives through a few tools: a plain drive for
little nudges, look to take a photo, stop, and two ways of going forward. Advance rolls up to
something and stops before it reaches it; navigate rolls off and steers around obstacles.
Keeping those two separate turned out to matter, because "come here and stop" and "go explore"
want opposite things when there's a wall in front. One bug that took me a while: I was doing
the image processing on the same thread as the audio, and it made the voice stutter. Moving
that work to a background thread fixed it.

## Watching it

Debugging a headless robot over SSH by squinting at log files got old fast, so I wrote a tiny
web dashboard (debug/debug_web.py) that I open in a browser on my laptop. It shows the live
camera frame, the left/middle/right closeness bars, the front distance in centimetres, whether
it's armed, and a running tail of each log, plus an arm/disarm button and an escape-key kill
switch. Across the top there's a health strip that tells me at a glance whether each of the
robot's programs is actually running, how fresh the camera / depth / distance readings are, and
how hot the board is getting — which turns "why isn't it reacting" from an SSH archaeology dig
into a look. There's a little d-pad too, so I can nudge it around with the arrow keys when I
just want to reposition it by hand, and a chat box for typing to it when saying things out loud
isn't practical (its replies show up there). It's plain Python with no extra dependencies, so
it just runs.

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

A subtler one came from the self-driving: the car kept nosing straight into a clear plastic
storage tub. It took me a while to realize the depth model was seeing right through the
plastic to the wall behind, so as far as the robot was concerned there was nothing there.
That's what pushed me to add the "how fast is the picture growing" cue, which doesn't care
whether it can actually measure the distance.

## Running it

```sh
# flash the Arduino firmware
arduino-cli upload --fqbn arduino:avr:uno -p /dev/ttyUSB0 firmware/mecanum_uno

# start the one program that owns the motors
python3 motor/motor_daemon.py

# start the seeing loop (needs a MiDaS-small ONNX model in models/)
python3 perception/perception.py

# let it drive itself around (optional; the voice loop can also drive)
python3 perception/nav.py

# start the voice loop (needs an OpenAI API key, kept out of the repo)
OPENAI_KEY_FILE=~/secrets/openai.key python3 voice/realtime_sidecar.py

# start the debug dashboard (optional; then open http://<jetson-ip>:8099 in a browser)
./debug/run.sh
```

You'll need OpenCV built with CUDA, NumPy, websockets, and PulseAudio. The depth model and
any API keys are deliberately not in this repo (see .gitignore), so you bring your own. The
voice needs an OpenAI key; the "think when stuck" planner needs an Anthropic one — both live in
a secrets folder that isn't committed.

## What I want to add next

- Give it a sense of angle. Right now it has no compass and the motors are open-loop, so it
  can't turn a known number of degrees — it just nudges and re-checks. A cheap gyro (an IMU)
  would let it turn to an actual heading, which would tidy up searching and stop the little
  overshoots.
- Build a simple map of a room from a spin, so it can remember where the openings and things
  are instead of only reacting to what's right in front of it. The angle sense above is the
  missing piece for this.
- Keep making "go to that specific thing" more robust — the lock-on tracker works, but it can
  still lose a target while squeezing past an obstacle and has to look again.

Thanks for reading. This has been the most fun I've had learning something in a while.
