#!/usr/bin/env python3
"""
Robot realtime sidecar — low-latency voice+drive copilot via the OpenAI Realtime API.

An alternative to the Claude-operative voice loop for the "hear me, react, move NOW"
use case: one bidirectional websocket to gpt-realtime, streaming mic audio up and
voice audio down, with server-side semantic VAD and barge-in. The model drives the
car through function tools (drive/stop/look) that bridge to the SAME `move` script
the operative uses — so the firmware 300ms watchdog and the `.disarmed` kill switch
still apply. Runs on the host (reuses parec/paplay + the bind-mounted workspace).

Env (all optional):
  OPENAI_KEY_FILE   path to the API key file      (default ~/robot/secrets/openai-realtime.key)
  REALTIME_MODEL    realtime model id             (default gpt-realtime-2)
  REALTIME_VOICE    output voice                  (default marin)
  LISTEN_SOURCE_MATCH / SPEECH_SINK_MATCH  pulse device substrings (default jabra)
  ROBOT_WORKSPACE   workspace dir (move/vision.txt)(default ~/robot/workspace)
  REALTIME_SMOKE=1  connect, say one line, exit (no mic) — for validation
  REALTIME_LOG      log file                      (default ~/robot/realtime.log)
"""
import os
import re
import sys
import json
import time
import base64
import socket
import asyncio
import subprocess
import urllib.request
from datetime import datetime

import avoid                      # shared reactive-avoidance brain (decide + recovery_plan); see avoid.py
import websockets
from websockets.asyncio.client import connect as ws_connect

try:
    import cv2
    import numpy as np
    _CV2 = True
except Exception:
    _CV2 = False

HOME = os.path.expanduser("~")


def env(n, d):
    v = os.environ.get(n)
    return v if v not in (None, "") else d


KEY_FILE = env("OPENAI_KEY_FILE", os.path.join(HOME, "robot", "secrets", "openai-realtime.key"))
MODEL = env("REALTIME_MODEL", "gpt-realtime-2")
VOICE = env("REALTIME_VOICE", "marin")
RATE = 24000
# The `think` tool routes hard spatial-planning decisions to a stronger reasoning model (Claude).
# The realtime voice model is fast but a weak planner; Claude reasons over the surroundings summary
# and hands back a short plan the voice model then executes with its normal tools.
PLANNER_KEY_FILE = env("ANTHROPIC_KEY_FILE", os.path.join(HOME, "robot", "secrets", "anthropic.key"))
PLANNER_MODEL = env("PLANNER_MODEL", "claude-opus-4-8")
PLANNER_MAXTOK = int(env("PLANNER_MAXTOK", "512"))
PLANNER_TIMEOUT = float(env("PLANNER_TIMEOUT", "8"))    # give up on the planner after this; the voice model
                                                        # then falls back to its own judgement. Kept short so a
                                                        # stalled call over flaky Wi-Fi can't leave the car
                                                        # deaf/mute waiting (motors already dead-man in 0.5 s)
THINK_COOLDOWN = float(env("THINK_COOLDOWN", "8"))      # don't re-plan within this of the last plan; carry the
                                                        # last one out first (stops the model spamming the planner)
# find_it: Claude (Opus, a much stronger vision model than the realtime voice one) locates a named
# object in the current frame and hands back a box we seed the tracker with — open-vocabulary, no
# YOLO/training. Called only to ACQUIRE/re-acquire (~1-2 s); the CSRT tracker follows in between.
LOCATOR_MODEL = env("LOCATOR_MODEL", "claude-opus-4-8")
LOCATOR_TIMEOUT = float(env("LOCATOR_TIMEOUT", "12"))
LOCATOR_MAXDIM = int(env("LOCATOR_MAXDIM", "1024"))     # frame long-edge sent to Claude (bigger = better boxes)
WORKSPACE = env("ROBOT_WORKSPACE", os.path.join(HOME, "robot", "workspace"))
MOVE = os.path.join(WORKSPACE, "move")
DISARM = os.path.join(WORKSPACE, ".disarmed")       # kill switch flag (move disarm)
FRAME_PATH = os.path.join(WORKSPACE, "frame.jpg")   # written by the perception loop
NAV_STATE = os.path.join(WORKSPACE, "nav_state.json")  # perception's structured free-space output
ULTRA = os.path.join(WORKSPACE, "ultrasonic.json")  # front sonar distance (motor daemon publishes it)
GOAL = os.path.join(WORKSPACE, "nav_goal.json")     # we WRITE this to seed perception's target tracker
LOCATE = os.path.join(WORKSPACE, "locate.json")     # we WRITE find_it's last result so the dashboard can draw it
CHAT_SOCK = os.path.join(WORKSPACE, "chat.sock")    # typed messages from the debug dashboard arrive here
# guarded advance = roll forward and STEER AROUND obstacles locally, using the SAME
# avoidance decision the autonomous loop uses (avoid.decide) — obstacle thresholds
# (STOP_NEAR/SIDE_NEAR/LOOM_BRAKE) all live in avoid.py now, shared, so there's one brain.
ADVANCE_MAX_SECS = float(env("ADVANCE_MAX_SECS", "20"))      # safety cap: navigating can cross a room; this bounds
                                                             # a single 'go' so it can't roam indefinitely
MAX_RECOVER = int(env("ADVANCE_MAX_RECOVER", "3"))           # give up (report wedged) after this many recoveries
# stall fallback: commanded to move but the camera view stops changing -> wedged on something
# depth can't see (a low box, chair leg, glass). decide() can't steer off it, so we recover.
ADVANCE_MOTION_MIN = float(env("ADVANCE_MOTION_MIN", "5.0"))  # smoothed scene-motion below this = not moving
ADVANCE_STALL_SECS = float(env("ADVANCE_STALL_SECS", "0.6"))  # ...for this long while driving = wedged
ULTRA_FRESH = float(env("ULTRA_FRESH", "0.5"))                # ignore sonar readings older than this (fall back to camera)
NAV_FRESH = float(env("NAV_FRESH", "0.6"))                   # treat perception's nav_state as dead past this, so a
                                                             # frozen depth frame (perception stalled/crashed) halts the
                                                             # drive loops instead of steering blind on a stale picture
# go_to (drive to a locked visual target): steer to keep it centred, skirt obstacles, stop on the sonar.
GOTO_MAX_SECS = float(env("GOTO_MAX_SECS", "60"))             # safety cap so it can't drive forever if it can't reach
                                                             # the thing; Claude-per-step is slow so give it room
GOTO_LOST_SECS = float(env("GOTO_LOST_SECS", "1.2"))          # tracker 'lost' this long -> give up, ask to re-find
GOTO_BEAR_DEAD = float(env("GOTO_BEAR_DEAD", "0.10"))         # tight 'centred' band used up CLOSE (precise final aim)
GOTO_BEAR_DEAD_FAR = float(env("GOTO_BEAR_DEAD_FAR", "0.22"))  # loose band when far/small: if the thing's roughly ahead,
                                                             # just DRIVE at it to close the distance; only turn to face
                                                             # it if it's clearly off to a side. Tightens to _DEAD as it
                                                             # fills the frame -> fine centring only when it's near.
GOTO_BLIND_STEPS = int(env("GOTO_BLIND_STEPS", "6"))         # keep committing toward the last-seen spot for this many
                                                             # not-found looks before giving up — a small/far object
                                                             # blinks in and out of Claude's view, so don't quit on a miss
# self-healing lock: while homing, use Claude (the strong eye) to keep the dumb CSRT tracker honest —
# re-box the named thing in the background and re-seed when the tracker drifts off it or loses it.
GOTO_RELOCK_EVERY = float(env("GOTO_RELOCK_EVERY", "4.0"))    # periodic background re-locate cadence while homing
GOTO_RELOCK_DRIFT = float(env("GOTO_RELOCK_DRIFT", "0.20"))   # re-seed if Claude's box centre is >this (norm) off the
                                                             # tracker's — catches drift onto background before it's lost
GOTO_RELOCK_MAX_FAILS = int(env("GOTO_RELOCK_MAX_FAILS", "2"))  # give up only after Claude ALSO can't see it this many times
# pulsed homing: drive in short bursts and PAUSE between them so the tracker reads a sharp frame, instead
# of trying to track through continuous motion blur (which was making it lose the object while approaching).
GOTO_PULSE_DRIVE = float(env("GOTO_PULSE_DRIVE", "0.45"))    # forward burst length — longer so it closes distance
GOTO_PULSE_TURN = float(env("GOTO_PULSE_TURN", "0.08"))      # MAX centring turn burst; scaled down near centre so a
                                                             # single step never swings past where the (1-2s old) box
                                                             # said the thing was -> no chasing the 'afterimage'
GOTO_PULSE_TURN_MIN = float(env("GOTO_PULSE_TURN_MIN", "0.05"))  # floor so a tiny turn still actually moves the wheels
                                                                 # (at full speed a 0.05s flick still breaks friction)
GOTO_HOME_TURN_SPEED = float(env("GOTO_HOME_TURN_SPEED", "1.0"))  # FULL speed: a short low-speed pulse can't overcome
                                                                 # the wheels' static friction (it just doesn't move) —
                                                                 # a crisp full-speed flick for a tiny time actually nudges
GOTO_PULSE_SETTLE = float(env("GOTO_PULSE_SETTLE", "0.45"))  # stop-and-look pause: let motion stop AND a fresh frame land
                                                             # before Claude looks, so it reads NOW, not a frame ago
GOTO_KP = float(env("GOTO_KP", "2.2"))                        # bearing -> turn-rate gain (proportional steering)
GOTO_WMAX = float(env("GOTO_WMAX", "0.7"))                    # cap on the turn component
GOTO_SIZE_ARRIVE = float(env("GOTO_SIZE_ARRIVE", "0.22"))    # target filling this frac of frame = we're there
GOTO_ARRIVE_Y = float(env("GOTO_ARRIVE_Y", "0.80"))          # target's bottom edge this far down the frame = it's at
                                                             # the robot's feet (floor objects drop low as you close in)
# --- sonar-guarded DASH: on a long clear straightaway, drive continuously (no per-step Claude call)
# under a fast local sonar/camera guard, to cut the ~1-2s-per-step Claude cost on the approach. It only
# ever goes STRAIGHT (never turns blind), only off a FRESH centred fix on a far target, hands back at
# 60cm (never enters the ~35cm standoff), and is hard-capped blind for a sonar-invisible target.
GOTO_DASH_ENABLE = env("GOTO_DASH_ENABLE", "1") == "1"       # master switch (0 = pure pulsed homing)
GOTO_DASH_SIZE_MAX = float(env("GOTO_DASH_SIZE_MAX", "0.06"))    # only dash when the target is this small in frame (far)
GOTO_DASH_MAX_BOTTOM = float(env("GOTO_DASH_MAX_BOTTOM", "0.55"))  # AND its bottom edge is this HIGH in frame. size alone
                                                                  # is a bad 'far' proxy — a physically small ball stays
                                                                  # tiny even up close; but a floor object that's genuinely
                                                                  # far sits HIGH (small box_bottom), a close one sits low.
                                                                  # This stops the dash running over a close small ball.
GOTO_DASH_BEAR_DEAD = float(env("GOTO_DASH_BEAR_DEAD", "0.08"))  # must be this centred to START (tighter than the drive band)
GOTO_DASH_MIN_CLEAR_CM = float(env("GOTO_DASH_MIN_CLEAR_CM", "120"))  # need a FRESH VALID sonar runway >= this to enter
GOTO_DASH_STOP_CM = float(env("GOTO_DASH_STOP_CM", "60"))       # hand back to the slow fine loop when sonar drops to this
GOTO_DASH_SONAR_FRESH = float(env("GOTO_DASH_SONAR_FRESH", "0.2"))  # abort if the sonar reading is older than this
GOTO_DASH_MAX_CM = float(env("GOTO_DASH_MAX_CM", "60"))        # HARD blind-distance cap per dash — kept short so even if
                                                              # the far-gate is fooled, an overshoot stays recoverable
GOTO_DASH_MAX_SECS = float(env("GOTO_DASH_MAX_SECS", "1.5"))   # hard wall-clock backstop per dash
GOTO_DASH_SPEED_CMS = float(env("GOTO_DASH_SPEED_CMS", "70"))  # CALIBRATE ON FLOOR: HIGH estimate of full-speed cm/s so
                                                              # the distance cap over-counts travel and fires early
GOTO_DASH_POLL = float(env("GOTO_DASH_POLL", "0.05"))         # dash inner-loop poll period (~20Hz)
GOTO_DASH_BURST = float(env("GOTO_DASH_BURST", "0.2"))        # self-expiring forward burst re-issued each tick (dead loop
                                                             # -> daemon stops the car within this)
GOTO_LOC_MAXDIM_FAR = int(env("GOTO_LOC_MAXDIM_FAR", "512"))  # locator frame size for coarse cruise steering (~4x faster)
GOTO_RES_NEAR_SIZE = float(env("GOTO_RES_NEAR_SIZE", "0.10")) # at/above this frame-fraction, use full res for precise aim
LOCK_BOX = {"small": (0.16, 0.20), "medium": (0.30, 0.36), "large": (0.50, 0.56)}  # centred seed-box (w,h)
# scan = "turn a small step, then look" — the search primitive (spin-until-you-see-it). Deliberately
# a small step so it doesn't overshoot; the model calls it repeatedly and reads the photo each time.
SCAN_TURN_SPEED = float(env("SCAN_TURN_SPEED", "0.65"))  # slower than before to curb scan overshoot; held near
                                                         # the motor floor (the left-wheel trim eats into this)
SCAN_TURN_SECS = float(env("SCAN_TURN_SECS", "0.35"))  # step size: turn roughly one camera field-of-view per
                                                       # step so consecutive frames tile ~edge-to-edge (the near
                                                       # side of each new view overlaps the far side of the last),
                                                       # covering the room in fewer steps without leaving gaps
SCAN_NUDGE_SECS = float(env("SCAN_NUDGE_SECS", "0.1"))     # fine-centering turn: a tiny step to slide a target
                                                           # that's already in view toward the middle of the frame
SCAN_SETTLE = float(env("SCAN_SETTLE", "1.0"))         # dwell after each turn step: let motion blur clear, a fresh
                                                       # frame.jpg land, AND give the model time to process before
                                                       # the next burst (0.35 -> 0.7 -> 1.0; was overshooting a step)
SCAN_NUDGE_SETTLE = float(env("SCAN_NUDGE_SETTLE", "0.4"))  # shorter dwell for centering — the target's already in
                                                           # view, so converge quickly instead of the full dwell
SCAN_NUDGE_INTERVAL = float(env("SCAN_NUDGE_INTERVAL", "0.6"))  # light pacing between centering nudges (NOT the
                                                               # full sweep floor) so a few taps can centre it
SCAN_SWEEP_GAP = float(env("SCAN_SWEEP_GAP", "5.0"))   # a scan after this long of no scanning starts a NEW sweep:
                                                       # look at the current view WITHOUT turning first, so the bot
                                                       # registers where it is before it ever moves
SCAN_MIN_INTERVAL = float(env("SCAN_MIN_INTERVAL", "3.0"))  # HARD floor between consecutive scan turns. The voice
                                                       # model fires scans as fast as it gets photos; this paces the
                                                       # actual turning so it can't outrun itself. Raise to slow scans.
SCAN_FULL_TURN_STEPS = int(env("SCAN_FULL_TURN_STEPS", "5"))  # ~how many scan steps make a full 360 (no compass, so
                                                              # we count steps as a crude 'how far around am I' and
                                                              # tell the model, so it doesn't quit before it's checked
                                                              # behind itself). Rough; tune on the floor.
# hybrid vision: on-demand look (crisp) + silent change-gated ambient push (cheap)
LOOK_MAXDIM = int(env("VISION_LOOK_MAXDIM", "512"))
LOOK_QUALITY = int(env("VISION_LOOK_QUALITY", "65"))
LOOK_DETAIL = env("VISION_LOOK_DETAIL", "low")   # "low" = ~2x faster model vision; "high" for fine detail
PUSH_MAXDIM = int(env("VISION_PUSH_MAXDIM", "384"))
PUSH_QUALITY = int(env("VISION_PUSH_QUALITY", "55"))
PUSH_POLL = float(env("VISION_PUSH_POLL", "0.7"))          # how often to check for change
PUSH_CHANGE_THRESH = float(env("VISION_PUSH_THRESH", "8"))  # mean abs pixel delta (0-255) to count as changed
PUSH_MIN_INTERVAL = float(env("VISION_PUSH_MIN_INTERVAL", "1.5"))  # rate cap between ambient pushes
VOICE_QUIET_GAP = float(env("VISION_QUIET_GAP", "3.0"))  # (legacy image push) don't push a frame within this many s of voice
TEXT_POLL = float(env("VISION_TEXT_POLL", "0.4"))          # how often to check perception state
TEXT_MIN_INTERVAL = float(env("VISION_TEXT_MIN_INTERVAL", "1.0"))  # min seconds between text situational updates
TEXT_QUIET_GAP = float(env("VISION_TEXT_QUIET_GAP", "1.0"))        # keep text updates just clear of live speech
LOOM_TEXT = float(env("VISION_LOOM_TEXT", "120"))                  # loom above this => 'something getting closer'
SOURCE_MATCH = env("LISTEN_SOURCE_MATCH", "jabra").lower()
SINK_MATCH = env("SPEECH_SINK_MATCH", "jabra").lower()
SMOKE = env("REALTIME_SMOKE", "0") == "1"
LOG_PATH = env("REALTIME_LOG", os.path.join(HOME, "robot", "realtime.log"))
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())

INSTRUCTIONS = (
    "You are the voice of a small four-wheeled robot car with a forward camera. "
    "Keep spoken replies short, natural, and free of filler: say only what matters. NEVER announce "
    "what you are about to do or narrate a tool call — no 'let me grab a snapshot', 'one moment', "
    "'I'll check', 'hold on'. Call tools silently, then speak ONCE with the real answer or result "
    "(what you actually see, the status) — never a placeholder acknowledgement. "
    "MOVING FORWARD: there are two forward tools — choose by whether the user wants the car to STOP near "
    "something or to GET somewhere. If they want it to come to them or approach something and stop (e.g. "
    "'come here', 'advance until you're close to me then stop', 'move forward until something's in the way'), "
    "call advance — it rolls forward and stops itself the instant something is close, then tells you why. If "
    "they want it to travel or make its way somewhere (e.g. 'go forward', 'head over there', 'make your way to "
    "the kitchen', 'explore'), call navigate — it rolls forward and steers around obstacles on its own, going "
    "around things instead of stopping. Both react on their own and are continuous, so never re-issue forward "
    "bursts or babysit them. Use the drive tool with direction forward ONLY for a tiny deliberate nudge. To "
    "TURN the car left or right in place use turn_left / turn_right; only slide sideways (strafe_left / "
    "strafe_right) when the user actually asks to move sideways, not when they say 'turn'. Give the drive tool "
    "a direction, "
    "a speed from 0.6 to 1.0 (NEVER below 0.6 — below that the motors just stall and buzz and the car "
    "won't move; default to full speed 1.0 unless the user asks to go slower), and a short duration "
    "in seconds — prefer brief bursts (0.5 to 1.5 s) and re-check rather than long "
    "blind drives. Call stop the instant the user says stop (it also cancels an advance). "
    "VISION: you continuously receive short text camera updates like '(camera) most open left; camera "
    "locked on something to the right' — which direction is most open, whether something is getting closer, "
    "and whether the camera has a lock. This is your CONSTANT background awareness for OPENNESS and MOVEMENT, "
    "so you almost always already know the layout without looking; use it directly for driving decisions. Do "
    "NOT call look for ordinary moves. BUT these notes CANNOT tell you WHAT a thing is: the camera lock grabs "
    "whatever sits in the middle of the view — often a person, a wall, furniture — NOT necessarily the thing "
    "you were asked to find. So NEVER conclude you've found or centred the thing you're searching for from a "
    "'locked'/'centred' note or any background text alone. Before you say you found it or drive to it, CONFIRM "
    "WITH YOUR OWN EYES — read the photo the scan returned, or call look — that the centred thing really is "
    "what you were asked for. If it isn't, keep searching. "
    "SEARCHING / FINDING: when asked to look around, spin until you see something, or find or identify a "
    "thing, use the scan tool, NOT drive. The FIRST scan just shows you the view from where you already "
    "are, without moving — look at it first. Each scan AFTER that turns the car a small step to the LEFT "
    "or RIGHT (you pick) and hands you a fresh photo, so scanning again means 'I've looked, keep going'; "
    "always read the photo and decide before you scan again. Choose the direction on purpose, toward where the thing "
    "should be: if you just turned right and it was ahead of you, scan left to bring it back; if you last "
    "saw it on one side, scan that way. Look at each photo; if what you're after isn't in it, scan again, "
    "step by step (a full turn is several scans). Getting it CLEARLY in view AND centered matters: if it "
    "first shows up as just a sliver at the edge of the frame, you've only caught the very start of it — do "
    "NOT act yet. Once it IS in view and roughly ahead, switch to centring: call scan 'center'. It locks on "
    "and turns the correct way BY ITSELF — you do NOT pick a side, and you do NOT reason about which way to "
    "turn. Just call scan 'center' again and again, reading each photo, until it tells you the thing is "
    "centred; then say what it is or go to it. If it says it couldn't lock on (the thing's near the edge), "
    "sweep one 'step' toward it to bring it inward, then centre again. Only once it's centred do you act. "
    "Don't take big blind turns to "
    "search. Chain the scans yourself — scan, read the view, decide, scan again — without waiting to be "
    "prompted. IMPORTANT — don't give up early: a few steps only covers a small arc, and something behind "
    "you takes about half a full circle of steps to come into view. Keep scanning the SAME direction, step "
    "by step, until you have gone a FULL circle (the view note tells you the step count and how many make a "
    "circle) before you ever conclude something isn't there. Finding nothing in the first few steps means "
    "keep going, NOT stop. "
    "GOING TO SOMETHING (a mission): when asked to find and go to a NAMED thing, the job is basically two "
    "tools: find_it, then go_to. Hold still and call find_it with what you want (e.g. 'a red ball'). It uses "
    "a sharper eye that finds the thing WHEREVER it is in view and locks on — you do NOT need to face, centre, "
    "or judge how far away it is yourself; do not decide the thing is close or lined up from your own sense of "
    "the scene. If find_it locks on, call go_to and then STOP acting: go_to turns to face the thing and drives "
    "all the way to it on its own, and calling any other tool (scan, look, find_it, drive) while it runs "
    "CANCELS it and wrecks the approach. Just wait for its update. If find_it says it can't see the thing, "
    "sweep ONE scan 'step' to show it a new part of the room and call find_it again; keep going around, and if "
    "a full circle turns up nothing, navigate a little to a new spot and sweep-and-find_it again. Do NOT use "
    "navigate to approach a specific target. If go_to reports it lost the thing, call find_it again, then "
    "go_to. Keep the loop going until you've reached it or it's clearly not findable. "
    "THINKING IT THROUGH: if you get genuinely stuck — a full sweep finds nothing, you're boxed into a tight "
    "space, or reaching a goal needs real multi-step planning around obstacles — call think with a short "
    "description of the situation and your goal. A stronger reasoning model hands back a short plan; carry it "
    "out with your normal tools, and tell the user in a few words what you're doing. Use think at real "
    "decision points, not for routine single moves."
)

TOOLS = [
    {"type": "function", "name": "drive",
     "description": "Drive/turn the car in one direction. Runs in the background (non-blocking) for the given duration; you stay aware while it moves. Any new drive replaces the current motion, and stop halts it early.",
     "parameters": {"type": "object", "properties": {
         "direction": {"type": "string",
                       "enum": ["forward", "back", "turn_left", "turn_right", "strafe_left", "strafe_right"],
                       "description": "forward/back; turn_left/turn_right ROTATE in place (this is what 'turn left/right' means); strafe_left/strafe_right slide sideways WITHOUT turning"},
         "speed": {"type": "number", "description": "0.6..1 (below 0.6 the motors stall and the car won't move), default 1"},
         "seconds": {"type": "number", "description": "burst length, keep <= 2"}},
         "required": ["direction"]}},
    {"type": "function", "name": "advance",
     "description": "Roll forward and automatically STOP the moment something is close ahead, using the live camera. Use this when the user wants the car to come toward them or approach something and STOP before reaching it — e.g. 'come here and stop', 'advance until you're close to me then stop', 'move forward until something's in the way'. It reports back when it stops and why, so you do not re-issue or babysit it.",
     "parameters": {"type": "object", "properties": {
         "speed": {"type": "number", "description": "0.6..1 (below 0.6 the motors stall), default 1"}},
         "required": []}},
    {"type": "function", "name": "navigate",
     "description": "Roll forward and STEER AROUND obstacles on its own — it turns toward the open side when something is ahead and keeps going, instead of stopping. Use this when the user wants the car to travel or get somewhere — e.g. 'go forward', 'head over there', 'make your way to the kitchen', 'explore'. It keeps moving and navigating by itself and reports back only if it has to give up (it gets wedged, or went as far as set), so you never re-issue or babysit it.",
     "parameters": {"type": "object", "properties": {
         "speed": {"type": "number", "description": "0.6..1 (below 0.6 the motors stall), default 1"}},
         "required": []}},
    {"type": "function", "name": "stop",
     "description": "Stop all motion immediately (also cancels a guarded advance).",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function", "name": "look",
     "description": "Attach a fresh photo from the robot's forward camera so you can see the scene with your own eyes. Call before moving toward something or when asked what you see.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function", "name": "scan",
     "description": "Look around for something. The FIRST scan of a search hands you the view from where you already are WITHOUT moving — look at it first. Each scan after that turns the car and hands you a fresh camera view, so calling scan again means 'I've looked at that view, keep going.' Two step sizes via amount: 'step' (default) turns about one camera-width so each new view picks up right where the last one ended — use this to sweep a room; 'center' AUTO-centres on the thing — use it once the thing is in view and roughly ahead (not jammed against an edge). It locks on and turns the correct way BY ITSELF using the camera, so you do NOT choose a side; just call scan 'center' again and again until it tells you the thing is centred, then go to it. (If it says it couldn't lock on, the thing is too near the edge — sweep one 'step' toward it first.) Always read each photo and decide before scanning again. When sweeping, pick the direction deliberately toward where the thing should be. Never use big blind turns to search.",
     "parameters": {"type": "object", "properties": {
         "direction": {"type": "string", "enum": ["left", "right"],
                       "description": "which way to turn — left or right (default right). Used for sweeping ('step'); IGNORED for 'center', which turns the correct way by itself"},
         "amount": {"type": "string", "enum": ["step", "center"],
                    "description": "'step' = a full camera-width sweep step (default); 'center' = a tiny centring nudge for a target that's already in view but off to one side"}},
         "required": []}},
    {"type": "function", "name": "find_it",
     "description": "Find and lock onto a specific thing by description, so you can then go to it — 'a red mug', 'my backpack', 'the mountain dew can', anything you can name. This uses a SHARPER eye than your quick camera glances: it finds the thing WHEREVER it is in the current view (you do NOT need to centre it first) and locks the tracker onto exactly it. This is the RIGHT way to start any 'go to / fetch / find the <named thing>' job. How to use: face the area and hold still, then call find_it with what you want. If it says it found and locked on, call go_to. If it says it can't see the thing, sweep one scan 'step' to show it a new part of the room, then call find_it again — repeat around the room. If go_to later says it lost sight of it, call find_it again to re-find it.",
     "parameters": {"type": "object", "properties": {
         "thing": {"type": "string", "description": "what to find, described plainly, e.g. 'a red coffee mug', 'the mountain dew can'"}},
         "required": ["thing"]}},
    {"type": "function", "name": "lock_on",
     "description": "Fallback lock: grabs whatever is in the MIDDLE of the view right now. Prefer find_it for anything you can name (it finds the thing wherever it is and won't grab the wrong object). Use lock_on only to grab whatever's already centred when find_it isn't suitable. After it locks, call go_to.",
     "parameters": {"type": "object", "properties": {
         "label": {"type": "string", "description": "what you're locking onto, e.g. 'the trash can' (for your own reference)"},
         "size": {"type": "string", "enum": ["small", "medium", "large"],
                  "description": "how big the target looks in the frame right now (default medium)"}},
         "required": ["label"]}},
    {"type": "function", "name": "go_to",
     "description": "Drive to the thing you locked onto with find_it. It does EVERYTHING itself: turns to face the thing (even if it's off to one side or far away — you do NOT need to face or centre it first), drives to it, steers around obstacles, re-checks with a sharper eye as it closes in, and stops just short when it arrives. CRITICAL: once you call go_to, WAIT — do not call scan, look, find_it, drive, or any other tool until it reports back. Any of those CANCELS the drive and it starts flailing. It self-reports when it arrives, or only if it truly loses the thing (then call find_it again). Requires a find_it lock first.",
     "parameters": {"type": "object", "properties": {
         "speed": {"type": "number", "description": "0.6..1 (below 0.6 the motors stall), default 1"}},
         "required": []}},
    {"type": "function", "name": "think",
     "description": "Ask a stronger reasoning model for a plan when you're stuck or a task needs real multi-step spatial planning — e.g. you've searched and can't find something, you're boxed into a tight space, or you must work out how to get somewhere around obstacles. It SEES the robot's current camera view, so it reasons about the actual space around you. Describe the situation and your goal; it hands back a short step-by-step plan that you then carry out with your other tools. Use it at genuine decision points, not for routine single moves (it takes a second or two).",
     "parameters": {"type": "object", "properties": {
         "situation": {"type": "string", "description": "what you're trying to do and why you're stuck or unsure"}},
         "required": ["situation"]}},
]


def log(msg):
    line = "%s  %s" % (datetime.now().isoformat(timespec="seconds"), msg)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def read_key():
    with open(KEY_FILE) as f:
        return f.read().strip()


def find_pulse(kind, match):
    """kind = 'sources' or 'sinks'; return device name containing `match` (skip monitors)."""
    try:
        out = subprocess.check_output(["pactl", "list", "short", kind], text=True, timeout=5)
        for line in out.splitlines():
            low = line.lower()
            if match in low and "monitor" not in low:
                return line.split()[1]
    except Exception as e:
        log("%s lookup failed: %s" % (kind, e))
    return None


def wait_pulse(kind, match, max_wait=20):
    """Retry find_pulse until the device shows up — the Jabra can take a couple
    seconds to register both its endpoints after (re-)enumeration."""
    start = time.time()
    while time.time() - start < max_wait:
        d = find_pulse(kind, match)
        if d:
            return d
        time.sleep(1)
    return None


# ---------- tool bridge: commands go to the motor daemon over its unix socket ----------
# The daemon is the single owner of the serial link: it sustains motion via its own
# watchdog resend, dead-mans if we stop sending, and enforces the .disarmed kill switch.
def _clamp(v, lo, hi, d):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return d


MOTOR_SOCK = os.path.join(WORKSPACE, "motor.sock")
_msock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)


def motor(cmd):
    """Fire a command to the motor daemon. Silent no-op if it's down -> the car stays
    still, which is the safe failure (never a second, racing serial writer)."""
    try:
        _msock.sendto(cmd.encode(), MOTOR_SOCK)
    except Exception:
        pass


def args_of(args_json):
    try:
        return json.loads(args_json or "{}")
    except Exception:
        return {}


# model-facing drive direction -> motor-daemon command. Turning is named turn_left/turn_right
# (what users actually say); the daemon rotates via cw/ccw and strafes via left/right, so "turn
# left" no longer collides with the sideways strafe. Raw daemon names stay accepted as a fallback.
DRIVE_MAP = {"forward": "forward", "back": "back",
             "turn_left": "ccw", "turn_right": "cw",
             "strafe_left": "left", "strafe_right": "right",
             "ccw": "ccw", "cw": "cw", "left": "left", "right": "right"}


def invoke_tool(name, args_json):
    args = args_of(args_json)
    try:
        if name == "drive":
            d = str(args.get("direction", "forward")).lower()
            cmd = DRIVE_MAP.get(d)
            if cmd is None:
                return {"ok": False, "error": "unknown direction %s" % d}
            spd = _clamp(args.get("speed", 1.0), 0.6, 1.0, 1.0)   # 0.6 floor: below this the motors stall/buzz
            secs = _clamp(args.get("seconds", 1.0), 0.1, 2.0, 1.0)
            motor("%s %.2f %.2f" % (cmd, spd, secs))   # timed move: the daemon drives for secs then stops
            return {"ok": True, "detail": "driving %s at %.2f for %.1fs" % (d, spd, secs)}
        if name == "stop":
            motor("stop")
            return {"ok": True, "detail": "stopped"}
        # note: "look" is handled specially in the receiver (it attaches an image), not here
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"error": "unknown tool %s" % name}


async def guarded_forward(speed, enqueue, steer):
    """Roll forward under local, on-the-box control — no cloud in this loop, so it reacts in
    ~one perception frame (~100 ms), not the ~1-2 s voice round-trip. Two modes, both using
    the SAME shared 'too close' thresholds (avoid.py):

      steer=False  (the `advance` tool): go straight and STOP the instant something is close
        ahead — for "come here and stop", "advance until you're near me then stop".
      steer=True   (the `navigate` tool): go forward and STEER AROUND obstacles (avoid.decide),
        turning toward the open side and continuing — for "go / head over there / get somewhere".

    Both end on the terminal conditions (kill switch, time cap, lost camera). Stop-mode also
    ends when blocked or bumped. Steer-mode instead turns, and if wedged on something depth
    can't see it recovers (reverse + turn via avoid.recovery_plan) and continues. Cancelling
    (stop / new command / disconnect) halts silently; a terminal end reports why so the model
    can speak it."""
    cancelled = False
    reason = "went as far as I could"
    start = time.monotonic()
    mot_ema = None
    stall_since = None
    last_recover = 0.0
    recover_count = 0
    cur = None                     # current action, for decide()'s hysteresis (steer mode)
    turn_started = 0.0
    try:
        while True:
            now = time.monotonic()
            if os.path.exists(DISARM):
                reason = "the kill switch is on"
                break
            if now - start > ADVANCE_MAX_SECS:
                reason = "went as far as I set"
                break
            st = read_nav_state()
            if st is None:
                reason = "I lost the camera feed"     # never keep driving blind
                break
            n = st.get("near", {}) or {}
            l, c, r = n.get("l", 0), n.get("c", 0), n.get("r", 0)
            loom = st.get("loom", 0) or 0

            if steer:
                act, turn_started = avoid.decide(l, c, r, loom, cur, turn_started, now)  # turn, don't stop
                # sonar backstop: if the camera says "clear ahead" but the front sonar sees something at
                # the standoff (a blank wall, clear plastic — where monocular depth fails), don't drive
                # into it; turn toward the more-open side instead (mirrors decide()'s open-side choice)
                if act == "forward":
                    ucm, uvalid = read_ultra()
                    if avoid.ultra_blocked(ucm, uvalid):
                        act = "ccw" if l <= r else "cw"
                        turn_started = now
                spd = speed if act == "forward" else max(0.6, speed)   # 0.6 floor: below it the motors stall
                motor("%s %.2f" % (act, spd))
                cur = act
            else:
                # stop-mode: stop on the front sonar's true standoff OR the camera/loom block.
                # sonar and camera are complementary — sonar sees clear/transparent surfaces the
                # camera looks through; camera sees off-axis things outside the sonar's narrow cone.
                ucm, uvalid = read_ultra()
                if avoid.ultra_blocked(ucm, uvalid):
                    reason = "I'm right up close"       # metric standoff reached
                    break
                if (c >= avoid.STOP_NEAR or l >= avoid.SIDE_NEAR
                        or r >= avoid.SIDE_NEAR or loom >= avoid.LOOM_BRAKE):
                    reason = "something's right ahead"
                    break
                spd = avoid.approach_speed(ucm, uvalid, speed)   # ease off as we close in (metric taper)
                motor("forward %.2f" % spd)            # refresh every loop (keeps the daemon dead-man alive)

            # bumped/wedged: driving but the view stops changing -> stuck on something depth can't
            # see. stop-mode stops + reports; steer-mode recovers (reverse+turn) and continues.
            mot = st.get("motion")
            if mot is not None:
                mot_ema = mot if mot_ema is None else 0.7 * mot_ema + 0.3 * mot
            if (mot_ema is not None and (now - start) > 0.6
                    and (now - last_recover) > (ADVANCE_STALL_SECS + 0.7)):   # grace / post-recover cooldown
                if mot_ema < ADVANCE_MOTION_MIN:
                    if stall_since is None:
                        stall_since = now
                    elif now - stall_since >= ADVANCE_STALL_SECS:
                        if not steer:
                            reason = "I bumped into something"
                            break
                        recover_count += 1
                        if recover_count >= MAX_RECOVER:
                            reason = "I got wedged and couldn't free myself"
                            break
                        log("navigate: wedged (motion~%.1f) -> recover #%d" % (mot_ema, recover_count))
                        for cmd, rspd, rsecs in avoid.recovery_plan(st):   # self-expiring timed moves
                            motor("%s %.2f %.2f" % (cmd, rspd, rsecs))
                            await asyncio.sleep(rsecs + 0.05)
                        motor("stop")
                        cur = None
                        stall_since = None
                        mot_ema = None
                        last_recover = time.monotonic()
                        continue
                else:
                    stall_since = None
                    recover_count = 0                  # real progress -> reset the give-up counter
            await asyncio.sleep(0.05)          # ~20Hz guard poll
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        motor("stop")
        log("%s stopped: %s" % ("navigate" if steer else "advance",
                                "cancelled" if cancelled else reason))
    # terminal end only (cancellation re-raises above): tell the model so it can speak it
    await enqueue({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "(camera) stopped moving — %s" % reason}]}})
    await enqueue({"type": "response.create"})


async def guarded_home(speed, enqueue):
    """Drive to the currently-locked visual target. Keeps it centred with proportional steering on
    target.bearing, rolls forward through the shared avoidance so it skirts obstacles, and stops on
    the sonar standoff when it arrives — the same local ~20Hz control as advance/navigate, so the
    cloud model just says 'go to it' and the box does the visual servoing. Ends (and reports) on:
    arrival, losing the target too long, no lock, kill switch, time cap, or lost camera."""
    cancelled = False
    reason = "I couldn't get to it"
    start = time.monotonic()
    label = read_goal_label() or "the target"     # what find_it locked; what we re-find each step
    fails = 0
    last_bearing = 0.0                             # last bearing we actually saw -> commit toward it through misses
    loc_maxdim = LOCATOR_MAXDIM                    # locator frame size for the NEXT look: full res to (re)acquire,
                                                   # small (fast) once cruising on a solid far fix
    try:
        tg = read_target()
        if not (tg and tg.get("active")):
            reason = "I don't have a lock — find it and lock onto it first"
        else:
            while True:
                now = time.monotonic()
                if os.path.exists(DISARM):
                    reason = "the kill switch is on"; break
                if now - start > GOTO_MAX_SECS:
                    reason = "I ran out of time getting to it"; break
                st = read_nav_state()
                if st is None:
                    reason = "I lost the camera feed"; break
                n = st.get("near", {}) or {}
                l, c, r = n.get("l", 0), n.get("c", 0), n.get("r", 0)
                loom = st.get("loom", 0) or 0
                ucm, uvalid = read_ultra()

                # CLAUDE-AUTHORITATIVE steering. The car is stopped here (the previous burst has
                # expired), so this frame is SHARP. We ask Claude where the thing is and steer on THAT,
                # never the CSRT tracker: the tracker can't follow the scene panning during a turn — it
                # silently re-latches on random texture and its bearing spins the car. Claude either
                # finds the thing or says not-found; it never hands back a random jump. We still re-seed
                # the tracker from Claude's box so the dashboard overlay stays honest.
                box = await to_thread(call_locator, label, loc_maxdim)
                size = 0.0
                box_bottom = 0.0
                if box is not None:
                    # got a fresh fix -> steer on it, and remember it to coast on if the next look misses
                    fails = 0
                    seed_box(box, label)
                    write_locate(label, True, box, True)
                    bearing = (box[0] + box[2] / 2.0) - 0.5   # Claude's pixel bearing: - = left, + = right
                    size = box[2] * box[3]
                    box_bottom = box[1] + box[3]              # how low the thing sits in frame; floor objects drop
                                                              # toward the bottom as we close in on them
                    last_bearing = bearing
                    # next look: full res for the precise final aim, fast small frame while cruising far. 'Near' mirrors
                    # the ARRIVAL signals (size, low-in-frame box_bottom, or sonar) so a small FLOOR object — which
                    # arrives via box_bottom, not size — still gets full res for its final approach.
                    near_now = (size >= GOTO_RES_NEAR_SIZE or box_bottom >= GOTO_ARRIVE_Y
                                or avoid.ultra_blocked(ucm, uvalid))
                    loc_maxdim = LOCATOR_MAXDIM if near_now else GOTO_LOC_MAXDIM_FAR
                    # 'centred enough to drive' — loose when the thing is far/small (close the distance first),
                    # tightening to GOTO_BEAR_DEAD as it fills the frame (fine aim only when it's near). So it
                    # drives AT a roughly-ahead target and saves microadjustments for when it's close & accurate.
                    dead = GOTO_BEAR_DEAD + (GOTO_BEAR_DEAD_FAR - GOTO_BEAR_DEAD) * (1.0 - min(1.0, size / GOTO_SIZE_ARRIVE))
                    centered = abs(bearing) < dead
                    # arrived only when lined up AND actually AT the THING: sonar standoff, it fills the frame,
                    # or it has dropped to the bottom of the view. Deliberately NOT camera depth (c>=STOP_NEAR):
                    # the low camera reads the near floorboards as 'close' and that faked arrival on a far ball.
                    if centered and (avoid.ultra_blocked(ucm, uvalid) or size >= GOTO_SIZE_ARRIVE
                                     or box_bottom >= GOTO_ARRIVE_Y):
                        reason = "I'm right up next to it"; break
                else:
                    # missed it this look (small/far things blink out after a move). DON'T quit — commit
                    # toward where we last saw it and re-check as we close in (bigger = easier to spot).
                    fails += 1
                    loc_maxdim = LOCATOR_MAXDIM                # a miss -> full res next look to re-acquire
                    write_locate(label, False, None, False)
                    if avoid.ultra_blocked(ucm, uvalid) and abs(last_bearing) < GOTO_BEAR_DEAD:
                        reason = "I'm right up next to it"; break   # lined up and something's right ahead
                    if fails >= GOTO_BLIND_STEPS:
                        reason = "I lost it and couldn't re-find it"; break
                    bearing = last_bearing
                    centered = abs(bearing) < GOTO_BEAR_DEAD_FAR   # blind: coast forward toward where we last saw it
                                                                   # unless it was clearly off to a side

                log("go_to: %s bear=%+.2f centred=%s size=%.3f bot=%.2f ucm=%s" % (
                    "SEE" if box is not None else "blind", bearing, centered, size, box_bottom, ucm))
                blocked = (c >= avoid.STOP_NEAR or l >= avoid.SIDE_NEAR
                           or r >= avoid.SIDE_NEAR or loom >= avoid.LOOM_BRAKE)
                # PULSED move toward CLAUDE's bearing: one short, timed burst then a pause. The burst
                # self-expires at the daemon, so by the top of the next loop the car is stopped and the
                # frame is SHARP for Claude's next look. 'Move a little, pause, look again' — no steering
                # on a tracker that lies mid-turn, and no driving through blur.
                if blocked and not centered:
                    # obstacle ahead while the target is off to a side -> skirt toward the more open side
                    act, _ = avoid.decide(l, c, r, loom, None, 0.0, now)
                    if act in ("cw", "ccw"):
                        motor("%s %.2f %.2f" % (act, max(0.6, avoid.TURN_SPEED), GOTO_PULSE_TURN))
                        pulse = GOTO_PULSE_TURN
                    else:
                        motor("forward %.2f %.2f" % (max(0.6, avoid.approach_speed(ucm, uvalid, speed)), GOTO_PULSE_DRIVE))
                        pulse = GOTO_PULSE_DRIVE
                elif not centered:
                    # off to a side -> a short, GENTLE turn toward it, its length SCALED by how far off
                    # centre we are: big correction when way off, a hair when nearly lined up. That keeps
                    # one step from swinging past the thing before the next (1-2s later) look lands.
                    turn = "cw" if bearing > 0 else "ccw"
                    frac = min(1.0, abs(bearing) / 0.35)
                    tsecs = max(GOTO_PULSE_TURN_MIN, GOTO_PULSE_TURN * frac)
                    motor("%s %.2f %.2f" % (turn, GOTO_HOME_TURN_SPEED, tsecs))
                    pulse = tsecs
                else:
                    # centred -> normally a short forward burst. BUT if this is a FRESH far/centred fix with a
                    # long CLEAR metric runway, DASH: drive straight continuously under a fast local sonar+camera
                    # guard (NO Claude call) to close the distance quickly, then hand back to the fine loop to
                    # re-centre. The dash never turns and never runs blind — see the entry gate + guards below.
                    cam_clear = (loom < avoid.LOOM_BRAKE and l < avoid.SIDE_NEAR and r < avoid.SIDE_NEAR)
                    if (GOTO_DASH_ENABLE and box is not None and cam_clear
                            and size < GOTO_DASH_SIZE_MAX and box_bottom < GOTO_DASH_MAX_BOTTOM
                            and abs(bearing) < GOTO_DASH_BEAR_DEAD
                            and uvalid and ucm is not None and ucm >= GOTO_DASH_MIN_CLEAR_CM):
                        dash_start = time.monotonic()
                        dwhy = "?"
                        while True:
                            dnow = time.monotonic()
                            if os.path.exists(DISARM):
                                dwhy = "disarm"; break                 # kill switch -> hand back; outer loop reports it
                            if dnow - start > GOTO_MAX_SECS:
                                dwhy = "timecap"; break                # outer 60s cap, re-checked every tick
                            delapsed = dnow - dash_start
                            if delapsed > GOTO_DASH_MAX_SECS:
                                dwhy = "maxsecs"; break                # wall-clock backstop
                            if delapsed * GOTO_DASH_SPEED_CMS >= GOTO_DASH_MAX_CM:
                                dwhy = "maxdist"; break                # HARD blind-distance cap (sonar-invisible target)
                            dcm, dvalid, dage = read_ultra_dash()
                            if dage > GOTO_DASH_SONAR_FRESH or not dvalid or dcm is None:
                                dwhy = "sonar-stale"; break            # never advance on a stale/dead sonar
                            if dcm <= GOTO_DASH_STOP_CM or avoid.ultra_blocked(dcm, dvalid):
                                dwhy = "near"; break                   # hand the final 60->35cm approach to the fine loop
                            dst = read_nav_state()
                            if dst is None:
                                dwhy = "no-cam"; break                 # lost camera feed
                            dn = dst.get("near", {}) or {}
                            if ((dst.get("loom", 0) or 0) >= avoid.LOOM_BRAKE
                                    or dn.get("l", 0) >= avoid.SIDE_NEAR or dn.get("r", 0) >= avoid.SIDE_NEAR):
                                dwhy = "obstacle"; break               # off-axis/looming obstacle the sonar cone may miss
                            # self-expiring timed burst: if this loop stalls/dies the daemon stops the car within
                            # GOTO_DASH_BURST (well under the dead-man), and the sonar taper eases speed near SLOW_CM
                            motor("forward %.2f %.2f" % (max(0.6, avoid.approach_speed(dcm, dvalid, speed)), GOTO_DASH_BURST))
                            await asyncio.sleep(GOTO_DASH_POLL)        # cancellation point (stop -> CancelledError -> finally)
                        motor("stop")
                        log("go_to: DASH end (%s) %.2fs" % (dwhy, time.monotonic() - dash_start))
                        await asyncio.sleep(GOTO_PULSE_SETTLE)         # let motion stop + a sharp frame land
                        continue                                       # back to the fine loop to re-centre / re-check
                    # not dashing -> the existing short forward burst
                    motor("forward %.2f %.2f" % (max(0.6, avoid.approach_speed(ucm, uvalid, speed)), GOTO_PULSE_DRIVE))
                    pulse = GOTO_PULSE_DRIVE
                await asyncio.sleep(pulse + GOTO_PULSE_SETTLE)   # burst runs, then a pause for a sharp frame
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        motor("stop")
        if not cancelled:
            clear_goal()                            # terminal end -> drop the lock
        log("go_to stopped: %s" % ("cancelled" if cancelled else reason))
    await enqueue({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "(camera) %s" % reason}]}})
    await enqueue({"type": "response.create"})


# ---------- camera frames -> data URLs, + cheap change detection ----------
def frame_data_url(maxdim, quality):
    """Read the vision sidecar's frame.jpg, downscale + JPEG-compress, return a data URL."""
    if not _CV2:
        return None
    try:
        img = cv2.imread(FRAME_PATH)  # BGR; atomic write on the producer side, so never partial
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = maxdim / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception as e:
        log("frame encode failed: %s" % e)
        return None


def frame_thumb():
    """Tiny grayscale thumbnail for change detection (None if unavailable)."""
    if not _CV2:
        return None
    try:
        img = cv2.imread(FRAME_PATH, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
    except Exception:
        return None


def image_item(data_url, text, detail):
    """A user message carrying a camera photo (+ short text) for the model to interpret."""
    return {"type": "conversation.item.create", "item": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": text},
                    {"type": "input_image", "image_url": data_url, "detail": detail}]}}


def func_output(call_id, obj):
    return {"type": "conversation.item.create", "item": {
        "type": "function_call_output", "call_id": call_id, "output": json.dumps(obj)}}


async def to_thread(fn, *args):
    """Run a blocking fn off the event loop. asyncio.to_thread is 3.9+; the Jetson
    is on Python 3.8, so use run_in_executor for compatibility."""
    return await asyncio.get_event_loop().run_in_executor(None, fn, *args)


def _log_task_death(task):
    """Surface a background task that died from an unhandled exception (otherwise
    asyncio swallows it and the loop just goes quiet)."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log("[task died] %r" % exc)


# ---------- perception state -> compact text (cheap, constant situational awareness) ----------
def read_nav_state():
    """Perception's latest structured free-space (None if missing or stale)."""
    try:
        with open(NAV_STATE) as f:
            st = json.load(f)
        if time.time() - st.get("ts", 0) > NAV_FRESH:
            return None
        return st
    except Exception:
        return None


def read_ultra():
    """Latest front-sonar reading as (cm, valid). Returns (None, False) if missing/stale, so a
    dead or silent sonar simply hands the stop decision back to the camera."""
    try:
        with open(ULTRA) as f:
            u = json.load(f)
        if time.time() - u.get("ts", 0) > ULTRA_FRESH:
            return None, False
        return u.get("cm"), bool(u.get("valid"))
    except Exception:
        return None, False


def read_ultra_dash():
    """Like read_ultra but ALSO returns the reading's age -> (cm, valid, age_secs). The dash needs the
    age so it can enforce its own stricter freshness (GOTO_DASH_SONAR_FRESH) than ULTRA_FRESH, and abort
    the instant the sonar feed goes quiet. Missing/unparseable -> (None, False, 999.0)."""
    try:
        with open(ULTRA) as f:
            u = json.load(f)
        return u.get("cm"), bool(u.get("valid")), time.time() - u.get("ts", 0)
    except Exception:
        return None, False, 999.0


def seed_box(box, label):
    """Seed perception's tracker with an explicit normalized [x,y,w,h] box (clamped to frame).
    Used by find_it to lock onto WHERE Claude actually saw the object, not a blind centred guess."""
    try:
        x, y, w, h = [float(v) for v in box]
    except Exception:
        return False
    x = max(0.0, min(0.98, x)); y = max(0.0, min(0.98, y))
    w = max(0.02, min(1.0 - x, w)); h = max(0.02, min(1.0 - y, h))
    box = [round(x, 3), round(y, 3), round(w, 3), round(h, 3)]
    tmp = GOAL + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"seed": box, "label": label or "target"}, f)
        os.replace(tmp, GOAL)
        return True
    except Exception:
        return False


def seed_goal(label, size="medium"):
    """Seed the tracker with a CENTRED box (legacy path: lock onto whatever's in the middle)."""
    w, h = LOCK_BOX.get(size, LOCK_BOX["medium"])
    return seed_box([(1.0 - w) / 2.0, (1.0 - h) / 2.0, w, h], label)


def write_locate(thing, found, box, locked):
    """Publish find_it's last result so the debug dashboard can draw the box Claude reported."""
    tmp = LOCATE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"ts": time.time(), "thing": thing, "found": bool(found),
                       "box": box, "locked": bool(locked)}, f)   # box = normalized [x,y,w,h] or None
        os.replace(tmp, LOCATE)
    except Exception:
        pass


def clear_goal():
    """Drop the lock -> perception clears its tracker (target goes inactive)."""
    try:
        os.remove(GOAL)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def read_target():
    """Latest tracker target dict (active/bearing/size/lost) or None."""
    st = read_nav_state()
    return (st or {}).get("target") if st else None


def read_goal_label():
    """The label of the thing currently seeded/locked (what find_it was asked for), or None."""
    try:
        with open(GOAL) as f:
            return json.load(f).get("label")
    except Exception:
        return None


def box_center_dist(a, b):
    """Distance between the centres of two normalized [x,y,w,h] boxes; 1.0 if either is missing."""
    try:
        ax, ay = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
        bx, by = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    except Exception:
        return 1.0


def surroundings_summary():
    """Compact text picture of what the box senses right now, for the planner to reason over."""
    st = read_nav_state() or {}
    if not st:
        return "no live sensor data"
    n = st.get("near", {}) or {}
    ucm, uvalid = read_ultra()
    tg = st.get("target") or {}
    where = {"l": "left", "c": "center", "r": "right"}.get(st.get("clearest", "c"), "center")
    parts = ["closeness (higher=closer; ~%d blocks forward): left=%d center=%d right=%d"
             % (avoid.STOP_NEAR, n.get("l", 0), n.get("c", 0), n.get("r", 0)),
             "most open direction: %s" % where]
    if st.get("boxed_in"):
        parts.append("BOXED IN — tight on all sides")
    if uvalid and ucm is not None:
        parts.append("sonar dead-ahead: %d cm" % round(ucm))
    elif ucm is None:
        parts.append("sonar: clear ahead (nothing within ~3.4 m)")
    if tg.get("active"):
        parts.append("tracked target " + ("LOST from view" if tg.get("lost")
                     else "in view (bearing %.2f, -=left/+=right)" % tg.get("bearing", 0.0)))
    if (st.get("loom", 0) or 0) > LOOM_TEXT:
        parts.append("something looming closer ahead")
    return "; ".join(parts)


PLANNER_SYSTEM = (
    "You are the reasoning planner for a small four-wheeled robot car with a forward camera and a "
    "front-facing ultrasonic sensor. A faster voice model drives the car and calls you when it is "
    "stuck or a move needs real multi-step spatial planning. Think spatially and hand back a short, "
    "concrete plan it can execute. The car has NO map, NO compass or odometry, and open-loop motors: "
    "it cannot turn precise angles or measure distance travelled, so plan in small steps with "
    "re-checks — never absolute angles or distances. When a camera image is attached, READ THE SCENE "
    "from it directly — what's where, which way is open, where the goal likely is — and ground your plan "
    "in what you actually see. Be concise: a few short numbered steps, no preamble.")


def grab_frame_b64(maxdim):
    """Latest camera frame as a base64 JPEG (long edge <= maxdim) for a vision API call, or None."""
    if not _CV2:
        return None
    try:
        img = cv2.imread(FRAME_PATH)
        if img is None:
            return None
        h0, w0 = img.shape[:2]
        s = maxdim / float(max(h0, w0))
        if s < 1.0:
            img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode() if ok else None
    except Exception:
        return None


def call_locator(thing, maxdim=LOCATOR_MAXDIM):
    """Ask Claude (Opus vision) for a bounding box of `thing` in the CURRENT frame. Returns a
    normalized [x,y,w,h] box to seed the tracker with, or None if Claude can't see it / on any
    failure (the caller then sweeps and retries). Blocking — run via to_thread."""
    if not _CV2:
        return None
    try:
        key = open(PLANNER_KEY_FILE).read().strip()
    except Exception:
        return None
    try:
        img = cv2.imread(FRAME_PATH)              # latest frame (atomic write on the producer side)
        if img is None:
            return None
        h0, w0 = img.shape[:2]
        scale = maxdim / float(max(h0, w0))
        if scale < 1.0:
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
    except Exception:
        return None
    prompt = ("This is a %dx%d camera image (origin top-left, +x right, +y down). Find: %s.\n"
              'If it is clearly visible, reply with ONLY {"found": true, "box": [x0, y0, x1, y1]} — a '
              'TIGHT pixel box around it. If it is NOT clearly visible, reply with ONLY {"found": false}. '
              "Do not guess or invent a box. No other text." % (W, H, thing))
    body = json.dumps({"model": LOCATOR_MODEL, "max_tokens": 300,
                       "messages": [{"role": "user", "content": [
                           {"type": "image", "source": {"type": "base64",
                            "media_type": "image/jpeg", "data": b64}},
                           {"type": "text", "text": prompt}]}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST")
    req.add_header("x-api-key", key)
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("content-type", "application/json")
    try:
        r = urllib.request.urlopen(req, timeout=LOCATOR_TIMEOUT)
        d = json.load(r)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        if not obj.get("found") or not obj.get("box"):
            return None
        x0, y0, x1, y1 = [float(v) for v in obj["box"]]
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]   # -> normalized [x,y,w,h]
    except Exception as e:
        log("locator error: %s" % type(e).__name__)
        return None


def call_planner(situation, summary, history=None):
    """Blocking Claude call (run via to_thread). Returns a short plan string, or None on any failure
    so the caller falls back to the voice model's own judgement. `history` = the recent plans this
    attempt so the planner can BUILD ON them (or change tack) instead of handing back the same plan."""
    try:
        key = open(PLANNER_KEY_FILE).read().strip()
    except Exception:
        return None
    prior = ""
    if history:
        prior = ("\n\nEarlier this attempt (these did NOT get it unstuck — build on them or try a "
                 "different tack, don't just repeat):\n" + "\n".join(
                     "- for \"%s\": %s" % (h["situation"], " ".join(h["plan"].split())[:200])
                     for h in history))
    frame_b64 = grab_frame_b64(LOCATOR_MAXDIM)   # give the planner the actual camera view to reason over
    seeing = ("You are looking at the robot's current forward camera view (the attached image).\n\n"
              if frame_b64 else "")
    user = ("%sSituation / goal: %s\n\nWhat its sensors report: %s%s\n\n"
            "Give a short plan (2-5 numbered steps) using ONLY these actions: find_it then go_to (find a "
            "named thing and drive to it), scan left/right (turn a small step and look), advance (roll up "
            "to something and stop), navigate (wander forward around obstacles), drive (one small timed "
            "nudge), stop. Small steps, re-check as you go." % (seeing, situation, summary, prior))
    content = []
    if frame_b64:
        content.append({"type": "image", "source": {"type": "base64",
                        "media_type": "image/jpeg", "data": frame_b64}})
    content.append({"type": "text", "text": user})
    body = json.dumps({"model": PLANNER_MODEL, "max_tokens": PLANNER_MAXTOK,
                       "system": PLANNER_SYSTEM,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST")
    req.add_header("x-api-key", key)
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("content-type", "application/json")
    try:
        r = urllib.request.urlopen(req, timeout=PLANNER_TIMEOUT)
        d = json.load(r)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
        return text or None
    except Exception as e:
        log("planner error: %s" % type(e).__name__)
        return None


def scene_summary(st):
    """One short line describing what the robot senses, for near-constant awareness."""
    clearest = st.get("clearest", "c")
    loom = st.get("loom", 0) or 0
    tg = st.get("target", {}) or {}
    dirword = {"l": "left", "c": "center", "r": "right"}.get(clearest, "center")
    parts = ["most open %s" % dirword]
    if st.get("boxed_in"):
        parts.append("boxed in — tight all around")
    ucm, uvalid = read_ultra()
    if uvalid and ucm is not None and ucm < 80:      # only when notably close, to keep the line short
        parts.append("something ~%d cm dead ahead" % round(ucm))
    if loom > LOOM_TEXT:
        parts.append("something getting closer ahead")
    if tg.get("active"):
        if tg.get("lost"):
            parts.append("lost the camera lock")
        else:
            b = tg.get("bearing", 0.0)
            where = "centred" if abs(b) < 0.08 else ("to the left" if b < 0 else "to the right")
            parts.append("camera locked on something %s (not confirmed — look to see what it is)" % where)
    return "; ".join(parts)


# ---------- audio playback (paplay raw); barge-in = kill+respawn to drop buffer ----------
class Player:
    def __init__(self, sink):
        self.sink = sink
        self.proc = None

    async def _spawn(self):
        cmd = ["paplay", "--raw", "--format=s16le", "--rate=%d" % RATE, "--channels=1"]
        if self.sink:
            cmd.append("--device=" + self.sink)
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)

    async def write(self, pcm):
        if self.proc is None or self.proc.returncode is not None:
            await self._spawn()
        try:
            self.proc.stdin.write(pcm)
            await self.proc.stdin.drain()
        except Exception:
            await self._spawn()

    async def barge_in(self):
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    async def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass


async def main():
    log("realtime sidecar starting: model=%s voice=%s smoke=%s" % (MODEL, VOICE, SMOKE))
    key = read_key()
    src = wait_pulse("sources", SOURCE_MATCH)
    sink = wait_pulse("sinks", SINK_MATCH)
    log("audio: mic=%s speaker=%s" % (src or "default", sink or "default"))
    if not src or not sink:
        log("WARNING: Jabra %s not found — using default (audio may go nowhere)"
            % ("mic" if not src else "speaker"))

    url = "wss://api.openai.com/v1/realtime?model=%s" % MODEL
    headers = {"Authorization": "Bearer " + key}   # GA API: no OpenAI-Beta header (beta shape is disabled)

    player = Player(sink)
    stop_event = asyncio.Event()

    async def session_once():
        """One connection lifecycle; returns when the socket drops or stop_event fires."""
        send_q = asyncio.Queue()          # fresh per-connection queue (drops stale events on reconnect)
        last_activity = {"t": 0.0}        # monotonic time of last voice activity; gates ambient pushes
        adv = {"task": None, "kind": None}  # in-flight guarded task (+ kind: "home" go_to / "fwd" advance-navigate)
        scan_st = {"last": 0.0, "steps": 0, "last_turn": 0.0}   # last scan time + steps + last actual-turn time
        think_st = {"last": 0.0, "history": []}                 # last plan time + recent plans (planner memory)

        async def enqueue(evt):
            await send_q.put(evt)

        async def cancel_advance():
            """Halt any running guarded advance and wait for it to release the motors,
            so a following stop/drive never races the advance's own stop."""
            tk = adv["task"]
            if tk is not None and not tk.done():
                tk.cancel()
                try:
                    await tk
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            adv["task"] = None

        async with ws_connect(url, additional_headers=headers, max_size=None,
                              ping_interval=15, ping_timeout=30) as ws:
            log("connected")

            # configure the session
            await enqueue({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": MODEL,
                    "instructions": INSTRUCTIONS,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": RATE},
                            "turn_detection": {"type": "semantic_vad"},
                            "transcription": {"model": "gpt-4o-mini-transcribe"},
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": RATE},
                            "voice": VOICE,
                        },
                    },
                    "tools": TOOLS,
                },
            })

            async def sender():
                while not stop_event.is_set():
                    evt = await send_q.get()
                    if evt is None:
                        break
                    await ws.send(json.dumps(evt))

            async def mic():
                cmd = ["parec", "--format=s16le", "--rate=%d" % RATE, "--channels=1"]
                if src:
                    cmd.append("--device=" + src)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                try:
                    while not stop_event.is_set():
                        data = await proc.stdout.read(4800)   # ~100ms
                        if not data:
                            break
                        await enqueue({"type": "input_audio_buffer.append",
                                       "audio": base64.b64encode(data).decode()})
                finally:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            async def receiver():
                async for raw in ws:
                    evt = json.loads(raw)
                    t = evt.get("type", "")
                    if t in ("input_audio_buffer.speech_started",
                             "conversation.item.input_audio_transcription.completed",
                             "response.output_audio.delta", "response.output_audio_transcript.done"):
                        last_activity["t"] = time.monotonic()   # keep ambient pushes clear of live conversation
                    if t == "session.created":
                        log("[session created]")
                    elif t == "session.updated":
                        log("[session configured]")
                        if SMOKE:
                            await enqueue({"type": "conversation.item.create", "item": {
                                "type": "message", "role": "user",
                                "content": [{"type": "input_text",
                                             "text": "Say exactly: Realtime link online."}]}})
                            await enqueue({"type": "response.create"})
                    elif t == "input_audio_buffer.speech_started":
                        await player.barge_in()          # user talking -> drop our audio
                    elif t == "response.output_audio.delta":
                        await player.write(base64.b64decode(evt["delta"]))
                    elif t == "conversation.item.input_audio_transcription.completed":
                        log("You: %s" % evt.get("transcript", "").strip())
                    elif t == "response.output_audio_transcript.done":
                        log("Car: %s" % evt.get("transcript", "").strip())
                    elif t == "response.function_call_arguments.done":
                        name = evt.get("name", "")
                        call_id = evt.get("call_id", "")
                        arguments = evt.get("arguments", "")
                        log("[tool] %s(%s)" % (name, arguments))
                        # while go_to is autonomously driving to the locked thing, the model tends to fire
                        # scan/look/find_it and CANCEL it mid-approach (any of them calls cancel_advance).
                        # Refuse those until it finishes, so it can't sabotage its own drive. stop still halts
                        # it (handled below), and it self-reports on arrival / real loss.
                        if (name in ("look", "scan", "find_it", "lock_on")
                                and adv["task"] is not None and not adv["task"].done()
                                and adv.get("kind") == "home"):
                            await enqueue(func_output(call_id, {"ok": False,
                                "note": "I'm driving to it right now — hold on. I'll tell you when I arrive, "
                                        "or if I lose it and need to look again."}))
                            await enqueue({"type": "response.create"})
                        elif name == "look":
                            # ack the call, attach a fresh crisp photo, then let it respond.
                            # cv2 encode runs in a thread so it never stalls audio playback.
                            durl = await to_thread(frame_data_url, LOOK_MAXDIM, LOOK_QUALITY)
                            await enqueue(func_output(call_id, {"ok": durl is not None,
                                                               "note": "camera photo attached" if durl else "camera unavailable"}))
                            if durl:
                                await enqueue(image_item(durl, "Current camera view (look):", LOOK_DETAIL))
                            await enqueue({"type": "response.create"})
                        elif name == "scan":
                            # search step. Registration-before-movement: the FIRST scan of a sweep just
                            # hands over the CURRENT view without turning, so the model registers where it
                            # is before the bot ever moves. Each later scan means "I've processed the last
                            # view, keep going" — so it turns a step, then hands over the new view. This
                            # gates every move on a registered image and stops it running a step ahead.
                            await cancel_advance()   # scanning owns the motors while it runs
                            sargs = args_of(arguments)
                            side = sargs.get("direction", "right")
                            if side not in ("left", "right"):
                                side = "right"
                            amount = sargs.get("amount", "step")
                            if amount not in ("step", "center"):
                                amount = "step"
                            turn = "ccw" if side == "left" else "cw"   # left = ccw, right = cw
                            disarmed = os.path.exists(DISARM)
                            now_m = time.monotonic()
                            prev_last = scan_st["last"]
                            scan_st["last"] = now_m
                            if amount == "center":
                                # AUTO-centre off the tracker's pixel-measured bearing, NOT the model's
                                # left/right (which it gets wrong). Lock on first if needed (a centred box
                                # captures a roughly-central target), then turn toward the measured bearing
                                # a hair at a time. The model just calls this until it's told 'centred'.
                                if disarmed:
                                    await asyncio.sleep(SCAN_NUDGE_SETTLE)
                                    durl = await to_thread(frame_data_url, LOOK_MAXDIM, LOOK_QUALITY)
                                    note = "kill switch is on — I can't centre, but here's the current view"
                                    cap = "Current camera view (can't turn — kill switch on):"
                                else:
                                    tg = read_target()
                                    if not (tg and tg.get("active")):
                                        seed_goal("target", "medium")   # centred box -> locks a roughly-central target
                                        for _ in range(12):              # give the tracker a few frames to grab it
                                            await asyncio.sleep(0.08)
                                            tg = read_target()
                                            if tg and tg.get("active"):
                                                break
                                    if tg and tg.get("active") and not tg.get("lost"):
                                        bearing = tg.get("bearing", 0.0)   # pixel-measured: - = left, + = right
                                        if abs(bearing) <= GOTO_BEAR_DEAD:
                                            durl = await to_thread(frame_data_url, LOOK_MAXDIM, LOOK_QUALITY)
                                            note = ("something is centred and locked — but the lock grabs whatever "
                                                    "is in the MIDDLE, which may NOT be what you're after. LOOK at "
                                                    "this photo and be sure the centred thing really is it: if yes, "
                                                    "go to it; if it's something else (a person, a chair...), the "
                                                    "thing you want is still off to a side — scan a 'step' to keep looking.")
                                            cap = "Camera view — target centred:"
                                        else:
                                            autoside = "right" if bearing > 0 else "left"
                                            turn = "cw" if bearing > 0 else "ccw"   # toward its TRUE side (bearing sign)
                                            wait = SCAN_NUDGE_INTERVAL - (now_m - scan_st["last_turn"])
                                            if wait > 0:
                                                await asyncio.sleep(wait)
                                            scan_st["last_turn"] = time.monotonic()
                                            motor("%s %.2f %.2f" % (turn, SCAN_TURN_SPEED, SCAN_NUDGE_SECS))
                                            await asyncio.sleep(SCAN_NUDGE_SECS + SCAN_NUDGE_SETTLE)
                                            durl = await to_thread(frame_data_url, LOOK_MAXDIM, LOOK_QUALITY)
                                            note = ("centring the locked thing automatically (it's to the %s of "
                                                    "centre) — scan 'center' again to keep going. Check the photo "
                                                    "that the thing I locked is the one you want; if it isn't, scan "
                                                    "a 'step' to keep looking instead." % autoside)
                                            cap = "Camera view after an auto-centring nudge %s:" % autoside
                                    else:
                                        durl = await to_thread(frame_data_url, LOOK_MAXDIM, LOOK_QUALITY)
                                        note = ("couldn't lock on to centre it — it's probably too near the edge. Sweep "
                                                "one 'step' toward it to bring it inward, then scan 'center' again.")
                                        cap = "Current camera view (no lock to centre on):"
                            else:
                                # sweeping = still SEARCHING, so drop any existing lock: it was at best a
                                # guess at whatever sat in the middle, and letting it linger makes the ambient
                                # notes keep claiming a "target" that may be the wrong thing (a person, a chair).
                                clear_goal()
                                fresh_sweep = (now_m - prev_last) > SCAN_SWEEP_GAP
                                if fresh_sweep:
                                    scan_st["steps"] = 0
                                did_turn = (not disarmed) and (not fresh_sweep)
                                if did_turn:
                                    # hard pace: don't turn again until SCAN_MIN_INTERVAL since the last turn, so
                                    # the bot can't scan faster than this no matter how fast the model calls
                                    wait = SCAN_MIN_INTERVAL - (now_m - scan_st["last_turn"])
                                    if wait > 0:
                                        await asyncio.sleep(wait)
                                    scan_st["steps"] += 1
                                    scan_st["last_turn"] = time.monotonic()
                                    motor("%s %.2f %.2f" % (turn, SCAN_TURN_SPEED, SCAN_TURN_SECS))  # timed, self-expiring
                                    await asyncio.sleep(SCAN_TURN_SECS + SCAN_SETTLE)
                                else:
                                    await asyncio.sleep(SCAN_SETTLE)   # settle for a clean frame even without turning
                                durl = await to_thread(frame_data_url, LOOK_MAXDIM, LOOK_QUALITY)
                                steps = scan_st["steps"]
                                if disarmed:
                                    note = "kill switch is on — I can't turn, but here's the current view"
                                    cap = "Current camera view (can't turn — kill switch on):"
                                elif did_turn:
                                    note = ("turned a step %s (step %d of ~%d for a full circle); decide before "
                                            "scanning again. If you haven't found it, keep scanning the same way — "
                                            "you've only covered part of the way around." % (side, steps, SCAN_FULL_TURN_STEPS))
                                    cap = "Camera view after turning a step %s (scan step %d/~%d):" % (side, steps, SCAN_FULL_TURN_STEPS)
                                else:
                                    note = "here's the view from where I am (start of sweep); scan again to turn a step %s" % side
                                    cap = "Current camera view (start of sweep, not turned yet):"
                            await enqueue(func_output(call_id, {"ok": durl is not None,
                                                                "note": note if durl else "camera unavailable"}))
                            if durl:
                                await enqueue(image_item(durl, cap, LOOK_DETAIL))
                            await enqueue({"type": "response.create"})
                        elif name in ("advance", "navigate"):
                            # continuous, locally-reactive driving: hand off to a background task.
                            # advance = roll forward + STOP when close; navigate = STEER AROUND
                            # obstacles (avoid.decide). Reports back on a terminal stop.
                            await cancel_advance()
                            spd = _clamp(args_of(arguments).get("speed", 1.0), 0.6, 1.0, 1.0)
                            steer = (name == "navigate")
                            adv["task"] = asyncio.create_task(guarded_forward(spd, enqueue, steer))
                            adv["kind"] = "fwd"
                            adv["task"].add_done_callback(_log_task_death)
                            await enqueue(func_output(call_id, {"ok": True, "detail": (
                                "on my way; steering around obstacles" if steer
                                else "rolling forward; I'll stop when I'm close to something")}))
                        elif name == "find_it":
                            # Claude (Opus vision) locates the NAMED thing in the current frame and hands
                            # back a box; we seed the tracker on exactly that, so it locks the real object
                            # wherever it is — not a blind centred guess. Stop + settle first for a crisp
                            # frame, then confirm the tracker latched before telling the model to drive.
                            thing = args_of(arguments).get("thing", "").strip()
                            if not thing:
                                await enqueue(func_output(call_id, {"ok": False,
                                    "note": "tell me what to find (e.g. 'a red mug')"}))
                                await enqueue({"type": "response.create"})
                            else:
                                await cancel_advance()
                                motor("stop")
                                await asyncio.sleep(0.25)                 # let a crisp, still frame land
                                box = await to_thread(call_locator, thing)
                                if box is None:
                                    clear_goal()                          # no stale lock left behind
                                    write_locate(thing, False, None, False)   # dashboard: "looked, didn't see it"
                                    await enqueue(func_output(call_id, {"ok": False, "found": False,
                                        "note": ("I don't see %s from here. Sweep one scan 'step' to look at a "
                                                 "new part of the room, then find_it again." % thing)}))
                                    await enqueue({"type": "response.create"})
                                else:
                                    locked = False
                                    if seed_box(box, thing):
                                        for _ in range(24):               # ~1.2 s for the tracker to latch
                                            await asyncio.sleep(0.05)
                                            tg = read_target()
                                            if tg and tg.get("active") and not tg.get("lost"):
                                                locked = True; break
                                    if not locked:
                                        clear_goal()
                                    write_locate(thing, True, box, locked)   # dashboard: draw the box Claude saw
                                    await enqueue(func_output(call_id, {"ok": locked, "found": True, "locked": locked,
                                        "note": ("found %s and locked onto it — call go_to to drive to it" % thing
                                                 if locked else
                                                 "I spotted %s but couldn't get a solid lock — try find_it again" % thing)}))
                                    await enqueue({"type": "response.create"})
                        elif name == "lock_on":
                            # seed perception's tracker on the (already-centred) target, then confirm
                            # it actually latched before telling the model to drive.
                            a = args_of(arguments)
                            locked = False
                            if seed_goal(a.get("label", "target"), a.get("size", "medium")):
                                for _ in range(24):            # ~1.2s for the tracker to latch
                                    await asyncio.sleep(0.05)
                                    tg = read_target()
                                    if tg and tg.get("active") and not tg.get("lost"):
                                        locked = True; break
                            if not locked:
                                clear_goal()
                            await enqueue(func_output(call_id, {"ok": locked, "note": (
                                "locked on — call go_to to drive to it" if locked
                                else "couldn't lock on; get it centred and try lock_on again")}))
                            await enqueue({"type": "response.create"})
                        elif name == "go_to":
                            # home in on the locked target (background task, like advance/navigate)
                            await cancel_advance()
                            spd = _clamp(args_of(arguments).get("speed", 1.0), 0.6, 1.0, 1.0)
                            adv["task"] = asyncio.create_task(guarded_home(spd, enqueue))
                            adv["kind"] = "home"
                            adv["task"].add_done_callback(_log_task_death)
                            await enqueue(func_output(call_id, {"ok": True,
                                "detail": "on it — I'll turn to face it and drive there myself; just wait for my update"}))
                        elif name == "think":
                            # route a hard spatial decision to the stronger planner (Claude), giving it
                            # the live surroundings AND the recent plans this attempt so it iterates
                            # instead of repeating itself; hand its plan back for the voice model to run.
                            sit = args_of(arguments).get("situation", "").strip() or "decide what to do next"
                            now_t = time.monotonic()
                            since = now_t - think_st["last"]
                            if since < THINK_COOLDOWN and think_st["history"]:
                                # just planned -> don't burn another call; carry the last plan out first
                                last = think_st["history"][-1]["plan"]
                                await enqueue(func_output(call_id, {"ok": True,
                                    "note": ("you planned %ds ago — carry this out and re-check before thinking "
                                             "again:\n%s" % (int(since), last))}))
                                await enqueue({"type": "response.create"})
                            else:
                                think_st["last"] = now_t
                                plan = await to_thread(call_planner, sit, surroundings_summary(),
                                                       think_st["history"])
                                if plan:
                                    think_st["history"].append({"situation": sit, "plan": plan})
                                    think_st["history"] = think_st["history"][-2:]   # keep the last couple
                                note = ("plan:\n" + plan if plan else
                                        "planner unavailable — use your own judgement: sweep with scan; if boxed "
                                        "in, head toward the most open direction and re-check as you go")
                                await enqueue(func_output(call_id, {"ok": plan is not None, "note": note}))
                                await enqueue({"type": "response.create"})
                        else:
                            await cancel_advance()   # drive/stop preempt a guarded advance
                            result = invoke_tool(name, arguments)
                            log("[tool result] %s" % json.dumps(result)[:160])
                            await enqueue(func_output(call_id, result))
                            await enqueue({"type": "response.create"})
                    elif t == "response.done":
                        if SMOKE:
                            await asyncio.sleep(3)        # let playback drain
                            stop_event.set()
                            return
                    elif t == "error":
                        log("[error] %s" % json.dumps(evt.get("error", evt))[:300])

            async def ambient_text():
                """Cheap, near-constant situational awareness: summarize perception's
                nav_state as one short line and drop it into context when it changes.
                Text (not images) -> ~free, fast, and doesn't clog the voice turn-taking."""
                prev = None
                last_push = 0.0
                while not stop_event.is_set():
                    await asyncio.sleep(TEXT_POLL)
                    st = read_nav_state()
                    if st is None:
                        continue
                    summary = scene_summary(st)
                    now = time.monotonic()
                    if (summary != prev and (now - last_push) > TEXT_MIN_INTERVAL
                            and (now - last_activity["t"]) > TEXT_QUIET_GAP):
                        await enqueue({"type": "conversation.item.create", "item": {
                            "type": "message", "role": "user",
                            "content": [{"type": "input_text", "text": "(camera) " + summary}]}})
                        prev = summary
                        last_push = now
                        log("ambient text: %s" % summary)

            # ---- text chat input: typed messages from the debug dashboard, injected into the
            # session exactly like spoken input (so the model replies + can call the same tools) ----
            loop = asyncio.get_event_loop()
            chat_sock = None
            try:
                try:
                    os.unlink(CHAT_SOCK)
                except OSError:
                    pass
                chat_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                chat_sock.setblocking(False)
                chat_sock.bind(CHAT_SOCK)
                os.chmod(CHAT_SOCK, 0o777)           # allow the dashboard (cross-uid) to send

                def _on_chat():
                    try:
                        data, _ = chat_sock.recvfrom(65536)
                    except Exception:
                        return
                    text = data.decode("utf-8", "ignore").strip()
                    if not text:
                        return
                    log("You (chat): %s" % text)
                    last_activity["t"] = time.monotonic()   # keep ambient pushes clear of this turn
                    try:
                        send_q.put_nowait({"type": "conversation.item.create", "item": {
                            "type": "message", "role": "user",
                            "content": [{"type": "input_text", "text": text}]}})
                        send_q.put_nowait({"type": "response.create"})
                    except Exception:
                        pass

                loop.add_reader(chat_sock.fileno(), _on_chat)
                log("chat input ready")
            except Exception as e:
                log("chat input setup failed: %s" % e)

            tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
            if not SMOKE:
                tasks.append(asyncio.create_task(mic()))
                tasks.append(asyncio.create_task(ambient_text()))
            for tk in tasks:
                tk.add_done_callback(_log_task_death)

            waiter = asyncio.create_task(stop_event.wait())
            try:
                # proceed as soon as stop_event fires OR any task ends (e.g. a disconnect)
                await asyncio.wait(tasks + [waiter], return_when=asyncio.FIRST_COMPLETED,
                                   timeout=35 if SMOKE else None)
                if SMOKE:
                    stop_event.set()
            finally:
                if chat_sock is not None:
                    try:
                        loop.remove_reader(chat_sock.fileno())
                    except Exception:
                        pass
                    try:
                        chat_sock.close()
                    except Exception:
                        pass
                await cancel_advance()   # never leave the car rolling if the socket drops
                for tk in tasks:
                    tk.cancel()
                waiter.cancel()
                await asyncio.gather(*tasks, waiter, return_exceptions=True)

    # ---- reconnect loop: survive Wi-Fi blips / keepalive ping timeouts ----
    try:
        while not stop_event.is_set():
            try:
                await session_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log("connection lost: %r" % e)
            if SMOKE or stop_event.is_set():
                break
            log("reconnecting in 2s...")
            await asyncio.sleep(2)
    finally:
        await player.close()
        log("realtime sidecar stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
