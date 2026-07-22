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
import sys
import json
import time
import base64
import socket
import asyncio
import subprocess
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
WORKSPACE = env("ROBOT_WORKSPACE", os.path.join(HOME, "robot", "workspace"))
MOVE = os.path.join(WORKSPACE, "move")
DISARM = os.path.join(WORKSPACE, ".disarmed")       # kill switch flag (move disarm)
FRAME_PATH = os.path.join(WORKSPACE, "frame.jpg")   # written by the perception loop
NAV_STATE = os.path.join(WORKSPACE, "nav_state.json")  # perception's structured free-space output
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
# scan = "turn a small step, then look" — the search primitive (spin-until-you-see-it). Deliberately
# a small step so it doesn't overshoot; the model calls it repeatedly and reads the photo each time.
SCAN_TURN_SPEED = float(env("SCAN_TURN_SPEED", "0.7"))
SCAN_TURN_SECS = float(env("SCAN_TURN_SECS", "0.2"))   # rotation per step; tune on the floor (turns are fast)
SCAN_SETTLE = float(env("SCAN_SETTLE", "0.35"))        # let blur clear + a fresh frame.jpg land before the photo
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
    "bursts or babysit them. Use the drive tool with direction forward ONLY for a tiny deliberate nudge. For "
    "turning and strafing, use the drive tool with a direction, "
    "a speed from 0.6 to 1.0 (NEVER below 0.6 — below that the motors just stall and buzz and the car "
    "won't move; default to full speed 1.0 unless the user asks to go slower), and a short duration "
    "in seconds — prefer brief bursts (0.5 to 1.5 s) and re-check rather than long "
    "blind drives. Call stop the instant the user says stop (it also cancels an advance). "
    "VISION: you continuously receive short text camera updates like '(camera) most open left; target "
    "in view on the right' — which direction is most open, whether something is getting closer, and "
    "whether a tracked target is in view. This is your CONSTANT background awareness, so you almost "
    "always already know the current scene without looking; use it directly. Do NOT call look for "
    "ordinary moves or to check those things — you already have them. Call the look tool ONLY when the "
    "user asks what you see or you must actually identify/describe something with your own eyes. "
    "SEARCHING / FINDING: when asked to look around, spin until you see something, or find or identify a "
    "thing, use the scan tool, NOT drive. Each scan call turns the car a small step to the LEFT or RIGHT "
    "(you pick) and hands you a fresh photo. Choose the direction on purpose, toward where the thing "
    "should be: if you just turned right and it was ahead of you, scan left to bring it back; if you last "
    "saw it on one side, scan that way. Look at each photo; if what you're after isn't in it, scan again, "
    "step by step (a full turn is several scans). Getting it CLEARLY in view matters: if it first shows up "
    "as just a sliver at the edge of the frame, you've only caught the very start of it — do NOT act yet. "
    "Keep scanning a small step or two more the same way until it is fully in view and roughly centered, "
    "and only THEN stop scanning and either say what it is or head toward it. Don't take big blind turns to "
    "search. Chain the scans yourself — scan, read the view, decide, scan again — without waiting to be "
    "prompted. "
    "GOING TO SOMETHING (a mission): when asked to find something and go to it, run the whole job yourself "
    "as a loop, speaking only a short update at each stage. First sweep with scan to find it. If a full "
    "sweep turns up nothing, turn toward the most open direction the camera reports and navigate forward a "
    "little to a new spot, then sweep again from there; repeat until you find it. Once it is clearly and "
    "fully in view (not just a sliver at the edge), turn to put it roughly straight ahead (scan toward it) "
    "before driving, then navigate toward it, and every so often scan "
    "again to re-center it as it drifts, since navigate only rolls forward and around obstacles. When you "
    "get close, use advance so you stop just short of it. Keep the loop going until you've reached it or "
    "it's clearly not findable — don't stop after one step or wait to be told to keep going."
)

TOOLS = [
    {"type": "function", "name": "drive",
     "description": "Drive/turn the car in one direction. Runs in the background (non-blocking) for the given duration; you stay aware while it moves. Any new drive replaces the current motion, and stop halts it early.",
     "parameters": {"type": "object", "properties": {
         "direction": {"type": "string", "enum": ["forward", "back", "left", "right", "cw", "ccw"],
                       "description": "forward/back, left/right = strafe, cw/ccw = rotate in place"},
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
     "description": "Turn a small step to the LEFT or RIGHT (you choose) and take a photo — the way to look around for something. Use this whenever asked to spin, turn, or look around until you see, find, or identify something. Each call rotates the car a little the chosen way and hands you a fresh camera view; look at it, and if what you want isn't there, call scan again. Keep calling it to sweep (a full turn is several scans). Pick the direction deliberately, toward where the thing should be: if you just turned right and it was ahead of you, scan left to bring it back; if you last saw it on your left, scan left. The instant you see it, stop scanning and either say what it is or head toward it. Never use big blind turns to search.",
     "parameters": {"type": "object", "properties": {
         "direction": {"type": "string", "enum": ["left", "right"],
                       "description": "which way to turn this step — left or right (default right)"}},
         "required": []}},
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


def invoke_tool(name, args_json):
    args = args_of(args_json)
    try:
        if name == "drive":
            d = args.get("direction", "forward")
            spd = _clamp(args.get("speed", 1.0), 0.6, 1.0, 1.0)   # 0.6 floor: below this the motors stall/buzz
            secs = _clamp(args.get("seconds", 1.0), 0.1, 2.0, 1.0)
            motor("%s %.2f %.2f" % (d, spd, secs))   # timed move: the daemon drives for secs then stops
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
                spd = speed if act == "forward" else max(0.6, speed)   # 0.6 floor: below it the motors stall
                motor("%s %.2f" % (act, spd))
                cur = act
            else:
                # stop-mode: same 'blocked' test decide() uses, but we STOP instead of turning
                if (c >= avoid.STOP_NEAR or l >= avoid.SIDE_NEAR
                        or r >= avoid.SIDE_NEAR or loom >= avoid.LOOM_BRAKE):
                    reason = "something's right ahead"
                    break
                motor("forward %.2f" % speed)          # refresh every loop (keeps the daemon dead-man alive)

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
        if time.time() - st.get("ts", 0) > 1.0:
            return None
        return st
    except Exception:
        return None


def scene_summary(st):
    """One short line describing what the camera sees, for near-constant awareness."""
    clearest = st.get("clearest", "c")
    loom = st.get("loom", 0) or 0
    tg = st.get("target", {}) or {}
    dirword = {"l": "left", "c": "center", "r": "right"}.get(clearest, "center")
    parts = ["most open %s" % dirword]
    if loom > LOOM_TEXT:
        parts.append("something getting closer ahead")
    if tg.get("active"):
        if tg.get("lost"):
            parts.append("target lost")
        else:
            b = tg.get("bearing", 0.0)
            where = "centered" if abs(b) < 0.08 else ("on the left" if b < 0 else "on the right")
            parts.append("target in view %s" % where)
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
        adv = {"task": None}              # in-flight guarded-advance task, if any

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
                        if name == "look":
                            # ack the call, attach a fresh crisp photo, then let it respond.
                            # cv2 encode runs in a thread so it never stalls audio playback.
                            durl = await to_thread(frame_data_url, LOOK_MAXDIM, LOOK_QUALITY)
                            await enqueue(func_output(call_id, {"ok": durl is not None,
                                                               "note": "camera photo attached" if durl else "camera unavailable"}))
                            if durl:
                                await enqueue(image_item(durl, "Current camera view (look):", LOOK_DETAIL))
                            await enqueue({"type": "response.create"})
                        elif name == "scan":
                            # search step: turn a small amount locally, let it settle, then hand over a
                            # fresh photo so the model can recognise the target and decide whether to
                            # keep scanning. This is what makes "spin until you see X" actually work.
                            await cancel_advance()   # scanning owns the motors while it runs
                            side = args_of(arguments).get("direction", "right")
                            if side not in ("left", "right"):
                                side = "right"
                            turn = "ccw" if side == "left" else "cw"   # left = ccw, right = cw
                            disarmed = os.path.exists(DISARM)
                            if not disarmed:
                                motor("%s %.2f %.2f" % (turn, SCAN_TURN_SPEED, SCAN_TURN_SECS))  # timed, self-expiring
                                await asyncio.sleep(SCAN_TURN_SECS + SCAN_SETTLE)
                            durl = await to_thread(frame_data_url, LOOK_MAXDIM, LOOK_QUALITY)
                            note = ("kill switch is on — I can't turn, but here's the current view" if disarmed
                                    else ("turned a step %s; view attached" % side if durl else "camera unavailable"))
                            await enqueue(func_output(call_id, {"ok": durl is not None, "note": note}))
                            if durl:
                                await enqueue(image_item(durl, "Camera view after turning a step %s (scanning):" % side, LOOK_DETAIL))
                            await enqueue({"type": "response.create"})
                        elif name in ("advance", "navigate"):
                            # continuous, locally-reactive driving: hand off to a background task.
                            # advance = roll forward + STOP when close; navigate = STEER AROUND
                            # obstacles (avoid.decide). Reports back on a terminal stop.
                            await cancel_advance()
                            spd = _clamp(args_of(arguments).get("speed", 1.0), 0.6, 1.0, 1.0)
                            steer = (name == "navigate")
                            adv["task"] = asyncio.create_task(guarded_forward(spd, enqueue, steer))
                            adv["task"].add_done_callback(_log_task_death)
                            await enqueue(func_output(call_id, {"ok": True, "detail": (
                                "on my way; steering around obstacles" if steer
                                else "rolling forward; I'll stop when I'm close to something")}))
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
