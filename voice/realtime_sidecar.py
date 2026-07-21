#!/usr/bin/env python3
"""
Robot realtime sidecar: low-latency voice+drive control via the OpenAI Realtime API.

For the "hear me, react, move NOW" use case: one bidirectional websocket to
gpt-realtime, streaming mic audio up and voice audio down, with server-side semantic
VAD and barge-in. The model drives the car through function tools (drive/stop/look)
that bridge to the `move` script, so the firmware 300ms watchdog and the `.disarmed`
kill switch still apply. Runs on the host (uses parec/paplay + the shared workspace).

Env (all optional):
  OPENAI_KEY_FILE   path to the API key file      (default ~/robot/secrets/openai-realtime.key)
  REALTIME_MODEL    realtime model id             (default gpt-realtime-2)
  REALTIME_VOICE    output voice                  (default marin)
  LISTEN_SOURCE_MATCH / SPEECH_SINK_MATCH  pulse device substrings (default jabra)
  ROBOT_WORKSPACE   workspace dir (move script + frame.jpg)(default ~/robot/workspace)
  REALTIME_SMOKE=1  connect, say one line, exit (no mic) — for validation
  REALTIME_LOG      log file                      (default ~/robot/realtime.log)
"""
import os
import sys
import json
import time
import base64
import asyncio
import subprocess
from datetime import datetime

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
FRAME_PATH = os.path.join(WORKSPACE, "frame.jpg")   # written by the perception loop
# hybrid vision: on-demand look (crisp) + silent change-gated ambient push (cheap)
LOOK_MAXDIM = int(env("VISION_LOOK_MAXDIM", "512"))
LOOK_QUALITY = int(env("VISION_LOOK_QUALITY", "65"))
LOOK_DETAIL = env("VISION_LOOK_DETAIL", "low")   # "low" = ~2x faster model vision; "high" for fine detail
PUSH_MAXDIM = int(env("VISION_PUSH_MAXDIM", "384"))
PUSH_QUALITY = int(env("VISION_PUSH_QUALITY", "55"))
PUSH_POLL = float(env("VISION_PUSH_POLL", "0.7"))          # how often to check for change
PUSH_CHANGE_THRESH = float(env("VISION_PUSH_THRESH", "8"))  # mean abs pixel delta (0-255) to count as changed
PUSH_MIN_INTERVAL = float(env("VISION_PUSH_MIN_INTERVAL", "1.5"))  # rate cap between ambient pushes
VOICE_QUIET_GAP = float(env("VISION_QUIET_GAP", "3.0"))  # never push an ambient frame within this many s of voice activity
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
    "To move, call the drive tool with a direction, "
    "a speed from 0.6 to 1.0 (NEVER below 0.6 — below that the motors just stall and buzz and the car "
    "won't move; default to full speed 1.0 unless the user asks to go slower), and a short duration "
    "in seconds — prefer brief bursts (0.5 to 1.5 s) and re-check rather than long "
    "blind drives. Call stop the instant the user says stop. "
    "VISION: you receive ambient camera snapshots automatically whenever the view changes, so you "
    "are already aware of the scene most of the time — treat those as background awareness and only "
    "mention them if something important appears (e.g. an obstacle). Do NOT call look before ordinary "
    "moves; short bursts followed by re-checking are safe. Call the look tool only when the user asks "
    "what you see, or when you are about to drive toward a specific target and need a crisp view to aim "
    "or confirm the path is clear. "
    "SEARCHING: when told to turn or move until you see something, do NOT tap in tiny increments. "
    "Start a CONTINUOUS turn at the SLOWEST speed that still reliably moves the car, which is 0.6 "
    "(never lower — below 0.6 it just stalls and buzzes), and keep re-issuing it so the motion never "
    "pauses, watching the ambient snapshots as it sweeps. Driving is non-blocking, so you stay aware "
    "while moving. There is ~1-2 s of reaction lag, so expect some overshoot even at 0.6; call stop "
    "the INSTANT the target first edges into frame (stop preempts the turn)."
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
    {"type": "function", "name": "stop",
     "description": "Stop all motion immediately.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function", "name": "look",
     "description": "Attach a fresh photo from the robot's forward camera so you can see the scene with your own eyes. Call before moving toward something or when asked what you see.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
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


# ---------- tool bridge (reuses the move script -> respects watchdog + kill switch) ----------
def _clamp(v, lo, hi, d):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return d


# background handle for the current motion, so a turn can run non-blocking (the event
# loop stays free to see + react mid-turn) and stop / the next command can preempt it.
_motion_proc = None


def _kill_motion():
    global _motion_proc
    if _motion_proc is not None and _motion_proc.poll() is None:
        try:
            _motion_proc.terminate()
        except Exception:
            pass
    _motion_proc = None


def invoke_tool(name, args_json):
    global _motion_proc
    try:
        args = json.loads(args_json or "{}")
    except Exception:
        args = {}
    try:
        if name == "drive":
            d = args.get("direction", "forward")
            spd = _clamp(args.get("speed", 1.0), 0.6, 1.0, 1.0)   # 0.6 floor: below this the motors stall/buzz
            secs = _clamp(args.get("seconds", 1.0), 0.1, 2.0, 1.0)
            _kill_motion()   # a new command preempts any in-flight turn
            # non-blocking: the move script keeps resending for `secs` in the background,
            # so the event loop stays free to receive frames and issue a stop mid-turn.
            _motion_proc = subprocess.Popen([MOVE, d, "%.2f" % spd, "%.2f" % secs],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"ok": True, "detail": "driving %s at %.2f for up to %.1fs (non-blocking)" % (d, spd, secs)}
        if name == "stop":
            _kill_motion()   # preempt the in-flight turn, then halt
            subprocess.run([MOVE, "stop"], capture_output=True, timeout=5)
            return {"ok": True, "detail": "stopped"}
        # note: "look" is handled specially in the receiver (it attaches an image), not here
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"error": "unknown tool %s" % name}


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

        async def enqueue(evt):
            await send_q.put(evt)

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
                        else:
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

            async def vision_push():
                """Ambient awareness: when the camera view changes, silently drop ONE
                small frame into context (no response.create -> no chatter, no extra
                response cost). Rate-capped. Static scene -> nothing sent."""
                if not _CV2:
                    log("cv2 unavailable -> ambient vision push disabled")
                    return
                prev = None
                last_push = 0.0
                while not stop_event.is_set():
                    await asyncio.sleep(PUSH_POLL)
                    # all cv2 work runs off the event loop so it never stalls audio playback
                    thumb = await to_thread(frame_thumb)
                    if thumb is None:
                        continue
                    change = 0.0 if prev is None else float(np.mean(np.abs(thumb - prev)))
                    prev = thumb
                    now = time.monotonic()
                    if (change > PUSH_CHANGE_THRESH and (now - last_push) > PUSH_MIN_INTERVAL
                            and (now - last_activity["t"]) > VOICE_QUIET_GAP):   # not mid-conversation
                        durl = await to_thread(frame_data_url, PUSH_MAXDIM, PUSH_QUALITY)
                        if durl:
                            await enqueue(image_item(
                                durl, "(ambient camera update — background awareness only)", "low"))
                            last_push = now
                            log("ambient frame pushed (change=%.1f)" % change)

            tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
            if not SMOKE:
                tasks.append(asyncio.create_task(mic()))
                tasks.append(asyncio.create_task(vision_push()))
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
