#!/usr/bin/env python3
"""
nav.py — local reactive navigation loop (the fast half of the split brain).

Reads the depth free-space that perception.py publishes (nav_state.json) and drives
the car to avoid obstacles ENTIRELY ON THE BOX — no cloud in the control path, so the
reaction time is ~one loop period (~100 ms), not the ~1 s cloud round-trip. Drives
through the same `move` script, so the firmware 300 ms watchdog and the `.disarmed`
kill switch still apply.

Gated by a mode file so it only touches the motors when armed:
    echo explore > ~/robot/workspace/nav_mode     # autonomous wander + avoid
    echo idle    > ~/robot/workspace/nav_mode      # hand motors back / stop
(no file = idle). While idle it never drives, so voice/manual control has the motors.

Explore behavior: go forward while the path ahead is clear; when something gets too
close ahead, turn toward the more open side; repeat. If vision goes stale it stops
(never drives blind).

CALIBRATION: STOP_NEAR is the center 'nearness' (MiDaS inverse depth, higher = closer)
that counts as 'too close'. It is scene/scale-relative, so it MUST be calibrated against
a real wall and passed via env; the default is only a starting guess.

Env: STOP_NEAR (1100), NAV_FWD_SPEED (0.6), NAV_TURN_SPEED (0.7), NAV_BURST (0.6),
     NAV_LOOP_HZ (10), NAV_STATE_STALE (0.7)
"""
import os
import json
import time
import signal
import subprocess

HOME = os.path.expanduser("~")


def env(n, d):
    v = os.environ.get(n)
    return v if v not in (None, "") else d


WORKSPACE = env("ROBOT_WORKSPACE", os.path.join(HOME, "robot", "workspace"))
MOVE = os.path.join(WORKSPACE, "move")
STATE = os.path.join(WORKSPACE, "nav_state.json")
MODE = os.path.join(WORKSPACE, "nav_mode")
DISARM = os.path.join(WORKSPACE, ".disarmed")

STOP_NEAR = float(env("STOP_NEAR", "1100"))
FWD_SPEED = float(env("NAV_FWD_SPEED", "1.0"))
TURN_SPEED = float(env("NAV_TURN_SPEED", "1.0"))
BURST = float(env("NAV_BURST", "0.6"))          # move duration per command (re-issued to sustain)
LOOP_HZ = float(env("NAV_LOOP_HZ", "10"))
STATE_STALE = float(env("NAV_STATE_STALE", "0.7"))

# --- stuck detection: commanded to move but the camera view isn't changing -> wedged on
# something the depth band can't see. Back up and turn to free ourselves. ---
MOTION_MIN = float(env("NAV_MOTION_MIN", "5.0"))     # smoothed scene-change below this = not actually moving
                                                     # (calibrated: static ~1.5, moving ~12)
STUCK_SECS = float(env("NAV_STUCK_SECS", "1.3"))     # low motion for this long while driving = stuck
BACK_SECS = float(env("NAV_BACK_SECS", "0.5"))       # reverse duration during recovery
RECOVER_TURN_SECS = float(env("NAV_RECOVER_TURN_SECS", "0.7"))
BACK_SPEED = float(env("NAV_BACK_SPEED", "0.7"))
MAX_STUCK = int(env("NAV_MAX_STUCK", "4"))           # give up (park) after this many failed recoveries in a row

# --- B4 steering quality (turn hysteresis + side-column brake + loom brake). Thresholds
# below are sensible starting points but MUST be tuned on the floor with motion. ---
SIDE_NEAR = float(env("NAV_SIDE_NEAR", "900"))       # a side column this close blocks forward (corner-clip guard).
                                                     # Deliberately high so parallel corridor walls don't paralyze it.
LOOM_BRAKE = float(env("NAV_LOOM_BRAKE", "120"))     # rapid approach -> brake even if absolute nearness is under STOP_NEAR
RESUME_NEAR = float(env("NAV_RESUME_NEAR", str(STOP_NEAR - 80)))   # hysteresis: resume forward only once clearly clear
TURN_COMMIT_SECS = float(env("NAV_TURN_COMMIT", "0.7"))           # hold a chosen turn direction at least this long (anti-dither)
TURN_SWITCH_MARGIN = float(env("NAV_TURN_SWITCH_MARGIN", "120"))  # only flip turn direction if the other side is clearer by this much

_run = True


def _stop(*_):
    global _run
    _run = False


_motion = None


def _kill_motion():
    global _motion
    if _motion is not None and _motion.poll() is None:
        try:
            _motion.terminate()
        except Exception:
            pass
    _motion = None


def drive(direction, speed, secs):
    """Non-blocking: move resends in the background so the loop stays free to react."""
    global _motion
    _kill_motion()
    _motion = subprocess.Popen([MOVE, direction, "%.2f" % speed, "%.2f" % secs],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_motors():
    _kill_motion()
    subprocess.run([MOVE, "stop"], capture_output=True, timeout=5)


def _drive_blocking(direction, speed, secs):
    """Run a timed maneuver to completion (move exits after `secs`). Honors the kill
    switch mid-move (the move script checks .disarmed each tick)."""
    _kill_motion()
    try:
        subprocess.run([MOVE, direction, "%.2f" % speed, "%.2f" % secs],
                       capture_output=True, timeout=secs + 3)
    except Exception:
        pass


def recover(state):
    """Free a wedged car: reverse a little, then rotate toward the more-open side."""
    _drive_blocking("back", BACK_SPEED, BACK_SECS)
    d = "cw"
    if state:
        n = state["near"]
        d = "ccw" if n["l"] <= n["r"] else "cw"   # spin toward whichever side reads more open
    _drive_blocking(d, TURN_SPEED, RECOVER_TURN_SECS)
    stop_motors()


def read_mode():
    try:
        with open(MODE) as f:
            return f.read().strip().lower()
    except Exception:
        return "idle"


def read_state():
    try:
        with open(STATE) as f:
            st = json.load(f)
        if time.time() - st.get("ts", 0) > STATE_STALE:
            return None                       # stale -> treat as no vision
        return st
    except Exception:
        return None


def decide(l, c, r, loom, cur_act, turn_started, now):
    """Choose forward/cw/ccw. Beyond the naive 'forward if center clear', B4 adds:
      - side-column braking: a very close L or R also blocks forward (corner-clip guard),
      - loom early-brake: a fast rise in nearness blocks forward before it's even close,
      - turn hysteresis: once turning, commit to that direction for TURN_COMMIT_SECS and
        only switch if the other side is clearer by TURN_SWITCH_MARGIN, and resume forward
        only once genuinely clear (RESUME_NEAR) — kills the cw/ccw dither when L~R.
    Returns (act, turn_started)."""
    side_block = (l >= SIDE_NEAR) or (r >= SIDE_NEAR)
    blocked = (c >= STOP_NEAR) or side_block or (loom >= LOOM_BRAKE)
    turning = cur_act in ("cw", "ccw")

    if not blocked:
        if turning:
            # coast through the hysteresis band still turning; resume forward only when clearly clear
            if c < RESUME_NEAR and l < SIDE_NEAR and r < SIDE_NEAR and loom < LOOM_BRAKE:
                return "forward", 0.0
            return cur_act, turn_started
        return "forward", 0.0

    # blocked -> turn toward the more-open side (lower nearness), with hysteresis
    want = "ccw" if l <= r else "cw"
    if turning:
        if (now - turn_started) < TURN_COMMIT_SECS:
            return cur_act, turn_started               # commit: no mid-turn flip
        if want != cur_act and abs(l - r) >= TURN_SWITCH_MARGIN:
            return want, now                           # other side clearly better -> switch
        return cur_act, turn_started
    return want, now                                   # start a new turn


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    period = 1.0 / LOOP_HZ
    cur = None                                # current action, so we only re-issue on change/expiry
    cur_started = 0.0
    last_log = 0.0
    low_since = None                          # when smoothed motion first dropped below MOTION_MIN while driving
    last_recover = 0.0                        # cooldown so we don't re-trigger during/right after a recovery
    stuck_count = 0                           # consecutive failed recoveries (motion never came back)
    mot_ema = None                            # smoothed scene-motion (spike-robust stuck signal)
    turn_started = 0.0                        # when the current turn direction was committed (hysteresis)
    print("nav: reactive loop up (STOP_NEAR=%.0f, %.0f Hz, MOTION_MIN=%.1f). arm: echo explore > %s"
          % (STOP_NEAR, LOOP_HZ, MOTION_MIN, MODE), flush=True)

    while _run:
        t0 = time.time()
        armed = read_mode() == "explore" and not os.path.exists(DISARM)
        if not armed:
            if cur is not None:
                stop_motors()
                cur = None
            low_since = None
            stuck_count = 0
            time.sleep(period)
            continue

        st = read_state()
        if st is None:
            if cur is not None:               # no fresh depth -> never drive blind
                stop_motors()
                cur = None
            low_since = None
            time.sleep(period)
            continue

        n = st["near"]
        l, c, r = n["l"], n["c"], n["r"]
        loom = st.get("loom", 0) or 0
        motion = st.get("motion")
        if motion is not None:
            mot_ema = motion if mot_ema is None else 0.7 * mot_ema + 0.3 * motion

        now = time.time()
        act, turn_started = decide(l, c, r, loom, cur, turn_started, now)  # B4: hysteresis + side/loom brake
        if act != cur or (now - cur_started) > BURST * 0.6:               # (re)issue on change or before expiry
            drive(act, FWD_SPEED if act == "forward" else TURN_SPEED, BURST)
            cur, cur_started = act, now

        # --- stuck detection: commanding motion but the view isn't changing -> wedged ---
        stuck = False
        if mot_ema is not None and cur is not None and (now - last_recover) > STUCK_SECS:
            if mot_ema < MOTION_MIN:
                if low_since is None:
                    low_since = now
                elif (now - low_since) >= STUCK_SECS:
                    stuck = True
            else:
                low_since = None              # view is changing -> we're really moving
                stuck_count = 0
        if stuck:
            stuck_count += 1
            print("nav: STUCK (motion~%.1f<%.1f for %.1fs) -> recover #%d"
                  % (mot_ema, MOTION_MIN, STUCK_SECS, stuck_count), flush=True)
            if stuck_count >= MAX_STUCK:
                print("nav: %d recoveries failed -> parking (idle). re-arm to retry."
                      % stuck_count, flush=True)
                stop_motors()
                try:
                    with open(MODE, "w") as f:
                        f.write("idle")
                except Exception:
                    pass
                cur = None
                low_since = None
                stuck_count = 0
                continue
            recover(st)                        # blocking: back up, then turn toward the open side
            cur = None                         # force a fresh decision next loop
            low_since = None
            last_recover = time.time()
            continue

        if now - last_log > 0.5:
            last_log = now
            why = []
            if c >= STOP_NEAR:
                why.append("c")
            if l >= SIDE_NEAR or r >= SIDE_NEAR:
                why.append("side")
            if loom >= LOOM_BRAKE:
                why.append("loom")
            print("explore near L/C/R=%d/%d/%d loom=%d motion~%s -> %s%s"
                  % (l, c, r, loom, ("%.1f" % mot_ema) if mot_ema is not None else "?",
                     act, (" [block:%s]" % "+".join(why)) if why else ""), flush=True)

        sl = period - (time.time() - t0)
        if sl > 0:
            time.sleep(sl)

    stop_motors()
    print("nav: stopped", flush=True)


if __name__ == "__main__":
    main()
