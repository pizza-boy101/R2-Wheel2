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
import socket

import avoid                      # shared avoidance brain (decide + recovery_plan); see avoid.py
from avoid import decide          # re-exported so `nav.decide` still resolves for callers/tests

HOME = os.path.expanduser("~")


def env(n, d):
    v = os.environ.get(n)
    return v if v not in (None, "") else d


WORKSPACE = env("ROBOT_WORKSPACE", os.path.join(HOME, "robot", "workspace"))
MOVE = os.path.join(WORKSPACE, "move")
STATE = os.path.join(WORKSPACE, "nav_state.json")
MODE = os.path.join(WORKSPACE, "nav_mode")
DISARM = os.path.join(WORKSPACE, ".disarmed")

# decide()/recovery thresholds now live in avoid.py (single source of truth, shared with the
# API control path). Alias the ones nav's own logging/loop references so those lines are unchanged.
STOP_NEAR = avoid.STOP_NEAR
SIDE_NEAR = avoid.SIDE_NEAR
LOOM_BRAKE = avoid.LOOM_BRAKE
TURN_SPEED = avoid.TURN_SPEED

FWD_SPEED = float(env("NAV_FWD_SPEED", "1.0"))
BURST = float(env("NAV_BURST", "0.6"))          # move duration per command (re-issued to sustain)
LOOP_HZ = float(env("NAV_LOOP_HZ", "20"))       # 20Hz: halves the decide->act latency vs 10 (reads are cheap)
STATE_STALE = float(env("NAV_STATE_STALE", "0.7"))

# --- stuck detection (nav-local): commanded to move but the camera view isn't changing -> wedged
# on something the depth band can't see. recover() (via avoid.recovery_plan) frees us. ---
MOTION_MIN = float(env("NAV_MOTION_MIN", "5.0"))     # smoothed scene-change below this = not actually moving
                                                     # (calibrated: static ~1.5, moving ~12)
STUCK_SECS = float(env("NAV_STUCK_SECS", "1.3"))     # low motion for this long while driving = stuck
MAX_STUCK = int(env("NAV_MAX_STUCK", "4"))           # give up (park) after this many failed recoveries in a row

_run = True


def _stop(*_):
    global _run
    _run = False


# ---- motor commands go to the persistent motor daemon over its unix socket ----
# (no per-command process spawn; the daemon owns the serial, sustains motion via its
#  own watchdog resend, and enforces the .disarmed kill switch. If the daemon is down,
#  sends silently no-op -> the car stays still, which is the safe failure.)
MOTOR_SOCK = os.path.join(WORKSPACE, "motor.sock")
_msock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)


def motor(cmd):
    try:
        _msock.sendto(cmd.encode(), MOTOR_SOCK)
    except Exception:
        pass


def stop_motors():
    motor("stop")


def recover(state):
    """Free a wedged car via avoid.recovery_plan (reverse, then rotate toward the open
    side). Timed daemon moves self-expire, so the blocking sleeps here don't trip the
    daemon dead-man."""
    for cmd, spd, secs in avoid.recovery_plan(state):
        motor("%s %.2f %.2f" % (cmd, spd, secs))
        time.sleep(secs + 0.05)
    motor("stop")


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


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    period = 1.0 / LOOP_HZ
    cur = None                                # current action (None = stopped/idle), for stuck logic + logging
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
        spd = FWD_SPEED if act == "forward" else TURN_SPEED
        motor("%s %.2f" % (act, spd))          # send every loop; the daemon sustains it and
        cur = act                              # dead-mans if we ever stop sending (e.g. nav dies)

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
