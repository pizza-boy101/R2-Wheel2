#!/usr/bin/env python3
"""
avoid.py — the shared reactive-avoidance brain (single source of truth).

Both the autonomous explore loop (nav.py) and the voice/API control path
(realtime_sidecar.py) import this, so the car makes the SAME local obstacle
decisions whether it is wandering on its own or being driven by the model. The
avoidance logic lives in ONE place, not two that can drift apart.

decide(): pick forward/cw/ccw from the per-column nearness + loom that perception
publishes (higher = closer). Beyond the naive 'forward if center clear' it adds
side-column braking, loom early-braking, and turn hysteresis (see its docstring).

recovery_plan(): the timed-move sequence to free a wedged car (reverse, then rotate
toward the more-open side), returned AS DATA so a synchronous caller (nav.py) and an
async caller (the sidecar) run identical recovery, each with its own sleep.

Thresholds come from the same env vars nav.py has always read, so existing deployment
environment keeps working unchanged; the sidecar inherits the same defaults.
"""
import os


def env(n, d):
    v = os.environ.get(n)
    return v if v not in (None, "") else d


# --- decide() thresholds (nearness = MiDaS inverse depth, higher = closer) ---
STOP_NEAR = float(env("STOP_NEAR", "680"))            # center 'too close' -> block forward (tuned on the floor)
SIDE_NEAR = float(env("NAV_SIDE_NEAR", "900"))        # a side column this close also blocks (corner-clip guard)
LOOM_BRAKE = float(env("NAV_LOOM_BRAKE", "120"))      # fast rise in nearness -> brake before it's even close
RESUME_NEAR = float(env("NAV_RESUME_NEAR", str(STOP_NEAR - 80)))   # hysteresis: resume forward only once clearly clear
TURN_COMMIT_SECS = float(env("NAV_TURN_COMMIT", "0.7"))           # hold a chosen turn at least this long (anti-dither)
TURN_SWITCH_MARGIN = float(env("NAV_TURN_SWITCH_MARGIN", "120"))  # only flip turn dir if the other side is clearer by this

# --- recovery_plan() params (freeing a wedged car) ---
BACK_SPEED = float(env("NAV_BACK_SPEED", "0.7"))
BACK_SECS = float(env("NAV_BACK_SECS", "0.5"))
TURN_SPEED = float(env("NAV_TURN_SPEED", "1.0"))
RECOVER_TURN_SECS = float(env("NAV_RECOVER_TURN_SECS", "0.7"))

# --- forward metric distance (HC-SR04 front sonar), fused into the `advance` stop ---
# The camera nearness is *relative* (varies with light/texture) and blind to clear/transparent
# surfaces; the front sonar gives a true centimetre distance straight ahead. `advance` ("roll up
# to something and stop") is exactly a forward-distance task, so the sonar sets a real standoff
# and tapers speed as we close in. Lives here in the one brain so nav/navigate can adopt it too.
STOP_CM = float(env("STOP_CM", "35"))            # stop this far (cm) from whatever's dead ahead
SLOW_CM = float(env("SLOW_CM", "90"))            # begin easing off speed here (needs runway: at
                                                 # full speed + ~150ms pipeline + coast, 45cm was
                                                 # too tight and it drove into walls)
MOTOR_FLOOR = float(env("MOTOR_FLOOR", "0.6"))   # below this the motors stall -> never command less


def ultra_blocked(cm, valid):
    """True when the front sonar says we're at/inside the standoff distance. An invalid reading
    (nothing within range, or stale/missing -> caller passes valid=False) never blocks, so a
    silent sonar hands control back to the camera instead of stranding the car."""
    return bool(valid) and cm is not None and cm <= STOP_CM


def approach_speed(cm, valid, speed):
    """Taper forward speed by distance: full speed beyond SLOW_CM, easing to the motor floor as
    it nears STOP_CM (0 at/inside it). No valid reading -> speed unchanged, so behaviour matches
    the pre-sonar code whenever the sonar isn't talking."""
    if not valid or cm is None or cm >= SLOW_CM:
        return speed
    if cm <= STOP_CM:
        return 0.0
    frac = (cm - STOP_CM) / (SLOW_CM - STOP_CM)          # 1.0 at SLOW_CM -> 0 at STOP_CM
    return max(MOTOR_FLOOR, speed * frac)


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


def recovery_plan(state):
    """Timed-move sequence to free a wedged car: reverse a little, then rotate toward the
    more-open side. Each item = (cmd, speed, seconds) for a self-expiring daemon move; the
    caller sleeps ~seconds between items. Returned as data so nav (sync) and the sidecar
    (async) run identical recovery."""
    d = "cw"
    if state:
        n = state.get("near", {}) or {}
        d = "ccw" if n.get("l", 0) <= n.get("r", 0) else "cw"   # spin toward whichever side reads more open
    return [("back", BACK_SPEED, BACK_SECS), (d, TURN_SPEED, RECOVER_TURN_SECS)]
