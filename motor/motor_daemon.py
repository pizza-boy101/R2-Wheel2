#!/usr/bin/env python3
"""
motor_daemon.py — the single persistent owner of the Arduino serial link.

Every other component (nav loop, voice sidecar, the `move` CLI) sends short text
commands to this daemon over a unix datagram socket instead of writing the serial
port itself. Benefits over spawning the `move` script per command:
  - the port is opened + configured ONCE (no per-command process spawn, no repeated
    `stty` re-init that can glitch the line),
  - ONE writer to the serial device -> no interleaved/garbled commands from nav and
    the sidecar racing each other,
  - the daemon resends the current velocity faster than the firmware's 300 ms
    watchdog, so a single command SUSTAINS motion (clients set intent, not bursts),
  - a dead-man deadline: a sustained command auto-stops if the client goes silent
    (crash / disconnect), and a timed command self-expires — motion never runs away,
  - the `.disarmed` kill switch is enforced centrally, every loop.

Socket: workspace/motor.sock (unix SOCK_DGRAM, world-writable for cross-uid clients).
Commands (one text datagram each):
    forward|back|left|right|cw|ccw <speed 0..1> [seconds]
        set velocity. WITH seconds -> timed move, auto-stops after `seconds`.
        WITHOUT seconds -> sustained until DEADMAN s unless another command refreshes it.
    stop                stop immediately
    raw <V ...>         pass a raw firmware line (advanced/manual)
    ping                no-op (keepalive)

Env: ROBOT_SERIAL (else auto-detect ttyUSB*/ttyACM*), MOTOR_SOCK, ROBOT_WORKSPACE,
     MOTOR_RESEND (0.12 s), MOTOR_DEADMAN (0.5 s)
"""
import os
import json
import time
import glob
import select
import socket
import signal
import subprocess

HOME = os.path.expanduser("~")


def env(n, d):
    v = os.environ.get(n)
    return v if v not in (None, "") else d


WORKSPACE = env("ROBOT_WORKSPACE", os.path.join(HOME, "robot", "workspace"))
SOCK = env("MOTOR_SOCK", os.path.join(WORKSPACE, "motor.sock"))
DISARM = os.path.join(WORKSPACE, ".disarmed")
ULTRA = env("ULTRA_STATE", os.path.join(WORKSPACE, "ultrasonic.json"))  # forward distance we publish
RESEND = float(env("MOTOR_RESEND", "0.04"))       # loop/resend cadence (< firmware 300ms watchdog);
                                                  # also sets how often we drain the serial for
                                                  # "U <cm>" lines, so keep it at/under the ping rate
DEADMAN = float(env("MOTOR_DEADMAN", "0.5"))      # sustained cmd auto-stops if not refreshed within this
ULTRA_MIN_WRITE = float(env("ULTRA_MIN_WRITE", "0"))   # publish every reading (firmware caps at ~20Hz;
                                                       # the write is tiny, so no throttle needed)
ULTRA_MEDIAN = int(env("ULTRA_MEDIAN", "3"))      # median window over recent pings: rejects a single
                                                  # spurious short/dropped echo (HC-SR04s glitch off soft or
                                                  # angled surfaces) so one bad frame can't fake a close stop
ULTRA_FAR = 999.0                                 # sentinel 'far' for an invalid ping, so the median is numeric
BAUD = 115200

# direction -> (vx, vy, w) unit velocity, matching the `move` script + firmware V protocol
DIRS = {"forward": (0, 1, 0), "back": (0, -1, 0), "right": (1, 0, 0),
        "left": (-1, 0, 0), "cw": (0, 0, 1), "ccw": (0, 0, -1)}

_run = True


def _stop(*_):
    global _run
    _run = False


def find_port():
    p = os.environ.get("ROBOT_SERIAL")
    if p:
        return p
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        g = sorted(glob.glob(pat))
        if g:
            return g[0]
    return None


def clamp(v, lo, hi, d):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return d


def write_ultra(cm, valid):
    """Publish the latest forward distance for the dashboard / perception (atomic rename)."""
    tmp = ULTRA + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"ts": time.time(), "cm": cm, "valid": valid}, f)
        os.replace(tmp, ULTRA)
    except Exception:
        pass


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    port = find_port()
    if not port:
        print("motor_daemon: FATAL — no serial port found", flush=True)
        return
    # configure the line once (mirrors the move script; -hupcl avoids reset churn, clocal so
    # opening for read never blocks waiting on carrier)
    subprocess.run(["stty", "-F", port, str(BAUD), "cs8", "-cstopb", "-parenb",
                    "raw", "-hupcl", "clocal", "-echo"], stderr=subprocess.DEVNULL)
    try:
        ser = open(port, "r+b", buffering=0)       # r+w: we write commands AND read "U <cm>" back
    except Exception as e:
        print("motor_daemon: FATAL — cannot open %s: %s" % (port, e), flush=True)
        return
    serial_fd = ser.fileno()
    time.sleep(0.2)

    def w(line):
        try:
            ser.write((line + "\n").encode())
        except Exception as e:
            print("motor_daemon: serial write failed: %s" % e, flush=True)

    try:
        os.unlink(SOCK)
    except OSError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(SOCK)
    try:
        os.chmod(SOCK, 0o777)                      # allow cross-uid clients (host + container)
    except OSError:
        pass
    s.settimeout(RESEND)
    print("motor_daemon: up on %s -> %s (resend %.2fs, deadman %.2fs)"
          % (SOCK, port, RESEND, DEADMAN), flush=True)

    target = None                                  # current V-string being commanded
    deadline = 0.0                                 # stop when now passes this
    was_disarmed = False
    rbuf = b""                                      # inbound serial accumulator (for "U <cm>" lines)
    ultra_pending = None                           # latest FILTERED (cm, valid), written throttled
    ultra_ring = []                                # recent pings (cm floats; invalid -> ULTRA_FAR) for the median
    last_ultra_write = 0.0
    w("S")                                         # start stopped

    while _run:
        now = time.time()
        disarmed = os.path.exists(DISARM)

        # ---- receive one command (or time out and just service the watchdog) ----
        try:
            data, _ = s.recvfrom(256)
            cmd = data.decode("utf-8", "ignore").strip()
        except socket.timeout:
            cmd = None
        except Exception:
            cmd = None

        if cmd:
            parts = cmd.split()
            op = parts[0].lower() if parts else ""
            if op == "stop":
                target = None
                w("S")
            elif op == "ping":
                pass
            elif op == "raw":
                if not disarmed:
                    raw = cmd[3:].strip()
                    if raw:
                        target, deadline = raw, now + DEADMAN
                        w(raw)
            elif op in DIRS:
                spd = clamp(parts[1] if len(parts) > 1 else "1", 0.0, 1.0, 1.0)
                vx, vy, wz = DIRS[op]
                V = "V %g %g %g" % (vx * spd, vy * spd, wz * spd)
                secs = None
                if len(parts) > 2:
                    try:
                        secs = float(parts[2])
                    except Exception:
                        secs = None
                if disarmed:
                    target = None
                    w("S")
                    print("motor_daemon: recv '%s' -> %s [DISARMED: refused]" % (cmd, V), flush=True)
                else:
                    target = V
                    deadline = now + (secs if secs and secs > 0 else DEADMAN)
                    w(V)
            # unknown op -> ignore

        # ---- drain inbound serial: pick up the firmware's "U <cm>" distance lines ----
        try:
            r, _, _ = select.select([serial_fd], [], [], 0)
            if r:
                chunk = os.read(serial_fd, 4096)
                if chunk:
                    rbuf += chunk
                    *lines, rbuf = rbuf.split(b"\n")   # keep any trailing partial line in rbuf
                    for line in lines:
                        line = line.strip()
                        if line[:2] == b"U ":
                            try:
                                cm = float(line[2:].strip())
                            except Exception:
                                continue
                            # median-of-N to reject single-frame glitches (see ULTRA_MEDIAN). Invalid
                            # pings map to ULTRA_FAR so a lone dropped echo can't fake "far" either.
                            ultra_ring.append(ULTRA_FAR if cm < 0 else round(cm, 1))
                            if len(ultra_ring) > ULTRA_MEDIAN:
                                del ultra_ring[:-ULTRA_MEDIAN]
                            med = sorted(ultra_ring)[len(ultra_ring) // 2]
                            ultra_pending = (None, False) if med >= ULTRA_FAR else (med, True)
                    if len(rbuf) > 512:               # runaway guard (no newline arriving)
                        rbuf = rbuf[-128:]
        except Exception:
            pass
        if ultra_pending is not None and (now - last_ultra_write) >= ULTRA_MIN_WRITE:
            write_ultra(*ultra_pending)
            last_ultra_write = now
            ultra_pending = None

        # ---- watchdog resend + dead-man + disarm enforcement ----
        if disarmed:
            if not was_disarmed:
                w("S")                              # immediate stop the instant the kill switch engages
            target = None
        elif target is not None:
            if now < deadline:
                w(target)                           # sustain motion (resend beats the 300ms watchdog)
            else:
                target = None
                w("S")                              # deadline passed -> stop (dead-man / timed-move end)
        was_disarmed = disarmed

    # ---- shutdown: always stop the motors and clean up ----
    try:
        w("S")
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass
    try:
        os.unlink(SOCK)
    except OSError:
        pass
    print("motor_daemon: stopped", flush=True)


if __name__ == "__main__":
    main()
