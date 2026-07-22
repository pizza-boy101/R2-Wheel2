#!/bin/bash
# run.sh — start the R2-Wheel2 debug dashboard.
#
# Serves on http://<this-machine-ip>:8099 — open that in a browser on the same
# network to watch the camera view, the left/middle/right closeness bars, the
# armed/disarmed state, and live log tails (plus an arm/disarm button + Esc kill).
# Run it alongside the perception / nav / voice loops. Stdlib only, no dependencies.
#
# Optional: DEBUG_WEB_PORT=9000 ./run.sh  to serve on a different port.

exec python3 "$(dirname "$0")/debug_web.py"
