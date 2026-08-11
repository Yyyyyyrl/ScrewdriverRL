#!/usr/bin/env bash
# Capture one D435 colour+depth snapshot, recovering from the stall the device
# falls into after an aborted streaming session.
#
# The failure looks like "Frame didn't arrive within 5000" and survives closing
# the process, because it is the device that is wedged, not the host.  Only a
# hardware_reset clears it.  A long A/B capture run is worth protecting from a
# single wedge, so retry a few times before giving up.
set -uo pipefail

PREFIX="${1:?usage: realsense_snapshot_retry.sh <out-prefix> [extra args...]}"
shift || true
PY="${REALSENSE_PYTHON:-$HOME/miniconda3/bin/python}"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTEMPTS="${REALSENSE_ATTEMPTS:-3}"

for attempt in $(seq 1 "$ATTEMPTS"); do
    if "$PY" "$TOOLS/realsense_capture.py" --snapshot "$PREFIX" \
            --width 1280 --height 720 --fps 30 \
            --rgb-exposure 166 --rgb-gain 32 --rgb-white-balance 4600 "$@"; then
        exit 0
    fi
    echo "[capture] attempt $attempt failed; resetting the device" >&2
    "$PY" - <<'PYEOF' || true
import time
import pyrealsense2 as rs
for device in rs.context().query_devices():
    device.hardware_reset()
time.sleep(12)
PYEOF
    sleep 3
done

echo "[capture] giving up after $ATTEMPTS attempts" >&2
exit 1
