#!/usr/bin/env python3
"""Put the hand into one finger's roll isolation posture and derive its detector geometry.

Extends the target finger's MCP pitch and bends every other long finger's pitch
away, so only the target sweeps through the roll plane. Then it takes one
read-only camera snapshot and derives the ROI, row band and depth gate for that
finger, because the geometry changes every time the posture changes: bending
three fingers has been observed to shift the whole hand in the image.

Pitch moves use the already-tested Phase A per-finger runners, one adjacent step
at a time, re-reading the true position before every step because the pitch axis
relaxes a couple of raw after each command settles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

SDK_PYTHON = "/home/user/miniconda3/envs/env_isaaclab/bin/python"
CAMERA_PYTHON = "/home/user/miniconda3/bin/python3"
SDK_ROOT = "/home/user/linkerhand-ros-sdk"

FINGERS = {
    # finger: (pitch slot, roll slot, pip slot, runner module, joint name)
    "index": (1, 6, 16, "tools.run_g20_index_mcp_pitch_candidate_step", None),
    "middle": (2, 7, 17, "tools.run_g20_middle_candidate_step", "middle_mcp_pitch"),
    "ring": (3, 8, 18, "tools.run_g20_ring_candidate_step", "ring_mcp_pitch"),
    "pinky": (4, 9, 19, "tools.run_g20_pinky_candidate_step", "pinky_mcp_pitch"),
}

EXTENDED_PITCH_RAW = 240
BENT_PITCH_RAW = 96
STEP_RAW = 16
# The index pitch runner only accepts the Phase A template targets, so index
# steps are snapped onto that ladder instead of using a free step size.
INDEX_TEMPLATE_RAW = (255, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96,
                      80, 64, 48, 32, 20, 12, 6, 0)


def read_state20(calib: Path, scratch: Path) -> list[int]:
    subprocess.run(
        [SDK_PYTHON, "-m", "tools.record_g20_sdk_snapshot",
         "--sdk-root", SDK_ROOT, "--side", "left", "--hand-joint", "G20",
         "--can", "can0", "--calib", str(calib), "--samples", "5", "--hz", "20",
         "--out", str(scratch)],
        check=True, capture_output=True)
    samples = json.loads(scratch.read_text())["samples"]
    return [int(statistics.median(s["state20"][i] for s in samples))
            for i in range(20)]


def move_pitch(finger: str, target: int, calib: Path, out_dir: Path,
               scratch: Path) -> int:
    pitch_slot, roll_slot, pip_slot, module, joint = FINGERS[finger]
    while True:
        state = read_state20(calib, scratch)
        current = state[pitch_slot]
        if abs(current - target) < 3:
            return current
        if finger == "index":
            descending = target < current
            reachable = [
                value for value in INDEX_TEMPLATE_RAW
                if (value < current if descending else value > current)
                and abs(value - current) <= 17
                and (value >= target if descending else value <= target)
            ]
            if not reachable:
                return current
            nxt = max(reachable) if descending else min(reachable)
        else:
            step = min(STEP_RAW, abs(target - current))
            nxt = current - step if target < current else current + step
        cmd = [SDK_PYTHON, "-m", module, "--sdk-root", SDK_ROOT,
               "--calib", str(calib), "--can", "can0"]
        if joint is not None:
            cmd += ["--joint", joint, "--side-hold-raw", str(state[roll_slot]),
                    "--other-flex-hold-raw", str(state[pip_slot])]
        else:
            cmd += ["--pip-tolerance-raw", "2"]
        cmd += ["--expected-start-raw", str(current), "--target-raw", str(nxt),
                "--speed", "5", "--samples", "8", "--hz", "5",
                "--out", str(out_dir / f"{finger}_pitch_{current}_to_{nxt}.json"),
                "--execute"]
        result = subprocess.run(cmd, capture_output=True)
        after = read_state20(calib, scratch)[pitch_slot]
        print(f"  {finger} pitch {current} -> {nxt}: readback {after} "
              f"(rc={result.returncode})", flush=True)
        if after == current:
            raise SystemExit(
                f"STOP: {finger} pitch did not move from {current}; "
                "a human must look before any further command")


def derive_geometry(snapshot_prefix: Path, expect_row: float | None) -> dict:
    """Find the one extended proximal tape and size the detector around it."""
    bgr = cv2.imread(f"{snapshot_prefix}_color.png")
    depth = np.load(f"{snapshot_prefix}_depth_raw.npy").astype(float) * 0.001
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([75, 40, 20]), np.array([165, 255, 255]))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 150 or height > 60:  # skip noise and the tall palm tape
            continue
        cx, cy = centroids[i]
        patch = depth[stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP] + height,
                      stats[i, cv2.CC_STAT_LEFT]:stats[i, cv2.CC_STAT_LEFT] + width]
        valid = (patch > 0.05) & (patch < 3.0)
        blobs.append({"area": int(area), "width": int(width),
                      "cx": float(cx), "cy": float(cy),
                      "depth": float(np.median(patch[valid])) if valid.sum() else None})
    if not blobs:
        raise SystemExit("no finger-sized blue markers found")
    # The extended finger projects at full length; bent ones are foreshortened.
    blobs.sort(key=lambda b: -b["width"])
    target = blobs[0]
    if len(blobs) > 1 and target["width"] < 1.4 * blobs[1]["width"]:
        raise SystemExit(
            f"cannot tell the extended finger apart by projected width: {blobs}")
    if expect_row is not None and abs(target["cy"] - expect_row) > 80:
        raise SystemExit(
            f"widest marker at row {target['cy']:.0f} is not near the expected "
            f"row {expect_row:.0f} for this finger")
    cy, dz = target["cy"], target["depth"]
    # Neighbouring fingers sit about 74 px apart, so a fixed +-70 px band lets a
    # bent neighbour's blob edge into the ROI and the frame then fails with two
    # markers. Size the band from the measured nearest-neighbour gap instead.
    others = [b["cy"] for b in blobs[1:]]
    gap = min((abs(cy - o) for o in others), default=140.0)
    half = max(25.0, min(60.0, gap * 0.45))
    # Keep the depth gate clear of the bent fingers, which sit deeper.
    deeper = [b["depth"] for b in blobs[1:]
              if b["depth"] is not None and b["depth"] > dz]
    depth_max = min(dz + 0.013, (min(deeper) - 0.004) if deeper else dz + 0.013)
    return {
        "target": target,
        "all_blobs": blobs,
        "nearest_neighbour_row_gap_px": gap,
        "roi": f"620,{int(cy - half - 5)},1010,{int(cy + half + 5)}",
        "row_band": f"{int(cy - half)},{int(cy + half)}",
        "depth_min": round(dz - 0.013, 3),
        "depth_max": round(depth_max, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-finger", required=True, choices=tuple(FINGERS))
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expect-row", type=float, default=None)
    parser.add_argument("--scratch", type=Path,
                        default=Path("/tmp/claude-1000/-home-user-dex-forge/"
                                     "61e08eee-f15e-489b-a731-7c98eec29ea3/"
                                     "scratchpad/roll_state.json"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== isolation posture for {args.target_finger}", flush=True)
    for finger in FINGERS:
        want = EXTENDED_PITCH_RAW if finger == args.target_finger else BENT_PITCH_RAW
        move_pitch(finger, want, args.calib, args.out_dir, args.scratch)

    prefix = args.out_dir / f"posture_snapshot_{args.target_finger}"
    subprocess.run(
        [CAMERA_PYTHON, "-m", "tools.realsense_capture", "--snapshot", str(prefix),
         "--width", "1280", "--height", "720", "--fps", "30",
         "--rgb-exposure", "166", "--rgb-gain", "32",
         "--rgb-white-balance", "4600", "--control-warmup-frames", "45"],
        check=True, capture_output=True)

    geometry = derive_geometry(prefix, args.expect_row)
    state = read_state20(args.calib, args.scratch)
    payload: dict[str, Any] = {
        "target_finger": args.target_finger,
        "extended_pitch_raw": EXTENDED_PITCH_RAW,
        "bent_pitch_raw": BENT_PITCH_RAW,
        "state20": state,
        "roll_slots": {f: state[FINGERS[f][1]] for f in FINGERS},
        "pitch_slots": {f: state[FINGERS[f][0]] for f in FINGERS},
        "pip_slots": {f: state[FINGERS[f][2]] for f in FINGERS},
        "detector_geometry": geometry,
        "snapshot_prefix": str(prefix),
    }
    out = args.out_dir / f"isolation_posture_{args.target_finger}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("roll_slots", "pitch_slots", "detector_geometry")},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
