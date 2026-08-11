#!/usr/bin/env python3
"""Drive a sequence of adjacent thumb CMC-yaw points, stopping at the first gate failure.

Same shape as the Phase B roll driver: it alternates the single-step runner
(CAN open, camera closed) with the yaw detector (camera open, CAN closed), never
both at once, and evaluates the runbook gates after every point. Fail-closed --
any non-zero tool exit or gate breach stops the run and leaves every record on
disk. It never retries, never widens a tolerance and never approves an endpoint.

Two yaw-specific gates on top of the roll ones:

* **Wrap proximity.** The reported angle is recentred by a fixed reference
  offset, which buys about 90 deg of headroom each way. Thumb yaw is roughly
  0.3 deg/raw, 2.5x steeper than roll, so a full sweep can consume most of it.
  The run stops before the angle gets close enough to the boundary to flip.
* **Out-of-plane drift.** The 2D angle is only valid while the yaw axis stays
  near the optical axis. The thumb marker's projected extent is the witness:
  it shrank only 0.7 percent over the 4.1 deg direction check, but if the
  projection collapses further along the sweep the 2D reading stops being an
  angle and the run must stop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

CAMERA_PYTHON = "/home/user/miniconda3/bin/python3"
SDK_PYTHON = "/home/user/miniconda3/envs/env_isaaclab/bin/python"
SDK_ROOT = "/home/user/linkerhand-ros-sdk"
THUMB_YAW_SLOT = 10

MAX_FAILURE_FRACTION = 0.10
MAX_BLOCK_RANGE_DEG = 0.5
MAX_PALM_DRIFT_DEG = 0.5
WRAP_LIMIT_DEG = 80.0
MAX_EXTENT_SHRINK_FRACTION = 0.08
MAX_AREA_SHRINK_FRACTION = 0.08
MAX_THUMB_L_ANGLE_CHANGE_DEG = 5.0
MAX_ADJACENT_STEP_RAW = 14
EXPECTED_SERIAL = "LHT20-010-415-L-B-1-D"
RESERVED_SLOTS = (11, 12, 13, 14)


def axis_delta_deg(value: float, reference: float) -> float:
    """Small signed change between two undirected line headings."""
    return (value - reference + 90.0) % 180.0 - 90.0


def cross_referenced_yaw_deg(
        angle_summary: dict[str, Any], zero: dict[str, float]) -> float:
    """Thumb-longitudinal change relative to the unobscured palm cross."""
    thumb_delta = axis_delta_deg(
        angle_summary["thumb_heading_deg_2d"]["median"],
        zero["thumb_heading_deg_2d"])
    palm_cross_delta = axis_delta_deg(
        angle_summary["palm_cross_heading_deg_2d"]["median"],
        zero["palm_cross_heading_deg_2d"])
    return thumb_delta - palm_cross_delta


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def zero_medians(reference: Path) -> dict[str, float]:
    caps = sorted(reference.glob("camera_*_fixed_exp_180f_summary.json"))
    if len(caps) != 2:
        raise SystemExit(f"expected two zero captures in {reference}")
    summaries = [read_json(c)["angle_summary_deg"] for c in caps]
    return {k: sum(s[k]["median"] for s in summaries) / len(summaries)
            for k in summaries[0]}


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label}\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"STOP: {label} exited {result.returncode}")


def check_gates(summary_path: Path, zero: dict[str, float],
                previous: float | None, expect_increasing: bool) -> float:
    s = read_json(summary_path)
    if s["capture"]["failure_fraction"] > MAX_FAILURE_FRACTION:
        raise SystemExit(
            f"STOP: camera failure fraction {s['capture']['failure_fraction']:.3f}")

    key = "thumb_cmc_yaw_deg_2d"
    blocks = s["block_medians_deg"][key]
    block_range = max(blocks) - min(blocks)
    if block_range > MAX_BLOCK_RANGE_DEG:
        raise SystemExit(f"STOP: yaw block median range {block_range:.4f} deg")

    yaw = cross_referenced_yaw_deg(s["angle_summary_deg"], zero)
    if abs(yaw) > WRAP_LIMIT_DEG:
        raise SystemExit(
            f"STOP: cross-referenced yaw {yaw:+.3f} deg is within "
            f"{90 - abs(yaw):.1f} deg of the line-heading wrap boundary")

    palm_drift = axis_delta_deg(
        s["angle_summary_deg"]["palm_cross_heading_deg_2d"]["median"],
        zero["palm_cross_heading_deg_2d"])
    if abs(palm_drift) > MAX_PALM_DRIFT_DEG:
        raise SystemExit(f"STOP: palm reference drifted {palm_drift:+.4f} deg")

    extent = s["angle_summary_deg"]["thumb_major_extent_px"]["median"]
    shrink = 1.0 - extent / zero["thumb_major_extent_px"]
    if shrink > MAX_EXTENT_SHRINK_FRACTION:
        raise SystemExit(
            f"STOP: thumb marker projected extent shrank {100 * shrink:.1f}%; "
            "the rotation is leaving the image plane and the 2D angle is no "
            "longer a faithful angle")

    area = s["angle_summary_deg"]["thumb_marker_area_px"]["median"]
    area_shrink = 1.0 - area / zero["thumb_marker_area_px"]
    if area_shrink > MAX_AREA_SHRINK_FRACTION:
        raise SystemExit(
            f"STOP: thumb marker area shrank {100 * area_shrink:.1f}%; "
            "the marker may be clipped or leaving the image plane")

    thumb_l = s["angle_summary_deg"]["thumb_l_angle_deg"]["median"]
    thumb_l_change = thumb_l - zero["thumb_l_angle_deg"]
    if abs(thumb_l_change) > MAX_THUMB_L_ANGLE_CHANGE_DEG:
        raise SystemExit(
            f"STOP: thumb L angle changed {thumb_l_change:+.3f} deg; "
            "the segment is leaving the calibrated image plane")

    if previous is not None:
        delta = yaw - previous
        if (expect_increasing and delta <= 0.0) or (
                not expect_increasing and delta >= 0.0):
            raise SystemExit(
                f"STOP: yaw moved {delta:+.4f} deg against the commanded direction")
    print(f"[gate] yaw {yaw:+.4f} deg from zero  block_range={block_range:.4f}  "
          f"palm_drift={palm_drift:+.4f}  extent_shrink={100 * shrink:+.2f}%  "
          f"area_shrink={100 * area_shrink:+.2f}%  "
          f"thumb_l_change={thumb_l_change:+.3f}  OK",
          flush=True)
    return yaw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--start-raw", type=int, required=True)
    p.add_argument("--direction", choices=("down", "up"), required=True)
    p.add_argument("--step-raw", type=int, default=14)
    p.add_argument("--min-step-raw", type=int, default=8)
    p.add_argument("--stop-at", type=int, required=True)
    p.add_argument("--roll-hold-raw", type=int, required=True)
    p.add_argument("--pitch-hold-raw", type=int, required=True)
    p.add_argument("--mcp-hold-raw", type=int, required=True)
    p.add_argument("--settle-tolerance-raw", type=int, default=2)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--camera-arg", action="append", default=[])
    p.add_argument("--previous-yaw-deg", type=float, default=None)
    p.add_argument("--tag", default="yaw")
    args = p.parse_args()
    if not 1 <= args.step_raw <= MAX_ADJACENT_STEP_RAW:
        p.error(f"--step-raw must be in 1..{MAX_ADJACENT_STEP_RAW}")
    if not 1 <= args.min_step_raw <= args.step_raw:
        p.error("--min-step-raw must be positive and no larger than --step-raw")

    zero = zero_medians(args.reference_dir)
    previous = args.previous_yaw_deg
    results: list[dict[str, Any]] = []
    iteration = 0

    while True:
        state20 = fresh_state(args.session, args.tag, iteration, args.calib)
        current = state20[THUMB_YAW_SLOT]
        if iteration == 0 and abs(current - args.start_raw) > 2:
            raise SystemExit(
                f"STOP: requested start raw{args.start_raw}, fresh read is "
                f"raw{current}")
        remaining = (current - args.stop_at if args.direction == "down"
                     else args.stop_at - current)
        if remaining <= 0:
            break
        step = min(args.step_raw, remaining)
        if step < args.min_step_raw:
            print(f"[driver] {remaining} raw short of the bound is below the "
                  f"{args.min_step_raw} raw floor; stopping cleanly", flush=True)
            break
        target = current - step if args.direction == "down" else current + step
        point = args.session / f"{args.tag}_raw{target:03d}"
        if point.exists():
            raise SystemExit(f"STOP: refusing to overwrite existing point {point}")
        point.mkdir(parents=True)

        motion_out = point / f"motion_raw{current:03d}_to_raw{target:03d}.json"
        run([SDK_PYTHON, "-m", "tools.run_g20_thumb_cmc_yaw_candidate_step",
             "--sdk-root", SDK_ROOT, "--calib", str(args.calib), "--can", "can0",
             "--expected-start-raw", str(current), "--target-raw", str(target),
             "--roll-hold-raw", str(args.roll_hold_raw),
             "--pitch-hold-raw", str(args.pitch_hold_raw),
             "--mcp-hold-raw", str(args.mcp_hold_raw),
             "--settle-tolerance-raw", str(args.settle_tolerance_raw),
             "--speed", "5", "--samples", "20", "--hz", "5",
             "--out", str(motion_out), "--execute"],
            f"motion {current} -> {target}")

        prefix = point / "camera_fixed_exp_180f"
        run([CAMERA_PYTHON, "-m", "tools.measure_g20_thumb_yaw",
             "--frames", "180", "--out-prefix", str(prefix), *args.camera_arg],
            f"camera at raw{target}")

        yaw = check_gates(Path(f"{prefix}_summary.json"), zero, previous,
                          args.direction == "down")
        readback = read_json(motion_out)["result"][
            "settled_state20_median"][THUMB_YAW_SLOT]
        if readback == current:
            raise SystemExit(
                f"STOP: command raw{target} produced no readback movement from "
                f"raw{current} with no fault; a human must look before any "
                "larger command")
        results.append({"command_raw": target, "stable_readback_raw": readback,
                        "yaw_deg_from_zero": yaw,
                        "camera_summary": str(Path(f"{prefix}_summary.json"))})
        print(f"[driver] cmd raw{target} -> readback raw{readback} "
              f"(undershoot {abs(readback - target)} raw)", flush=True)
        previous = yaw
        iteration += 1

    print("\n=== sweep complete", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return 0


def fresh_state(session: Path, tag: str, iteration: int,
                calib: Path) -> list[int]:
    """Read a new state after the previous CAN process has closed.

    The yaw axis can relax after a runner exits, so the previous runner's
    settled readback is evidence for that point, not a safe starting value for
    the next command.
    """
    out = session / f"{tag}_prestep_{iteration:03d}_readonly.json"
    if out.exists():
        raise SystemExit(f"STOP: refusing to overwrite {out}")
    run([SDK_PYTHON, "-m", "tools.record_g20_sdk_snapshot",
         "--sdk-root", SDK_ROOT, "--side", "left", "--hand-joint", "G20",
         "--can", "can0", "--calib", str(calib),
         "--samples", "5", "--hz", "20", "--out", str(out)],
        f"fresh read before {tag} step {iteration}")
    snapshot = read_json(out)
    serial = snapshot.get("identity", {}).get("serial")
    if serial != EXPECTED_SERIAL:
        raise SystemExit(
            f"STOP: serial mismatch, expected {EXPECTED_SERIAL}, got {serial}")
    states: list[list[int]] = []
    for sample in snapshot.get("samples", []):
        state = sample.get("state20")
        faults = sample.get("faults20")
        if not isinstance(state, list) or len(state) != 20:
            raise SystemExit("STOP: fresh read returned invalid state20")
        if not isinstance(faults, list) or len(faults) != 20:
            raise SystemExit("STOP: fresh read returned invalid fault telemetry")
        state20 = [int(round(value)) for value in state]
        if any(faults):
            raise SystemExit(
                "STOP: fresh read has nonzero faults "
                f"{[(i, value) for i, value in enumerate(faults) if value]}")
        if any(state20[slot] != 0 for slot in RESERVED_SLOTS):
            raise SystemExit(
                "STOP: fresh read has nonzero reserved slots "
                f"{[(slot, state20[slot]) for slot in RESERVED_SLOTS]}")
        states.append(state20)
    if len(states) != 5:
        raise SystemExit(f"STOP: expected 5 fresh state samples, got {len(states)}")
    return [int(statistics.median(column)) for column in zip(*states)]

if __name__ == "__main__":
    sys.exit(main())

