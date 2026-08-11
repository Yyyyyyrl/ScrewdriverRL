#!/usr/bin/env python3
"""Drive a sequence of adjacent MCP-roll points, stopping at the first gate failure.

This only orchestrates already-tested tools: it alternates
``run_g20_mcp_roll_candidate_step`` (CAN open, camera closed) with
``measure_g20_mcp_roll`` (camera open, CAN closed), never both at once, and
evaluates the runbook gates after every point.

It is fail-closed. Any non-zero tool exit, any gate breach, or any unexpected
exception stops the run immediately and leaves every record on disk. It never
retries, never widens a tolerance, and never approves a mechanical endpoint:
the caller chooses the target list and a human approves end regions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

CAMERA_PYTHON = "/home/user/miniconda3/bin/python3"
SDK_PYTHON = "/home/user/miniconda3/envs/env_isaaclab/bin/python"

OTHER_FINGERS = {
    "index": ("middle", "ring", "pinky"),
    "middle": ("index", "ring", "pinky"),
    "ring": ("index", "middle", "pinky"),
    "pinky": ("index", "middle", "ring"),
}

# Runbook section 9 gates that this driver can evaluate from the artifacts.
MAX_FAILURE_FRACTION = 0.10
MAX_BLOCK_RANGE_DEG = 0.5
MAX_PALM_DRIFT_DEG = 0.5
# Only meaningful in four-finger reference mode. Once the non-target fingers
# are pitched out of the way their tapes are no longer measured, so this cannot
# act as the contact detector any more. That role passes to the step runner's
# +-2 raw settle criterion: when the target loads against something it stops
# further and further short of its command, which is exactly how the
# 2026-08-03 index-to-middle contact first showed up (undershoot 0, 1, 2, 3).
MAX_NON_TARGET_MARKER_DRIFT_DEG = 0.15


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def zero_medians(reference_dir: Path) -> dict[str, float]:
    """Mean of the two 180-frame zero captures, per angle key."""
    summaries = [
        read_json(path)["angle_summary_deg"]
        for path in sorted(reference_dir.glob("camera_*_fixed_exp_180f_summary.json"))
    ]
    if len(summaries) != 2:
        raise SystemExit(f"expected two zero captures in {reference_dir}")
    return {
        key: sum(summary[key]["median"] for summary in summaries) / len(summaries)
        for key in summaries[0]
    }


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label}\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"STOP: {label} exited {result.returncode}")


def check_gates(finger: str, summary_path: Path, zero: dict[str, float],
                previous_abduction: float | None, expect_increasing: bool,
                check_non_target: bool = True) -> float:
    summary = read_json(summary_path)
    capture = summary["capture"]
    if capture["failure_fraction"] > MAX_FAILURE_FRACTION:
        raise SystemExit(
            f"STOP: camera failure fraction {capture['failure_fraction']:.3f}"
        )

    key = f"{finger}_abduction_deg_2d"
    blocks = summary["block_medians_deg"][key]
    block_range = max(blocks) - min(blocks)
    if block_range > MAX_BLOCK_RANGE_DEG:
        raise SystemExit(f"STOP: {key} block median range {block_range:.4f} deg")

    palm_key = "palm_reference_heading_deg_2d"
    palm_drift = summary["angle_summary_deg"][palm_key]["median"] - zero[palm_key]
    if abs(palm_drift) > MAX_PALM_DRIFT_DEG:
        raise SystemExit(f"STOP: palm reference drifted {palm_drift:+.4f} deg")

    for other in (OTHER_FINGERS[finger] if check_non_target else ()):
        other_key = f"{other}_abduction_deg_2d"
        drift = summary["angle_summary_deg"][other_key]["median"] - zero[other_key]
        if abs(drift) > MAX_NON_TARGET_MARKER_DRIFT_DEG:
            raise SystemExit(
                f"STOP: non-target {other} moved {drift:+.4f} deg from zero"
            )

    abduction = summary["angle_summary_deg"][key]["median"] - zero[key]
    if previous_abduction is not None:
        delta = abduction - previous_abduction
        if expect_increasing and delta <= 0.0:
            raise SystemExit(
                f"STOP: {key} moved {delta:+.4f} deg against the commanded "
                "direction"
            )
        if not expect_increasing and delta >= 0.0:
            raise SystemExit(
                f"STOP: {key} moved {delta:+.4f} deg against the commanded "
                "direction"
            )
    print(f"[gate] {finger} abduction {abduction:+.4f} deg from zero  "
          f"block_range={block_range:.4f}  palm_drift={palm_drift:+.4f}  OK",
          flush=True)
    return abduction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--finger", required=True, choices=tuple(OTHER_FINGERS))
    parser.add_argument("--start-raw", type=int, required=True)
    parser.add_argument("--direction", choices=("down", "up"), required=True)
    parser.add_argument("--step-raw", type=int, default=16,
                        help="nominal raw per step, measured from the readback")
    parser.add_argument("--min-step-raw", type=int, default=8,
                        help="refuse steps at or below the observed deadband")
    parser.add_argument("--stop-at", type=int, required=True,
                        help="raw bound this run will not pass without approval")
    parser.add_argument("--pitch-hold-raw", type=int, required=True)
    parser.add_argument("--pip-hold-raw", type=int, required=True)
    parser.add_argument("--other-roll-hold", action="append", required=True)
    parser.add_argument("--sdk-root", type=Path,
                        default=Path("/home/user/linkerhand-ros-sdk"))
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument(
        "--target-row-band", default=None, metavar="Y0,Y1",
        help=("sweep mode: measure only this finger, whose marker must fall in "
              "this image row band. Required once the other fingers are "
              "pitched away; disables the non-target marker gate."))
    parser.add_argument(
        "--camera-arg", action="append", default=[],
        help="extra flag passed straight through to the camera tool; repeat. "
             "The detector geometry changes whenever the isolation posture or "
             "the hand pose changes, so it is never hard-coded here.")
    parser.add_argument("--settle-tolerance-raw", type=int, default=2)
    parser.add_argument("--previous-abduction-deg", type=float, default=None)
    parser.add_argument("--tag", default="flex")
    args = parser.parse_args()

    zero = zero_medians(args.reference_dir)
    current = args.start_raw
    previous = args.previous_abduction_deg
    results: list[dict[str, Any]] = []

    # This actuator stops one to three raw short of every command, and a
    # follow-up step small enough to close that gap lands inside the deadband
    # and does not move at all. Chasing exact template knots therefore
    # manufactures dead steps. Instead each step is a fixed size measured from
    # the previous stable readback, and wherever it lands becomes the knot.
    while True:
        remaining = (current - args.stop_at if args.direction == "down"
                     else args.stop_at - current)
        if remaining <= 0:
            break
        step = min(args.step_raw, remaining)
        if step < args.min_step_raw:
            print(f"[driver] {remaining} raw short of the bound is below the "
                  f"{args.min_step_raw} raw deadband floor; stopping cleanly",
                  flush=True)
            break
        target = current - step if args.direction == "down" else current + step
        expect_increasing = args.direction == "down"  # raw down is positive
        point = args.session / f"{args.tag}_raw{target:03d}"
        point.mkdir(parents=True, exist_ok=True)

        cmd = [SDK_PYTHON, "-m", "tools.run_g20_mcp_roll_candidate_step",
               "--sdk-root", str(args.sdk_root), "--calib", str(args.calib),
               "--can", "can0", "--finger", args.finger,
               "--expected-start-raw", str(current),
               "--target-raw", str(target),
               "--pitch-hold-raw", str(args.pitch_hold_raw),
               "--pip-hold-raw", str(args.pip_hold_raw),
               "--settle-tolerance-raw", str(args.settle_tolerance_raw)]
        for hold in args.other_roll_hold:
            cmd += ["--other-roll-hold", hold]
        cmd += ["--speed", "5", "--samples", "20", "--hz", "5",
                "--out", str(point / f"motion_raw{current:03d}_to_raw{target:03d}.json"),
                "--execute"]
        run(cmd, f"motion {current} -> {target}")

        prefix = point / "camera_fixed_exp_180f"
        camera_cmd = [CAMERA_PYTHON, "-m", "tools.measure_g20_mcp_roll",
                      "--frames", "180", "--out-prefix", str(prefix)]
        if args.target_row_band:
            camera_cmd += ["--target-finger", args.finger,
                           "--target-row-band", args.target_row_band]
        camera_cmd += args.camera_arg
        run(camera_cmd, f"camera at raw{target}")

        abduction = check_gates(
            args.finger, Path(f"{prefix}_summary.json"), zero, previous,
            expect_increasing, check_non_target=not args.target_row_band,
        )
        motion = read_json(
            point / f"motion_raw{current:03d}_to_raw{target:03d}.json")
        readback = motion["result"]["settled_state20_median"][
            {"index": 6, "middle": 7, "ring": 8, "pinky": 9}[args.finger]]
        if readback == current:
            raise SystemExit(
                f"STOP: command raw{target} produced no readback movement from "
                f"raw{current} with no fault. Either the deadband swallowed a "
                f"{step} raw step or this is the mechanical end. A human must "
                "approve the endpoint before any larger command."
            )
        results.append({
            "command_raw": target,
            "stable_readback_raw": readback,
            "abduction_deg_from_zero": abduction,
            "target_error_raw": motion["result"]["target_error_raw"],
            "camera_summary": str(Path(f"{prefix}_summary.json")),
        })
        print(f"[driver] cmd raw{target} -> readback raw{readback} "
              f"(undershoot {abs(readback - target)} raw)", flush=True)
        previous = abduction
        current = readback

    print("\n=== sweep complete", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
