#!/usr/bin/env python3
"""Audit random G20 specs through the production physical-LUT mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_g20_random_dynamic_specs import JOINTS, SAFE_ENVELOPE
from tools.run_g20_random_dynamic_trajectory import _smooth_segment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-raw-step", type=int, default=5)
    args = parser.parse_args()

    from screwdriver_rl.deploy import linker_sdk_map as sdkmap

    sdkmap.apply_calibration(str(args.calib))
    joints = sdkmap.active_joints()
    if tuple(joint.name for joint in joints) != JOINTS:
        raise RuntimeError("production calibration joint order mismatch")
    rows = []
    global_worst: dict[str, object] = {"step": 0}
    for path in sorted(args.spec_dir.glob("0*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        waypoints = data["waypoints"]
        for row in waypoints:
            for name, value in zip(JOINTS, row["semantic"]):
                lo, hi = SAFE_ENVELOPE[name]
                if not lo <= float(value) <= hi:
                    raise ValueError(f"{path.name}/{row['name']}: {name} outside envelope")
        frames: list[tuple[str, list[float]]] = []
        previous = [float(value) for value in waypoints[0]["semantic"]]
        for row in waypoints[1:]:
            target = [float(value) for value in row["semantic"]]
            frames.extend(
                (f"to_{row['name']}", semantic)
                for semantic in _smooth_segment(
                    previous, target, float(row["duration_s"]), args.rate_hz
                )
            )
            previous = target
        previous_raw = sdkmap.joints16_to_sdk_range(waypoints[0]["semantic"])
        local_worst: dict[str, object] = {"step": 0}
        for index, (phase, semantic) in enumerate(frames):
            raw = sdkmap.joints16_to_sdk_range(semantic)
            diffs = [abs(a - b) for a, b in zip(raw, previous_raw)]
            step = max(diffs)
            if step > int(local_worst["step"]):
                slot = diffs.index(step)
                local_worst = {
                    "step": step,
                    "frame": index,
                    "phase": phase,
                    "slot": slot,
                    "before": previous_raw[slot],
                    "after": raw[slot],
                }
            previous_raw = raw
        if int(local_worst["step"]) > int(global_worst["step"]):
            global_worst = {"spec": data["name"], **local_worst}
        spans = {
            name: max(float(row["semantic"][i]) for row in waypoints)
            - min(float(row["semantic"][i]) for row in waypoints)
            for i, name in enumerate(JOINTS)
        }
        rows.append(
            {
                "name": data["name"],
                "path": str(path),
                "waypoint_count": len(waypoints),
                "interpolated_transition_frames": len(frames),
                "worst_interpolated_raw_step": local_worst,
                "joint_span_rad": spans,
            }
        )
    result = {
        "schema_version": 1,
        "calibration": str(args.calib.resolve()),
        "joint_order16": list(JOINTS),
        "rate_hz": args.rate_hz,
        "max_raw_step_limit": args.max_raw_step,
        "global_worst_interpolated_raw_step": global_worst,
        "passed": int(global_worst["step"]) <= args.max_raw_step,
        "note": "Waypoint-to-waypoint audit only; the hardware executor separately gates the fresh measured start-to-first-waypoint segment.",
        "trajectories": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"[audit] {len(rows)} trajectories, worst raw step "
        f"{global_worst['step']}/{args.max_raw_step}, passed={result['passed']}"
    )
    if not result["passed"]:
        raise RuntimeError("random trajectory raw-step audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
