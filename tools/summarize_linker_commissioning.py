#!/usr/bin/env python3
"""Summarize LinkerHand startup readback and a deployment CSV as JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _number(row: dict[str, str], key: str) -> float | None:
    value = (row.get(key) or "").strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _vector(row: dict[str, str], prefix: str, size: int) -> list[float] | None:
    result = [_number(row, f"{prefix}{index}") for index in range(size)]
    return None if any(value is None for value in result) else list(result)  # type: ignore[arg-type]


def _max_consecutive(values: list[bool]) -> int:
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase", default="policy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from screwdriver_rl.deploy import linker_sdk_map as sdkmap
    from screwdriver_rl.deploy.policy import DeployPolicy

    sdkmap.apply_calibration(args.calib)
    policy = DeployPolicy(args.checkpoint, device="cpu")
    joint_names = [joint.name for joint in sdkmap.active_joints()]
    lower = policy.finger_lower[0].tolist()
    upper = policy.finger_upper[0].tolist()
    home = policy.home_targets[0].tolist()

    state_payload = json.loads(args.state_json.read_text(encoding="utf-8"))
    state20 = state_payload.get("state20")
    if not isinstance(state20, list) or len(state20) != 20:
        raise ValueError(f"{args.state_json} does not contain a valid state20")
    q_startup = sdkmap.sdk_range_to_joints16(state20)
    startup_errors = [
        measured - target for measured, target in zip(q_startup, home)
    ]

    with args.trace.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if args.phase is None or row.get("phase") == args.phase
        ]
    if not rows:
        raise ValueError(f"no {args.phase!r} rows in {args.trace}")

    actions = [value for row in rows if (value := _vector(row, "act", 16)) is not None]
    targets = [value for row in rows if (value := _vector(row, "tgt", 16)) is not None]
    measured = [value for row in rows if (value := _vector(row, "q", 16)) is not None]
    commands = [value for row in rows if (value := _vector(row, "cmd", 20)) is not None]
    states = [value for row in rows if (value := _vector(row, "state", 20)) is not None]
    times = [
        value for row in rows if (value := _number(row, "t_mono")) is not None
    ]
    period_s = policy.codec.spec.control_period_ns / 1_000_000_000.0
    intervals = [later - earlier for earlier, later in zip(times, times[1:])]

    action_flat = [value for row in actions for value in row]
    command_flat = [value for row in commands for value in row]
    per_joint = []
    for index, name in enumerate(joint_names):
        joint_actions = [row[index] for row in actions]
        joint_targets = [row[index] for row in targets]
        joint_measured = [row[index] for row in measured]
        rail_flags = [
            value <= lower[index] + 1.0e-6 or value >= upper[index] - 1.0e-6
            for value in joint_targets
        ]
        per_joint.append(
            {
                "joint": name,
                "action_abs_max": max(map(abs, joint_actions)),
                "action_near_unit_count": sum(
                    abs(value) >= 0.999 for value in joint_actions
                ),
                "target_min_rad": min(joint_targets),
                "target_max_rad": max(joint_targets),
                "target_rail_count": sum(rail_flags),
                "target_rail_max_consecutive": _max_consecutive(rail_flags),
                "measured_span_rad": max(joint_measured) - min(joint_measured),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "calibration": str(Path(args.calib).resolve()),
        "startup_readback": {
            "state_json": str(args.state_json.resolve()),
            "target_rad16": home,
            "measured_rad16": q_startup,
            "error_measured_minus_target_rad16": startup_errors,
            "mean_abs_error_rad": sum(map(abs, startup_errors)) / len(startup_errors),
            "max_abs_error_rad": max(map(abs, startup_errors)),
        },
        "trace": {
            "path": str(args.trace.resolve()),
            "phase": args.phase,
            "rows": len(rows),
            "valid_action_rows": len(actions),
            "valid_target_rows": len(targets),
            "valid_measured_rows": len(measured),
            "valid_command_rows": len(commands),
            "valid_state_rows": len(states),
            "control_period_s": period_s,
            "interval_mean_s": (
                sum(intervals) / len(intervals) if intervals else None
            ),
            "interval_max_s": max(intervals) if intervals else None,
            "intervals_over_period_count": sum(
                interval > period_s for interval in intervals
            ),
            "action_abs_max": max(map(abs, action_flat)),
            "action_near_unit_count": sum(
                abs(value) >= 0.999 for value in action_flat
            ),
            "action_near_unit_fraction": sum(
                abs(value) >= 0.999 for value in action_flat
            )
            / len(action_flat),
            "target_rail_values": sum(
                item["target_rail_count"] for item in per_joint
            ),
            "raw_command_rail_values": sum(
                value <= 1.0 or value >= 254.0 for value in command_flat
            ),
            "per_joint": per_joint,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[commissioning-summary] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
