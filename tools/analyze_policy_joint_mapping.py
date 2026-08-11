#!/usr/bin/env python3
"""Analyze policy target -> SDK command -> next state -> decoded joint mapping."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _vector(row: dict[str, str], prefix: str, size: int) -> list[float]:
    values = []
    for index in range(size):
        text = (row.get(f"{prefix}{index}") or "").strip()
        if not text:
            raise ValueError(f"missing {prefix}{index} in policy row")
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"non-finite {prefix}{index} in policy row")
        values.append(value)
    return values


def _median_state(snapshot: dict[str, Any]) -> tuple[list[float], list[float]]:
    samples = snapshot.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("post snapshot has no samples")
    states = [
        sample.get("state20")
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("state20"), list)
        and len(sample["state20"]) == 20
    ]
    if not states:
        raise ValueError("post snapshot has no valid state20 samples")
    median = [statistics.median(column) for column in zip(*states)]
    span = [max(column) - min(column) for column in zip(*states)]
    return median, span


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--post-snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from screwdriver_rl.deploy import linker_sdk_map as sdkmap

    sdkmap.apply_calibration(args.calib)
    joints = list(sdkmap.active_joints())

    with args.trace.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row.get("phase") == "policy"
        ]
    if not rows:
        raise ValueError(f"no policy rows in {args.trace}")

    decoded_rows = []
    encode_raw_max_error = 0.0
    decode_rad_max_error = 0.0
    for row in rows:
        target = _vector(row, "tgt", 16)
        command = _vector(row, "cmd", 20)
        state = _vector(row, "state", 20)
        measured = _vector(row, "q", 16)
        expected_command = sdkmap.joints16_to_sdk_range(target)
        decoded_state = sdkmap.sdk_range_to_joints16(state)
        encode_raw_max_error = max(
            encode_raw_max_error,
            max(abs(actual - expected) for actual, expected in zip(command, expected_command)),
        )
        decode_rad_max_error = max(
            decode_rad_max_error,
            max(abs(actual - expected) for actual, expected in zip(measured, decoded_state)),
        )
        decoded_rows.append(
            {
                "tick": int(float(row["tick"])),
                "target": target,
                "command": command,
                "state": state,
                "measured": measured,
            }
        )

    dynamic = []
    for current, following in zip(decoded_rows, decoded_rows[1:]):
        per_joint = []
        for index, joint in enumerate(joints):
            raw_error = following["state"][joint.slot] - current["command"][joint.slot]
            rad_error = following["measured"][index] - current["target"][index]
            per_joint.append(
                {
                    "joint": joint.name,
                    "slot": joint.slot,
                    "command_raw": current["command"][joint.slot],
                    "next_state_raw": following["state"][joint.slot],
                    "raw_error_state_minus_command": raw_error,
                    "target_rad": current["target"][index],
                    "next_measured_rad": following["measured"][index],
                    "rad_error_measured_minus_target": rad_error,
                }
            )
        dynamic.append(
            {
                "command_tick": current["tick"],
                "state_tick": following["tick"],
                "per_joint": per_joint,
            }
        )

    snapshot = json.loads(args.post_snapshot.read_text(encoding="utf-8"))
    post_state, post_state_span = _median_state(snapshot)
    post_q = sdkmap.sdk_range_to_joints16(post_state)
    last = decoded_rows[-1]
    final_per_joint = []
    for index, joint in enumerate(joints):
        command_raw = last["command"][joint.slot]
        state_raw = post_state[joint.slot]
        target_rad = last["target"][index]
        measured_rad = post_q[index]
        quantized_rad = sdkmap.sdk_range_to_joints16(last["command"])[index]
        final_per_joint.append(
            {
                "joint": joint.name,
                "slot": joint.slot,
                "target_rad": target_rad,
                "command_raw": command_raw,
                "post_state_raw_median": state_raw,
                "post_state_raw_span": post_state_span[joint.slot],
                "post_measured_rad": measured_rad,
                "raw_error_state_minus_command": state_raw - command_raw,
                "rad_error_measured_minus_target": measured_rad - target_rad,
                "mapping_quantization_error_rad": quantized_rad - target_rad,
            }
        )

    final_abs_rad = [
        abs(item["rad_error_measured_minus_target"]) for item in final_per_joint
    ]
    final_abs_raw = [
        abs(item["raw_error_state_minus_command"]) for item in final_per_joint
    ]
    payload = {
        "schema_version": 1,
        "trace": str(args.trace.resolve()),
        "calibration": str(Path(args.calib).resolve()),
        "post_snapshot": str(args.post_snapshot.resolve()),
        "policy_ticks": len(decoded_rows),
        "software_mapping_checks": {
            "target_to_recorded_command_max_abs_raw_error": encode_raw_max_error,
            "recorded_state_to_recorded_q_max_abs_rad_error": decode_rad_max_error,
        },
        "dynamic_one_tick_alignment": dynamic,
        "final_command_to_stable_readback": {
            "command_tick": last["tick"],
            "mean_abs_rad_error": statistics.fmean(final_abs_rad),
            "max_abs_rad_error": max(final_abs_rad),
            "mean_abs_raw_error": statistics.fmean(final_abs_raw),
            "max_abs_raw_error": max(final_abs_raw),
            "per_joint": final_per_joint,
        },
        "interpretation_limit": (
            "This validates software slot/order/range mapping and actuator encoder "
            "tracking. Because command and readback use the same calibration, it "
            "does not independently prove the physical link angle."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        "software mapping: "
        f"target->cmd max {encode_raw_max_error:.3f} raw, "
        f"state->q max {decode_rad_max_error:.6f} rad"
    )
    print(
        "final command -> stable readback: "
        f"MAE {statistics.fmean(final_abs_rad):.5f} rad, "
        f"max {max(final_abs_rad):.5f} rad; "
        f"raw MAE {statistics.fmean(final_abs_raw):.2f}, "
        f"raw max {max(final_abs_raw):.2f}"
    )
    print(
        f"{'joint':18s} {'slot':>4s} {'tgt_rad':>8s} {'q_rad':>8s} "
        f"{'err_rad':>8s} {'cmd':>5s} {'state':>6s} {'rawerr':>7s} {'span':>5s}"
    )
    for item in final_per_joint:
        print(
            f"{item['joint']:18s} {item['slot']:4d} "
            f"{item['target_rad']:8.4f} {item['post_measured_rad']:8.4f} "
            f"{item['rad_error_measured_minus_target']:8.4f} "
            f"{item['command_raw']:5.0f} {item['post_state_raw_median']:6.0f} "
            f"{item['raw_error_state_minus_command']:7.0f} "
            f"{item['post_state_raw_span']:5.0f}"
        )
    print(f"[joint-mapping] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
