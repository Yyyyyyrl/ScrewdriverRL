#!/usr/bin/env python3
"""Move the three active G20 index axes by one small clearance-pose step.

The index 0x42 frame is ``[side(s6), r11, pitch(s1), r13, r14, pip(s16)]``.
Every changed active axis is limited to 17 raw counts from fresh readback.
The tool sends exactly one index frame and records state/fault/temperature.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Sequence

from tools.run_g20_index_pip_candidate_step import (
    ACTIVE_SLOTS,
    EXPECTED_SERIAL,
    RESERVED_SLOTS,
    _fault_snapshot,
    _fresh_index_state6,
    _int_vector,
    _median20,
    _read_state20,
    _sha256,
    _temperature_snapshot,
    _write_json,
)


INDEX_SIDE_SLOT = 6
INDEX_PITCH_SLOT = 1
INDEX_PIP_SLOT = 16
INDEX_FRAME_SLOTS = (6, 11, 1, 13, 14, 16)
ACTIVE_INDEX_SLOTS = (INDEX_SIDE_SLOT, INDEX_PITCH_SLOT, INDEX_PIP_SLOT)
APPROVED_INDEX_TORQUE6 = [80, 40, 80, 40, 40, 40]


def build_index_clearance_command(
    state20: Sequence[int],
    index_state6: Sequence[int],
    *,
    expected_start_side_raw: int,
    expected_start_pitch_raw: int,
    expected_start_pip_raw: int,
    target_side_raw: int,
    target_pitch_raw: int,
    target_pip_raw: int,
    start_tolerance_raw: int = 2,
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    before = _int_vector(state20, length=20, label="state20")
    index_before = _int_vector(index_state6, length=6, label="index_state6")
    expected_frame = [before[slot] for slot in INDEX_FRAME_SLOTS]
    for frame_index, (actual, expected) in enumerate(
        zip(index_before, expected_frame)
    ):
        tolerance = 0 if frame_index in (1, 3, 4) else start_tolerance_raw
        if abs(actual - expected) > tolerance:
            raise ValueError(
                "fresh index frame disagrees with raw20: "
                f"index {frame_index}, {actual} != {expected}"
            )
    if any(before[slot] != 0 for slot in RESERVED_SLOTS):
        raise ValueError(
            "reserved slots 11..14 are not zero: "
            f"{[before[slot] for slot in RESERVED_SLOTS]}"
        )

    expected = (
        expected_start_side_raw,
        expected_start_pitch_raw,
        expected_start_pip_raw,
    )
    targets = (target_side_raw, target_pitch_raw, target_pip_raw)
    for slot, start, target in zip(ACTIVE_INDEX_SLOTS, expected, targets):
        if not 0 <= target <= 255:
            raise ValueError(f"target slot {slot} raw must be in 0..255")
        if abs(before[slot] - start) > start_tolerance_raw:
            raise ValueError(
                f"slot {slot} must start within {start_tolerance_raw} raw "
                f"of {start}, got {before[slot]}"
            )
        step = abs(target - before[slot])
        if step > 17:
            raise ValueError(
                f"actual slot {slot} step must be 0..17 raw, got "
                f"{before[slot]} to {target} ({step})"
            )
    if all(before[slot] == target for slot, target in zip(ACTIVE_INDEX_SLOTS, targets)):
        raise ValueError("clearance step must change at least one active axis")

    after = list(before)
    for slot, target in zip(ACTIVE_INDEX_SLOTS, targets):
        after[slot] = target
    index_after = [after[slot] for slot in INDEX_FRAME_SLOTS]
    diff = [
        {"slot": slot, "before": old, "after": new}
        for slot, (old, new) in enumerate(zip(before, after))
        if old != new
    ]
    if any(item["slot"] not in ACTIVE_INDEX_SLOTS for item in diff):
        raise AssertionError(f"unsafe raw20 diff: {diff}")
    if index_after[1:2] + index_after[3:5] != [0, 0, 0]:
        raise AssertionError(f"unsafe index reserved values: {index_after}")
    return after, index_after, diff


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--expected-serial", default=EXPECTED_SERIAL)
    for name in ("side", "pitch", "pip"):
        parser.add_argument(f"--expected-start-{name}-raw", type=int, required=True)
        parser.add_argument(f"--target-{name}-raw", type=int, required=True)
    parser.add_argument("--start-tolerance-raw", type=int, default=2)
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    if args.expected_serial != EXPECTED_SERIAL:
        parser.error(f"expected serial must remain {EXPECTED_SERIAL}")
    for name in ("side", "pitch", "pip"):
        for kind in ("expected_start", "target"):
            value = getattr(args, f"{kind}_{name}_raw")
            if not 0 <= value <= 255:
                parser.error(f"--{kind.replace('_', '-')}-{name}-raw must be in 0..255")
    if args.start_tolerance_raw not in (0, 1, 2):
        parser.error("--start-tolerance-raw must be 0, 1, or 2")
    if not 1 <= args.speed <= 10:
        parser.error("--speed must be in 1..10")
    if args.samples < 5 or args.hz <= 0:
        parser.error("--samples must be >=5 and --hz must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from screwdriver_rl.deploy import hw_utils
    from screwdriver_rl.deploy import linker_sdk_map as sdkmap

    sdkmap.apply_calibration(str(args.calib))
    hw_utils.bootstrap_sdk(str(args.sdk_root))
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    sdk_g20 = (
        args.sdk_root
        / "linker_hand_sdk_ros/scripts/LinkerHand/core/can/"
        "linker_hand_g20_can.py"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "operation": "g20_index_three_axis_clearance_step",
        "approved_scope": {
            "joint": "index_clearance_pose",
            "raw20_slots": list(ACTIVE_INDEX_SLOTS),
            "start_raw": {
                "side": args.expected_start_side_raw,
                "pitch": args.expected_start_pitch_raw,
                "pip": args.expected_start_pip_raw,
            },
            "target_raw": {
                "side": args.target_side_raw,
                "pitch": args.target_pitch_raw,
                "pip": args.target_pip_raw,
            },
            "continue_beyond_this_target": False,
            "automatic_return": False,
        },
        "transport": {
            "side": "left",
            "hand_joint": "G20",
            "can": args.can,
            "sdk_root": str(args.sdk_root.resolve()),
        },
        "safety_contract": {
            "position_method": "api.hand.set_index_positions",
            "position_can_frame": "0x42",
            "frame_slots": list(INDEX_FRAME_SLOTS),
            "finger_move_called": False,
            "other_finger_position_frames_sent": False,
            "index_speed_only": args.speed,
            "index_torque6": APPROVED_INDEX_TORQUE6,
            "max_actual_step_per_axis_raw": 17,
            "clear_faults_called": False,
            "single_position_frame": True,
        },
        "hashes": {
            "calibration_sha256": _sha256(args.calib),
            "sdk_g20_sha256": _sha256(sdk_g20),
            "runner_sha256": _sha256(Path(__file__)),
        },
        "started_wall_time_s": time.time(),
        "motion_sent": False,
    }
    api: Any | None = None
    try:
        api = LinkerHandApi(hand_type="left", hand_joint="G20", can=args.can)
        serial = str(api.get_serial_number()).strip().strip("\x00")
        payload["identity"] = {
            "serial": serial,
            "embedded_version": [int(value) for value in api.get_embedded_version()],
            "touch_type": int(api.get_touch_type()),
            "sdk_version": "3.1.0",
        }
        if serial != args.expected_serial:
            raise RuntimeError(
                f"serial mismatch: expected {args.expected_serial}, got {serial}"
            )
        pre_rows = [_read_state20(api, hw_utils) for _ in range(5)]
        before20 = _median20(pre_rows)
        fault_groups, faults20 = _fault_snapshot(api)
        if any(faults20):
            raise RuntimeError(
                "nonzero preflight faults: "
                f"{[(i, value) for i, value in enumerate(faults20) if value]}"
            )
        temperature_before20 = _temperature_snapshot(api)
        index_before6 = _fresh_index_state6(api)
        after20, index_command6, raw20_diff = build_index_clearance_command(
            before20,
            index_before6,
            expected_start_side_raw=args.expected_start_side_raw,
            expected_start_pitch_raw=args.expected_start_pitch_raw,
            expected_start_pip_raw=args.expected_start_pip_raw,
            target_side_raw=args.target_side_raw,
            target_pitch_raw=args.target_pitch_raw,
            target_pip_raw=args.target_pip_raw,
            start_tolerance_raw=args.start_tolerance_raw,
        )
        payload["preflight"] = {
            "state20_samples": pre_rows,
            "state20_median": before20,
            "index_state6": index_before6,
            "faults_by_finger": fault_groups,
            "faults20": faults20,
            "temperature20": temperature_before20,
        }
        payload["command"] = {
            "raw20_before": before20,
            "raw20_after_contract": after20,
            "raw20_diff": raw20_diff,
            "index_frame6_before": index_before6,
            "index_frame6_sent": index_command6,
        }
        _write_json(args.out, payload)
        print(f"[g20-index-clearance] only diff: {raw20_diff}", flush=True)
        print(f"[g20-index-clearance] index 0x42: {index_command6}", flush=True)

        api.hand.set_index_speed([args.speed] * 6)
        api.hand.set_index_torque(APPROVED_INDEX_TORQUE6)
        time.sleep(0.05)
        payload["motion_command_wall_time_s"] = time.time()
        api.hand.set_index_positions(index_command6)
        payload["motion_sent"] = True
        _write_json(args.out, payload)

        records: list[dict[str, Any]] = []
        period = 1.0 / args.hz
        targets = (
            args.target_side_raw,
            args.target_pitch_raw,
            args.target_pip_raw,
        )
        for sample_index in range(args.samples):
            started = time.monotonic()
            state20 = _read_state20(api, hw_utils)
            _, sample_faults20 = _fault_snapshot(api)
            temperature20 = _temperature_snapshot(api)
            records.append({
                "sample": sample_index,
                "wall_time_s": time.time(),
                "monotonic_time_s": time.monotonic(),
                "state20": state20,
                "faults20": sample_faults20,
                "temperature20": temperature20,
            })
            print(
                f"[g20-index-clearance] sample={sample_index:02d} "
                f"side={state20[INDEX_SIDE_SLOT]} "
                f"pitch={state20[INDEX_PITCH_SLOT]} "
                f"pip={state20[INDEX_PIP_SLOT]} "
                f"faults={sum(value != 0 for value in sample_faults20)}",
                flush=True,
            )
            if any(sample_faults20):
                payload["automatic_stop_reason"] = "nonzero fault after command"
                break
            if any(
                temperature20[slot] - temperature_before20[slot] >= 8
                for slot in ACTIVE_INDEX_SLOTS
            ):
                payload["automatic_stop_reason"] = (
                    "index active-axis temperature increased by at least 8 degC"
                )
                break
            remaining = period - (time.monotonic() - started)
            if remaining > 0 and sample_index + 1 < args.samples:
                time.sleep(remaining)

        payload["post_command_samples"] = records
        post_median = _median20([row["state20"] for row in records])
        settled_window = records[-min(10, len(records)):]
        settled_median = _median20([row["state20"] for row in settled_window])
        errors = {
            str(slot): settled_median[slot] - target
            for slot, target in zip(ACTIVE_INDEX_SLOTS, targets)
        }
        fault_free = not any(
            value for row in records for value in row["faults20"]
        )
        settled = (
            all(abs(error) <= 2 for error in errors.values())
            and fault_free
            and "automatic_stop_reason" not in payload
        )
        payload["result"] = {
            "state20_median": post_median,
            "settled_state20_median": settled_median,
            "target_error_raw_by_slot": errors,
            "other_fingers_max_abs_change_raw": max(
                abs(post_median[slot] - before20[slot])
                for slot in ACTIVE_SLOTS
                if slot not in ACTIVE_INDEX_SLOTS
            ),
            "temperature_delta20": [
                after - before
                for before, after in zip(
                    temperature_before20, records[-1]["temperature20"]
                )
            ],
            "fault_free": fault_free,
            "settled": settled,
        }
        payload["ended_wall_time_s"] = time.time()
        _write_json(args.out, payload)
        print(f"[g20-index-clearance] wrote {args.out}", flush=True)
        return 0 if settled else 2
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["ended_wall_time_s"] = time.time()
        _write_json(args.out, payload)
        raise
    finally:
        if api is not None:
            close = getattr(api.hand, "close_can_interface", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
