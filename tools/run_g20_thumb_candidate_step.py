#!/usr/bin/env python3
"""Run one adjacent, isolated G20 thumb MCP or CMC-pitch calibration step.

The G20 thumb 0x41 frame is ``[roll(s5), yaw(s10), pitch(s0), r11, r12,
mcp(s15)]``.  This tool fixes roll/yaw at the session's camera-view pose,
holds the other flexion axis at raw 255, and changes exactly one flexion axis.
It sends one thumb position frame and never continues or returns automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

from tools.run_g20_index_pip_candidate_step import (
    ACTIVE_SLOTS,
    EXPECTED_SERIAL,
    RESERVED_SLOTS,
    _fault_snapshot,
    _int_vector,
    _median20,
    _read_state20,
    _sha256,
    _temperature_snapshot,
    _write_json,
)


THUMB_PITCH_SLOT = 0
THUMB_ROLL_SLOT = 5
THUMB_YAW_SLOT = 10
THUMB_MCP_SLOT = 15
THUMB_FRAME_SLOTS = (5, 10, 0, 11, 12, 15)
APPROVED_THUMB_TORQUE6 = [40, 40, 60, 20, 20, 40]
RAW_CANDIDATES = (
    255, 240, 224, 208, 192, 176, 160, 144, 128, 112,
    96, 80, 64, 48, 32, 20, 12, 6, 0,
)
JOINT_TO_SLOT = {
    "thumb_mcp": THUMB_MCP_SLOT,
    "thumb_cmc_pitch": THUMB_PITCH_SLOT,
}


def _fresh_thumb_state6(api: Any) -> list[int]:
    api.hand.get_thumb_positions()
    time.sleep(0.04)
    return _int_vector(api.hand.x41, length=6, label="thumb_state6")


def build_thumb_command(
    state20: Sequence[int],
    thumb_state6: Sequence[int],
    *,
    joint: str,
    expected_start_raw: int,
    target_raw: int,
    roll_hold_raw: int,
    yaw_hold_raw: int,
    other_flex_hold_raw: int = 255,
    start_tolerance_raw: int = 2,
    hold_tolerance_raw: int = 2,
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Validate raw20/thumb-frame identity and build one isolated command."""

    if joint not in JOINT_TO_SLOT:
        raise ValueError(f"unsupported joint {joint!r}")
    if not 0 <= target_raw <= 255:
        raise ValueError("target raw must be in 0..255")
    before = _int_vector(state20, length=20, label="state20")
    thumb_before = _int_vector(
        thumb_state6, length=6, label="thumb_state6"
    )
    expected_frame = [before[slot] for slot in THUMB_FRAME_SLOTS]
    for index, (actual, expected) in enumerate(
        zip(thumb_before, expected_frame)
    ):
        tolerance = 0 if index in (3, 4) else hold_tolerance_raw
        if abs(actual - expected) > tolerance:
            raise ValueError(
                "fresh thumb frame disagrees with raw20: "
                f"index {index}, {actual} != {expected}"
            )

    target_slot = JOINT_TO_SLOT[joint]
    other_slot = (
        THUMB_PITCH_SLOT
        if target_slot == THUMB_MCP_SLOT
        else THUMB_MCP_SLOT
    )
    if abs(before[target_slot] - expected_start_raw) > start_tolerance_raw:
        raise ValueError(
            f"slot {target_slot} must start within {start_tolerance_raw} raw "
            f"of {expected_start_raw}, got {before[target_slot]}"
        )
    actual_start_raw = before[target_slot]
    step_raw = abs(target_raw - actual_start_raw)
    if not 1 <= step_raw <= 17:
        raise ValueError(
            f"actual {joint} step must be 1..17 raw, got "
            f"{actual_start_raw} to {target_raw} ({step_raw})"
        )
    holds = {
        THUMB_ROLL_SLOT: roll_hold_raw,
        THUMB_YAW_SLOT: yaw_hold_raw,
        other_slot: other_flex_hold_raw,
    }
    for slot, hold in holds.items():
        if abs(before[slot] - hold) > hold_tolerance_raw:
            raise ValueError(
                f"hold slot {slot} must be within {hold_tolerance_raw} raw "
                f"of {hold}, got {before[slot]}"
            )
    if any(before[slot] != 0 for slot in RESERVED_SLOTS):
        raise ValueError(
            "reserved slots 11..14 are not zero: "
            f"{[before[slot] for slot in RESERVED_SLOTS]}"
        )

    after = list(before)
    for slot, hold in holds.items():
        after[slot] = hold
    after[target_slot] = target_raw
    thumb_after = [after[slot] for slot in THUMB_FRAME_SLOTS]
    diff = [
        {"slot": slot, "before": old, "after": new}
        for slot, (old, new) in enumerate(zip(before, after))
        if old != new
    ]
    allowed = {target_slot, *holds}
    if any(item["slot"] not in allowed for item in diff):
        raise AssertionError(f"unsafe raw20 diff: {diff}")
    if thumb_after[3:5] != [0, 0]:
        raise AssertionError(f"unsafe thumb reserved values: {thumb_after}")
    return after, thumb_after, diff


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--expected-serial", default=EXPECTED_SERIAL)
    parser.add_argument(
        "--joint", choices=tuple(JOINT_TO_SLOT), required=True
    )
    parser.add_argument("--expected-start-raw", type=int, required=True)
    parser.add_argument(
        "--expected-start-tolerance-raw", type=int, default=2
    )
    parser.add_argument("--target-raw", type=int, required=True)
    parser.add_argument("--roll-hold-raw", type=int, required=True)
    parser.add_argument("--yaw-hold-raw", type=int, required=True)
    parser.add_argument("--other-flex-hold-raw", type=int, default=255)
    parser.add_argument("--hold-tolerance-raw", type=int, default=2)
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
    for value, label in (
        (args.expected_start_raw, "expected start"),
        (args.target_raw, "target"),
        (args.roll_hold_raw, "roll hold"),
        (args.yaw_hold_raw, "yaw hold"),
        (args.other_flex_hold_raw, "other flex hold"),
    ):
        if not 0 <= value <= 255:
            parser.error(f"{label} raw must be in 0..255")
    if not 1 <= abs(args.target_raw - args.expected_start_raw) <= 17:
        parser.error("candidate step must move 1..17 raw counts")
    if args.expected_start_tolerance_raw not in (0, 1, 2):
        parser.error("--expected-start-tolerance-raw must be 0, 1, or 2")
    if args.hold_tolerance_raw not in (0, 1, 2):
        parser.error("--hold-tolerance-raw must be 0, 1, or 2")
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

    target_slot = JOINT_TO_SLOT[args.joint]
    other_slot = (
        THUMB_PITCH_SLOT
        if target_slot == THUMB_MCP_SLOT
        else THUMB_MCP_SLOT
    )
    sdk_g20 = (
        args.sdk_root
        / "linker_hand_sdk_ros/scripts/LinkerHand/core/can/"
        "linker_hand_g20_can.py"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "operation": (
            f"g20_{args.joint}_candidate_step_raw"
            f"{args.expected_start_raw}_to_raw{args.target_raw}"
        ),
        "approved_scope": {
            "joint": args.joint,
            "raw20_slot": target_slot,
            "start_raw": args.expected_start_raw,
            "target_raw": args.target_raw,
            "roll_hold_raw": args.roll_hold_raw,
            "yaw_hold_raw": args.yaw_hold_raw,
            "other_flex_slot": other_slot,
            "other_flex_hold_raw": args.other_flex_hold_raw,
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
            "position_method": "api.hand.set_thumb_positions",
            "position_can_frame": "0x41",
            "frame_slots": list(THUMB_FRAME_SLOTS),
            "finger_move_called": False,
            "other_finger_position_frames_sent": False,
            "thumb_speed_only": args.speed,
            "thumb_torque6": APPROVED_THUMB_TORQUE6,
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
        api = LinkerHandApi(
            hand_type="left", hand_joint="G20", can=args.can
        )
        serial = str(api.get_serial_number()).strip().strip("\x00")
        payload["identity"] = {
            "serial": serial,
            "embedded_version": [
                int(value) for value in api.get_embedded_version()
            ],
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
        thumb_before6 = _fresh_thumb_state6(api)
        after20, thumb_command6, raw20_diff = build_thumb_command(
            before20,
            thumb_before6,
            joint=args.joint,
            expected_start_raw=args.expected_start_raw,
            target_raw=args.target_raw,
            roll_hold_raw=args.roll_hold_raw,
            yaw_hold_raw=args.yaw_hold_raw,
            other_flex_hold_raw=args.other_flex_hold_raw,
            start_tolerance_raw=args.expected_start_tolerance_raw,
            hold_tolerance_raw=args.hold_tolerance_raw,
        )
        payload["preflight"] = {
            "state20_samples": pre_rows,
            "state20_median": before20,
            "thumb_state6": thumb_before6,
            "faults_by_finger": fault_groups,
            "faults20": faults20,
            "temperature20": temperature_before20,
        }
        payload["command"] = {
            "raw20_before": before20,
            "raw20_after_contract": after20,
            "raw20_diff": raw20_diff,
            "thumb_frame6_before": thumb_before6,
            "thumb_frame6_sent": thumb_command6,
        }
        _write_json(args.out, payload)
        print(f"[g20-thumb-step] raw20 before: {before20}", flush=True)
        print(f"[g20-thumb-step] only diff:    {raw20_diff}", flush=True)
        print(f"[g20-thumb-step] thumb 0x41:  {thumb_command6}", flush=True)

        api.hand.set_thumb_speed([args.speed] * 6)
        api.hand.set_thumb_torque(APPROVED_THUMB_TORQUE6)
        time.sleep(0.05)
        payload["motion_command_wall_time_s"] = time.time()
        api.hand.set_thumb_positions(thumb_command6)
        payload["motion_sent"] = True
        _write_json(args.out, payload)

        records: list[dict[str, Any]] = []
        period = 1.0 / args.hz
        for sample_index in range(args.samples):
            started = time.monotonic()
            state20 = _read_state20(api, hw_utils)
            _, sample_faults20 = _fault_snapshot(api)
            temperature20 = _temperature_snapshot(api)
            record = {
                "sample": sample_index,
                "wall_time_s": time.time(),
                "monotonic_time_s": time.monotonic(),
                "state20": state20,
                "faults20": sample_faults20,
                "temperature20": temperature20,
            }
            records.append(record)
            print(
                f"[g20-thumb-step] sample={sample_index:02d} "
                f"target={state20[target_slot]} "
                f"roll={state20[THUMB_ROLL_SLOT]} "
                f"yaw={state20[THUMB_YAW_SLOT]} "
                f"other={state20[other_slot]} "
                f"faults={sum(value != 0 for value in sample_faults20)} "
                f"temp={temperature20[target_slot]}",
                flush=True,
            )
            stop_reason = None
            if any(sample_faults20):
                stop_reason = "nonzero fault after command"
            elif (
                temperature20[target_slot]
                - temperature_before20[target_slot]
                >= 8
            ):
                stop_reason = "target temperature increased by at least 8 degC"
            else:
                for slot, hold in (
                    (THUMB_ROLL_SLOT, args.roll_hold_raw),
                    (THUMB_YAW_SLOT, args.yaw_hold_raw),
                    (other_slot, args.other_flex_hold_raw),
                ):
                    if abs(state20[slot] - hold) > args.hold_tolerance_raw + 2:
                        stop_reason = f"hold slot {slot} left its reference"
                        break
            if stop_reason is not None:
                payload["automatic_stop_reason"] = stop_reason
                break
            remaining = period - (time.monotonic() - started)
            if remaining > 0 and sample_index + 1 < args.samples:
                time.sleep(remaining)

        payload["post_command_samples"] = records
        post_median = _median20(
            [record["state20"] for record in records]
        )
        post_span = [
            max(column) - min(column)
            for column in zip(
                *(record["state20"] for record in records)
            )
        ]
        final_temperature = records[-1]["temperature20"]
        settled_window = records[-min(10, len(records)):]
        settled_median = _median20(
            [record["state20"] for record in settled_window]
        )
        excluded = {
            target_slot,
            THUMB_ROLL_SLOT,
            THUMB_YAW_SLOT,
            other_slot,
        }
        target_error = settled_median[target_slot] - args.target_raw
        roll_error = settled_median[THUMB_ROLL_SLOT] - args.roll_hold_raw
        yaw_error = settled_median[THUMB_YAW_SLOT] - args.yaw_hold_raw
        other_flex_error = (
            settled_median[other_slot] - args.other_flex_hold_raw
        )
        fault_free = not any(
            value
            for record in records
            for value in record["faults20"]
        )
        settled = (
            abs(target_error) <= 2
            and abs(roll_error) <= 2
            and abs(yaw_error) <= 2
            and abs(other_flex_error) <= 2
            and fault_free
            and "automatic_stop_reason" not in payload
        )
        payload["result"] = {
            "state20_median": post_median,
            "state20_span": post_span,
            "settled_window_samples": len(settled_window),
            "settled_state20_median": settled_median,
            "target_error_raw": target_error,
            "roll_hold_error_raw": roll_error,
            "yaw_hold_error_raw": yaw_error,
            "other_flex_hold_error_raw": other_flex_error,
            "other_fingers_max_abs_change_raw": max(
                abs(post_median[slot] - before20[slot])
                for slot in ACTIVE_SLOTS
                if slot not in excluded
            ),
            "temperature_delta20": [
                after - before
                for before, after in zip(
                    temperature_before20, final_temperature
                )
            ],
            "fault_free": fault_free,
            "settled": settled,
        }
        payload["ended_wall_time_s"] = time.time()
        _write_json(args.out, payload)
        print(f"[g20-thumb-step] wrote {args.out}", flush=True)
        if not settled:
            print(
                "[g20-thumb-step] NOT SETTLED; refusing automatic continuation",
                flush=True,
            )
            return 2
        return 0
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
