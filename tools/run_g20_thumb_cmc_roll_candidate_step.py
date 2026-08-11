#!/usr/bin/env python3
"""Run one adjacent, isolated G20 thumb CMC-roll calibration step.

The vendor G20 thumb 0x41 frame is
``[roll(s5), yaw(s10), pitch(s0), r11, r12, mcp(s15)]``. Roll is element 0.

Unlike the Phase B long-finger roll runner, every non-target thumb axis lives in
this same frame, so yaw, pitch and MCP are actively commanded to their
isolation holds rather than only verified. Isolation here is therefore stronger
than it was for the four-finger rolls.

Roll has no raw255 home: like roll it is bidirectional with a mid-range
reference, so ``--expected-start-raw`` must come from a measured read-only
snapshot and never from a constant.

It never moves another finger, continues automatically, returns automatically,
or calls ``finger_move``.
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
# Element index -> raw20 slot, verified against run_g20_thumb_candidate_step.
THUMB_FRAME_SLOTS = (5, 10, 0, 11, 12, 15)
FRAME_ROLL_INDEX = 0
FRAME_RESERVED_INDICES = (3, 4)
APPROVED_THUMB_TORQUE6 = [40, 40, 60, 20, 20, 40]
MAX_ADJACENT_STEP_RAW = 14


def _fresh_thumb_state6(api: Any) -> list[int]:
    api.hand.get_thumb_positions()
    time.sleep(0.04)
    return _int_vector(api.hand.x41, length=6, label="thumb_state6")


def build_thumb_roll_command(
    state20: Sequence[int],
    thumb_state6: Sequence[int],
    *,
    expected_start_raw: int,
    target_raw: int,
    yaw_hold_raw: int,
    pitch_hold_raw: int,
    mcp_hold_raw: int,
    start_tolerance_raw: int = 2,
    hold_tolerance_raw: int = 2,
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Validate raw20/0x41 identity and build one isolated roll command."""

    if not 0 <= target_raw <= 255:
        raise ValueError("target raw must be in 0..255")
    before = _int_vector(state20, length=20, label="state20")
    frame_before = _int_vector(thumb_state6, length=6, label="thumb_state6")

    if any(before[slot] != 0 for slot in RESERVED_SLOTS):
        raise ValueError(
            "reserved slots 11..14 are not zero: "
            f"{[before[slot] for slot in RESERVED_SLOTS]}")

    for index in FRAME_RESERVED_INDICES:
        slot = THUMB_FRAME_SLOTS[index]
        if frame_before[index] != before[slot]:
            raise ValueError(
                f"fresh thumb frame element {index} must equal reserved slot "
                f"{slot} ({before[slot]}), got {frame_before[index]}")
        if frame_before[index] != 0:
            raise ValueError(
                f"fresh thumb frame reserved element {index} must be zero, "
                f"got {frame_before[index]}")

    holds = {
        THUMB_YAW_SLOT: yaw_hold_raw,
        THUMB_PITCH_SLOT: pitch_hold_raw,
        THUMB_MCP_SLOT: mcp_hold_raw,
    }
    for index, slot, tolerance in (
        (1, THUMB_YAW_SLOT, hold_tolerance_raw),
        (FRAME_ROLL_INDEX, THUMB_ROLL_SLOT, start_tolerance_raw),
        (2, THUMB_PITCH_SLOT, hold_tolerance_raw),
        (5, THUMB_MCP_SLOT, hold_tolerance_raw),
    ):
        if abs(frame_before[index] - before[slot]) > tolerance:
            raise ValueError(
                f"fresh thumb frame disagrees with raw20 at element {index}/"
                f"slot {slot}: {frame_before[index]} != {before[slot]}")

    if abs(before[THUMB_ROLL_SLOT] - expected_start_raw) > start_tolerance_raw:
        raise ValueError(
            f"slot {THUMB_ROLL_SLOT} must start within {start_tolerance_raw} raw "
            f"of {expected_start_raw}, got {before[THUMB_ROLL_SLOT]}")
    actual_start_raw = before[THUMB_ROLL_SLOT]
    step_raw = abs(target_raw - actual_start_raw)
    if not 1 <= step_raw <= MAX_ADJACENT_STEP_RAW:
        raise ValueError(
            f"actual thumb roll step must be 1..{MAX_ADJACENT_STEP_RAW} raw, got "
            f"{actual_start_raw} to {target_raw} ({step_raw})")

    for slot, hold in holds.items():
        if abs(before[slot] - hold) > hold_tolerance_raw:
            raise ValueError(
                f"hold slot {slot} must be within {hold_tolerance_raw} raw of "
                f"{hold}, got {before[slot]}")

    after = list(before)
    after[THUMB_ROLL_SLOT] = target_raw
    for slot, hold in holds.items():
        after[slot] = hold
    frame_after = list(frame_before)
    frame_after[FRAME_ROLL_INDEX] = target_raw
    frame_after[1] = yaw_hold_raw
    frame_after[2] = pitch_hold_raw
    frame_after[5] = mcp_hold_raw

    diff = [{"slot": slot, "before": old, "after": new}
            for slot, (old, new) in enumerate(zip(before, after)) if old != new]
    allowed = {THUMB_ROLL_SLOT, *holds}
    if any(item["slot"] not in allowed for item in diff):
        raise AssertionError(f"unsafe raw20 diff: {diff}")
    if any(frame_after[i] != 0 for i in FRAME_RESERVED_INDICES):
        raise AssertionError(f"unsafe thumb reserved values: {frame_after}")
    return after, frame_after, diff


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--expected-serial", default=EXPECTED_SERIAL)
    parser.add_argument("--expected-start-raw", type=int, required=True)
    parser.add_argument("--expected-start-tolerance-raw", type=int, default=2)
    parser.add_argument("--target-raw", type=int, required=True)
    parser.add_argument("--yaw-hold-raw", type=int, required=True)
    parser.add_argument("--pitch-hold-raw", type=int, required=True)
    parser.add_argument("--mcp-hold-raw", type=int, required=True)
    parser.add_argument("--hold-tolerance-raw", type=int, default=2)
    parser.add_argument("--settle-tolerance-raw", type=int, default=2)
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
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

    sdk_g20 = (args.sdk_root / "linker_hand_sdk_ros/scripts/LinkerHand/core/can/"
               "linker_hand_g20_can.py")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "operation": (f"g20_thumb_cmc_roll_candidate_step_raw"
                      f"{args.expected_start_raw}_to_raw{args.target_raw}"),
        "approved_scope": {
            "joint": "thumb_cmc_roll",
            "raw20_slot": THUMB_ROLL_SLOT,
            "start_raw": args.expected_start_raw,
            "target_raw": args.target_raw,
            "yaw_hold_raw": args.yaw_hold_raw,
            "pitch_hold_raw": args.pitch_hold_raw,
            "mcp_hold_raw": args.mcp_hold_raw,
            "continue_beyond_this_target": False,
            "automatic_return": False,
        },
        "transport": {"side": "left", "hand_joint": "G20", "can": args.can,
                      "sdk_root": str(args.sdk_root.resolve())},
        "safety_contract": {
            "position_method": "api.hand.set_thumb_positions",
            "position_can_frame": "0x41",
            "frame_order": "[roll, yaw, pitch, reserved, reserved, mcp]",
            "frame_slots": list(THUMB_FRAME_SLOTS),
            "non_target_thumb_axes_are_commanded_not_only_verified": True,
            "finger_move_called": False,
            "other_finger_position_frames_sent": False,
            "thumb_speed_only": args.speed,
            "thumb_torque6": APPROVED_THUMB_TORQUE6,
            "clear_faults_called": False,
            "single_position_frame": True,
        },
        "hashes": {"calibration_sha256": _sha256(args.calib),
                   "sdk_g20_sha256": _sha256(sdk_g20),
                   "runner_sha256": _sha256(Path(__file__))},
        "started_wall_time_s": time.time(),
        "motion_sent": False,
    }

    api: Any | None = None
    try:
        api = LinkerHandApi(hand_type="left", hand_joint="G20", can=args.can)
        serial = str(api.get_serial_number()).strip().strip("\x00")
        payload["identity"] = {
            "serial": serial,
            "embedded_version": [int(v) for v in api.get_embedded_version()],
            "touch_type": int(api.get_touch_type()),
            "sdk_version": "3.1.0",
        }
        if serial != args.expected_serial:
            raise RuntimeError(
                f"serial mismatch: expected {args.expected_serial}, got {serial}")

        pre_rows = [_read_state20(api, hw_utils) for _ in range(5)]
        before20 = _median20(pre_rows)
        fault_groups, faults20 = _fault_snapshot(api)
        if any(faults20):
            raise RuntimeError(
                "nonzero preflight faults: "
                f"{[(i, v) for i, v in enumerate(faults20) if v]}")
        temperature_before20 = _temperature_snapshot(api)
        frame_before6 = _fresh_thumb_state6(api)
        after20, frame_command6, raw20_diff = build_thumb_roll_command(
            before20, frame_before6,
            expected_start_raw=args.expected_start_raw,
            target_raw=args.target_raw,
            yaw_hold_raw=args.yaw_hold_raw,
            pitch_hold_raw=args.pitch_hold_raw,
            mcp_hold_raw=args.mcp_hold_raw,
            start_tolerance_raw=args.expected_start_tolerance_raw,
            hold_tolerance_raw=args.hold_tolerance_raw)
        payload["preflight"] = {
            "state20_samples": pre_rows, "state20_median": before20,
            "thumb_state6": frame_before6, "faults_by_finger": fault_groups,
            "faults20": faults20, "temperature20": temperature_before20,
        }
        payload["command"] = {
            "raw20_before": before20, "raw20_after_contract": after20,
            "raw20_diff": raw20_diff, "frame6_before": frame_before6,
            "frame6_sent": frame_command6,
        }
        _write_json(args.out, payload)
        print(f"[g20-thumb-roll] raw20 before: {before20}", flush=True)
        print(f"[g20-thumb-roll] only diff:    {raw20_diff}", flush=True)
        print(f"[g20-thumb-roll] 0x41:         {frame_command6}", flush=True)
        if not args.execute:
            payload["refused_reason"] = "--execute not supplied"
            payload["ended_wall_time_s"] = time.time()
            _write_json(args.out, payload)
            print("[g20-thumb-roll] preflight only; no frame sent", flush=True)
            return 0

        api.hand.set_thumb_speed([args.speed] * 6)
        api.hand.set_thumb_torque(APPROVED_THUMB_TORQUE6)
        time.sleep(0.05)
        payload["motion_command_wall_time_s"] = time.time()
        api.hand.set_thumb_positions(frame_command6)
        payload["motion_sent"] = True
        _write_json(args.out, payload)

        records: list[dict[str, Any]] = []
        period = 1.0 / args.hz
        holds = {THUMB_YAW_SLOT: args.yaw_hold_raw,
                 THUMB_PITCH_SLOT: args.pitch_hold_raw,
                 THUMB_MCP_SLOT: args.mcp_hold_raw}
        for sample_index in range(args.samples):
            started = time.monotonic()
            state20 = _read_state20(api, hw_utils)
            _, sample_faults20 = _fault_snapshot(api)
            temperature20 = _temperature_snapshot(api)
            records.append({
                "sample": sample_index, "wall_time_s": time.time(),
                "monotonic_time_s": time.monotonic(), "state20": state20,
                "faults20": sample_faults20, "temperature20": temperature20})
            print(f"[g20-thumb-roll] sample={sample_index:02d} "
                  f"roll={state20[THUMB_ROLL_SLOT]} "
                  f"yaw={state20[THUMB_YAW_SLOT]} "
                  f"pitch={state20[THUMB_PITCH_SLOT]} "
                  f"mcp={state20[THUMB_MCP_SLOT]} "
                  f"faults={sum(v != 0 for v in sample_faults20)} "
                  f"temp={temperature20[THUMB_ROLL_SLOT]}", flush=True)
            stop_reason = None
            if any(sample_faults20):
                stop_reason = "nonzero fault after command"
            elif (temperature20[THUMB_ROLL_SLOT]
                  - temperature_before20[THUMB_ROLL_SLOT] >= 8):
                stop_reason = "target temperature increased by at least 8 degC"
            else:
                for slot, hold in holds.items():
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
        post_median = _median20([r["state20"] for r in records])
        settled_window = records[-min(10, len(records)):]
        settled_median = _median20([r["state20"] for r in settled_window])
        target_error = settled_median[THUMB_ROLL_SLOT] - args.target_raw
        fault_free = not any(v for r in records for v in r["faults20"])
        excluded = {THUMB_ROLL_SLOT, *holds}
        other_slots_max_abs_change_raw = max(
            abs(post_median[s] - before20[s])
            for s in ACTIVE_SLOTS if s not in excluded)
        settled = (abs(target_error) <= args.settle_tolerance_raw
                   and all(abs(settled_median[s] - h) <= 2 for s, h in holds.items())
                   and other_slots_max_abs_change_raw <= 2
                   and fault_free and "automatic_stop_reason" not in payload)
        payload["result"] = {
            "state20_median": post_median,
            "state20_span": [max(c) - min(c) for c in
                             zip(*(r["state20"] for r in records))],
            "settled_window_samples": len(settled_window),
            "settled_state20_median": settled_median,
            "target_error_raw": target_error,
            "hold_error_raw": {str(s): settled_median[s] - h
                               for s, h in sorted(holds.items())},
            "other_slots_max_abs_change_raw": max(
                abs(post_median[s] - before20[s])
                for s in ACTIVE_SLOTS if s not in excluded),
            "temperature_delta20": [a - b for b, a in zip(
                temperature_before20, records[-1]["temperature20"])],
            "fault_free": fault_free,
            "settled": settled,
        }
        payload["ended_wall_time_s"] = time.time()
        _write_json(args.out, payload)
        print(f"[g20-thumb-roll] wrote {args.out}", flush=True)
        if not settled:
            print("[g20-thumb-roll] NOT SETTLED; refusing automatic continuation",
                  flush=True)
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
