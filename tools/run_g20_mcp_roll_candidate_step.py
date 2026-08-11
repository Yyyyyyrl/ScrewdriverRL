#!/usr/bin/env python3
"""Run one adjacent, isolated G20 long-finger MCP-roll calibration step.

Every vendor long-finger position frame is
``[side(roll), reserved, pitch, reserved, reserved, pip]`` and can only address
its own finger. This tool changes the side element of exactly one finger, holds
that finger's pitch and pip at the session isolation reference, and sends
exactly one frame.

Two things differ from the Phase A flexion runners:

* the target is the side axis, which has no raw255 home. ``--expected-start-raw``
  therefore has to come from a measured read-only snapshot, never a constant.
* the other three roll slots are the nearest non-target active axes. A
  single-finger frame cannot command them, so they are *verified* against
  ``--other-roll-hold`` and must not appear in the raw20 diff. They are never
  written.

It never moves another finger, continues automatically, returns automatically,
or calls ``finger_move``.

Deviation from runbook section 6.5, recorded deliberately: this is one
table-driven runner for all four long fingers rather than four near-identical
copies. The rule exists so that no finger inherits another finger's slot or
frame identity by changing a command-line flag. Here every finger has its own
explicit verified entry in ``ROLL_FINGERS``, the frame-to-raw20 agreement is
revalidated on each run, and the offline tests cover all four fingers, so the
identity risk the rule guards against is covered without the copy-paste drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, NamedTuple, Sequence

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


class RollFinger(NamedTuple):
    roll_slot: int
    pitch_slot: int
    pip_slot: int
    can_frame: str
    state_attr: str
    state_getter: str
    speed_setter: str
    torque_setter: str
    position_setter: str
    torque6: list[int]


# Slot and frame identity per finger, each verified against the Phase A runner
# for that finger. Frame order is [side, reserved, pitch, reserved, reserved,
# pip] for all four.
ROLL_FINGERS: dict[str, RollFinger] = {
    "index": RollFinger(6, 1, 16, "0x42", "x42", "get_index_positions",
                        "set_index_speed", "set_index_torque",
                        "set_index_positions", [80, 40, 80, 40, 40, 40]),
    "middle": RollFinger(7, 2, 17, "0x43", "x43", "get_middle_positions",
                         "set_middle_speed", "set_middle_torque",
                         "set_middle_positions", [40, 20, 60, 20, 20, 40]),
    "ring": RollFinger(8, 3, 18, "0x44", "x44", "get_ring_positions",
                       "set_ring_speed", "set_ring_torque",
                       "set_ring_positions", [40, 20, 60, 20, 20, 40]),
    "pinky": RollFinger(9, 4, 19, "0x45", "x45", "get_little_positions",
                        "set_little_speed", "set_little_torque",
                        "set_little_positions", [40, 20, 60, 20, 20, 40]),
}

FRAME_SIDE_INDEX = 0
FRAME_PITCH_INDEX = 2
FRAME_PIP_INDEX = 5
FRAME_RESERVED_INDICES = (1, 3, 4)
MAX_ADJACENT_STEP_RAW = 17


def _fresh_finger_state6(api: Any, finger: str) -> list[int]:
    spec = ROLL_FINGERS[finger]
    getattr(api.hand, spec.state_getter)()
    time.sleep(0.04)
    return _int_vector(
        getattr(api.hand, spec.state_attr), length=6, label=f"{finger}_state6"
    )


def build_roll_command(
    state20: Sequence[int],
    finger_state6: Sequence[int],
    *,
    finger: str,
    expected_start_raw: int,
    target_raw: int,
    pitch_hold_raw: int,
    pip_hold_raw: int,
    other_roll_holds: dict[int, int],
    start_tolerance_raw: int = 2,
    hold_tolerance_raw: int = 2,
    other_roll_tolerance_raw: int = 2,
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Validate raw20/frame identity and build one isolated roll command."""

    if finger not in ROLL_FINGERS:
        raise ValueError(f"unsupported finger {finger!r}")
    spec = ROLL_FINGERS[finger]
    if not 0 <= target_raw <= 255:
        raise ValueError("target raw must be in 0..255")

    before = _int_vector(state20, length=20, label="state20")
    frame_before = _int_vector(
        finger_state6, length=6, label=f"{finger}_state6"
    )

    for index in FRAME_RESERVED_INDICES:
        if frame_before[index] != 0:
            raise ValueError(
                f"fresh {finger} frame reserved byte {index} must be zero, "
                f"got {frame_before[index]}"
            )
    for index, slot, tolerance in (
        (FRAME_SIDE_INDEX, spec.roll_slot, start_tolerance_raw),
        (FRAME_PITCH_INDEX, spec.pitch_slot, hold_tolerance_raw),
        (FRAME_PIP_INDEX, spec.pip_slot, hold_tolerance_raw),
    ):
        if abs(frame_before[index] - before[slot]) > tolerance:
            raise ValueError(
                f"fresh {finger} frame disagrees with raw20 at element "
                f"{index}/slot {slot}: {frame_before[index]} != {before[slot]}"
            )

    if abs(before[spec.roll_slot] - expected_start_raw) > start_tolerance_raw:
        raise ValueError(
            f"slot {spec.roll_slot} must start within {start_tolerance_raw} "
            f"raw of {expected_start_raw}, got {before[spec.roll_slot]}"
        )
    actual_start_raw = before[spec.roll_slot]
    step_raw = abs(target_raw - actual_start_raw)
    if not 1 <= step_raw <= MAX_ADJACENT_STEP_RAW:
        raise ValueError(
            f"actual {finger} roll step must be 1..{MAX_ADJACENT_STEP_RAW} "
            f"raw, got {actual_start_raw} to {target_raw} ({step_raw})"
        )

    for slot, hold in ((spec.pitch_slot, pitch_hold_raw),
                       (spec.pip_slot, pip_hold_raw)):
        if abs(before[slot] - hold) > hold_tolerance_raw:
            raise ValueError(
                f"hold slot {slot} must be within {hold_tolerance_raw} raw "
                f"of {hold}, got {before[slot]}"
            )

    # The other three roll axes belong to other fingers' frames. They are
    # checked here and must stay untouched; this runner cannot and must not
    # write them.
    expected_other = {
        other.roll_slot
        for name, other in ROLL_FINGERS.items()
        if name != finger
    }
    if set(other_roll_holds) != expected_other:
        raise ValueError(
            f"--other-roll-hold must cover exactly slots "
            f"{sorted(expected_other)}, got {sorted(other_roll_holds)}"
        )
    for slot, hold in sorted(other_roll_holds.items()):
        if abs(before[slot] - hold) > other_roll_tolerance_raw:
            raise ValueError(
                f"non-target roll slot {slot} must be within "
                f"{other_roll_tolerance_raw} raw of {hold}, "
                f"got {before[slot]}"
            )

    if any(before[slot] != 0 for slot in RESERVED_SLOTS):
        raise ValueError(
            "reserved slots 11..14 are not zero: "
            f"{[before[slot] for slot in RESERVED_SLOTS]}"
        )

    after = list(before)
    after[spec.roll_slot] = target_raw
    after[spec.pitch_slot] = pitch_hold_raw
    after[spec.pip_slot] = pip_hold_raw
    frame_after = list(frame_before)
    frame_after[FRAME_SIDE_INDEX] = target_raw
    frame_after[FRAME_PITCH_INDEX] = pitch_hold_raw
    frame_after[FRAME_PIP_INDEX] = pip_hold_raw

    diff = [
        {"slot": slot, "before": old, "after": new}
        for slot, (old, new) in enumerate(zip(before, after))
        if old != new
    ]
    allowed = {spec.roll_slot, spec.pitch_slot, spec.pip_slot}
    if any(item["slot"] not in allowed for item in diff):
        raise AssertionError(f"unsafe raw20 diff: {diff}")
    if any(item["slot"] in expected_other for item in diff):
        raise AssertionError(f"roll step would move another finger: {diff}")
    if any(frame_after[index] != 0 for index in FRAME_RESERVED_INDICES):
        raise AssertionError(f"unsafe {finger} reserved values: {frame_after}")
    return after, frame_after, diff


def _parse_other_roll(text: str) -> tuple[int, int]:
    try:
        slot_text, raw_text = text.split("=", 1)
        return int(slot_text), int(raw_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--other-roll-hold must be SLOT=RAW"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--expected-serial", default=EXPECTED_SERIAL)
    parser.add_argument("--finger", choices=tuple(ROLL_FINGERS), required=True)
    parser.add_argument("--expected-start-raw", type=int, required=True)
    parser.add_argument("--expected-start-tolerance-raw", type=int, default=2)
    parser.add_argument("--target-raw", type=int, required=True)
    parser.add_argument("--pitch-hold-raw", type=int, required=True)
    parser.add_argument("--pip-hold-raw", type=int, required=True)
    parser.add_argument(
        "--other-roll-hold", type=_parse_other_roll, action="append",
        required=True, metavar="SLOT=RAW",
        help="verified-only hold for each non-target roll slot; repeat 3 times",
    )
    parser.add_argument(
        "--settle-tolerance-raw", type=int, default=2,
        help=("raw the target may settle short of its command and still count "
              "as settled. The Phase A flexion default is 2; the roll axes "
              "were measured to undershoot 0..3 raw with the physical angle "
              "still tracking normally, so a roll sweep may pass 3. Raising "
              "this weakens an already weak proxy for finger-to-finger "
              "contact: with the non-target fingers pitched away their tapes "
              "are not measured, and neither their readbacks nor the local "
              "slope changed during the one contact observed on 2026-08-03. "
              "Fault, temperature and hold-slot gates are unaffected."))
    parser.add_argument("--hold-tolerance-raw", type=int, default=2)
    parser.add_argument("--other-roll-tolerance-raw", type=int, default=2)
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

    spec = ROLL_FINGERS[args.finger]
    other_roll_holds = dict(args.other_roll_hold)
    sdk_g20 = (
        args.sdk_root
        / "linker_hand_sdk_ros/scripts/LinkerHand/core/can/"
        "linker_hand_g20_can.py"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "operation": (
            f"g20_{args.finger}_mcp_roll_candidate_step_raw"
            f"{args.expected_start_raw}_to_raw{args.target_raw}"
        ),
        "approved_scope": {
            "joint": f"{args.finger}_mcp_roll",
            "raw20_slot": spec.roll_slot,
            "start_raw": args.expected_start_raw,
            "target_raw": args.target_raw,
            "pitch_hold_slot": spec.pitch_slot,
            "pitch_hold_raw": args.pitch_hold_raw,
            "pip_hold_slot": spec.pip_slot,
            "pip_hold_raw": args.pip_hold_raw,
            "other_roll_holds_verified_not_commanded": other_roll_holds,
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
            "position_method": f"api.hand.{spec.position_setter}",
            "position_can_frame": spec.can_frame,
            "frame_order": "[side, reserved, pitch, reserved, reserved, pip]",
            "frame_slots": [spec.roll_slot, None, spec.pitch_slot,
                            None, None, spec.pip_slot],
            "finger_move_called": False,
            "other_finger_position_frames_sent": False,
            "other_roll_slots_written": False,
            "speed_only": args.speed,
            "torque6": spec.torque6,
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
        frame_before6 = _fresh_finger_state6(api, args.finger)
        after20, frame_command6, raw20_diff = build_roll_command(
            before20,
            frame_before6,
            finger=args.finger,
            expected_start_raw=args.expected_start_raw,
            target_raw=args.target_raw,
            pitch_hold_raw=args.pitch_hold_raw,
            pip_hold_raw=args.pip_hold_raw,
            other_roll_holds=other_roll_holds,
            start_tolerance_raw=args.expected_start_tolerance_raw,
            hold_tolerance_raw=args.hold_tolerance_raw,
            other_roll_tolerance_raw=args.other_roll_tolerance_raw,
        )
        payload["preflight"] = {
            "state20_samples": pre_rows,
            "state20_median": before20,
            f"{args.finger}_state6": frame_before6,
            "faults_by_finger": fault_groups,
            "faults20": faults20,
            "temperature20": temperature_before20,
        }
        payload["command"] = {
            "raw20_before": before20,
            "raw20_after_contract": after20,
            "raw20_diff": raw20_diff,
            "frame6_before": frame_before6,
            "frame6_sent": frame_command6,
        }
        _write_json(args.out, payload)
        tag = f"g20-{args.finger}-roll"
        print(f"[{tag}] raw20 before: {before20}", flush=True)
        print(f"[{tag}] only diff:    {raw20_diff}", flush=True)
        print(f"[{tag}] {spec.can_frame}: {frame_command6}", flush=True)
        if not args.execute:
            payload["refused_reason"] = "--execute not supplied"
            payload["ended_wall_time_s"] = time.time()
            _write_json(args.out, payload)
            print(f"[{tag}] preflight only; no frame sent", flush=True)
            return 0

        getattr(api.hand, spec.speed_setter)([args.speed] * 6)
        getattr(api.hand, spec.torque_setter)(spec.torque6)
        time.sleep(0.05)
        payload["motion_command_wall_time_s"] = time.time()
        getattr(api.hand, spec.position_setter)(frame_command6)
        payload["motion_sent"] = True
        _write_json(args.out, payload)

        records: list[dict[str, Any]] = []
        period = 1.0 / args.hz
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
                f"[{tag}] sample={sample_index:02d} "
                f"roll={state20[spec.roll_slot]} "
                f"pitch={state20[spec.pitch_slot]} "
                f"pip={state20[spec.pip_slot]} "
                f"others={[state20[slot] for slot in sorted(other_roll_holds)]} "
                f"faults={sum(value != 0 for value in sample_faults20)} "
                f"temp={temperature20[spec.roll_slot]}",
                flush=True,
            )
            stop_reason = None
            if any(sample_faults20):
                stop_reason = "nonzero fault after command"
            elif (temperature20[spec.roll_slot]
                  - temperature_before20[spec.roll_slot] >= 8):
                stop_reason = "target temperature increased by at least 8 degC"
            else:
                for slot, hold in ((spec.pitch_slot, args.pitch_hold_raw),
                                   (spec.pip_slot, args.pip_hold_raw)):
                    if abs(state20[slot] - hold) > args.hold_tolerance_raw + 2:
                        stop_reason = f"hold slot {slot} left its reference"
                        break
                for slot, hold in sorted(other_roll_holds.items()):
                    if (abs(state20[slot] - hold)
                            > args.other_roll_tolerance_raw + 2):
                        stop_reason = (
                            f"non-target roll slot {slot} moved during the step"
                        )
                        break
            if stop_reason is not None:
                payload["automatic_stop_reason"] = stop_reason
                break
            remaining = period - (time.monotonic() - started)
            if remaining > 0 and sample_index + 1 < args.samples:
                time.sleep(remaining)

        payload["post_command_samples"] = records
        post_median = _median20([record["state20"] for record in records])
        post_span = [
            max(column) - min(column)
            for column in zip(*(record["state20"] for record in records))
        ]
        settled_window = records[-min(10, len(records)):]
        settled_median = _median20(
            [record["state20"] for record in settled_window]
        )
        excluded = {spec.roll_slot, spec.pitch_slot, spec.pip_slot,
                    *other_roll_holds}
        target_error = settled_median[spec.roll_slot] - args.target_raw
        fault_free = not any(
            value for record in records for value in record["faults20"]
        )
        settled = (
            abs(target_error) <= args.settle_tolerance_raw
            and abs(settled_median[spec.pitch_slot] - args.pitch_hold_raw) <= 2
            and abs(settled_median[spec.pip_slot] - args.pip_hold_raw) <= 2
            and all(abs(settled_median[slot] - hold) <= 2
                    for slot, hold in other_roll_holds.items())
            and fault_free
            and "automatic_stop_reason" not in payload
        )
        payload["result"] = {
            "state20_median": post_median,
            "state20_span": post_span,
            "settled_window_samples": len(settled_window),
            "settled_state20_median": settled_median,
            "target_error_raw": target_error,
            "pitch_hold_error_raw": (
                settled_median[spec.pitch_slot] - args.pitch_hold_raw),
            "pip_hold_error_raw": (
                settled_median[spec.pip_slot] - args.pip_hold_raw),
            "other_roll_error_raw": {
                str(slot): settled_median[slot] - hold
                for slot, hold in sorted(other_roll_holds.items())
            },
            "other_slots_max_abs_change_raw": max(
                abs(post_median[slot] - before20[slot])
                for slot in ACTIVE_SLOTS
                if slot not in excluded
            ),
            "temperature_delta20": [
                after - before
                for before, after in zip(
                    temperature_before20, records[-1]["temperature20"])
            ],
            "fault_free": fault_free,
            "settled": settled,
        }
        payload["ended_wall_time_s"] = time.time()
        _write_json(args.out, payload)
        print(f"[{tag}] wrote {args.out}", flush=True)
        if not settled:
            print(f"[{tag}] NOT SETTLED; refusing automatic continuation",
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
