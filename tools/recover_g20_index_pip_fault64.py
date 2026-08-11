#!/usr/bin/env python3
"""Clear the known G20 index fault and retreat its PIP to raw 12.

This is a deliberately one-purpose, fail-closed recovery tool. It accepts only
the expected left G20 hand and the observed fault state (slot 16 at raw 4 with
fault value 64). It sends one index-only fault-clear frame. Only after repeated
fault-free readback does it send one low-speed index frame that holds the root
at raw 250 and retreats the PIP to raw 12.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence


EXPECTED_SERIAL = "LHT20-010-415-L-B-1-D"
INDEX_ROOT_SLOT = 1
INDEX_SIDE_SLOT = 6
INDEX_PIP_SLOT = 16
INDEX_ACTIVE_SLOTS = (INDEX_ROOT_SLOT, INDEX_SIDE_SLOT, INDEX_PIP_SLOT)
RESERVED_SLOTS = (11, 12, 13, 14)
FAULT_VALUE = 64
EXPECTED_START_RAW = 4
TARGET_RAW = 12
ROOT_HOLD_RAW = 250
INDEX_CLEAR_MASK = [0, 1, 0, 0, 0]
INDEX_SPEED6 = [5, 5, 5, 5, 5, 5]
INDEX_TORQUE6 = [80, 40, 80, 40, 40, 40]
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_vector(
    values: Any,
    *,
    length: int,
    label: str,
    lo: int = 0,
    hi: int = 255,
) -> list[int]:
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not a numeric vector") from exc
    if len(result) != length:
        raise RuntimeError(f"{label} length is {len(result)}, expected {length}")
    if any(value < lo or value > hi for value in result):
        raise RuntimeError(f"{label} contains values outside {lo}..{hi}: {result}")
    return result


def _median20(rows: Sequence[Sequence[int]]) -> list[int]:
    if not rows:
        raise RuntimeError("no valid state rows")
    return [int(round(statistics.median(column))) for column in zip(*rows)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_state20(api: Any, hw_utils: Any) -> list[int]:
    state = hw_utils.validate_state20(api.get_state())
    if state is None:
        raise RuntimeError("invalid state20")
    return [int(round(value)) for value in state]


def _fault_snapshot(api: Any) -> tuple[dict[str, list[int]], list[int]]:
    for name in FINGER_NAMES:
        getattr(api.hand, f"get_{name}_fault")()
    time.sleep(0.05)
    groups = {
        name: _int_vector(
            getattr(api.hand, attr),
            length=6,
            label=f"{name}_fault6",
        )
        for name, attr in zip(FINGER_NAMES, ("x59", "x5A", "x5B", "x5C", "x5D"))
    }
    flat = _int_vector(
        api.hand.joint_state_to_cmd_state(list(groups.values())),
        length=20,
        label="faults20",
    )
    return groups, flat


def _temperature_snapshot(api: Any) -> list[int]:
    api.get_temperature()
    time.sleep(0.05)
    return _int_vector(api.get_temperature(), length=20, label="temperature20")


def _fresh_index_state6(api: Any) -> list[int]:
    api.hand.get_index_positions()
    time.sleep(0.05)
    return _int_vector(api.hand.x42, length=6, label="index_state6")


def build_recovery_command(
    state20: Sequence[int],
    index_state6: Sequence[int],
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Return the exact index frame and its raw20 contract."""

    before = _int_vector(state20, length=20, label="state20")
    index_before = _int_vector(index_state6, length=6, label="index_state6")
    if before[INDEX_PIP_SLOT] != EXPECTED_START_RAW:
        raise RuntimeError(
            f"slot {INDEX_PIP_SLOT} must be raw {EXPECTED_START_RAW}, "
            f"got {before[INDEX_PIP_SLOT]}"
        )
    expected_visible = (
        before[INDEX_SIDE_SLOT],
        before[INDEX_ROOT_SLOT],
        before[INDEX_PIP_SLOT],
    )
    actual_visible = (index_before[0], index_before[2], index_before[5])
    if actual_visible != expected_visible:
        raise RuntimeError(
            f"fresh index frame disagrees with raw20: "
            f"{actual_visible} != {expected_visible}"
        )
    if any(index_before[element] != 0 for element in (1, 3, 4)):
        raise RuntimeError(
            f"index reserved elements 1,3,4 are not zero: {index_before}"
        )

    after = list(before)
    after[INDEX_ROOT_SLOT] = ROOT_HOLD_RAW
    after[INDEX_PIP_SLOT] = TARGET_RAW
    index_command = list(index_before)
    index_command[2] = ROOT_HOLD_RAW
    index_command[5] = TARGET_RAW
    diff = [
        {"slot": slot, "before": old, "after": new}
        for slot, (old, new) in enumerate(zip(before, after))
        if old != new
    ]
    expected_diff = [
        {
            "slot": INDEX_ROOT_SLOT,
            "before": before[INDEX_ROOT_SLOT],
            "after": ROOT_HOLD_RAW,
        },
        {
            "slot": INDEX_PIP_SLOT,
            "before": EXPECTED_START_RAW,
            "after": TARGET_RAW,
        },
    ]
    if before[INDEX_ROOT_SLOT] == ROOT_HOLD_RAW:
        expected_diff.pop(0)
    if diff != expected_diff:
        raise AssertionError(f"unsafe raw20 diff: {diff}")
    return after, index_command, diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement for the approved clear-and-retreat action.",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required")
    if args.can != "can0":
        parser.error("--can must remain can0")
    if args.out.exists():
        parser.error(f"refusing to overwrite existing record: {args.out}")
    return args


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from screwdriver_rl.deploy import hw_utils

    hw_utils.bootstrap_sdk(str(args.sdk_root))
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    sdk_g20 = (
        args.sdk_root
        / "linker_hand_sdk_ros/scripts/LinkerHand/core/can/linker_hand_g20_can.py"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "operation": "clear_index_fault64_then_retreat_raw4_to_raw12",
        "approved_scope": {
            "side": "left",
            "hand_joint": "G20",
            "serial": EXPECTED_SERIAL,
            "can": "can0",
            "fault_slot": INDEX_PIP_SLOT,
            "expected_fault_value": FAULT_VALUE,
            "fault_clear_mask": INDEX_CLEAR_MASK,
            "start_raw": EXPECTED_START_RAW,
            "target_raw": TARGET_RAW,
            "root_hold_raw": ROOT_HOLD_RAW,
            "automatic_continuation": False,
            "automatic_return": False,
        },
        "safety_contract": {
            "public_api_clear_faults_called": False,
            "low_level_clear_method": "api.hand.clear_finger_faults",
            "low_level_clear_calls_allowed": 1,
            "low_level_clear_calls_made": 0,
            "position_method": "api.hand.set_index_positions",
            "position_frames_allowed": 1,
            "position_frames_sent": 0,
            "finger_move_called": False,
            "other_finger_position_frames_sent": False,
            "index_speed6": INDEX_SPEED6,
            "index_torque6": INDEX_TORQUE6,
            "temperature_limit_slots": INDEX_ACTIVE_SLOTS,
            "max_index_temperature_c": 60,
            "max_temperature_rise_c": 8,
        },
        "transport": {
            "sdk_root": str(args.sdk_root.resolve()),
            "calibration_path": str(args.calib.resolve()),
        },
        "hashes": {
            "calibration_sha256": _sha256(args.calib),
            "sdk_g20_sha256": _sha256(sdk_g20),
            "runner_sha256": _sha256(Path(__file__)),
        },
        "started_wall_time_s": time.time(),
        "fault_clear_sent": False,
        "motion_sent": False,
    }
    _write_json(args.out, payload)

    api: Any | None = None
    try:
        api = LinkerHandApi(hand_type="left", hand_joint="G20", can="can0")
        serial = str(api.get_serial_number()).strip().strip("\x00")
        payload["identity"] = {
            "serial": serial,
            "embedded_version": [int(value) for value in api.get_embedded_version()],
            "sdk_version": "3.1.0",
        }
        if serial != EXPECTED_SERIAL:
            raise RuntimeError(
                f"serial mismatch: expected {EXPECTED_SERIAL}, got {serial}"
            )

        state_rows = [_read_state20(api, hw_utils) for _ in range(5)]
        state_median = _median20(state_rows)
        if any(row[INDEX_PIP_SLOT] != EXPECTED_START_RAW for row in state_rows):
            raise RuntimeError(
                f"slot {INDEX_PIP_SLOT} is not stably raw {EXPECTED_START_RAW}: "
                f"{[row[INDEX_PIP_SLOT] for row in state_rows]}"
            )
        if not all(248 <= row[INDEX_ROOT_SLOT] <= 250 for row in state_rows):
            raise RuntimeError(
                f"root slot {INDEX_ROOT_SLOT} is outside 248..250: "
                f"{[row[INDEX_ROOT_SLOT] for row in state_rows]}"
            )
        if any(
            row[slot] != 0 for row in state_rows for slot in RESERVED_SLOTS
        ):
            raise RuntimeError("reserved raw20 slots 11..14 are not all zero")

        fault_rows = []
        for _ in range(5):
            groups, faults20 = _fault_snapshot(api)
            fault_rows.append({"by_finger": groups, "faults20": faults20})
        expected_faults = [0] * 20
        expected_faults[INDEX_PIP_SLOT] = FAULT_VALUE
        if any(row["faults20"] != expected_faults for row in fault_rows):
            raise RuntimeError(
                "preflight fault is not exclusively persistent slot16=64: "
                f"{[row['faults20'] for row in fault_rows]}"
            )

        temperature_before = _temperature_snapshot(api)
        payload["preflight"] = {
            "state20_samples": state_rows,
            "state20_median": state_median,
            "fault_samples": fault_rows,
            "temperature20": temperature_before,
        }
        _write_json(args.out, payload)
        if max(temperature_before[slot] for slot in INDEX_ACTIVE_SLOTS) >= 60:
            raise RuntimeError(
                "preflight index temperature is at or above 60 C: "
                f"{[(slot, temperature_before[slot]) for slot in INDEX_ACTIVE_SLOTS]}"
            )
        index_before6 = _fresh_index_state6(api)
        after20, index_command6, raw20_diff = build_recovery_command(
            state_median,
            index_before6,
        )
        payload["preflight"]["index_state6"] = index_before6
        payload["command"] = {
            "raw20_before": state_median,
            "raw20_after_contract": after20,
            "raw20_diff": raw20_diff,
            "index_frame6_before": index_before6,
            "index_frame6_sent": index_command6,
        }
        _write_json(args.out, payload)
        print(f"[g20-recovery] preflight state: {state_median}", flush=True)
        print(f"[g20-recovery] preflight fault: slot16={FAULT_VALUE}", flush=True)
        print(f"[g20-recovery] planned diff: {raw20_diff}", flush=True)
        print(f"[g20-recovery] index frame: {index_command6}", flush=True)

        payload["fault_clear_wall_time_s"] = time.time()
        payload["safety_contract"]["low_level_clear_calls_made"] = 1
        clear_response = api.hand.clear_finger_faults(INDEX_CLEAR_MASK)
        payload["fault_clear_sent"] = True
        payload["fault_clear_immediate_response"] = (
            list(clear_response)
            if isinstance(clear_response, (list, tuple))
            else clear_response
        )
        _write_json(args.out, payload)
        time.sleep(0.15)

        post_clear_rows = []
        for sample_index in range(5):
            groups, faults20 = _fault_snapshot(api)
            post_clear_rows.append(
                {
                    "sample": sample_index,
                    "wall_time_s": time.time(),
                    "by_finger": groups,
                    "faults20": faults20,
                }
            )
        payload["post_clear_fault_samples"] = post_clear_rows
        if any(
            value
            for row in post_clear_rows
            for value in row["faults20"]
        ):
            payload["automatic_stop_reason"] = (
                "fault clear did not produce five all-zero fault snapshots"
            )
            _write_json(args.out, payload)
            raise RuntimeError(payload["automatic_stop_reason"])

        state_after_clear = _read_state20(api, hw_utils)
        temperature_after_clear = _temperature_snapshot(api)
        payload["post_clear_state20"] = state_after_clear
        payload["post_clear_temperature20"] = temperature_after_clear
        if state_after_clear[INDEX_PIP_SLOT] != EXPECTED_START_RAW:
            raise RuntimeError(
                "slot16 changed before the position command: "
                f"{state_after_clear[INDEX_PIP_SLOT]}"
            )
        if (
            max(
                temperature_after_clear[slot]
                for slot in INDEX_ACTIVE_SLOTS
            )
            >= 60
        ):
            raise RuntimeError(
                "post-clear index temperature is at or above 60 C: "
                f"{[(slot, temperature_after_clear[slot]) for slot in INDEX_ACTIVE_SLOTS]}"
            )

        fresh_index6 = _fresh_index_state6(api)
        _, fresh_command6, _ = build_recovery_command(
            state_after_clear,
            fresh_index6,
        )
        if fresh_command6 != index_command6:
            raise RuntimeError(
                f"index frame changed after clear: "
                f"{fresh_command6} != {index_command6}"
            )
        payload["post_clear_index_state6"] = fresh_index6
        _write_json(args.out, payload)

        api.hand.set_index_speed(INDEX_SPEED6)
        api.hand.set_index_torque(INDEX_TORQUE6)
        time.sleep(0.05)
        payload["motion_command_wall_time_s"] = time.time()
        api.hand.set_index_positions(index_command6)
        payload["motion_sent"] = True
        payload["safety_contract"]["position_frames_sent"] = 1
        _write_json(args.out, payload)

        records = []
        period = 0.2
        for sample_index in range(10):
            started = time.monotonic()
            state20 = _read_state20(api, hw_utils)
            _, faults20 = _fault_snapshot(api)
            temperature20 = _temperature_snapshot(api)
            record = {
                "sample": sample_index,
                "wall_time_s": time.time(),
                "monotonic_time_s": time.monotonic(),
                "state20": state20,
                "faults20": faults20,
                "temperature20": temperature20,
            }
            records.append(record)
            print(
                f"[g20-recovery] sample={sample_index:02d} "
                f"slot16={state20[INDEX_PIP_SLOT]} "
                f"root={state20[INDEX_ROOT_SLOT]} "
                f"faults={sum(value != 0 for value in faults20)} "
                f"tip_temp={temperature20[INDEX_PIP_SLOT]}",
                flush=True,
            )
            if any(faults20):
                payload["automatic_stop_reason"] = "nonzero fault after motion"
                break
            if (
                max(temperature20[slot] for slot in INDEX_ACTIVE_SLOTS)
                >= 60
            ):
                payload["automatic_stop_reason"] = (
                    "index temperature reached or exceeded 60 C"
                )
                break
            if (
                temperature20[INDEX_PIP_SLOT]
                - temperature_before[INDEX_PIP_SLOT]
                >= 8
            ):
                payload["automatic_stop_reason"] = (
                    "index PIP temperature increased by at least 8 C"
                )
                break
            remaining = period - (time.monotonic() - started)
            if remaining > 0 and sample_index + 1 < 10:
                time.sleep(remaining)

        payload["post_motion_samples"] = records
        post_median = _median20([record["state20"] for record in records])
        payload["result"] = {
            "state20_median": post_median,
            "slot16_target_error_raw": post_median[INDEX_PIP_SLOT] - TARGET_RAW,
            "root_target_error_raw": post_median[INDEX_ROOT_SLOT] - ROOT_HOLD_RAW,
            "fault_free": not any(
                value
                for record in records
                for value in record["faults20"]
            ),
            "temperature_delta20": [
                after - before
                for before, after in zip(
                    temperature_before,
                    records[-1]["temperature20"],
                )
            ],
            "completed_all_samples": len(records) == 10,
        }
        payload["ended_wall_time_s"] = time.time()
        _write_json(args.out, payload)
        print(f"[g20-recovery] wrote {args.out}", flush=True)
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
