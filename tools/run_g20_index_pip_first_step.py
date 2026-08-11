#!/usr/bin/env python3
"""Run the single approved G20 index-PIP calibration step: raw 255 -> 240.

This tool is intentionally narrow and fail-closed:

* it accepts only the left G20 hand with the expected serial number;
* it requires the index PIP state (slot 16) to start at raw 255;
* it requires every fault and reserved slot 11--14 to be zero;
* it changes only index-finger element 5 in the vendor's six-element 0x42
  position frame, preserving all other elements from a fresh hardware read;
* it configures speed/torque only for the index finger;
* it never calls ``finger_move`` or commands another finger;
* it does not continue to raw 224 and does not automatically return to 255.

The JSON output contains the full raw20 before/after contract, the exact
six-element index frame, temperatures, faults, and post-command state samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence


EXPECTED_SERIAL = "LHT20-010-415-L-B-1-D"
EXPECTED_START_RAW = 255
ONLY_ALLOWED_TARGET_RAW = 240
INDEX_PIP_SLOT = 16
RESERVED_SLOTS = (11, 12, 13, 14)
ACTIVE_SLOTS = tuple(range(0, 11)) + tuple(range(15, 20))
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


def build_index_pip_command(
    state20: Sequence[int],
    index_state6: Sequence[int],
    target_raw: int = ONLY_ALLOWED_TARGET_RAW,
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Build and validate the full raw20 contract and isolated index frame."""

    before = _int_vector(state20, length=20, label="state20")
    index_before = _int_vector(index_state6, length=6, label="index_state6")
    if target_raw != ONLY_ALLOWED_TARGET_RAW:
        raise ValueError(
            f"this first-step tool permits only target raw {ONLY_ALLOWED_TARGET_RAW}"
        )
    if before[INDEX_PIP_SLOT] != EXPECTED_START_RAW:
        raise ValueError(
            f"slot {INDEX_PIP_SLOT} must start at raw {EXPECTED_START_RAW}, "
            f"got {before[INDEX_PIP_SLOT]}"
        )
    # Vendor G20 index order is [side, reserved, root, reserved, reserved, tip].
    expected_visible = (before[6], before[1], before[16])
    actual_visible = (index_before[0], index_before[2], index_before[5])
    if actual_visible != expected_visible:
        raise ValueError(
            "fresh index frame disagrees with raw20 at side/root/tip: "
            f"{actual_visible} != {expected_visible}"
        )

    after = list(before)
    after[INDEX_PIP_SLOT] = target_raw
    index_after = list(index_before)
    index_after[5] = target_raw
    diff = [
        {"slot": slot, "before": old, "after": new}
        for slot, (old, new) in enumerate(zip(before, after))
        if old != new
    ]
    if diff != [
        {
            "slot": INDEX_PIP_SLOT,
            "before": EXPECTED_START_RAW,
            "after": ONLY_ALLOWED_TARGET_RAW,
        }
    ]:
        raise AssertionError(f"unsafe raw20 diff: {diff}")
    return after, index_after, diff


def _fault_snapshot(api: Any) -> tuple[dict[str, list[int]], list[int]]:
    for name in FINGER_NAMES:
        getattr(api.hand, f"get_{name}_fault")()
    time.sleep(0.04)
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
    # Prime the five asynchronous response caches, then read them once more.
    api.get_temperature()
    time.sleep(0.04)
    return _int_vector(
        api.get_temperature(),
        length=20,
        label="temperature20",
    )


def _fresh_index_state6(api: Any) -> list[int]:
    api.hand.get_index_positions()
    time.sleep(0.04)
    return _int_vector(api.hand.x42, length=6, label="index_state6")


def _read_state20(api: Any, hw_utils: Any) -> list[int]:
    state = hw_utils.validate_state20(api.get_state())
    if state is None:
        raise RuntimeError("invalid state20")
    return [int(round(value)) for value in state]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--expected-serial", default=EXPECTED_SERIAL)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--torque", type=int, default=40)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this process may send the one motion.",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required")
    if args.expected_serial != EXPECTED_SERIAL:
        parser.error(f"expected serial must remain {EXPECTED_SERIAL}")
    if not 1 <= args.speed <= 20:
        parser.error("--speed must be in 1..20 for this first step")
    if not 1 <= args.torque <= 40:
        parser.error("--torque must be in 1..40 for this first step")
    if args.samples < 5 or args.hz <= 0:
        parser.error("--samples must be >=5 and --hz must be positive")
    return args


def main() -> int:
    args = parse_args()
    from screwdriver_rl.deploy import hw_utils
    from screwdriver_rl.deploy import linker_sdk_map as sdkmap

    sdkmap.apply_calibration(str(args.calib))
    hw_utils.bootstrap_sdk(str(args.sdk_root))
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    sdk_g20 = (
        args.sdk_root
        / "linker_hand_sdk_ros/scripts/LinkerHand/core/can/linker_hand_g20_can.py"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "operation": "g20_index_pip_first_step_raw255_to_raw240",
        "approved_scope": {
            "joint": "index_pip",
            "raw20_slot": INDEX_PIP_SLOT,
            "start_raw": EXPECTED_START_RAW,
            "target_raw": ONLY_ALLOWED_TARGET_RAW,
            "continue_to_raw224": False,
            "automatic_return_to_raw255": False,
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
            "finger_move_called": False,
            "other_finger_position_frames_sent": False,
            "index_speed_only": args.speed,
            "index_torque_limit_only": args.torque,
            "clear_faults_called": False,
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
            hand_type="left",
            hand_joint="G20",
            can=args.can,
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
        if any(before20[slot] != 0 for slot in RESERVED_SLOTS):
            raise RuntimeError(
                f"reserved slots 11..14 are not zero: "
                f"{[before20[slot] for slot in RESERVED_SLOTS]}"
            )
        fault_groups, faults20 = _fault_snapshot(api)
        if any(faults20):
            raise RuntimeError(
                f"nonzero preflight faults: "
                f"{[(i, value) for i, value in enumerate(faults20) if value]}"
            )
        temperature_before20 = _temperature_snapshot(api)
        index_before6 = _fresh_index_state6(api)
        after20, index_command6, raw20_diff = build_index_pip_command(
            before20,
            index_before6,
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
        print(f"[g20-first-step] raw20 before: {before20}", flush=True)
        print(f"[g20-first-step] raw20 after:  {after20}", flush=True)
        print(f"[g20-first-step] only diff:    {raw20_diff}", flush=True)
        print(f"[g20-first-step] index 0x42:   {index_command6}", flush=True)

        api.hand.set_index_speed([args.speed] * 6)
        api.hand.set_index_torque([args.torque] * 6)
        time.sleep(0.05)
        payload["motion_command_wall_time_s"] = time.time()
        api.hand.set_index_positions(index_command6)
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
                f"[g20-first-step] sample={sample_index:02d} "
                f"slot16={state20[INDEX_PIP_SLOT]} "
                f"faults={sum(value != 0 for value in sample_faults20)} "
                f"index_tip_temp={temperature20[INDEX_PIP_SLOT]}",
                flush=True,
            )
            if any(sample_faults20):
                payload["automatic_stop_reason"] = "nonzero fault after command"
                break
            if (
                temperature20[INDEX_PIP_SLOT]
                - temperature_before20[INDEX_PIP_SLOT]
                >= 8
            ):
                payload["automatic_stop_reason"] = (
                    "index PIP temperature increased by at least 8 degC"
                )
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
        final_temperature = records[-1]["temperature20"]
        payload["result"] = {
            "state20_median": post_median,
            "state20_span": post_span,
            "slot16_target_error_raw": post_median[INDEX_PIP_SLOT]
            - ONLY_ALLOWED_TARGET_RAW,
            "non_target_max_abs_change_raw": max(
                abs(post_median[slot] - before20[slot])
                for slot in ACTIVE_SLOTS
                if slot != INDEX_PIP_SLOT
            ),
            "temperature_delta20": [
                after - before
                for before, after in zip(
                    temperature_before20,
                    final_temperature,
                )
            ],
            "fault_free": not any(
                value
                for record in records
                for value in record["faults20"]
            ),
        }
        payload["ended_wall_time_s"] = time.time()
        _write_json(args.out, payload)
        print(f"[g20-first-step] wrote {args.out}", flush=True)
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
