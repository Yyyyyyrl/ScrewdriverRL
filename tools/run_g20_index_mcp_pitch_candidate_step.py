#!/usr/bin/env python3
"""Run one adjacent G20 index-MCP-pitch calibration step.

The tool is deliberately fail-closed:

* only the expected left G20 serial is accepted;
* raw20 slot 1 and the fresh index 0x42 frame must agree at the requested
  starting MCP value;
* index PIP/raw20 slot 16 must be at the raw-255 isolation reference;
* all faults and reserved raw20 slots 11--14 must be zero;
* only index-frame element 2 (MCP root/pitch) is changed, while element 5
  explicitly holds PIP at raw 255;
* speed and the approved six-element torque vector are set only for index;
* exactly one position target is sent, with no continuation or return.

The output JSON is written before motion and updated after every safety phase.
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
    _fresh_index_state6,
    _int_vector,
    _median20,
    _read_state20,
    _sha256,
    _temperature_snapshot,
    _write_json,
)


INDEX_ROOT_SLOT = 1
INDEX_PIP_SLOT = 16
PIP_HOLD_RAW = 255
APPROVED_INDEX_TORQUE6 = [80, 40, 80, 40, 40, 40]
MCP_RAW_CANDIDATES = (
    255,
    240,
    224,
    208,
    192,
    176,
    160,
    144,
    128,
    112,
    96,
    80,
    64,
    48,
    32,
    20,
    12,
    6,
    0,
)


def build_index_mcp_pitch_command(
    state20: Sequence[int],
    index_state6: Sequence[int],
    *,
    expected_start_raw: int,
    target_raw: int,
    start_tolerance_raw: int = 0,
    pip_tolerance_raw: int = 0,
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Validate and build the isolated raw20/index-frame MCP command."""

    before = _int_vector(state20, length=20, label="state20")
    index_before = _int_vector(
        index_state6, length=6, label="index_state6"
    )
    if target_raw not in MCP_RAW_CANDIDATES:
        raise ValueError(f"target raw must be one of {MCP_RAW_CANDIDATES}")
    if (
        abs(before[INDEX_ROOT_SLOT] - expected_start_raw)
        > start_tolerance_raw
    ):
        raise ValueError(
            f"slot {INDEX_ROOT_SLOT} must start within "
            f"{start_tolerance_raw} raw of {expected_start_raw}, got "
            f"{before[INDEX_ROOT_SLOT]}"
        )
    actual_start_raw = before[INDEX_ROOT_SLOT]
    step_raw = abs(target_raw - actual_start_raw)
    if not 1 <= step_raw <= 17:
        raise ValueError(
            f"actual MCP pitch step must be 1..17 raw, got "
            f"{actual_start_raw} to {target_raw} ({step_raw})"
        )
    if abs(before[INDEX_PIP_SLOT] - PIP_HOLD_RAW) > pip_tolerance_raw:
        raise ValueError(
            f"slot {INDEX_PIP_SLOT} PIP must be within {pip_tolerance_raw} "
            f"raw of {PIP_HOLD_RAW}, got {before[INDEX_PIP_SLOT]}"
        )

    # Vendor G20 index order is [side, reserved, root, reserved, reserved, tip].
    expected_visible = (
        before[6],
        before[INDEX_ROOT_SLOT],
        before[INDEX_PIP_SLOT],
    )
    actual_visible = (
        index_before[0],
        index_before[2],
        index_before[5],
    )
    if (
        actual_visible[0] != expected_visible[0]
        or abs(actual_visible[1] - expected_visible[1])
        > start_tolerance_raw
        or abs(actual_visible[2] - expected_visible[2])
        > pip_tolerance_raw
    ):
        raise ValueError(
            "fresh index frame disagrees with raw20 at side/root/tip: "
            f"{actual_visible} != {expected_visible}"
        )

    after = list(before)
    after[INDEX_ROOT_SLOT] = target_raw
    after[INDEX_PIP_SLOT] = PIP_HOLD_RAW
    index_after = list(index_before)
    index_after[2] = target_raw
    index_after[5] = PIP_HOLD_RAW
    diff = [
        {"slot": slot, "before": old, "after": new}
        for slot, (old, new) in enumerate(zip(before, after))
        if old != new
    ]
    expected_diff = [
        {
            "slot": INDEX_ROOT_SLOT,
            "before": actual_start_raw,
            "after": target_raw,
        }
    ]
    if before[INDEX_PIP_SLOT] != PIP_HOLD_RAW:
        expected_diff.append(
            {
                "slot": INDEX_PIP_SLOT,
                "before": before[INDEX_PIP_SLOT],
                "after": PIP_HOLD_RAW,
            }
        )
    if diff != expected_diff:
        raise AssertionError(f"unsafe raw20 diff: {diff}")
    return after, index_after, diff


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--expected-serial", default=EXPECTED_SERIAL)
    parser.add_argument("--expected-start-raw", type=int, required=True)
    parser.add_argument(
        "--expected-start-tolerance-raw", type=int, default=2
    )
    parser.add_argument("--pip-tolerance-raw", type=int, default=1)
    parser.add_argument("--target-raw", type=int, required=True)
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this process sends one motion.",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    if args.expected_serial != EXPECTED_SERIAL:
        parser.error(f"expected serial must remain {EXPECTED_SERIAL}")
    if not 0 <= args.expected_start_raw <= 255:
        parser.error("--expected-start-raw must be in 0..255")
    if args.expected_start_tolerance_raw not in (0, 1, 2):
        parser.error("--expected-start-tolerance-raw must be 0, 1, or 2")
    if args.pip_tolerance_raw not in (0, 1, 2):
        parser.error("--pip-tolerance-raw must be 0, 1, or 2")
    if args.target_raw not in MCP_RAW_CANDIDATES:
        parser.error(f"--target-raw must be one of {MCP_RAW_CANDIDATES}")
    if not (
        1
        <= abs(args.target_raw - args.expected_start_raw)
        <= 17
    ):
        parser.error("candidate step must move 1..17 raw counts")
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
        "operation": (
            f"g20_index_mcp_pitch_candidate_step_raw"
            f"{args.expected_start_raw}_to_raw{args.target_raw}"
        ),
        "approved_scope": {
            "joint": "index_mcp_pitch",
            "raw20_slot": INDEX_ROOT_SLOT,
            "start_raw": args.expected_start_raw,
            "expected_start_tolerance_raw": (
                args.expected_start_tolerance_raw
            ),
            "target_raw": args.target_raw,
            "pip_hold_raw": PIP_HOLD_RAW,
            "pip_tolerance_raw": args.pip_tolerance_raw,
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
            "finger_move_called": False,
            "other_finger_position_frames_sent": False,
            "index_speed_only": args.speed,
            "index_torque6": APPROVED_INDEX_TORQUE6,
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
                "reserved slots 11..14 are not zero: "
                f"{[before20[slot] for slot in RESERVED_SLOTS]}"
            )
        fault_groups, faults20 = _fault_snapshot(api)
        if any(faults20):
            raise RuntimeError(
                "nonzero preflight faults: "
                f"{[(i, value) for i, value in enumerate(faults20) if value]}"
            )
        temperature_before20 = _temperature_snapshot(api)
        index_before6 = _fresh_index_state6(api)
        after20, index_command6, raw20_diff = (
            build_index_mcp_pitch_command(
                before20,
                index_before6,
                expected_start_raw=args.expected_start_raw,
                target_raw=args.target_raw,
                start_tolerance_raw=args.expected_start_tolerance_raw,
                pip_tolerance_raw=args.pip_tolerance_raw,
            )
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
        print(f"[g20-mcp-step] raw20 before: {before20}", flush=True)
        print(f"[g20-mcp-step] raw20 after:  {after20}", flush=True)
        print(f"[g20-mcp-step] only diff:    {raw20_diff}", flush=True)
        print(f"[g20-mcp-step] index 0x42:   {index_command6}", flush=True)

        api.hand.set_index_speed([args.speed] * 6)
        api.hand.set_index_torque(APPROVED_INDEX_TORQUE6)
        time.sleep(0.05)
        payload["motion_command_wall_time_s"] = time.time()
        api.hand.set_index_positions(index_command6)
        payload["motion_sent"] = True
        _write_json(args.out, payload)

        records: list[dict[str, Any]] = []
        period = 1.0 / args.hz
        for sample_index in range(args.samples):
            sample_started = time.monotonic()
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
                f"[g20-mcp-step] sample={sample_index:02d} "
                f"root={state20[INDEX_ROOT_SLOT]} "
                f"pip={state20[INDEX_PIP_SLOT]} "
                f"faults={sum(value != 0 for value in sample_faults20)} "
                f"root_temp={temperature20[INDEX_ROOT_SLOT]}",
                flush=True,
            )
            if any(sample_faults20):
                payload["automatic_stop_reason"] = (
                    "nonzero fault after command"
                )
                break
            if (
                temperature20[INDEX_ROOT_SLOT]
                - temperature_before20[INDEX_ROOT_SLOT]
                >= 8
            ):
                payload["automatic_stop_reason"] = (
                    "index MCP temperature increased by at least 8 degC"
                )
                break
            if (
                abs(state20[INDEX_PIP_SLOT] - PIP_HOLD_RAW)
                > args.pip_tolerance_raw + 2
            ):
                payload["automatic_stop_reason"] = (
                    "index PIP left raw-255 isolation hold"
                )
                break
            remaining = period - (time.monotonic() - sample_started)
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
        payload["result"] = {
            "state20_median": post_median,
            "state20_span": post_span,
            "root_target_error_raw": (
                post_median[INDEX_ROOT_SLOT] - args.target_raw
            ),
            "pip_hold_target_error_raw": (
                post_median[INDEX_PIP_SLOT] - PIP_HOLD_RAW
            ),
            "non_target_max_abs_change_raw": max(
                abs(post_median[slot] - before20[slot])
                for slot in ACTIVE_SLOTS
                if slot not in (INDEX_ROOT_SLOT, INDEX_PIP_SLOT)
            ),
            "temperature_delta20": [
                after - before
                for before, after in zip(
                    temperature_before20, final_temperature
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
        print(f"[g20-mcp-step] wrote {args.out}", flush=True)
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
