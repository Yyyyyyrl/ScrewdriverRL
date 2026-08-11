#!/usr/bin/env python3
"""Record a read-only LinkerHand G20 SDK health snapshot as JSON.

This tool opens the CAN interface and sends query frames only.  It never calls
``set_speed``, ``set_torque``, ``finger_move``, or ``clear_faults``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _fault_snapshot(api: Any) -> tuple[dict[str, list[int]], list[int] | None]:
    hand = api.hand
    for name in FINGER_NAMES:
        getattr(hand, f"get_{name}_fault")()
    time.sleep(0.04)
    groups = {
        name: [int(value) for value in getattr(hand, attr)]
        for name, attr in zip(FINGER_NAMES, ("x59", "x5A", "x5B", "x5C", "x5D"))
    }
    flat = hand.joint_state_to_cmd_state(list(groups.values()))
    if not isinstance(flat, list) or len(flat) != 20:
        return groups, None
    try:
        return groups, [int(value) for value in flat]
    except (TypeError, ValueError):
        return groups, None


def _tactile_masses(api: Any) -> list[float] | None:
    matrices = api.get_matrix_touch()
    time.sleep(0.02)
    masses: list[float] = []
    try:
        for matrix in matrices:
            values = [float(value) for row in matrix for value in row]
            if len(values) != 72 or any(
                not math.isfinite(value) or value < 0.0 for value in values
            ):
                return None
            masses.append(sum(values))
    except (TypeError, ValueError):
        return None
    return masses if len(masses) == 5 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--side", default="left", choices=("left", "right"))
    parser.add_argument("--hand-joint", default="G20", choices=("G20",))
    parser.add_argument("--can", default="can0")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.hz <= 0.0:
        parser.error("--samples and --hz must be positive")

    from screwdriver_rl.deploy import hw_utils
    from screwdriver_rl.deploy import linker_sdk_map as sdkmap

    sdkmap.apply_calibration(args.calib)
    hw_utils.bootstrap_sdk(args.sdk_root)
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    api = LinkerHandApi(
        hand_type=args.side,
        hand_joint=args.hand_joint,
        can=args.can,
    )
    try:
        identity = {
            "serial": str(api.get_serial_number()).strip().strip("\x00"),
            "embedded_version": [int(value) for value in api.get_embedded_version()],
            "touch_type": int(api.get_touch_type()),
            "sdk_version": "3.1.0",
        }
        records = []
        period = 1.0 / args.hz
        for sample_index in range(args.samples):
            started = time.monotonic()
            state = hw_utils.validate_state20(api.get_state())
            groups, faults20 = _fault_snapshot(api)
            tactile = _tactile_masses(api)
            decoded = (
                sdkmap.sdk_range_to_joints16(state) if state is not None else None
            )
            record = {
                "sample": sample_index,
                "wall_time": time.time(),
                "monotonic_time": time.monotonic(),
                "state20": state,
                "joint_rad16": decoded,
                "faults_by_finger": groups,
                "faults20": faults20,
                "tactile_mass5": tactile,
            }
            records.append(record)
            active = (
                None
                if faults20 is None
                else {
                    str(slot): value
                    for slot, value in enumerate(faults20)
                    if value != 0
                }
            )
            print(
                f"[sdk-snapshot] sample={sample_index} "
                f"state={'ok' if state is not None else 'invalid'} "
                f"active_faults={active} tactile={tactile}",
                flush=True,
            )
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0 and sample_index + 1 < args.samples:
                time.sleep(remaining)

        payload = {
            "schema_version": 1,
            "read_only": True,
            "identity": identity,
            "transport": {
                "side": args.side,
                "hand_joint": args.hand_joint,
                "can": args.can,
                "sdk_root": str(Path(args.sdk_root).resolve()),
            },
            "calibration": str(Path(args.calib).resolve()),
            "joint_order16": [joint.name for joint in sdkmap.active_joints()],
            "samples": records,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[sdk-snapshot] wrote {args.out}", flush=True)
    finally:
        close = getattr(api.hand, "close_can_interface", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
