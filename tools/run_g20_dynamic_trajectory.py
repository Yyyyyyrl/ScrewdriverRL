#!/usr/bin/env python3
"""Run one bounded G20 trajectory with fail-closed health and raw-step gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SERIAL = "LHT20-010-415-L-B-1-D"
RESERVED_SLOTS = (11, 12, 13, 14)


def median20(rows: list[list[int]]) -> list[int]:
    return [int(round(statistics.median(column))) for column in zip(*rows)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", default="/home/user/linkerhand-ros-sdk")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-raw-step", type=int, default=8)
    parser.add_argument("--max-temperature-c", type=int, default=60)
    parser.add_argument("--settle-s", type=float, default=0.8)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.rate_hz <= 0 or args.samples <= 0:
        parser.error("rate and samples must be positive")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    name = str(spec["name"])
    endpoint_a = [float(value) for value in spec["endpoint_a"]]
    endpoint_b = [float(value) for value in spec["endpoint_b"]]
    if len(endpoint_a) != 16 or len(endpoint_b) != 16:
        raise ValueError("each endpoint must contain 16 semantic joints")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from screwdriver_rl.deploy import hw_utils
    from screwdriver_rl.deploy import linker_sdk_map as sdkmap
    from tools.run_g20_index_pip_candidate_step import (
        _fault_snapshot,
        _read_state20,
        _temperature_snapshot,
        _write_json,
    )

    sdkmap.apply_calibration(str(args.calib))
    joints = sdkmap.active_joints()
    for label, endpoint in (("endpoint_a", endpoint_a), ("endpoint_b", endpoint_b)):
        for joint, value in zip(joints, endpoint):
            if not joint.lo - 1e-9 <= value <= joint.hi + 1e-9:
                raise ValueError(
                    f"{label}: {joint.name}={value:.6f} outside "
                    f"[{joint.lo:.6f}, {joint.hi:.6f}]"
                )
        raw = sdkmap.joints16_to_sdk_range(endpoint)
        if any(raw[slot] != 0 for slot in RESERVED_SLOTS):
            raise ValueError(f"{label} writes reserved slots: {raw}")

    hw_utils.bootstrap_sdk(args.sdk_root)
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    api = LinkerHandApi(hand_joint="G20", hand_type="left", can=args.can)
    records: list[dict[str, Any]] = []
    try:
        serial = str(api.get_serial_number()).strip().strip("\x00")
        if serial != EXPECTED_SERIAL:
            raise RuntimeError(f"serial mismatch: {serial!r}")
        start_raw = _read_state20(api, hw_utils)
        start_q = [float(value) for value in sdkmap.sdk_range_to_joints16(start_raw)]
        faults_before_by_finger, faults_before = _fault_snapshot(api)
        temperature_before = _temperature_snapshot(api)
        if any(faults_before):
            raise RuntimeError(f"pre-existing faults: {faults_before}")
        if max(temperature_before) > args.max_temperature_c:
            raise RuntimeError(
                f"pre-existing temperature {max(temperature_before)}C exceeds limit"
            )

        phases: list[tuple[str, list[float]]] = []

        def ramp(phase: str, start: list[float], end: list[float], duration_s: float) -> None:
            for frame in hw_utils.ramp_frames(start, end, duration_s, args.rate_hz):
                phases.append((phase, [float(value) for value in frame]))

        def hold(phase: str, pose: list[float], duration_s: float) -> None:
            for _ in range(max(1, round(duration_s * args.rate_hz))):
                phases.append((phase, list(pose)))

        ramp("preposition", start_q, endpoint_a, float(spec.get("preposition_s", 1.5)))
        hold("hold_a_before", endpoint_a, float(spec.get("hold_s", 0.5)))
        ramp("a_to_b", endpoint_a, endpoint_b, float(spec.get("leg_s", 2.5)))
        hold("hold_b", endpoint_b, float(spec.get("hold_s", 0.5)))
        ramp("b_to_a", endpoint_b, endpoint_a, float(spec.get("leg_s", 2.5)))
        hold("hold_a_after", endpoint_a, float(spec.get("hold_s", 0.5)))
        ramp("return", endpoint_a, start_q, float(spec.get("return_s", 1.5)))

        raw_frames = [sdkmap.joints16_to_sdk_range(frame) for _, frame in phases]
        previous = start_raw
        worst_raw_step = 0
        worst_raw_step_record: dict[str, Any] | None = None
        for index, (raw, (phase, _)) in enumerate(zip(raw_frames, phases)):
            diffs = [abs(int(a) - int(b)) for a, b in zip(raw, previous)]
            step = max(diffs)
            if step > worst_raw_step:
                slot = diffs.index(step)
                worst_raw_step = step
                worst_raw_step_record = {
                    "frame": index,
                    "phase": phase,
                    "slot": slot,
                    "before": previous[slot],
                    "after": raw[slot],
                    "step": step,
                }
            previous = raw
        if worst_raw_step > args.max_raw_step:
            raise RuntimeError(
                f"trajectory max raw step {worst_raw_step} exceeds "
                f"{args.max_raw_step}: {worst_raw_step_record}"
            )

        preflight = {
            "schema_version": 1,
            "name": name,
            "motion_sent": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "serial": serial,
            "calibration": str(args.calib),
            "joint_order16": [joint.name for joint in joints],
            "spec": spec,
            "start_raw20": start_raw,
            "start_semantic": start_q,
            "faults_before_by_finger": faults_before_by_finger,
            "faults20_before": faults_before,
            "temperature20_before": temperature_before,
            "frame_count": len(phases),
            "nominal_duration_s": len(phases) / args.rate_hz,
            "rate_hz": args.rate_hz,
            "worst_planned_raw_step": worst_raw_step_record,
        }
        _write_json(args.out_dir / "preflight.json", preflight)

        api.set_speed(speed=[args.speed] * 5)
        time.sleep(0.1)
        started_mono = time.monotonic() + 0.2
        for index, ((phase, semantic), raw) in enumerate(zip(phases, raw_frames)):
            due = started_mono + index / args.rate_hz
            remaining = due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            wall_ns = time.time_ns()
            mono = time.monotonic()
            api.finger_move(pose=raw)
            records.append(
                {
                    "index": index,
                    "phase": phase,
                    "wall_time_ns": wall_ns,
                    "monotonic_time": mono,
                    "target_semantic": semantic,
                    "target_raw20": raw,
                }
            )
        time.sleep(args.settle_s)

        state_rows = []
        for _ in range(args.samples):
            state_rows.append(_read_state20(api, hw_utils))
            time.sleep(0.05)
        settled_raw = median20(state_rows)
        settled_q = [float(value) for value in sdkmap.sdk_range_to_joints16(settled_raw)]
        faults_after_by_finger, faults_after = _fault_snapshot(api)
        temperature_after = _temperature_snapshot(api)
        result = dict(preflight)
        result.update(
            {
                "motion_sent": True,
                "commands": records,
                "settled_raw20": settled_raw,
                "settled_semantic": settled_q,
                "faults_after_by_finger": faults_after_by_finger,
                "faults20_after": faults_after,
                "temperature20_after": temperature_after,
                "max_return_abs_error_rad": max(
                    abs(a - b) for a, b in zip(settled_q, start_q)
                ),
            }
        )
        _write_json(args.out_dir / "trajectory_log.json", result)
        if any(faults_after):
            raise RuntimeError(f"trajectory raised faults: {faults_after}")
        if max(temperature_after) > args.max_temperature_c:
            raise RuntimeError(
                f"post temperature {max(temperature_after)}C exceeds limit"
            )
        print(
            f"[trajectory] {name}: {len(records)} commands, "
            f"Tmax={max(temperature_after)}C, faults=0, "
            f"return_error={result['max_return_abs_error_rad']:.4f} rad",
            flush=True,
        )
    finally:
        close = getattr(api.hand, "close_can_interface", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
