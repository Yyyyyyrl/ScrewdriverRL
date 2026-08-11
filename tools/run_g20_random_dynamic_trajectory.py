#!/usr/bin/env python3
"""Run one bounded multi-waypoint G20 trajectory with fail-closed health gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_g20_random_dynamic_specs import JOINTS, SAFE_ENVELOPE


EXPECTED_SERIAL = "LHT20-010-415-L-B-1-D"
RESERVED_SLOTS = (11, 12, 13, 14)


def _median20(rows: list[list[int]]) -> list[int]:
    return [int(round(statistics.median(column))) for column in zip(*rows)]


def _smooth_segment(
    start: list[float], end: list[float], duration_s: float, rate_hz: float
) -> list[list[float]]:
    count = max(1, round(duration_s * rate_hz))
    frames = []
    for index in range(1, count + 1):
        t = index / count
        alpha = t * t * (3.0 - 2.0 * t)
        frames.append([a + alpha * (b - a) for a, b in zip(start, end)])
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", default="/home/user/linkerhand-ros-sdk")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speed", type=int, default=15)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-raw-step", type=int, default=5)
    parser.add_argument("--max-temperature-c", type=int, default=60)
    parser.add_argument("--max-tactile-mass", type=float, default=250.0)
    parser.add_argument("--tactile-poll-frames", type=int, default=5)
    parser.add_argument("--settle-s", type=float, default=0.8)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if (
        args.rate_hz <= 0
        or args.samples <= 0
        or args.max_tactile_mass <= 0
        or args.tactile_poll_frames <= 0
    ):
        parser.error("rate and samples must be positive")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 2:
        raise ValueError("random trajectory spec must use schema_version 2")
    if tuple(spec.get("joint_order16", ())) != JOINTS:
        raise ValueError("spec joint order does not match the production semantic order")
    if spec.get("safety_envelope_rad") != {
        name: list(bounds) for name, bounds in SAFE_ENVELOPE.items()
    }:
        raise ValueError("spec safety envelope differs from the reviewed executor envelope")
    waypoint_rows = spec.get("waypoints", [])
    if not 4 <= len(waypoint_rows) <= 8:
        raise ValueError("trajectory must contain 4..8 waypoints")
    waypoints = [[float(v) for v in row["semantic"]] for row in waypoint_rows]
    for row, pose in zip(waypoint_rows, waypoints):
        if len(pose) != 16:
            raise ValueError(f"{row['name']} must contain 16 semantic joints")
        if float(row["duration_s"]) < 1.0 or float(row["hold_s"]) < 0.0:
            raise ValueError(f"{row['name']} has unsafe timing")
        for name, value in zip(JOINTS, pose):
            lo, hi = SAFE_ENVELOPE[name]
            if not lo <= value <= hi:
                raise ValueError(f"{row['name']}: {name}={value} outside {lo}..{hi}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from screwdriver_rl.deploy import hw_utils
    from screwdriver_rl.deploy import linker_sdk_map as sdkmap
    from tools.run_g20_index_pip_candidate_step import (
        _fault_snapshot,
        _read_state20,
        _temperature_snapshot,
        _write_json,
    )
    from tools.record_g20_sdk_snapshot import _tactile_masses

    sdkmap.apply_calibration(str(args.calib))
    joints = sdkmap.active_joints()
    if tuple(joint.name for joint in joints) != JOINTS:
        raise RuntimeError("active calibration joint order differs from spec")
    for row, pose in zip(waypoint_rows, waypoints):
        for joint, value in zip(joints, pose):
            if not joint.lo - 1e-9 <= value <= joint.hi + 1e-9:
                raise ValueError(f"{row['name']}: {joint.name} outside active limits")
        raw = sdkmap.joints16_to_sdk_range(pose)
        if any(raw[slot] != 0 for slot in RESERVED_SLOTS):
            raise ValueError(f"{row['name']} writes reserved slots: {raw}")

    hw_utils.bootstrap_sdk(args.sdk_root)
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    api = LinkerHandApi(hand_joint="G20", hand_type="left", can=args.can)
    commands: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    tactile_checks: list[dict[str, Any]] = []
    rollback_commands: list[dict[str, Any]] = []
    preflight: dict[str, Any] | None = None
    motion_sent = False
    try:
        serial = str(api.get_serial_number()).strip().strip("\x00")
        if serial != EXPECTED_SERIAL:
            raise RuntimeError(f"serial mismatch: {serial!r}")
        start_raw = _read_state20(api, hw_utils)
        start_q = [float(value) for value in sdkmap.sdk_range_to_joints16(start_raw)]
        outside_start = [
            {
                "joint": joint.name,
                "value": value,
                "limits": [joint.lo, joint.hi],
            }
            for joint, value in zip(joints, start_q)
            if not joint.lo - 1e-9 <= value <= joint.hi + 1e-9
        ]
        if outside_start:
            raise RuntimeError(
                "measured start is outside active command limits; normalize the "
                f"hand before trajectory construction: {outside_start}"
            )
        faults_before_by_finger, faults_before = _fault_snapshot(api)
        temperature_before = _temperature_snapshot(api)
        tactile_before = _tactile_masses(api)
        if any(faults_before):
            raise RuntimeError(f"pre-existing faults: {faults_before}")
        if max(temperature_before) > args.max_temperature_c:
            raise RuntimeError(f"pre-existing temperature exceeds {args.max_temperature_c}C")
        if tactile_before is None:
            raise RuntimeError("invalid pre-existing tactile telemetry")
        if max(tactile_before) > args.max_tactile_mass:
            raise RuntimeError(
                f"pre-existing tactile mass {max(tactile_before):.1f} exceeds limit"
            )

        first = waypoints[0]
        endpoint_timing_scale = 1.0
        for _ in range(8):
            phases: list[dict[str, Any]] = []
            for semantic in _smooth_segment(
                start_q,
                first,
                float(spec.get("preposition_s", 1.8)) * endpoint_timing_scale,
                args.rate_hz,
            ):
                phases.append(
                    {"phase": "preposition", "semantic": semantic, "gate": False}
                )
            previous = first
            for index, (row, waypoint) in enumerate(zip(waypoint_rows, waypoints)):
                if index > 0:
                    for semantic in _smooth_segment(
                        previous, waypoint, float(row["duration_s"]), args.rate_hz
                    ):
                        phases.append(
                            {
                                "phase": f"to_{row['name']}",
                                "semantic": semantic,
                                "gate": False,
                            }
                        )
                hold_count = max(1, round(float(row["hold_s"]) * args.rate_hz))
                for hold_index in range(hold_count):
                    phases.append(
                        {
                            "phase": f"hold_{row['name']}",
                            "semantic": list(waypoint),
                            "gate": hold_index == hold_count - 1,
                        }
                    )
                previous = waypoint
            for semantic in _smooth_segment(
                previous,
                start_q,
                float(spec.get("return_s", 1.8)) * endpoint_timing_scale,
                args.rate_hz,
            ):
                phases.append({"phase": "return", "semantic": semantic, "gate": False})

            raw_frames = [
                sdkmap.joints16_to_sdk_range(row["semantic"]) for row in phases
            ]
            previous_raw = start_raw
            worst: dict[str, Any] = {"step": 0}
            for index, (raw, phase) in enumerate(zip(raw_frames, phases)):
                diffs = [abs(int(a) - int(b)) for a, b in zip(raw, previous_raw)]
                step = max(diffs)
                if step > worst["step"]:
                    slot = diffs.index(step)
                    worst = {
                        "frame": index,
                        "phase": phase["phase"],
                        "slot": slot,
                        "before": previous_raw[slot],
                        "after": raw[slot],
                        "step": step,
                    }
                previous_raw = raw
            if worst["step"] <= args.max_raw_step:
                break
            if worst["phase"] not in ("preposition", "return"):
                raise RuntimeError(
                    f"waypoint trajectory max raw step {worst['step']} exceeds "
                    f"{args.max_raw_step}: {worst}"
                )
            endpoint_timing_scale *= max(
                1.15, 1.10 * float(worst["step"]) / args.max_raw_step
            )
            if endpoint_timing_scale > 4.0:
                raise RuntimeError(
                    "endpoint timing adaptation exceeded 4x; normalize the start pose"
                )
        else:
            raise RuntimeError("failed to adapt endpoint timing below raw-step limit")

        preflight = {
            "schema_version": 2,
            "name": str(spec["name"]),
            "motion_sent": False,
            "completed": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "serial": serial,
            "calibration": str(args.calib.resolve()),
            "joint_order16": list(JOINTS),
            "spec": spec,
            "start_raw20": start_raw,
            "start_semantic": start_q,
            "faults_before_by_finger": faults_before_by_finger,
            "faults20_before": faults_before,
            "temperature20_before": temperature_before,
            "tactile_mass5_before": tactile_before,
            "frame_count": len(phases),
            "nominal_duration_s": len(phases) / args.rate_hz,
            "rate_hz": args.rate_hz,
            "endpoint_timing_scale": endpoint_timing_scale,
            "max_raw_step_limit": args.max_raw_step,
            "max_tactile_mass_limit": args.max_tactile_mass,
            "tactile_poll_frames": args.tactile_poll_frames,
            "worst_planned_raw_step": worst,
        }
        _write_json(args.out_dir / "preflight.json", preflight)

        api.set_speed(speed=[args.speed] * 5)
        time.sleep(0.1)
        next_due = time.monotonic() + 0.2
        for index, (phase, raw) in enumerate(zip(phases, raw_frames)):
            remaining = next_due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            wall_ns = time.time_ns()
            api.finger_move(pose=raw)
            motion_sent = True
            commands.append(
                {
                    "index": index,
                    "phase": phase["phase"],
                    "wall_time_ns": wall_ns,
                    "monotonic_time": time.monotonic(),
                    "target_semantic": phase["semantic"],
                    "target_raw20": raw,
                }
            )
            next_due = time.monotonic() + 1.0 / args.rate_hz
            if index % args.tactile_poll_frames == 0 or phase["gate"]:
                tactile = _tactile_masses(api)
                tactile_checks.append(
                    {
                        "after_command_index": index,
                        "phase": phase["phase"],
                        "wall_time_ns": time.time_ns(),
                        "tactile_mass5": tactile,
                    }
                )
                if tactile is None:
                    raise RuntimeError(f"invalid tactile telemetry at {phase['phase']}")
                if max(tactile) > args.max_tactile_mass:
                    # Reverse the already-reviewed trajectory one frame at a
                    # time (therefore still <= max_raw_step) until pressure is
                    # relieved or ten frames have been undone, then stop.
                    relieved = False
                    for rollback_index in range(index - 1, max(-1, index - 11), -1):
                        time.sleep(1.0 / args.rate_hz)
                        rollback_raw = raw_frames[rollback_index]
                        api.finger_move(pose=rollback_raw)
                        rollback_commands.append(
                            {
                                "source_command_index": rollback_index,
                                "wall_time_ns": time.time_ns(),
                                "target_raw20": rollback_raw,
                            }
                        )
                        if (index - rollback_index) % 2 == 0:
                            rollback_tactile = _tactile_masses(api)
                            if (
                                rollback_tactile is not None
                                and max(rollback_tactile) <= 0.6 * args.max_tactile_mass
                            ):
                                relieved = True
                                break
                    raise RuntimeError(
                        f"tactile mass {max(tactile):.1f} exceeded "
                        f"{args.max_tactile_mass} at {phase['phase']}; "
                        f"rollback_relieved={relieved}"
                    )
                next_due = time.monotonic() + 1.0 / args.rate_hz
            if phase["gate"]:
                gate_fault_groups, gate_faults = _fault_snapshot(api)
                gate_temperatures = _temperature_snapshot(api)
                gate_state = _read_state20(api, hw_utils)
                gate = {
                    "after_command_index": index,
                    "phase": phase["phase"],
                    "wall_time_ns": time.time_ns(),
                    "faults_by_finger": gate_fault_groups,
                    "faults20": gate_faults,
                    "temperature20": gate_temperatures,
                    "state_raw20": gate_state,
                }
                health.append(gate)
                if any(gate_faults):
                    raise RuntimeError(f"mid-trajectory faults at {phase['phase']}: {gate_faults}")
                if max(gate_temperatures) > args.max_temperature_c:
                    raise RuntimeError(
                        f"mid-trajectory temperature {max(gate_temperatures)}C exceeds limit"
                    )
                next_due = time.monotonic() + 1.0 / args.rate_hz

        time.sleep(args.settle_s)
        state_rows = []
        for _ in range(args.samples):
            state_rows.append(_read_state20(api, hw_utils))
            time.sleep(0.05)
        settled_raw = _median20(state_rows)
        settled_q = [float(value) for value in sdkmap.sdk_range_to_joints16(settled_raw)]
        faults_after_by_finger, faults_after = _fault_snapshot(api)
        temperature_after = _temperature_snapshot(api)
        result = dict(preflight)
        result.update(
            {
                "motion_sent": True,
                "completed": True,
                "commands": commands,
                "health_gates": health,
                "tactile_checks": tactile_checks,
                "rollback_commands": rollback_commands,
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
            raise RuntimeError(f"post temperature exceeds {args.max_temperature_c}C")
        print(
            f"[trajectory] {spec['name']}: {len(commands)} commands, "
            f"{len(health)} health gates, Tmax={max(temperature_after)}C, "
            f"faults=0, return_error={result['max_return_abs_error_rad']:.4f} rad",
            flush=True,
        )
    except Exception as exc:
        if preflight is not None:
            aborted = dict(preflight)
            aborted.update(
                {
                    "motion_sent": motion_sent,
                    "completed": False,
                    "commands": commands,
                    "health_gates": health,
                    "tactile_checks": tactile_checks,
                    "rollback_commands": rollback_commands,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_json(args.out_dir / "trajectory_log.json", aborted)
        raise
    finally:
        close = getattr(api.hand, "close_can_interface", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
