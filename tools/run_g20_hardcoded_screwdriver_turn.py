#!/usr/bin/env python3
"""Run the reviewed 64 mm G20 screwdriver-turn sequence.

The trajectory is intentionally fixed.  Its home pose is the 2026-08-06
top-down posture and its four waypoint offsets are the commissioning-DR
candidate selected by the Isaac closed-cycle search.  The default invocation is offline-only; hardware
motion requires both ``--arm`` and the exact hand serial as an acknowledgement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_SERIAL = "LHT20-010-415-L-B-1-D"
EXPECTED_CALIBRATION_SHA256 = (
    "b728cd58f15408193080a0bd35dc6ae281adff9a96d6115ded012498b6f7ecb7"
)
SEARCH_ARTIFACT = (
    ROOT
    / "records/g20_checkpoint_netturn_search_20260806/"
    "option_d_cycle_candidate5_third_dr16_seed20260812.json"
)
EXPECTED_SEARCH_SHA256 = (
    "3cda14cfcd32c3b165b6e33d5cb7d5da7e4b7296bf7828e5d14b5f03d7f4b27b"
)
RESERVED_SLOTS = (11, 12, 13, 14)

JOINTS = (
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
    "thumb_cmc_yaw",
    "thumb_cmc_roll",
    "thumb_cmc_pitch",
    "thumb_mcp",
)

# TOPDOWN_RESET_POSITIONS from the current 2026-08-06 promoted-calibration
# posture refit. This is the collision-valid reset pose; the Isaac target
# window below is centered on the separate compliant preload target.
HOME = (
    0.05414624541908073,
    0.5461605996458497,
    0.9885106328788478,
    0.0006154768077150623,
    0.4511338392713365,
    0.9935342729324822,
    -0.015733775224370507,
    0.6507791025586955,
    0.7018963003365457,
    0.0458935652917647,
    0.7322323193351579,
    1.14897143528816,
    0.89291068971524,
    1.0608869680859287,
    0.20380961699860453,
    0.31407836112665954,
)

# Hard local-q limits from the exact URDF used by Isaac.
URDF_JOINT_LIMITS = (
    (-0.17, 0.17), (0.0, 1.26), (0.04, 1.57),
    (-0.17, 0.17), (0.0, 1.24), (0.06, 1.57),
    (-0.17, 0.17), (0.0, 1.24), (0.02, 1.57),
    (-0.17, 0.17), (0.0, 1.14), (0.03, 1.57),
    (0.0, 1.12), (0.42, 1.22), (0.0, 0.79), (0.0, 1.05),
)

# Effective target limits exported from the exact top-down Isaac runtime.
# They combine soft URDF limits with the task's home +/- 0.35 rad window.
ISAAC_TARGET_LIMITS = (
    (-0.149999991, 0.149999991),
    (0.196160585, 0.896160603),
    (0.638510585, 1.338510633),
    (-0.149999991, 0.149999991),
    (0.101133838, 0.801133871),
    (0.643534243, 1.343534231),
    (-0.149999991, 0.149999991),
    (0.300779134, 1.000779152),
    (0.351896316, 1.051896334),
    (-0.149999991, 0.149999991),
    (0.382232338, 1.082232356),
    (0.798971415, 1.498971462),
    (0.542910695, 1.100000024),
    (0.710886955, 1.199999928),
    (0.020000000, 0.553809643),
    (0.020000000, 0.664078355),
)

SELECTED_JOINTS = JOINTS


def _archived_waypoint_offsets() -> tuple[tuple[float, ...], ...]:
    payload = json.loads(SEARCH_ARTIFACT.read_text(encoding="utf-8"))
    if tuple(payload["joint_order"]) != JOINTS:
        raise RuntimeError("Isaac cycle-search joint order mismatch")
    if tuple(payload["selected_joints"]) != SELECTED_JOINTS:
        raise RuntimeError("Isaac cycle-search selected-joint order mismatch")
    candidate = payload["best"]
    if int(candidate["candidate_id"]) != 0 or int(candidate["iteration"]) != 0:
        raise RuntimeError("unexpected final Isaac candidate identity")
    offsets = tuple(
        tuple(float(value) for value in row)
        for row in candidate["waypoint_offsets_rad"]
    )
    if len(offsets) != 5 or any(len(row) != len(JOINTS) for row in offsets):
        raise RuntimeError("unexpected final Isaac waypoint shape")
    return offsets


WAYPOINT_OFFSETS = _archived_waypoint_offsets()

ALLOWED_AMPLITUDES = (0.35, 0.65, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smooth_segment(
    start: list[float],
    end: list[float],
    duration_s: float,
    rate_hz: float,
) -> list[list[float]]:
    count = max(1, round(duration_s * rate_hz))
    rows = []
    for index in range(1, count + 1):
        t = index / count
        alpha = t * t * (3.0 - 2.0 * t)
        rows.append([a + alpha * (b - a) for a, b in zip(start, end)])
    return rows


def _linear_segment(
    start: list[float],
    end: list[float],
    duration_s: float,
    rate_hz: float,
) -> list[list[float]]:
    count = max(1, round(duration_s * rate_hz))
    return [
        [a + (index / count) * (b - a) for a, b in zip(start, end)]
        for index in range(1, count + 1)
    ]


def _raw_linear_frames(
    start: list[int] | list[float],
    end: list[int],
    *,
    duration_s: float,
    rate_hz: float,
    max_raw_step: int,
) -> list[list[int]]:
    """Interpolate directly in SDK raw space, including outside LUT coverage."""
    largest = max(
        abs(int(round(b)) - int(round(a))) for a, b in zip(start, end)
    )
    count = max(
        1,
        round(duration_s * rate_hz),
        (largest + max_raw_step - 1) // max_raw_step,
    )
    rows = []
    for index in range(1, count + 1):
        alpha = index / count
        rows.append(
            [
                int(round(float(a) + alpha * (float(b) - float(a))))
                for a, b in zip(start, end)
            ]
        )
    return rows


def waypoints(
    amplitude: float,
    *,
    anchor: tuple[float, ...] | list[float] = HOME,
) -> list[list[float]]:
    if not any(abs(amplitude - allowed) < 1.0e-9 for allowed in ALLOWED_AMPLITUDES):
        raise ValueError(f"amplitude must be one of {ALLOWED_AMPLITUDES}")
    if len(anchor) != len(JOINTS):
        raise ValueError("anchor must contain 16 joints")
    rows = []
    for offsets in WAYPOINT_OFFSETS:
        row = []
        for index, value in enumerate(anchor):
            offset = offsets[index]
            requested = float(value + amplitude * offset)
            lower, upper = ISAAC_TARGET_LIMITS[index]
            row.append(min(max(requested, lower), upper))
        rows.append(row)
    return rows


def build_phases(
    *,
    start: list[float],
    mode: str,
    amplitude: float,
    cycles: int,
    rate_hz: float,
    preposition_s: float,
    leg_s: float,
    hold_s: float,
    profile: str = "linear",
    return_to_anchor: bool = False,
) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []

    if profile not in ("linear", "smooth"):
        raise ValueError("profile must be linear or smooth")

    def add_segment(
        label: str,
        a: list[float],
        b: list[float],
        duration: float,
        *,
        force_smooth: bool = False,
    ) -> None:
        segment_fn = _smooth_segment if force_smooth or profile == "smooth" else _linear_segment
        rows = segment_fn(a, b, duration, rate_hz)
        for index, row in enumerate(rows):
            phases.append(
                {
                    "phase": label,
                    "semantic": row,
                    "gate": index == len(rows) - 1,
                }
            )

    def add_hold(label: str, pose: list[float]) -> None:
        if hold_s <= 0.0:
            return
        count = max(1, round(hold_s * rate_hz))
        for index in range(count):
            phases.append(
                {
                    "phase": label,
                    "semantic": list(pose),
                    "gate": index == count - 1,
                }
            )

    home = list(HOME)
    if mode == "preposition":
        add_segment("preposition_to_home", start, home, preposition_s, force_smooth=True)
        add_hold("hold_home", home)
        return phases

    # Search offsets are relative to the zero-tension settled measured pose.
    # Use the measured start as the cycle anchor on both Isaac and hardware.
    anchor = list(start)
    points = waypoints(amplitude, anchor=anchor)
    add_segment("acquire_wp1", start, points[0], preposition_s)
    add_hold("hold_wp1_initial", points[0])
    previous = points[0]
    for cycle in range(cycles):
        for index in range(1, len(points)):
            add_segment(
                f"cycle{cycle + 1}_to_wp{index + 1}",
                previous,
                points[index],
                leg_s,
            )
            add_hold(f"cycle{cycle + 1}_hold_wp{index + 1}", points[index])
            previous = points[index]
        add_segment(f"cycle{cycle + 1}_close_to_wp1", previous, points[0], leg_s)
        add_hold(f"cycle{cycle + 1}_hold_wp1", points[0])
        previous = points[0]
    if return_to_anchor:
        add_segment("return_anchor", previous, anchor, preposition_s)
        add_hold("hold_anchor_final", anchor)
    return phases


def _median20(rows: list[list[int]]) -> list[int]:
    return [int(round(statistics.median(column))) for column in zip(*rows)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", default="/home/user/linkerhand-ros-sdk")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--calib", type=Path, default=ROOT / "linker_calib_deploy.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preposition", "turn"), default="turn")
    parser.add_argument("--amplitude", type=float, choices=ALLOWED_AMPLITUDES, default=0.35)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--preposition-s", type=float, default=6.0)
    parser.add_argument("--acquire-s", type=float, default=0.8)
    parser.add_argument("--leg-s", type=float, default=0.8)
    parser.add_argument("--hold-s", type=float, default=0.0)
    parser.add_argument("--max-raw-step", type=int, default=7)
    parser.add_argument("--max-temperature-c", type=int, default=58)
    parser.add_argument("--max-tactile-mass", type=float, default=250.0)
    parser.add_argument("--max-tactile-rise", type=float, default=120.0)
    parser.add_argument("--tactile-poll-frames", type=int, default=2)
    parser.add_argument("--max-start-home-error-rad", type=float, default=0.22)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--confirm-serial", default="")
    args = parser.parse_args()

    if args.offline and args.arm:
        parser.error("--offline and --arm are mutually exclusive")
    if args.cycles not in (1, 2, 3):
        parser.error("--cycles must be 1, 2, or 3")
    if min(
        args.rate_hz,
        args.preposition_s,
        args.acquire_s,
        args.leg_s,
        args.max_tactile_mass,
        args.max_tactile_rise,
    ) <= 0:
        parser.error("timing, rate, and tactile limits must be positive")
    if args.arm and args.confirm_serial != EXPECTED_SERIAL:
        parser.error(
            "--arm requires --confirm-serial "
            f"{EXPECTED_SERIAL}"
        )

    calibration = args.calib.resolve()
    if _sha256(calibration) != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("production calibration SHA256 mismatch")
    if not SEARCH_ARTIFACT.exists() or _sha256(SEARCH_ARTIFACT) != EXPECTED_SEARCH_SHA256:
        raise RuntimeError("Isaac cycle-search artifact SHA256 mismatch")

    from screwdriver_rl.deploy import linker_sdk_map as sdkmap

    sdkmap.apply_calibration(str(calibration))
    active = sdkmap.active_joints()
    if tuple(joint.name for joint in active) != JOINTS:
        raise RuntimeError("active calibration joint order mismatch")

    for label, pose in [("home", list(HOME))] + [
        (f"wp{index + 1}", pose)
        for index, pose in enumerate(waypoints(args.amplitude))
    ]:
        for joint, value in zip(active, pose):
            if not joint.lo - 1.0e-9 <= value <= joint.hi + 1.0e-9:
                raise RuntimeError(
                    f"{label}: {joint.name}={value:.6f} outside "
                    f"[{joint.lo:.6f}, {joint.hi:.6f}]"
                )
        raw = sdkmap.joints16_to_sdk_range(pose)
        if any(raw[slot] != 0 for slot in RESERVED_SLOTS):
            raise RuntimeError(f"{label} writes reserved SDK slots")

    if args.offline:
        start_raw = sdkmap.joints16_to_sdk_range(list(HOME))
        start_q = list(HOME)
        serial = "offline"
        faults_before = [0] * 20
        temperatures_before = [0] * 20
        tactile_before = [0.0] * 5
        api = None
    else:
        from screwdriver_rl.deploy import hw_utils
        from tools.run_g20_index_pip_candidate_step import (
            _fault_snapshot,
            _read_state20,
            _temperature_snapshot,
        )
        from tools.record_g20_sdk_snapshot import _tactile_masses

        hw_utils.bootstrap_sdk(args.sdk_root)
        from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

        api = LinkerHandApi(hand_joint="G20", hand_type="left", can=args.can)
        serial = str(api.get_serial_number()).strip().strip("\x00")
        if serial != EXPECTED_SERIAL:
            raise RuntimeError(f"serial mismatch or hand unavailable: {serial!r}")
        start_raw = _read_state20(api, hw_utils)
        start_q = [float(value) for value in sdkmap.sdk_range_to_joints16(start_raw)]
        _fault_groups, faults_before = _fault_snapshot(api)
        temperatures_before = _temperature_snapshot(api)
        tactile_before = _tactile_masses(api)
        if any(faults_before):
            raise RuntimeError(f"pre-existing faults: {faults_before}")
        if max(temperatures_before) > args.max_temperature_c:
            raise RuntimeError("pre-existing temperature exceeds limit")
        if tactile_before is None:
            raise RuntimeError("invalid pre-existing tactile telemetry")
        if max(tactile_before) > args.max_tactile_mass:
            raise RuntimeError("pre-existing tactile mass exceeds limit")
        if args.mode == "turn":
            start_error = max(abs(a - b) for a, b in zip(start_q, HOME))
            if start_error > args.max_start_home_error_rad:
                raise RuntimeError(
                    f"turn mode start is {start_error:.4f} rad from home; "
                    "run --mode preposition first"
                )

    if args.mode == "preposition":
        target_raw = sdkmap.joints16_to_sdk_range(list(HOME))
        raw_frames = _raw_linear_frames(
            start_raw,
            target_raw,
            duration_s=args.preposition_s,
            rate_hz=args.rate_hz,
            max_raw_step=args.max_raw_step,
        )
        phases = [
            {
                "phase": "preposition_raw_to_home",
                "semantic": [
                    float(value)
                    for value in sdkmap.sdk_range_to_joints16(raw)
                ],
                "gate": index == len(raw_frames) - 1,
            }
            for index, raw in enumerate(raw_frames)
        ]
    else:
        phases = build_phases(
            start=start_q,
            mode=args.mode,
            amplitude=args.amplitude,
            cycles=args.cycles,
            rate_hz=args.rate_hz,
            preposition_s=args.acquire_s,
            leg_s=args.leg_s,
            hold_s=args.hold_s,
        )
        raw_frames = [
            sdkmap.joints16_to_sdk_range(row["semantic"]) for row in phases
        ]
    previous = start_raw
    worst: dict[str, Any] = {"step": 0}
    for index, (phase, raw) in enumerate(zip(phases, raw_frames)):
        diffs = [abs(int(a) - int(b)) for a, b in zip(raw, previous)]
        step = max(diffs)
        if step > worst["step"]:
            slot = diffs.index(step)
            worst = {
                "frame": index,
                "phase": phase["phase"],
                "slot": slot,
                "before": previous[slot],
                "after": raw[slot],
                "step": step,
            }
        previous = raw
    if worst["step"] > args.max_raw_step:
        raise RuntimeError(
            f"planned raw step {worst['step']} exceeds {args.max_raw_step}: {worst}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "amplitude": args.amplitude,
        "cycles": args.cycles,
        "armed": bool(args.arm),
        "motion_sent": False,
        "completed": False,
        "serial": serial,
        "calibration": str(calibration),
        "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
        "search_artifact": str(SEARCH_ARTIFACT),
        "search_artifact_sha256": EXPECTED_SEARCH_SHA256,
        "joint_order16": list(JOINTS),
        "home": list(HOME),
        "waypoints": waypoints(args.amplitude, anchor=start_q),
        "turn_anchor_semantic": start_q if args.mode == "turn" else None,
        "start_raw20": start_raw,
        "start_semantic": start_q,
        "faults20_before": faults_before,
        "temperature20_before": temperatures_before,
        "tactile_mass5_before": tactile_before,
        "frame_count": len(phases),
        "nominal_duration_s": len(phases) / args.rate_hz,
        "worst_planned_raw_step": worst,
        "limits": {
            "max_raw_step": args.max_raw_step,
            "max_temperature_c": args.max_temperature_c,
            "max_tactile_mass": args.max_tactile_mass,
            "max_tactile_rise": args.max_tactile_rise,
        },
        "commands": [],
        "health_gates": [],
        "tactile_checks": [],
        "rollback_commands": [],
    }
    _write_json(args.out_dir / "preflight.json", result)

    if not args.arm:
        result["completed"] = True
        result["offline_or_read_only"] = True
        _write_json(args.out_dir / "trajectory_log.json", result)
        print(
            f"[preflight] {args.mode}: {len(phases)} frames, "
            f"{len(phases) / args.rate_hz:.2f}s, max_raw_step={worst['step']}; "
            "motion_sent=False",
            flush=True,
        )
        if api is not None:
            close = getattr(api.hand, "close_can_interface", None)
            if callable(close):
                close()
        return 0

    assert api is not None
    from tools.run_g20_index_pip_candidate_step import (
        _fault_snapshot,
        _read_state20,
        _temperature_snapshot,
    )
    from tools.record_g20_sdk_snapshot import _tactile_masses
    from screwdriver_rl.deploy import hw_utils

    def rollback(command_index: int) -> None:
        for source_index in range(command_index - 1, -1, -1):
            time.sleep(1.0 / args.rate_hz)
            raw = start_raw if source_index == 0 else raw_frames[source_index - 1]
            api.finger_move(pose=raw)
            result["rollback_commands"].append(
                {
                    "source_index": source_index,
                    "wall_time_ns": time.time_ns(),
                    "target_raw20": raw,
                }
            )

    try:
        api.set_speed(speed=[args.speed] * 5)
        time.sleep(0.1)
        next_due = time.monotonic() + 0.2
        last_index = 0
        for index, (phase, raw) in enumerate(zip(phases, raw_frames)):
            last_index = index
            remaining = next_due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            api.finger_move(pose=raw)
            result["motion_sent"] = True
            result["commands"].append(
                {
                    "index": index,
                    "phase": phase["phase"],
                    "wall_time_ns": time.time_ns(),
                    "target_semantic": phase["semantic"],
                    "target_raw20": raw,
                }
            )
            next_due = time.monotonic() + 1.0 / args.rate_hz

            if index % args.tactile_poll_frames == 0 or phase["gate"]:
                tactile = _tactile_masses(api)
                result["tactile_checks"].append(
                    {
                        "after_command_index": index,
                        "phase": phase["phase"],
                        "tactile_mass5": tactile,
                        "wall_time_ns": time.time_ns(),
                    }
                )
                if tactile is None:
                    raise RuntimeError("invalid tactile telemetry")
                worst_mass = max(tactile)
                worst_rise = max(
                    value - baseline
                    for value, baseline in zip(tactile, tactile_before)
                )
                if (
                    worst_mass > args.max_tactile_mass
                    or worst_rise > args.max_tactile_rise
                ):
                    raise RuntimeError(
                        f"tactile gate: mass={worst_mass:.1f}, rise={worst_rise:.1f}"
                    )
                next_due = time.monotonic() + 1.0 / args.rate_hz

            if phase["gate"]:
                fault_groups, faults = _fault_snapshot(api)
                temperatures = _temperature_snapshot(api)
                state = _read_state20(api, hw_utils)
                result["health_gates"].append(
                    {
                        "after_command_index": index,
                        "phase": phase["phase"],
                        "faults_by_finger": fault_groups,
                        "faults20": faults,
                        "temperature20": temperatures,
                        "state_raw20": state,
                        "wall_time_ns": time.time_ns(),
                    }
                )
                if any(faults):
                    raise RuntimeError(f"fault gate: {faults}")
                if max(temperatures) > args.max_temperature_c:
                    raise RuntimeError("temperature gate")
                next_due = time.monotonic() + 1.0 / args.rate_hz

        settled_rows = []
        for _ in range(7):
            settled_rows.append(_read_state20(api, hw_utils))
            time.sleep(0.05)
        settled_raw = _median20(settled_rows)
        result["settled_raw20"] = settled_raw
        result["settled_semantic"] = [
            float(value) for value in sdkmap.sdk_range_to_joints16(settled_raw)
        ]
        result["completed"] = True
        _write_json(args.out_dir / "trajectory_log.json", result)
        print(
            f"[hardware] completed {args.mode}, amplitude={args.amplitude}, "
            f"cycles={args.cycles}, frames={len(phases)}",
            flush=True,
        )
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            rollback(last_index)
        except BaseException as rollback_exc:
            result["rollback_error"] = (
                f"{type(rollback_exc).__name__}: {rollback_exc}"
            )
        _write_json(args.out_dir / "trajectory_log.json", result)
        raise
    finally:
        close = getattr(api.hand, "close_can_interface", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
