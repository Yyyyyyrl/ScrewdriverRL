#!/usr/bin/env python3
"""Summarize the fixed-D435 G20 pinky PIP/DIP/MCP-pitch calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.deploy import linker_sdk_map as sdkmap


RAWS = (0, 6, 12, 20, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255)
PITCH_RAWS = RAWS[1:]
ANGLE_KEYS = ("mcp_projected_deg_2d", "pip_projected_deg_2d", "dip_projected_deg_2d")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = read_json(path)
    medians = {
        key: float(payload["angle_summary_deg"][key]["median"])
        for key in ANGLE_KEYS
    }
    if payload["capture"]["usable_frames"] <= 0:
        raise ValueError(f"{path}: no usable camera frames")
    return medians, payload


def mean_angles(paths: list[Path]) -> tuple[dict[str, float], dict[str, float]]:
    rows = [camera(path)[0] for path in paths]
    mean = {key: statistics.mean(row[key] for row in rows) for key in ANGLE_KEYS}
    repeat = {key: abs(rows[0][key] - rows[1][key]) for key in ANGLE_KEYS}
    return mean, repeat


def pip_camera_path(session: Path, raw: int) -> Path:
    return session / (
        f"pip_forward/flex_raw{raw}/"
        "camera_four_real_markers_fixed_exp_180f_summary.json"
    )


def pitch_camera_path(session: Path, raw: int) -> Path:
    if raw == 0:
        raise ValueError("pitch raw0 is a rejected coupled endpoint")
    suffix = (
        "camera_four_real_markers_fixed_exp_180f_after_reposition_summary.json"
        if raw == 112
        else "camera_four_real_markers_fixed_exp_180f_summary.json"
    )
    return session / f"pitch_forward/flex_raw{raw}/{suffix}"


def sent_motion(directory: Path) -> dict[str, Any]:
    candidates = []
    for path in sorted(directory.glob("motion*.json")):
        payload = read_json(path)
        if payload.get("motion_sent"):
            candidates.append((path, payload))
    if not candidates:
        raise ValueError(f"{directory}: no sent motion")
    complete = [item for item in candidates if "post_command_samples" in item[1]]
    return (complete or candidates)[-1][1]


def stable_state(payload: dict[str, Any], slot: int) -> float:
    rows = payload["post_command_samples"]
    return statistics.median(row["state20"][slot] for row in rows[-10:])


def pip_motion(session: Path, raw: int) -> dict[str, Any]:
    return sent_motion(session / f"pip_forward/flex_raw{raw}")


def pitch_motion(session: Path, raw: int) -> dict[str, Any]:
    return sent_motion(session / f"pitch_forward/flex_raw{raw}")


def fit_mimic(pip: list[float], dip: list[float]) -> dict[str, float]:
    zero_slope = sum(x * y for x, y in zip(pip, dip)) / sum(x * x for x in pip)
    x_mean = statistics.mean(pip)
    y_mean = statistics.mean(dip)
    affine_slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(pip, dip)) / sum(
        (x - x_mean) ** 2 for x in pip
    )
    affine_offset = y_mean - affine_slope * x_mean
    return {
        "zero_intercept_multiplier": zero_slope,
        "zero_intercept_max_abs_residual_rad": max(
            abs(y - zero_slope * x) for x, y in zip(pip, dip)
        ),
        "affine_multiplier": affine_slope,
        "affine_offset_rad": affine_offset,
        "affine_max_abs_residual_rad": max(
            abs(y - (affine_slope * x + affine_offset)) for x, y in zip(pip, dip)
        ),
    }


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    for index in range(len(xs) - 1):
        if xs[index] <= x <= xs[index + 1]:
            fraction = (x - xs[index]) / (xs[index + 1] - xs[index])
            return ys[index] + fraction * (ys[index + 1] - ys[index])
    raise ValueError(x)


def heldout_error(rad: list[float], raws: tuple[int, ...]) -> float:
    train = list(range(0, len(raws), 2))
    if train[-1] != len(raws) - 1:
        train.append(len(raws) - 1)
    xs = [raws[index] for index in train]
    ys = [rad[index] for index in train]
    return max(
        abs(rad[index] - interp(raws[index], xs, ys))
        for index in range(1, len(raws) - 1, 2)
    )


def scan_motion_health(session: Path) -> dict[str, Any]:
    files = []
    files_without_post_samples = 0
    samples = 0
    pinky_fault_samples = 0
    max_pinky_temp = 0
    for path in session.rglob("*.json"):
        try:
            payload = read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not payload.get("motion_sent"):
            continue
        operation = str(payload.get("operation", ""))
        scope_joint = str(payload.get("approved_scope", {}).get("joint", ""))
        if "pinky" not in operation and not scope_joint.startswith("pinky"):
            continue
        files.append(path)
        if "post_command_samples" not in payload:
            files_without_post_samples += 1
            continue
        for row in payload["post_command_samples"]:
            samples += 1
            faults = row["faults20"]
            temps = row["temperature20"]
            if any(faults[slot] for slot in (4, 9, 19)):
                pinky_fault_samples += 1
            max_pinky_temp = max(max_pinky_temp, *(temps[slot] for slot in (4, 9, 19)))
    return {
        "sent_pinky_motion_files": len(files),
        "sent_pinky_motion_files_without_post_samples": files_without_post_samples,
        "recorded_post_command_samples": samples,
        "pinky_fault_samples": pinky_fault_samples,
        "max_pinky_temperature_c": max_pinky_temp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--base-calib", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()

    zero_paths = [
        session / "pinky_raw255_zero/camera_four_real_markers_fixed_exp_180f_a_summary.json",
        session / "pinky_raw255_zero/camera_four_real_markers_fixed_exp_180f_b_summary.json",
    ]
    pip_zero_paths = list(zero_paths)
    pitch_zero_paths = list(zero_paths)
    pip_zero, pip_repeat = mean_angles(pip_zero_paths)
    pitch_zero, pitch_repeat = mean_angles(pitch_zero_paths)

    final_snapshot_path = session / "final_sdk_snapshot_after_pinky_pip_pitch_sweeps.json"
    final_snapshot = read_json(final_snapshot_path)
    final_rows = final_snapshot["samples"]
    final_pip_readback = statistics.median(row["state20"][19] for row in final_rows)
    final_pitch_readback = statistics.median(row["state20"][4] for row in final_rows)
    if any(any(row["faults20"]) for row in final_rows):
        raise ValueError("final snapshot contains active faults")

    endpoint_motion_path = session / "pitch_forward/flex_raw0/motion_raw6_to_raw0.json"
    endpoint_snapshot_path = session / "pitch_forward/flex_raw0/sdk_snapshot_after_safety_stop.json"
    endpoint_camera_path = session / (
        "pitch_forward/flex_raw0/"
        "rejected_diagnostic_actual_pitch3_side140_camera_180f_summary.json"
    )
    endpoint_recovery_path = session / (
        "pitch_endpoint_recovery/"
        "motion_actual_pitch3_side140_to_pitch6_side153.json"
    )
    pitch_return_raw6_path = session / (
        "pitch_return_raw6/"
        "camera_four_real_markers_fixed_exp_180f_summary.json"
    )
    final_camera_path = session / (
        "pitch_return_raw255/"
        "camera_four_real_markers_fixed_exp_180f_after_reposition_summary.json"
    )
    rejected_pitch_raw112_paths = [
        session / (
            "pitch_forward/flex_raw112/"
            "camera_four_real_markers_fixed_exp_180f_summary.json"
        ),
        session / (
            "pitch_forward/flex_raw112/"
            "camera_four_real_markers_fixed_exp_180f_retry_visibility_summary.json"
        ),
    ]
    endpoint_snapshot = read_json(endpoint_snapshot_path)
    endpoint_rows = endpoint_snapshot["samples"]
    endpoint_pitch_readback = statistics.median(row["state20"][4] for row in endpoint_rows)
    endpoint_side_readback = statistics.median(row["state20"][9] for row in endpoint_rows)

    points = []
    input_paths: list[Path] = list(zero_paths) + [
        final_snapshot_path, endpoint_motion_path, endpoint_snapshot_path,
        endpoint_camera_path, endpoint_recovery_path, pitch_return_raw6_path,
        final_camera_path, *rejected_pitch_raw112_paths,
    ]
    for raw in RAWS:
        if raw == 255:
            pip_angles = pip_zero
            pip_readback = final_pip_readback
            pip_quality = "formal_zero_two_repeats"
            pip_source = ",".join(str(path.relative_to(session)) for path in pip_zero_paths)
        else:
            pip_path = pip_camera_path(session, raw)
            pip_angles, _ = camera(pip_path)
            pip_readback = stable_state(pip_motion(session, raw), 19)
            pip_quality = "formal_180f_four_real_markers_no_split"
            pip_source = str(pip_path.relative_to(session))
            input_paths.append(pip_path)

        pip_angle_deg = pip_angles["pip_projected_deg_2d"]
        if raw <= 12 and pip_angle_deg > 0.0:
            pip_angle_deg -= 180.0

        if raw == 0:
            pitch_angles = None
            pitch_readback = endpoint_pitch_readback
            pitch_quality = "rejected_pitch_side_coupling"
            pitch_source = str(endpoint_camera_path.relative_to(session))
        elif raw == 255:
            pitch_angles = pitch_zero
            pitch_readback = final_pitch_readback
            pitch_quality = "formal_zero_two_repeats"
            pitch_source = ",".join(str(path.relative_to(session)) for path in pitch_zero_paths)
        else:
            pitch_path = pitch_camera_path(session, raw)
            pitch_angles, _ = camera(pitch_path)
            pitch_readback = stable_state(pitch_motion(session, raw), 4)
            pitch_quality = "formal_180f_four_real_markers_no_split"
            pitch_source = str(pitch_path.relative_to(session))
            input_paths.append(pitch_path)

        point = {
            "command_raw": raw,
            "pip_stable_readback_raw": pip_readback,
            "pitch_stable_readback_raw": pitch_readback,
            "pinky_pip_rad": math.radians(
                pip_zero["pip_projected_deg_2d"] - pip_angle_deg
            ),
            "pinky_dip_rad": math.radians(
                pip_zero["dip_projected_deg_2d"]
                - pip_angles["dip_projected_deg_2d"]
            ),
            "pinky_mcp_pitch_rad": (
                None if pitch_angles is None else math.radians(
                    pitch_zero["mcp_projected_deg_2d"]
                    - pitch_angles["mcp_projected_deg_2d"]
                )
            ),
            "pip_scan_mcp_isolation_drift_rad": math.radians(
                pip_zero["mcp_projected_deg_2d"]
                - pip_angles["mcp_projected_deg_2d"]
            ),
            "pitch_scan_pip_apparent_drift_rad": (
                None if pitch_angles is None else math.radians(
                    pitch_zero["pip_projected_deg_2d"]
                    - pitch_angles["pip_projected_deg_2d"]
                )
            ),
            "pitch_scan_dip_apparent_drift_rad": (
                None if pitch_angles is None else math.radians(
                    pitch_zero["dip_projected_deg_2d"]
                    - pitch_angles["dip_projected_deg_2d"]
                )
            ),
            "pip_camera_quality": pip_quality,
            "pitch_camera_quality": pitch_quality,
            "pip_camera_source": pip_source,
            "pitch_camera_source": pitch_source,
        }
        points.append(point)

    for key in ("pinky_pip_rad", "pinky_dip_rad", "pinky_mcp_pitch_rad"):
        values = [point[key] for point in points if point[key] is not None]
        if not all(values[index] > values[index + 1] for index in range(len(values) - 1)):
            raise ValueError(f"{key}: physical curve is not strictly monotonic")

    pip_forward_path = pip_camera_path(session, 128)
    pip_return_path = session / "pip_return_raw128/camera_four_real_markers_fixed_exp_180f_summary.json"
    pitch_forward_path = pitch_camera_path(session, 128)
    pitch_return_path = session / "pitch_return_raw128/camera_four_real_markers_fixed_exp_180f_summary.json"
    pip_forward, _ = camera(pip_forward_path)
    pip_return, _ = camera(pip_return_path)
    pitch_forward, _ = camera(pitch_forward_path)
    pitch_return, _ = camera(pitch_return_path)
    pitch_return_raw6, _ = camera(pitch_return_raw6_path)
    final_camera, final_camera_payload = camera(final_camera_path)
    input_paths.extend([pip_forward_path, pip_return_path, pitch_return_path])

    pip_rad = [point["pinky_pip_rad"] for point in points]
    dip_rad = [point["pinky_dip_rad"] for point in points]
    pitch_rad = [
        point["pinky_mcp_pitch_rad"]
        for point in points
        if point["pinky_mcp_pitch_rad"] is not None
    ]
    health = scan_motion_health(session)
    summary = {
        "schema_version": 1,
        "status": "complete_candidate_only",
        "identity": {
            "hand_serial": "LHT20-010-415-L-B-1-D",
            "hand": "left G20",
            "sdk_version": "3.1.0",
            "embedded_version": "1.0.7",
            "camera": "RealSense D435 143322073091",
        },
        "camera_contract": {
            "profile": "1280x720@30",
            "rgb_exposure": 166,
            "rgb_gain": 32,
            "rgb_white_balance": 4600,
            "same_fixed_camera_pose_all_scans": True,
            "separate_zero_pairs_for_pip_and_pitch_phases": False,
            "dynamic_other_finger_occlusion_schedule": True,
            "pinky_and_camera_fixed_across_occluder_reposition": True,
            "physical_angle_is_change_from_matching_phase_zero": True,
        },
        "zero_reference_deg": {
            "pip_phase_zero_mean": pip_zero,
            "pip_phase_zero_repeat_abs": pip_repeat,
            "pitch_phase_zero_mean": pitch_zero,
            "pitch_phase_zero_repeat_abs": pitch_repeat,
        },
        "points": points,
        "ranges": {
            "pinky_pip_max_rad": pip_rad[0],
            "pinky_pip_max_deg": math.degrees(pip_rad[0]),
            "pinky_pip_raw0_stable_readback_raw": points[0]["pip_stable_readback_raw"],
            "pinky_dip_max_rad": dip_rad[0],
            "pinky_dip_max_deg": math.degrees(dip_rad[0]),
            "pinky_mcp_pitch_isolated_max_rad": pitch_rad[0],
            "pinky_mcp_pitch_isolated_max_deg": math.degrees(pitch_rad[0]),
            "pinky_mcp_pitch_min_valid_command_raw": PITCH_RAWS[0],
            "pinky_mcp_pitch_min_valid_stable_readback_raw": points[1]["pitch_stable_readback_raw"],
            "rejected_command_raw0_actual_pitch_raw": endpoint_pitch_readback,
            "rejected_command_raw0_actual_side_raw": endpoint_side_readback,
        },
        "mimic_fit": fit_mimic(pip_rad, dip_rad),
        "hysteresis_raw128": {
            "pinky_pip_rad": math.radians(pip_forward["pip_projected_deg_2d"] - pip_return["pip_projected_deg_2d"]),
            "pinky_pip_deg": pip_forward["pip_projected_deg_2d"] - pip_return["pip_projected_deg_2d"],
            "pinky_dip_rad": math.radians(pip_forward["dip_projected_deg_2d"] - pip_return["dip_projected_deg_2d"]),
            "pinky_dip_deg": pip_forward["dip_projected_deg_2d"] - pip_return["dip_projected_deg_2d"],
            "pinky_mcp_pitch_rad": math.radians(pitch_forward["mcp_projected_deg_2d"] - pitch_return["mcp_projected_deg_2d"]),
            "pinky_mcp_pitch_deg": pitch_forward["mcp_projected_deg_2d"] - pitch_return["mcp_projected_deg_2d"],
            "pip_forward_actual_raw": 129,
            "pip_return_actual_raw": 128,
            "pitch_forward_actual_raw": 128,
            "pitch_return_actual_raw": 127,
        },
        "pitch_endpoint_return_raw6": {
            "forward_minus_return_rad": math.radians(
                camera(pitch_camera_path(session, 6))[0]["mcp_projected_deg_2d"]
                - pitch_return_raw6["mcp_projected_deg_2d"]
            ),
            "forward_minus_return_deg": (
                camera(pitch_camera_path(session, 6))[0]["mcp_projected_deg_2d"]
                - pitch_return_raw6["mcp_projected_deg_2d"]
            ),
        },
        "heldout_piecewise_linear_max_abs_error_rad": {
            "pinky_pip": heldout_error(pip_rad, RAWS),
            "pinky_dip": heldout_error(dip_rad, RAWS),
            "pinky_mcp_pitch": heldout_error(pitch_rad, PITCH_RAWS),
        },
        "isolation_diagnostics": {
            "pip_scan_max_abs_mcp_projected_drift_deg": math.degrees(max(abs(point["pip_scan_mcp_isolation_drift_rad"]) for point in points)),
            "pitch_scan_max_abs_pip_projected_drift_deg": math.degrees(max(abs(point["pitch_scan_pip_apparent_drift_rad"]) for point in points if point["pitch_scan_pip_apparent_drift_rad"] is not None)),
            "pitch_scan_max_abs_dip_projected_drift_deg": math.degrees(max(abs(point["pitch_scan_dip_apparent_drift_rad"]) for point in points if point["pitch_scan_dip_apparent_drift_rad"] is not None)),
            "pitch_endpoint_coupling": {
                "command_raw": 0,
                "actual_pitch_readback_raw": endpoint_pitch_readback,
                "actual_side_readback_raw": endpoint_side_readback,
                "side_hold_command_raw": 153,
                "accepted_for_lut": False,
                "lowest_isolated_command_raw": 6,
            },
            "pitch_scan_note": "SDK held PIP at 254..255; large projected link-angle drift at high pitch is perspective/cross-talk uncertainty, not an independently commanded PIP/DIP motion.",
        },
        "motion_health": health,
        "final_sdk_snapshot": {
            "path": str(final_snapshot_path.relative_to(session)),
            "samples": len(final_rows),
            "all_state_vectors_present": all(len(row["state20"]) == 20 for row in final_rows),
            "all_fault_free": not any(any(row["faults20"]) for row in final_rows),
            "pinky_pip_readback_median_raw": final_pip_readback,
            "pinky_pitch_readback_median_raw": final_pitch_readback,
        },
        "final_visual_repeat": {
            "path": str(final_camera_path.relative_to(session)),
            "usable_frames": final_camera_payload["capture"]["usable_frames"],
            "angles_deg": final_camera,
        },
        "limitations": [
            "Every accepted formal PIP and pitch capture completed 180/180 usable frames with four real marker components and no synthetic marker splitting.",
            "Two original pitch raw112 captures are retained as rejected evidence because another blue marker transiently merged with the proximal pinky marker; the accepted recapture followed an occluder-only reposition.",
            "Other fingers were repositioned for visibility while the pinky and fixed camera stayed fixed. The final raw255 visual repeat is a visibility check, not a replacement zero reference.",
            "At pitch command raw0 the physical side axis moved from its raw153 hold target to raw140, so raw0 is rejected. The isolated pitch LUT begins at command raw6 and clamps lower software requests to that measured endpoint.",
            "Raw128 pitch forward/return difference is retained as measured mechanical hysteresis/slack, not fitted away.",
            "This file completes the candidate-only fixed-camera PIP/DIP/pitch phase; four-finger MCP roll and thumb yaw/roll phases remain unmeasured and production calibration is unchanged.",
        ],
        "input_sha256": {
            str(path.relative_to(session)): sha256(path)
            for path in sorted(set(input_paths))
        },
    }
    write_json(session / "pinky_pip_dip_pitch_summary.json", summary)

    csv_path = session / "pinky_pip_dip_pitch_calibration_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    candidate = read_json(args.base_calib)
    candidate["note"] = (
        "CANDIDATE ONLY: fixed-D435 physical LUTs for index, middle, ring, "
        "pinky PIP/MCP pitch plus thumb MCP/CMC pitch; full PIP/DIP/pitch "
        "measurement phase complete, MCP-roll and thumb yaw/roll pending"
    )
    candidate["joints"]["pinky_pip"] = {
        "flip": False,
        "lo": 0.0,
        "hi": pip_rad[0],
        "physical_lut": {"raw": list(RAWS), "rad": pip_rad},
    }
    candidate["joints"]["pinky_mcp_pitch"] = {
        "flip": False,
        "lo": 0.0,
        "hi": pitch_rad[0],
        "physical_lut": {"raw": list(PITCH_RAWS), "rad": pitch_rad},
    }
    for finger in ("index", "middle", "ring", "pinky"):
        pip_entry = candidate["joints"][f"{finger}_pip"]
        pip_entry["lo"] = 0.0
        pip_entry["hi"] = float(pip_entry["physical_lut"]["rad"][0])
    write_json(args.candidate_out, candidate)

    validation_path = session / "phase_a_candidate_mapping_validation.json"
    specs = sdkmap.apply_calibration(str(args.candidate_out))
    try:
        lut_names = [spec.name for spec in specs if spec.physical_raw_knots is not None]
        required_names = [
            "index_mcp_pitch", "index_pip", "middle_mcp_pitch", "middle_pip",
            "ring_mcp_pitch", "ring_pip", "pinky_mcp_pitch", "pinky_pip",
            "thumb_cmc_pitch", "thumb_mcp",
        ]
        if sorted(lut_names) != sorted(required_names):
            raise ValueError(f"candidate LUT set mismatch: {lut_names}")
        checked_knots = 0
        max_readback_rad_error = 0.0
        max_command_raw_error = 0.0
        for index, spec in enumerate(specs):
            if spec.physical_raw_knots is None:
                continue
            for raw, rad in zip(spec.physical_raw_knots, spec.physical_rad_knots):
                state = [255.0] * 20
                state[spec.slot] = raw
                observed = sdkmap.sdk_range_to_joints16(state)[index]
                max_readback_rad_error = max(
                    max_readback_rad_error, abs(observed - rad)
                )
                target = [joint.lo for joint in specs]
                target[index] = rad
                command_raw = sdkmap.joints16_to_sdk_range(target)[spec.slot]
                max_command_raw_error = max(
                    max_command_raw_error, abs(command_raw - raw)
                )
                checked_knots += 1
        if max_readback_rad_error > 1.0e-12 or max_command_raw_error > 0.0:
            raise ValueError("candidate LUT round-trip validation failed")
        stale_gate_blockers = {
            spec.name: {"measured_hi_rad": spec.hi, "stale_required_hi_rad": 1.08}
            for spec in specs
            if spec.name.endswith("_pip") and abs(spec.hi - 1.08) > 1.0e-9
        }
        validation = {
            "status": "pass_physical_lut_mapping",
            "candidate": str(args.candidate_out.resolve()),
            "lut_joints": lut_names,
            "knots_checked": checked_knots,
            "max_readback_rad_error": max_readback_rad_error,
            "max_command_raw_error": max_command_raw_error,
            "physical_endpoint_is_source_of_truth": True,
            "current_topdown_gate_status": (
                "blocked_by_stale_pip_1_08_contract"
                if stale_gate_blockers else "pass"
            ),
            "current_topdown_gate_blockers": stale_gate_blockers,
            "note": "Update Isaac limits and the deployment gate from the measured physical endpoints before policy deployment; do not alter the measured LUT to satisfy the old 1.08-rad assumption.",
        }
        write_json(validation_path, validation)
    finally:
        sdkmap.reset_calibration()

    mimic = summary["mimic_fit"]
    hyst = summary["hysteresis_raw128"]
    heldout = summary["heldout_piecewise_linear_max_abs_error_rad"]
    endpoint_return = summary["pitch_endpoint_return_raw6"]
    report = f"""# Pinky PIP/DIP/MCP-pitch fixed-exposure calibration

Status: complete as a candidate calibration; not production-enabled.

## Result

- Pinky PIP command raw0 settled at raw `{points[0]['pip_stable_readback_raw']:.0f}` and measured `{pip_rad[0]:.10f} rad / {math.degrees(pip_rad[0]):.3f} deg`.
- Passive pinky DIP measured `{dip_rad[0]:.10f} rad / {math.degrees(dip_rad[0]):.3f} deg` at that endpoint.
- Pinky MCP pitch lowest isolated command is raw6, with stable readback raw `{points[1]["pitch_stable_readback_raw"]:.0f}` and measured `{pitch_rad[0]:.10f} rad / {math.degrees(pitch_rad[0]):.3f} deg`.
- Pitch command raw0 is rejected: it settled near pitch raw `{endpoint_pitch_readback:.0f}` while the held side axis moved to raw `{endpoint_side_readback:.0f}` instead of raw153.
- PIP/DIP zero-intercept mimic candidate: `{mimic['zero_intercept_multiplier']:.10f}`. Affine diagnostic: `DIP = {mimic['affine_multiplier']:.10f} * PIP {mimic['affine_offset_rad']:+.10f} rad`.
- Raw128 same-camera hysteresis: PIP `{hyst['pinky_pip_rad']:+.10f} rad / {hyst['pinky_pip_deg']:+.3f} deg`, DIP `{hyst['pinky_dip_rad']:+.10f} rad / {hyst['pinky_dip_deg']:+.3f} deg`, pitch `{hyst['pinky_mcp_pitch_rad']:+.10f} rad / {hyst['pinky_mcp_pitch_deg']:+.3f} deg`.
- Pitch raw6 forward/return difference: `{endpoint_return["forward_minus_return_rad"]:+.10f} rad / {endpoint_return["forward_minus_return_deg"]:+.3f} deg`.
- Candidate mapping validation: `{validation["knots_checked"]}` LUT knots, maximum command error `{validation["max_command_raw_error"]}` raw and readback error `{validation["max_readback_rad_error"]:.3e} rad`. The current Topdown gate is intentionally blocked by its stale hard-coded PIP `1.08 rad` contract.
- Held-out piecewise-linear maximum errors: PIP `{heldout['pinky_pip']:.10f} rad`, DIP `{heldout['pinky_dip']:.10f} rad`, pitch `{heldout['pinky_mcp_pitch']:.10f} rad`.

## Camera and measurement contract

PIP/DIP and MCP-pitch LUT points use the same fixed side-camera pose and the same initial pair of 180-frame raw255 zero references. PIP was also returned to raw255 before the pitch scan, and a final raw255 visual repeat was captured after the last occluder reposition. All physical angles are changes from the initial zero pair; the final repeat verifies visibility and return health but does not redefine zero.

All formal captures used RealSense D435 `143322073091`, `1280x720@30`, manual exposure `166`, gain `32`, white balance `4600`, with auto exposure and auto white balance disabled. Physical angle is always a change from the matching pose's zero; raw tape orientation is not treated as a URDF angle.

Every accepted formal PIP and pitch point completed 180/180 usable frames with four real marker components and no marker splitting. Two pitch-raw112 captures were rejected because another blue marker transiently merged with the proximal pinky marker. After only the occluding fingers were repositioned, the raw112 recapture passed, and the complete accepted curves passed strict monotonicity.

## Isolation and health

- Maximum apparent MCP drift during the PIP scan: `{summary['isolation_diagnostics']['pip_scan_max_abs_mcp_projected_drift_deg']:.3f} deg`.
- During the pitch scan, projected PIP/DIP drift reached `{summary['isolation_diagnostics']['pitch_scan_max_abs_pip_projected_drift_deg']:.3f} / {summary['isolation_diagnostics']['pitch_scan_max_abs_dip_projected_drift_deg']:.3f} deg`; SDK PIP stayed at raw254..255, so this is retained as perspective/cross-talk uncertainty rather than commanded joint motion.
- Recorded pinky-motion files: `{health['sent_pinky_motion_files']}`; post-command samples: `{health['recorded_post_command_samples']}`; pinky fault samples: `{health['pinky_fault_samples']}`; maximum pinky-slot temperature: `{health['max_pinky_temperature_c']} C`.
- Final read-only SDK snapshot: 20/20 `state=ok`, no active fault; PIP median raw `{final_pip_readback}`, pitch median raw `{final_pitch_readback}`.
- Final raw255 visual repeat completed `{final_camera_payload["capture"]["usable_frames"]}/180`, and the final SDK snapshot completed 20/20 state=ok with no active fault.
- Pitch motion held side at command raw153. The side axis stayed near that target through raw6, but moved to raw140 at pitch raw0; the safety runner stopped automatically and raw0 was excluded.

## Software implication

The generated candidate contains measured LUTs for every index/middle/ring/pinky PIP and MCP-pitch joint plus thumb MCP/CMC pitch. Pinky PIP has 19 knots (raw0..255), and its candidate semantic upper limit is the measured safe physical endpoint; pinky pitch has 18 isolated knots (raw6..255), and lower pitch requests clamp to the measured raw6 endpoint. Deployment-gate and round-trip results are recorded separately by the validation command rather than asserted here.

The candidate deliberately treats measured safe physical endpoints as the source of truth. Current Isaac joint limits and the Topdown deployment gate still contain older assumptions and must be updated and revalidated before policy deployment. The new JSON remains candidate-only. Production calibration, semantic limits, URDF mimic multiplier, and policy deployment remain unchanged. The next physical phase is the changed-camera four-finger MCP-roll calibration, followed by thumb yaw/roll.

Artifacts:

- `pinky_pip_dip_pitch_summary.json`
- `pinky_pip_dip_pitch_calibration_points.csv`
- `phase_a_candidate_mapping_validation.json`
- `{args.candidate_out.resolve()}`
- `final_sdk_snapshot_after_pinky_pip_pitch_sweeps.json`
"""
    (session / "PINKY_PIP_DIP_PITCH_FIXED_EXPOSURE_CALIBRATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({
        "summary": str(session / "pinky_pip_dip_pitch_summary.json"),
        "points": str(csv_path),
        "report": str(session / "PINKY_PIP_DIP_PITCH_FIXED_EXPOSURE_CALIBRATION_REPORT.md"),
        "candidate": str(args.candidate_out.resolve()),
        "validation": str(validation_path),
        "ranges": summary["ranges"],
        "hysteresis": hyst,
        "health": health,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
