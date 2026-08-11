#!/usr/bin/env python3
"""Summarize the fixed-D435 G20 middle PIP/DIP/MCP-pitch calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


RAWS = (0, 6, 12, 20, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255)
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
    if raw == 0:
        return session / "resume_after_manual_index_clearance/pip_flex_raw000/camera_actual_raw003_angle_prior_fixed_exp_180f_summary.json"
    if raw == 6:
        return session / "resume_after_manual_index_clearance/pip_flex_raw006/camera_angle_prior_fixed_exp_180f_lower_confidence_summary.json"
    if raw == 12:
        return session / "resume_after_manual_index_clearance/pip_flex_raw012/camera_angle_prior_fixed_exp_180f_summary.json"
    suffix = "camera_angle_prior_fixed_exp_180f_summary.json" if raw <= 128 else "camera_fixed_exp_180f_summary.json"
    return session / f"valid_dynamic_index_schedule/pip_flex_raw{raw:03d}/{suffix}"


def pitch_camera_path(session: Path, raw: int) -> Path:
    suffix = (
        "camera_fixed_exp_180f_after_manual_index_clearance_summary.json"
        if raw <= 64
        else "camera_fixed_exp_180f_summary.json"
    )
    return session / f"middle_pitch_new_camera/flex_raw{raw:03d}/{suffix}"


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
    if raw == 12:
        return read_json(session / "resume_after_manual_index_clearance/descent_to_raw012/motion_raw020_to_raw012.json")
    if raw in (0, 6, 12):
        directory = session / f"resume_after_manual_index_clearance/pip_flex_raw{raw:03d}"
    else:
        directory = session / f"valid_dynamic_index_schedule/pip_flex_raw{raw:03d}"
    return sent_motion(directory)


def pitch_motion(session: Path, raw: int) -> dict[str, Any]:
    return sent_motion(session / f"middle_pitch_new_camera/flex_raw{raw:03d}")


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


def heldout_error(rad: list[float]) -> float:
    train = list(range(0, len(RAWS), 2))
    xs = [RAWS[index] for index in train]
    ys = [rad[index] for index in train]
    return max(
        abs(rad[index] - interp(RAWS[index], xs, ys))
        for index in range(1, len(RAWS) - 1, 2)
    )


def scan_motion_health(session: Path) -> dict[str, Any]:
    files = []
    files_without_post_samples = 0
    samples = 0
    middle_fault_samples = 0
    max_middle_temp = 0
    for path in session.rglob("*.json"):
        try:
            payload = read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not payload.get("motion_sent"):
            continue
        operation = str(payload.get("operation", ""))
        scope_joint = str(payload.get("approved_scope", {}).get("joint", ""))
        if "middle" not in operation and not scope_joint.startswith("middle"):
            continue
        files.append(path)
        if "post_command_samples" not in payload:
            files_without_post_samples += 1
            continue
        for row in payload["post_command_samples"]:
            samples += 1
            faults = row["faults20"]
            temps = row["temperature20"]
            if any(faults[slot] for slot in (2, 7, 17)):
                middle_fault_samples += 1
            max_middle_temp = max(max_middle_temp, *(temps[slot] for slot in (2, 7, 17)))
    return {
        "sent_middle_motion_files": len(files),
        "sent_middle_motion_files_without_post_samples": files_without_post_samples,
        "recorded_post_command_samples": samples,
        "middle_fault_samples": middle_fault_samples,
        "max_middle_temperature_c": max_middle_temp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--base-calib", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()

    pip_zero_paths = [
        session / "valid_dynamic_index_schedule/pip_raw255_zero_new_palm/camera_fixed_exp_180f_a_summary.json",
        session / "valid_dynamic_index_schedule/pip_raw255_zero_new_palm/camera_fixed_exp_180f_b_summary.json",
    ]
    pitch_zero_paths = [
        session / "camera_reposition_diagnostic/raw255_new_zero_a_fixed_exp_180f_summary.json",
        session / "camera_reposition_diagnostic/raw255_new_zero_b_fixed_exp_180f_summary.json",
    ]
    pip_zero, pip_repeat = mean_angles(pip_zero_paths)
    pitch_zero, pitch_repeat = mean_angles(pitch_zero_paths)

    final_snapshot_path = session / "final_readonly/sdk_snapshot_after_middle_pip_return255.json"
    final_snapshot = read_json(final_snapshot_path)
    final_rows = final_snapshot["samples"]
    final_pip_readback = statistics.median(row["state20"][17] for row in final_rows)
    final_pitch_readback = statistics.median(row["state20"][2] for row in final_rows)
    if any(any(row["faults20"]) for row in final_rows):
        raise ValueError("final snapshot contains active faults")

    points = []
    input_paths: list[Path] = pip_zero_paths + pitch_zero_paths + [final_snapshot_path]
    for raw in RAWS:
        if raw == 255:
            pip_angles = pip_zero
            pitch_angles = pitch_zero
            pip_readback = final_pip_readback
            pitch_readback = final_pitch_readback
            pip_quality = "formal_zero_two_repeats"
            pitch_quality = "formal_zero_two_repeats"
            pip_source = ",".join(str(path.relative_to(session)) for path in pip_zero_paths)
            pitch_source = ",".join(str(path.relative_to(session)) for path in pitch_zero_paths)
        else:
            pip_path = pip_camera_path(session, raw)
            pitch_path = pitch_camera_path(session, raw)
            pip_angles, _ = camera(pip_path)
            pitch_angles, _ = camera(pitch_path)
            pip_payload = pip_motion(session, raw)
            pitch_payload = pitch_motion(session, raw)
            pip_readback = stable_state(pip_payload, 17)
            if raw == 64:
                next_payload = pitch_motion(session, 48)
                pitch_readback = statistics.median(
                    row[2]
                    for row in next_payload["preflight"]["state20_samples"]
                )
            else:
                pitch_readback = stable_state(pitch_payload, 2)
            pip_quality = "lower_confidence_angle_prior" if raw in (0, 6) else "formal_180f"
            pitch_quality = "formal_180f"
            pip_source = str(pip_path.relative_to(session))
            pitch_source = str(pitch_path.relative_to(session))
            input_paths.extend([pip_path, pitch_path])
        point = {
            "command_raw": raw,
            "pip_stable_readback_raw": pip_readback,
            "pitch_stable_readback_raw": pitch_readback,
            "middle_pip_rad": math.radians(pip_zero["pip_projected_deg_2d"] - pip_angles["pip_projected_deg_2d"]),
            "middle_dip_rad": math.radians(pip_zero["dip_projected_deg_2d"] - pip_angles["dip_projected_deg_2d"]),
            "middle_mcp_pitch_rad": math.radians(pitch_zero["mcp_projected_deg_2d"] - pitch_angles["mcp_projected_deg_2d"]),
            "pip_scan_mcp_isolation_drift_rad": math.radians(pip_zero["mcp_projected_deg_2d"] - pip_angles["mcp_projected_deg_2d"]),
            "pitch_scan_pip_apparent_drift_rad": math.radians(pitch_zero["pip_projected_deg_2d"] - pitch_angles["pip_projected_deg_2d"]),
            "pitch_scan_dip_apparent_drift_rad": math.radians(pitch_zero["dip_projected_deg_2d"] - pitch_angles["dip_projected_deg_2d"]),
            "pip_camera_quality": pip_quality,
            "pitch_camera_quality": pitch_quality,
            "pip_camera_source": pip_source,
            "pitch_camera_source": pitch_source,
        }
        points.append(point)

    for key in ("middle_pip_rad", "middle_dip_rad", "middle_mcp_pitch_rad"):
        values = [point[key] for point in points]
        if not all(values[index] > values[index + 1] for index in range(len(values) - 1)):
            raise ValueError(f"{key}: physical curve is not strictly monotonic")

    pip_forward_path = session / "pip_hysteresis_new_camera/camera_raw128_forward_actual130_fixed_exp_180f_summary.json"
    pip_return_path = session / "pip_hysteresis_new_camera/camera_raw128_return_actual127_fixed_exp_180f_summary.json"
    pitch_forward_path = pitch_camera_path(session, 128)
    pitch_return_path = session / "middle_pitch_new_camera/return_to_raw255/camera_raw128_return_fixed_exp_180f_summary.json"
    pip_forward, _ = camera(pip_forward_path)
    pip_return, _ = camera(pip_return_path)
    pitch_forward, _ = camera(pitch_forward_path)
    pitch_return, _ = camera(pitch_return_path)
    input_paths.extend([pip_forward_path, pip_return_path, pitch_return_path])

    pip_rad = [point["middle_pip_rad"] for point in points]
    dip_rad = [point["middle_dip_rad"] for point in points]
    pitch_rad = [point["middle_mcp_pitch_rad"] for point in points]
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
            "old_pose_used_for_pip_lut": True,
            "new_pose_used_for_pitch_lut_and_same_pose_hysteresis": True,
            "absolute_projected_angles_across_camera_poses_must_not_be_mixed": True,
        },
        "zero_reference_deg": {
            "pip_old_camera_mean": pip_zero,
            "pip_old_camera_repeat_abs": pip_repeat,
            "pitch_new_camera_mean": pitch_zero,
            "pitch_new_camera_repeat_abs": pitch_repeat,
        },
        "points": points,
        "ranges": {
            "middle_pip_max_rad": pip_rad[0],
            "middle_pip_max_deg": math.degrees(pip_rad[0]),
            "middle_pip_raw0_stable_readback_raw": points[0]["pip_stable_readback_raw"],
            "middle_dip_max_rad": dip_rad[0],
            "middle_dip_max_deg": math.degrees(dip_rad[0]),
            "middle_mcp_pitch_max_rad": pitch_rad[0],
            "middle_mcp_pitch_max_deg": math.degrees(pitch_rad[0]),
        },
        "mimic_fit": fit_mimic(pip_rad, dip_rad),
        "hysteresis_raw128": {
            "middle_pip_rad": math.radians(pip_forward["pip_projected_deg_2d"] - pip_return["pip_projected_deg_2d"]),
            "middle_pip_deg": pip_forward["pip_projected_deg_2d"] - pip_return["pip_projected_deg_2d"],
            "middle_dip_rad": math.radians(pip_forward["dip_projected_deg_2d"] - pip_return["dip_projected_deg_2d"]),
            "middle_dip_deg": pip_forward["dip_projected_deg_2d"] - pip_return["dip_projected_deg_2d"],
            "middle_mcp_pitch_rad": math.radians(pitch_forward["mcp_projected_deg_2d"] - pitch_return["mcp_projected_deg_2d"]),
            "middle_mcp_pitch_deg": pitch_forward["mcp_projected_deg_2d"] - pitch_return["mcp_projected_deg_2d"],
            "pip_forward_actual_raw": 130,
            "pip_return_actual_raw": 127,
        },
        "heldout_piecewise_linear_max_abs_error_rad": {
            "middle_pip": heldout_error(pip_rad),
            "middle_dip": heldout_error(dip_rad),
            "middle_mcp_pitch": heldout_error(pitch_rad),
        },
        "isolation_diagnostics": {
            "pip_scan_max_abs_mcp_projected_drift_deg": math.degrees(max(abs(point["pip_scan_mcp_isolation_drift_rad"]) for point in points)),
            "pitch_scan_max_abs_pip_projected_drift_deg": math.degrees(max(abs(point["pitch_scan_pip_apparent_drift_rad"]) for point in points)),
            "pitch_scan_max_abs_dip_projected_drift_deg": math.degrees(max(abs(point["pitch_scan_dip_apparent_drift_rad"]) for point in points)),
            "pitch_scan_note": "SDK held PIP at 254..255; large projected link-angle drift at high pitch is perspective/cross-talk uncertainty, not an independently commanded PIP/DIP motion.",
        },
        "motion_health": health,
        "final_sdk_snapshot": {
            "path": str(final_snapshot_path.relative_to(session)),
            "samples": len(final_rows),
            "all_state_vectors_present": all(len(row["state20"]) == 20 for row in final_rows),
            "all_fault_free": not any(any(row["faults20"]) for row in final_rows),
            "middle_pip_readback_median_raw": final_pip_readback,
            "middle_pitch_readback_median_raw": final_pitch_readback,
        },
        "limitations": [
            "PIP raw0/6 image association is accepted with lower confidence; all 180-frame captures still completed.",
            "The camera was repositioned between the PIP LUT and pitch LUT; each uses its own two-repeat zero.",
            "A final visual raw255 repeat after pitch was unusable because the manually placed index occluded markers; motor return and final SDK state are verified.",
            "Candidate calibration is not production-enabled; ring and pinky remain uncalibrated.",
        ],
        "input_sha256": {
            str(path.relative_to(session)): sha256(path)
            for path in sorted(set(input_paths))
        },
    }
    write_json(session / "middle_pip_dip_pitch_summary.json", summary)

    csv_path = session / "middle_pip_dip_pitch_calibration_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    candidate = read_json(args.base_calib)
    candidate["note"] = (
        "CANDIDATE ONLY: fixed-D435 physical LUTs for index and middle "
        "PIP/MCP pitch plus thumb MCP/CMC pitch; ring and pinky incomplete"
    )
    candidate["joints"]["middle_pip"] = {
        "flip": False,
        "lo": 0.0,
        "hi": pip_rad[0],
        "physical_lut": {"raw": list(RAWS), "rad": pip_rad},
    }
    candidate["joints"]["middle_mcp_pitch"] = {
        "flip": False,
        "lo": 0.0,
        "hi": pitch_rad[0],
        "physical_lut": {"raw": list(RAWS), "rad": pitch_rad},
    }
    write_json(args.candidate_out, candidate)

    mimic = summary["mimic_fit"]
    hyst = summary["hysteresis_raw128"]
    heldout = summary["heldout_piecewise_linear_max_abs_error_rad"]
    report = f"""# Middle PIP/DIP/MCP-pitch fixed-exposure calibration

Status: complete as a candidate calibration; not production-enabled.

## Result

- Middle PIP command raw0 settled at raw `{points[0]['pip_stable_readback_raw']:.0f}` and measured `{pip_rad[0]:.10f} rad / {math.degrees(pip_rad[0]):.3f} deg`.
- Passive middle DIP measured `{dip_rad[0]:.10f} rad / {math.degrees(dip_rad[0]):.3f} deg` at that endpoint.
- Middle MCP pitch raw0 measured `{pitch_rad[0]:.10f} rad / {math.degrees(pitch_rad[0]):.3f} deg`; raw0 readback was `{points[0]['pitch_stable_readback_raw']:.0f}`.
- PIP/DIP zero-intercept mimic candidate: `{mimic['zero_intercept_multiplier']:.10f}`. Affine diagnostic: `DIP = {mimic['affine_multiplier']:.10f} * PIP {mimic['affine_offset_rad']:+.10f} rad`.
- Raw128 same-camera hysteresis: PIP `{hyst['middle_pip_rad']:+.10f} rad / {hyst['middle_pip_deg']:+.3f} deg`, DIP `{hyst['middle_dip_rad']:+.10f} rad / {hyst['middle_dip_deg']:+.3f} deg`, pitch `{hyst['middle_mcp_pitch_rad']:+.10f} rad / {hyst['middle_mcp_pitch_deg']:+.3f} deg`.
- Held-out piecewise-linear maximum errors: PIP `{heldout['middle_pip']:.10f} rad`, DIP `{heldout['middle_dip']:.10f} rad`, pitch `{heldout['middle_mcp_pitch']:.10f} rad`.

## Camera and measurement contract

PIP/DIP LUT points use the original fixed side-camera pose and their own two 180-frame raw255 zero references. The camera was then physically repositioned; MCP-pitch points and PIP same-pose hysteresis use the new pose and a separate two-repeat raw255 zero. Absolute projected angles from the two camera poses were never mixed.

All formal captures used RealSense D435 `143322073091`, `1280x720@30`, manual exposure `166`, gain `32`, white balance `4600`, with auto exposure and auto white balance disabled. Physical angle is always a change from the matching pose's zero; raw tape orientation is not treated as a URDF angle.

Raw12 had 176/180 usable frames. Raw6 had 150/180 and raw0 had 135/180; raw6/raw0 remain explicitly lower-confidence angle-prior associations. Their monotonicity, motor state and fault state passed, but they should be visually rechecked if the tape or camera is changed.

## Isolation and health

- Maximum apparent MCP drift during the PIP scan: `{summary['isolation_diagnostics']['pip_scan_max_abs_mcp_projected_drift_deg']:.3f} deg`.
- During the pitch scan, projected PIP/DIP drift reached `{summary['isolation_diagnostics']['pitch_scan_max_abs_pip_projected_drift_deg']:.3f} / {summary['isolation_diagnostics']['pitch_scan_max_abs_dip_projected_drift_deg']:.3f} deg`; SDK PIP stayed at raw254..255, so this is retained as perspective/cross-talk uncertainty rather than commanded joint motion.
- Recorded middle-motion files: `{health['sent_middle_motion_files']}`; post-command samples: `{health['recorded_post_command_samples']}`; middle fault samples: `{health['middle_fault_samples']}`; maximum middle-slot temperature: `{health['max_middle_temperature_c']} C`.
- Final read-only SDK snapshot: 20/20 `state=ok`, no active fault; PIP median raw `{final_pip_readback}`, pitch median raw `{final_pitch_readback}`.
- A final raw255 visual repeat after the pitch return was rejected because the manually positioned index occluded link markers. The initial new-pose double zero remains valid; motor return and final SDK health are independently recorded.

## Software implication

The measured middle PIP physical endpoint is above both the old candidate `1.08 rad` and the current generic `1.57 rad` assumption. The measured middle pitch endpoint is `{pitch_rad[0]:.10f} rad`. The new JSON is deliberately candidate-only: production calibration, semantic limits, URDF mimic multiplier and policy deployment remain unchanged until ring and pinky are measured and the full-hand gate is reviewed.

Artifacts:

- `middle_pip_dip_pitch_summary.json`
- `middle_pip_dip_pitch_calibration_points.csv`
- `{args.candidate_out.resolve()}`
- `final_readonly/sdk_snapshot_after_middle_pip_return255.json`
"""
    (session / "MIDDLE_PIP_DIP_PITCH_FIXED_EXPOSURE_CALIBRATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({
        "summary": str(session / "middle_pip_dip_pitch_summary.json"),
        "points": str(csv_path),
        "report": str(session / "MIDDLE_PIP_DIP_PITCH_FIXED_EXPOSURE_CALIBRATION_REPORT.md"),
        "candidate": str(args.candidate_out.resolve()),
        "ranges": summary["ranges"],
        "hysteresis": hyst,
        "health": health,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
