#!/usr/bin/env python3
"""Summarize the 2026-08-01 G20 index PIP/DIP physical calibration sweep.

The input session is intentionally explicit: every camera CSV and SDK motion
record used in the fit is listed below.  Camera angles are reduced with three
contiguous block medians so the broad tape contours do not make a single-frame
PCA angle look more precise than it is.

Outputs:

* ``index_pip_dip_calibration_points.csv``: one row per measured pose;
* ``index_pip_dip_calibration_fit.json``: provenance, piecewise mappings,
  hysteresis, health gates, and comparisons with the current affine map;
* ``index_pip_dip_calibration_curve.png``: review plot when matplotlib exists;
* ``INDEX_PIP_DIP_CALIBRATION_REPORT.md``: concise, auditable report.

This script is offline-only.  It does not import the hand SDK, open CAN, open
the camera, or command hardware.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PIP_FIELD = "pip_projected_deg_2d"
DIP_FIELD = "dip_projected_deg_2d"
PIP_URDF_UPPER_RAD = 1.08
DIP_URDF_MIMIC_MULTIPLIER = 0.8917
OPERATIONAL_RAW_FLOOR = 20


@dataclass(frozen=True)
class PointSpec:
    direction: str
    target_raw: int
    directory: str
    sdk_file: str
    camera_prefix: str
    quality_role: str = "fit"


POINTS: tuple[PointSpec, ...] = (
    PointSpec("flexing", 240, "step_00_return_225_to_240", "sdk_motion.json", "camera_masked_raw240_official_180f"),
    PointSpec("flexing", 224, "step_01_down_target224", "sdk_motion.json", "camera_masked_raw224_after_torque_fix_180f"),
    PointSpec("flexing", 208, "step_02_down_target208", "sdk_motion.json", "camera_180f"),
    PointSpec("flexing", 192, "step_03_down_target192", "sdk_motion.json", "camera_180f"),
    PointSpec("flexing", 176, "step_04_down_target176", "sdk_motion.json", "camera_180f"),
    PointSpec("flexing", 160, "step_05_down_target160", "sdk_motion.json", "camera_after_root_hold_180f"),
    PointSpec("flexing", 144, "step_06_down_target144", "sdk_motion.json", "camera_180f"),
    PointSpec("flexing", 128, "step_07_down_target128", "sdk_motion.json", "camera_repeat_180f"),
    PointSpec("flexing", 112, "step_08_down_target112", "sdk_motion.json", "camera_expanded_roi_minarea500_official_180f"),
    PointSpec("flexing", 96, "step_09_down_target096", "sdk_motion.json", "camera_180f"),
    PointSpec("flexing", 80, "step_10_down_target080", "sdk_motion.json", "camera_180f"),
    PointSpec("flexing", 64, "step_11_down_target064", "sdk_motion.json", "camera_180f"),
    PointSpec("flexing", 48, "step_12_down_target048", "sdk_motion.json", "camera_anatomical_official_180f"),
    PointSpec("flexing", 32, "step_13_down_target032", "sdk_motion.json", "camera_180f"),
    PointSpec("flexing", 20, "step_14_down_target020", "sdk_motion.json", "camera_final_candidate_official_180f"),
    PointSpec("flexing", 12, "step_15_down_target012", "sdk_motion.json", "camera_diagnostic_180f", "limit_characterization"),
    PointSpec("extending", 20, "return_01_target020", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 32, "return_02_target032", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 48, "return_03_target048", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 64, "return_04_target064", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 80, "return_05_target080", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 96, "return_06_target096", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 112, "return_07_target112", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 128, "return_08_target128", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 144, "return_09_target144", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 160, "return_10_target160", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 176, "return_11_target176", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 192, "return_12_target192", "sdk_motion_retry.json", "camera_180f"),
    PointSpec("extending", 208, "return_13_target208", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 224, "return_14_target224", "sdk_motion.json", "camera_180f"),
    PointSpec("extending", 240, "return_15_target240", "sdk_motion_tolerance_retry.json", "camera_180f"),
    PointSpec("extending", 255, "return_16_target255", "sdk_motion.json", "camera_180f"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def block_median(values: Sequence[float], blocks: int = 3) -> tuple[float, list[float]]:
    if len(values) < blocks:
        raise ValueError(f"need at least {blocks} values, got {len(values)}")
    base, extra = divmod(len(values), blocks)
    out: list[float] = []
    start = 0
    for index in range(blocks):
        width = base + (1 if index < extra else 0)
        chunk = values[start : start + width]
        out.append(median(chunk))
        start += width
    return median(out), out


def load_camera(csv_path: Path, summary_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"camera CSV has no usable rows: {csv_path}")
    pip_values = [float(row[PIP_FIELD]) for row in rows]
    dip_values = [float(row[DIP_FIELD]) for row in rows]
    pip_robust, pip_blocks = block_median(pip_values)
    dip_robust, dip_blocks = block_median(dip_values)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    capture = summary["capture"]
    return {
        "usable_frames": len(rows),
        "requested_frames": int(capture["requested_frames"]),
        "failure_fraction": float(capture["failure_fraction"]),
        "quality_pass_5pct": float(capture["failure_fraction"]) <= 0.05,
        "pip_tape_deg": pip_robust,
        "pip_block_medians_deg": pip_blocks,
        "pip_block_range_deg": max(pip_blocks) - min(pip_blocks),
        "dip_tape_deg": dip_robust,
        "dip_block_medians_deg": dip_blocks,
        "dip_block_range_deg": max(dip_blocks) - min(dip_blocks),
    }


def load_sdk(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("motion_sent"):
        raise ValueError(f"selected SDK record did not send motion: {path}")
    rows = payload["post_command_samples"]
    stable = rows[-10:]
    raw = [float(row["state20"][16]) for row in stable]
    root = [float(row["state20"][1]) for row in stable]
    temp = [float(row["temperature20"][16]) for row in stable]
    all_faults = [
        int(value)
        for row in rows
        for value in row["faults20"]
    ]
    return {
        "sdk_readback_raw": median(raw),
        "sdk_readback_raw_min": min(raw),
        "sdk_readback_raw_max": max(raw),
        "sdk_readback_raw_span": max(raw) - min(raw),
        "root_hold_readback_raw": median(root),
        "index_tip_temp_c": median(temp),
        "index_tip_temp_max_c": max(temp),
        "fault_free": not any(all_faults),
        "motion_sent": True,
        "command_target_error_raw": median(raw) - float(payload["approved_scope"]["target_raw"]),
        "temperature_delta_max_c": max(abs(float(v)) for v in payload["result"]["temperature_delta20"]),
    }


def interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("invalid interpolation table")
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for left in range(len(xs) - 1):
        if xs[left] <= x <= xs[left + 1]:
            frac = (x - xs[left]) / (xs[left + 1] - xs[left])
            return float(ys[left] + frac * (ys[left + 1] - ys[left]))
    raise AssertionError("unreachable interpolation interval")


def isotonic_nonincreasing(values: Sequence[float]) -> list[float]:
    """Least-squares non-increasing projection using unit-weight PAVA."""
    blocks: list[dict[str, float | int]] = []
    for value in values:
        blocks.append({"sum": -float(value), "count": 1})
        while len(blocks) >= 2:
            prev = float(blocks[-2]["sum"]) / int(blocks[-2]["count"])
            curr = float(blocks[-1]["sum"]) / int(blocks[-1]["count"])
            if prev <= curr:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                {
                    "sum": float(left["sum"]) + float(right["sum"]),
                    "count": int(left["count"]) + int(right["count"]),
                }
            )
    result: list[float] = []
    for block in blocks:
        value = -(float(block["sum"]) / int(block["count"]))
        result.extend([value] * int(block["count"]))
    if len(result) != len(values):
        raise AssertionError("PAVA output length mismatch")
    return result


def branch_table(points: Sequence[dict], field: str) -> tuple[list[float], list[float]]:
    grouped: dict[float, list[float]] = {}
    for point in points:
        grouped.setdefault(float(point["sdk_readback_raw"]), []).append(float(point[field]))
    xs = sorted(grouped)
    ys = [median(grouped[x]) for x in xs]
    return xs, isotonic_nonincreasing(ys)


def inverse_nonincreasing(q: float, raw: Sequence[float], angle: Sequence[float]) -> tuple[float, bool]:
    """Invert a non-increasing angle(raw) table; return raw and clamp flag."""
    if q >= angle[0]:
        return float(raw[0]), q > angle[0]
    if q <= angle[-1]:
        return float(raw[-1]), q < angle[-1]
    for left in range(len(raw) - 1):
        hi_q, lo_q = angle[left], angle[left + 1]
        if hi_q >= q >= lo_q:
            if hi_q == lo_q:
                return 0.5 * (raw[left] + raw[left + 1]), False
            frac = (hi_q - q) / (hi_q - lo_q)
            return raw[left] + frac * (raw[left + 1] - raw[left]), False
    raise AssertionError("unreachable inverse interpolation interval")


def stats(values: Sequence[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "count": len(values),
        "median": median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "min": min(values),
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()

    points: list[dict] = []
    input_hashes: dict[str, str] = {}
    for spec in POINTS:
        directory = session / spec.directory
        sdk_path = directory / spec.sdk_file
        csv_path = directory / f"{spec.camera_prefix}_samples.csv"
        summary_path = directory / f"{spec.camera_prefix}_summary.json"
        for path in (sdk_path, csv_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            input_hashes[str(path.relative_to(session))] = sha256(path)
        row = {
            "direction": spec.direction,
            "quality_role": spec.quality_role,
            "target_raw": spec.target_raw,
            "sdk_record": str(sdk_path.relative_to(session)),
            "camera_csv": str(csv_path.relative_to(session)),
            "camera_summary": str(summary_path.relative_to(session)),
            **load_sdk(sdk_path),
            **load_camera(csv_path, summary_path),
        }
        points.append(row)

    endpoint = next(
        point
        for point in points
        if point["direction"] == "extending" and point["target_raw"] == 255
    )
    pip_reference_deg = float(endpoint["pip_tape_deg"])
    dip_reference_deg = float(endpoint["dip_tape_deg"])
    for point in points:
        point["pip_flex_deg"] = pip_reference_deg - float(point["pip_tape_deg"])
        point["dip_flex_deg"] = dip_reference_deg - float(point["dip_tape_deg"])
        point["pip_flex_rad"] = math.radians(float(point["pip_flex_deg"]))
        point["dip_flex_rad"] = math.radians(float(point["dip_flex_deg"]))
        raw = float(point["sdk_readback_raw"])
        point["current_affine_pip_rad"] = (255.0 - raw) / 255.0 * PIP_URDF_UPPER_RAD
        point["current_affine_physical_error_rad"] = (
            float(point["pip_flex_rad"]) - float(point["current_affine_pip_rad"])
        )
        point["dip_mimic_residual_rad"] = float(point["dip_flex_rad"]) - (
            DIP_URDF_MIMIC_MULTIPLIER * float(point["pip_flex_rad"])
        )
        point["dip_to_pip_ratio"] = (
            float(point["dip_flex_rad"]) / float(point["pip_flex_rad"])
            if float(point["pip_flex_rad"]) > math.radians(3.0)
            else None
        )

    fit_points = [
        point
        for point in points
        if point["quality_role"] == "fit"
        and float(point["sdk_readback_raw"]) >= OPERATIONAL_RAW_FLOOR
    ]
    flexing = [point for point in fit_points if point["direction"] == "flexing"]
    extending = [point for point in fit_points if point["direction"] == "extending"]
    down_raw, down_pip = branch_table(flexing, "pip_flex_rad")
    up_raw, up_pip = branch_table(extending, "pip_flex_rad")
    _, down_dip = branch_table(flexing, "dip_flex_rad")
    _, up_dip = branch_table(extending, "dip_flex_rad")

    common_raw = [20, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]
    hysteresis_rows: list[dict] = []
    midpoint_pip: list[float] = []
    midpoint_dip: list[float] = []
    for raw in common_raw:
        down_p = interp(raw, down_raw, down_pip)
        up_p = interp(raw, up_raw, up_pip)
        down_d = interp(raw, down_raw, down_dip)
        up_d = interp(raw, up_raw, up_dip)
        midpoint_pip.append(0.5 * (down_p + up_p))
        midpoint_dip.append(0.5 * (down_d + up_d))
        hysteresis_rows.append(
            {
                "raw": raw,
                "pip_flexing_rad": down_p,
                "pip_extending_rad": up_p,
                "pip_signed_extending_minus_flexing_rad": up_p - down_p,
                "pip_abs_hysteresis_rad": abs(up_p - down_p),
                "dip_flexing_rad": down_d,
                "dip_extending_rad": up_d,
                "dip_signed_extending_minus_flexing_rad": up_d - down_d,
                "dip_abs_hysteresis_rad": abs(up_d - down_d),
            }
        )
    midpoint_raw = [float(value) for value in common_raw] + [255.0]
    midpoint_pip = isotonic_nonincreasing(midpoint_pip + [0.0])
    midpoint_dip = isotonic_nonincreasing(midpoint_dip + [0.0])

    requested_q = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.5197473168373108,
        0.5280907168623586,
        0.6,
        0.7,
        0.8,
        0.9,
        0.97,
        1.0,
        1.08,
    ]
    inverse_rows: list[dict] = []
    for q in requested_q:
        recommended_raw, recommended_clamped = inverse_nonincreasing(
            q, midpoint_raw, midpoint_pip
        )
        current_raw = 255.0 * (1.0 - q / PIP_URDF_UPPER_RAD)
        current_in_measured_domain = current_raw >= OPERATIONAL_RAW_FLOOR
        actual_at_current = (
            interp(current_raw, midpoint_raw, midpoint_pip)
            if current_in_measured_domain
            else None
        )
        inverse_rows.append(
            {
                "requested_sim_pip_rad": q,
                "recommended_midpoint_raw": recommended_raw,
                "recommended_clamped": recommended_clamped,
                "current_affine_raw": current_raw,
                "current_affine_in_measured_operational_domain": current_in_measured_domain,
                "measured_physical_pip_at_current_affine_raw": actual_at_current,
                "current_affine_physical_error_rad": (
                    actual_at_current - q if actual_at_current is not None else None
                ),
            }
        )

    operational_points = [
        point
        for point in fit_points
        if float(point["pip_flex_rad"]) > math.radians(3.0)
    ]
    mimic_residual_abs = [abs(float(point["dip_mimic_residual_rad"])) for point in operational_points]
    ratios = [
        float(point["dip_to_pip_ratio"])
        for point in operational_points
        if point["dip_to_pip_ratio"] is not None
    ]
    command_errors_abs = [abs(float(point["command_target_error_raw"])) for point in points]

    flex13 = next(point for point in points if point["direction"] == "flexing" and point["target_raw"] == 12)
    flex20 = next(point for point in points if point["direction"] == "flexing" and point["target_raw"] == 20)
    flex32 = next(point for point in points if point["direction"] == "flexing" and point["target_raw"] == 32)
    slope_32_20 = (
        float(flex20["pip_flex_deg"]) - float(flex32["pip_flex_deg"])
    ) / (
        float(flex32["sdk_readback_raw"]) - float(flex20["sdk_readback_raw"])
    )
    slope_20_12 = (
        float(flex13["pip_flex_deg"]) - float(flex20["pip_flex_deg"])
    ) / (
        float(flex20["sdk_readback_raw"]) - float(flex13["sdk_readback_raw"])
    )
    plateau_slope_ratio = slope_20_12 / slope_32_20

    final_snapshot_path = session / "final_sdk_readonly_at_raw255.json"
    final_snapshot = json.loads(final_snapshot_path.read_text(encoding="utf-8"))
    input_hashes[str(final_snapshot_path.relative_to(session))] = sha256(final_snapshot_path)
    final_raw = [float(row["state20"][16]) for row in final_snapshot["samples"]]
    final_faults = [
        int(value)
        for row in final_snapshot["samples"]
        for value in row["faults20"]
    ]

    pip_hysteresis = [float(row["pip_abs_hysteresis_rad"]) for row in hysteresis_rows]
    dip_hysteresis = [float(row["dip_abs_hysteresis_rad"]) for row in hysteresis_rows]
    output = {
        "schema_version": 1,
        "offline_only": True,
        "session": str(session),
        "method": {
            "camera_angle": "fixed-camera projected 2D tape-axis angle",
            "reduction": "median of three contiguous frame-block medians",
            "physical_zero": "final extending target raw255 camera endpoint",
            "physical_flexion": "endpoint tape angle minus measured tape angle",
            "fit": "direction-specific monotone piecewise linear plus midpoint LUT",
            "why_not_3d": "aligned-depth 3D axes were materially noisier than fixed-plane 2D axes",
        },
        "constants": {
            "index_pip_urdf_range_rad": [0.0, PIP_URDF_UPPER_RAD],
            "index_dip_urdf_mimic_multiplier": DIP_URDF_MIMIC_MULTIPLIER,
            "operational_raw_floor": OPERATIONAL_RAW_FLOOR,
            "lowest_characterized_target_raw": 12,
            "targets_raw6_and_raw0_commanded": False,
        },
        "reference": {
            "target_raw": 255,
            "sdk_steady_readback_raw": endpoint["sdk_readback_raw"],
            "pip_tape_reference_deg": pip_reference_deg,
            "dip_tape_reference_deg": dip_reference_deg,
            "interpretation": "endpoint-referenced flexion; not an external metrology absolute zero",
        },
        "health": {
            "all_selected_motion_records_fault_free": all(bool(point["fault_free"]) for point in points),
            "max_index_tip_temperature_c": max(float(point["index_tip_temp_max_c"]) for point in points),
            "max_per_step_temperature_delta_c": max(float(point["temperature_delta_max_c"]) for point in points),
            "final_readonly_samples": len(final_snapshot["samples"]),
            "final_readonly_fault_free": not any(final_faults),
            "final_readonly_raw16_median": median(final_raw),
            "final_readonly_raw16_min": min(final_raw),
            "final_readonly_raw16_max": max(final_raw),
            "command_vs_steady_readback_abs_raw": stats(command_errors_abs),
        },
        "camera_quality": {
            "fit_points_all_pass_5pct_failure_gate": all(
                bool(point["quality_pass_5pct"]) for point in fit_points
            ),
            "limit_characterization_raw12_passes_5pct_gate": bool(flex13["quality_pass_5pct"]),
            "raw12_usable_frames": flex13["usable_frames"],
            "raw12_requested_frames": flex13["requested_frames"],
            "max_fit_pip_block_range_deg": max(float(point["pip_block_range_deg"]) for point in fit_points),
            "max_fit_dip_block_range_deg": max(float(point["dip_block_range_deg"]) for point in fit_points),
        },
        "physical_limit": {
            "pip_deg_per_raw_33_to_21": slope_32_20,
            "pip_deg_per_raw_21_to_13": slope_20_12,
            "late_to_previous_slope_ratio": plateau_slope_ratio,
            "late_slope_collapse_percent": 100.0 * (1.0 - plateau_slope_ratio),
            "reason_for_stop": "PIP flexion plateau while DIP continued; no fault or temperature excursion",
            "recommended_initial_operational_floor_raw": OPERATIONAL_RAW_FLOOR,
        },
        "directional_piecewise_lut": {
            "flexing": {
                "raw": down_raw,
                "pip_rad": down_pip,
                "dip_rad": down_dip,
            },
            "extending": {
                "raw": up_raw,
                "pip_rad": up_pip,
                "dip_rad": up_dip,
            },
            "midpoint": {
                "raw": midpoint_raw,
                "pip_rad": midpoint_pip,
                "dip_rad": midpoint_dip,
            },
        },
        "hysteresis": {
            "comparison_rows": hysteresis_rows,
            "pip_abs_rad": stats(pip_hysteresis),
            "pip_max_deg": math.degrees(max(pip_hysteresis)),
            "dip_abs_rad": stats(dip_hysteresis),
            "dip_max_deg": math.degrees(max(dip_hysteresis)),
        },
        "current_affine_mapping_comparison": {
            "formula": "pip_rad=(255-raw)/255*1.08",
            "inverse_rows": inverse_rows,
        },
        "dip_coupling_vs_urdf": {
            "measured_dip_to_pip_ratio": stats(ratios),
            "abs_residual_rad_vs_dip_eq_0p8917_pip": stats(mimic_residual_abs),
            "max_abs_residual_deg": math.degrees(max(mimic_residual_abs)),
        },
        "points": points,
        "input_sha256": input_hashes,
    }

    csv_path = session / "index_pip_dip_calibration_points.csv"
    scalar_fields = [
        "direction",
        "quality_role",
        "target_raw",
        "sdk_readback_raw",
        "sdk_readback_raw_min",
        "sdk_readback_raw_max",
        "command_target_error_raw",
        "root_hold_readback_raw",
        "usable_frames",
        "requested_frames",
        "failure_fraction",
        "quality_pass_5pct",
        "pip_tape_deg",
        "pip_block_range_deg",
        "dip_tape_deg",
        "dip_block_range_deg",
        "pip_flex_deg",
        "dip_flex_deg",
        "pip_flex_rad",
        "dip_flex_rad",
        "current_affine_pip_rad",
        "current_affine_physical_error_rad",
        "dip_to_pip_ratio",
        "dip_mimic_residual_rad",
        "index_tip_temp_max_c",
        "temperature_delta_max_c",
        "fault_free",
        "sdk_record",
        "camera_csv",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)

    json_path = session / "index_pip_dip_calibration_fit.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    plot_path = session / "index_pip_dip_calibration_curve.png"
    plot_status = "not generated"
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
        for direction, marker in (("flexing", "o"), ("extending", "s")):
            selected = [point for point in fit_points if point["direction"] == direction]
            axes[0].plot(
                [point["sdk_readback_raw"] for point in selected],
                [point["pip_flex_rad"] for point in selected],
                marker=marker,
                label=direction,
            )
            axes[1].plot(
                [point["sdk_readback_raw"] for point in selected],
                [point["dip_flex_rad"] for point in selected],
                marker=marker,
                label=direction,
            )
        affine_raw = list(range(OPERATIONAL_RAW_FLOOR, 256))
        axes[0].plot(
            affine_raw,
            [(255.0 - raw) / 255.0 * PIP_URDF_UPPER_RAD for raw in affine_raw],
            linestyle="--",
            label="current affine report",
        )
        axes[0].axhline(PIP_URDF_UPPER_RAD, color="red", linestyle=":", label="URDF PIP upper")
        axes[0].set_ylabel("PIP flexion (rad)")
        axes[1].set_ylabel("DIP flexion (rad)")
        axes[1].set_xlabel("SDK readback raw (255=open, lower=flexed)")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.suptitle("G20 index PIP/DIP physical calibration")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plot_status = str(plot_path)
    except ImportError:
        pass

    q_contact = next(
        row
        for row in inverse_rows
        if math.isclose(float(row["requested_sim_pip_rad"]), 0.5280907168623586)
    )
    q_pregrasp = next(
        row
        for row in inverse_rows
        if math.isclose(float(row["requested_sim_pip_rad"]), 0.97)
    )
    report_path = session / "INDEX_PIP_DIP_CALIBRATION_REPORT.md"
    report = f"""# G20 食指 PIP / DIP 完整物理校准报告

## 结论

- 32 个姿态记录已纳入审计：16 个屈曲方向点（其中 raw12 仅用于极限表征）和 16 个伸展回程点。
- 全部选中动作记录故障为 0；食指末端执行器最高温度 {fmt(output['health']['max_index_tip_temperature_c'], 1)} °C；单步最大温升 {fmt(output['health']['max_per_step_temperature_delta_c'], 1)} °C。
- 最终停在伸直端 target raw255；随后 {output['health']['final_readonly_samples']} 次只读回读的 slot16 中位数为 {fmt(output['health']['final_readonly_raw16_median'], 1)}，范围 {fmt(output['health']['final_readonly_raw16_min'], 0)}–{fmt(output['health']['final_readonly_raw16_max'], 0)}，全部无故障。
- PIP 在 raw21→13 区间的视觉角增量斜率比 raw33→21 下降 {fmt(output['physical_limit']['late_slope_collapse_percent'], 1)}%，而 DIP 仍继续弯曲，因此 raw6、raw0 没有发送。初始部署下限建议保持 target raw20。
- 当前生产映射 `pip=(255-raw)/255*1.08` 只是代数双向一致，不是物理角一致。它在训练 contact target 0.5281 rad 处会给出 raw {fmt(q_contact['current_affine_raw'], 1)}，相机测得该 raw 附近的真实 PIP 约 {fmt(q_contact['measured_physical_pip_at_current_affine_raw'], 3)} rad；推荐非线性中线约 raw {fmt(q_contact['recommended_midpoint_raw'], 1)}。
- 旧 pregrasp 0.970 rad 会给出 raw {fmt(q_pregrasp['current_affine_raw'], 1)}，真实 PIP 约 {fmt(q_pregrasp['measured_physical_pip_at_current_affine_raw'], 3)} rad，物理过屈曲约 {fmt(q_pregrasp['current_affine_physical_error_rad'], 3)} rad。这足以造成明显 sim-to-real 姿态偏差。

## 测量定义

- 固定 D435，使用每帧四块蓝胶带的 2D PCA 长轴角；深度 3D 角仅留作诊断，因为本次深度轴角噪声明显更大。
- 每个姿态用 3 个连续时间块的中位数，再取这 3 个中位数的中位数。
- 最终伸展 raw255 定义为本轮物理屈曲零点。所有 PIP/DIP 角均为相对此端点的角度，因此消除了胶带粘贴固定偏置；它不是外部量角器给出的绝对机械零位。
- 真手一个 tip actuator 同时驱动 PIP 与 DIP；SDK slot16 只提供一个 raw 回读，但相机分别测出了两个物理关节角。

## 关键质量与滞回

| 指标 | 结果 |
|---|---:|
| 拟合点全部通过相机失败率 ≤5% | {output['camera_quality']['fit_points_all_pass_5pct_failure_gate']} |
| raw12 极限诊断可用帧 | {output['camera_quality']['raw12_usable_frames']}/{output['camera_quality']['raw12_requested_frames']} |
| 指令与稳态回读绝对偏差最大值 | {fmt(output['health']['command_vs_steady_readback_abs_raw']['max'], 1)} raw |
| PIP 去程/回程滞回中位数 | {fmt(math.degrees(output['hysteresis']['pip_abs_rad']['median']), 2)}° |
| PIP 去程/回程滞回最大值 | {fmt(output['hysteresis']['pip_max_deg'], 2)}° |
| DIP 去程/回程滞回中位数 | {fmt(math.degrees(output['hysteresis']['dip_abs_rad']['median']), 2)}° |
| DIP 去程/回程滞回最大值 | {fmt(output['hysteresis']['dip_max_deg'], 2)}° |

用户观察到的“松、旷”与上述两类现象一致：SDK 稳态量化/回差是 raw 级别；同一 raw 的物理方向滞回则是角度级别。部署时不应把二者混为一个误差。

## 当前线性映射与建议中线

| sim PIP 请求 (rad) | 当前 raw | 当前 raw 的实测物理 PIP (rad) | 物理误差 (rad) | 建议中线 raw |
|---:|---:|---:|---:|---:|
"""
    for row in inverse_rows:
        q = float(row["requested_sim_pip_rad"])
        if q not in (0.0, 0.3, 0.5, 0.5197473168373108, 0.5280907168623586, 0.7, 0.9, 0.97, 1.0, 1.08):
            continue
        actual = row["measured_physical_pip_at_current_affine_raw"]
        error = row["current_affine_physical_error_rad"]
        report += (
            f"| {fmt(q, 4)} | {fmt(float(row['current_affine_raw']), 1)} | "
            f"{fmt(float(actual), 3) if actual is not None else '未测/低于安全下限'} | "
            f"{fmt(float(error), 3) if error is not None else '未测'} | "
            f"{fmt(float(row['recommended_midpoint_raw']), 1)} |\n"
        )
    report += f"""

## DIP 机械耦合

- Isaac URDF 假设 `index_dip = 0.8917 × index_pip`。
- 本轮相机测得 DIP/PIP 比值中位数为 {fmt(output['dip_coupling_vs_urdf']['measured_dip_to_pip_ratio']['median'], 3)}，范围 {fmt(output['dip_coupling_vs_urdf']['measured_dip_to_pip_ratio']['min'], 3)}–{fmt(output['dip_coupling_vs_urdf']['measured_dip_to_pip_ratio']['max'], 3)}。
- 相对 URDF mimic 的绝对残差最大为 {fmt(output['dip_coupling_vs_urdf']['max_abs_residual_deg'], 2)}°。所以只修 PIP raw↔rad 仍不够：仿真中的 DIP mimic 曲线也需要按物理耦合重新拟合或至少重新选择线性 multiplier。

## 可部署性判断

现有 `linker_calib_thumbfit.json` 的 schema 只支持每个 SDK slot 的 affine `lo/hi/flip`，无法表达本次测到的非线性和方向滞回，也无法从一个 slot16 回读同时重建不同的 PIP、DIP 物理角。因此：

1. 不应直接把一个新的 `hi` 数字覆盖到生产校准文件并宣称校准完成。
2. 下一版部署映射应支持按 raw 查询的单调分段 PIP LUT；指令使用其逆表，回读使用同一物理表，从而真正做到 sim rad 与物理 PIP rad 一致。
3. DIP 在仿真侧应改用本报告的耦合曲线，或先用拟合后的线性 multiplier；部署观测若策略只含 16 个独立关节，仍由 PIP 状态派生，不新增不存在的 SDK 传感器。
4. 方向滞回可先用中线 LUT，误差带用于 domain randomization；若后续需要更高精度，再按 raw 变化方向选择 flexing/extending 两张表。

## 产物

- 数据点：`{csv_path.name}`
- 拟合、哈希和全部分段表：`{json_path.name}`
- 曲线图：`{plot_status}`
- 原始证据仍保留在每个 step/return 子目录，失败的安全门记录没有覆盖或删除。
"""
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps({
        "points_csv": str(csv_path),
        "fit_json": str(json_path),
        "plot": plot_status,
        "report": str(report_path),
        "health": output["health"],
        "physical_limit": output["physical_limit"],
        "hysteresis": output["hysteresis"],
        "contact_mapping": q_contact,
        "pregrasp_mapping": q_pregrasp,
        "dip_coupling": output["dip_coupling_vs_urdf"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
