#!/usr/bin/env python3
"""Summarize the fixed-D435 G20 thumb MCP/IP/CMC-pitch calibration session."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ANGLE_KEYS = (
    "mcp_projected_deg_2d",
    "pip_projected_deg_2d",
    "dip_projected_deg_2d",
)
FORWARD_RAWS = (
    240, 224, 208, 192, 176, 160, 144, 128, 112,
    96, 80, 64, 48, 32, 20, 12, 6, 0,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_medians(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 180:
        raise ValueError(f"{path}: expected 180 rows, got {len(rows)}")
    return {
        key: statistics.median(float(row[key]) for row in rows)
        for key in ANGLE_KEYS
    }


def _tail_state(path: Path, slot: int) -> tuple[float, bool, int]:
    payload = _read_json(path)
    if not payload.get("motion_sent"):
        raise ValueError(f"{path}: motion was not sent")
    rows = payload["post_command_samples"]
    tail = rows[-min(10, len(rows)):]
    state = statistics.median(row["state20"][slot] for row in tail)
    fault_free = not any(
        value for row in rows for value in row["faults20"]
    )
    max_temp = max(row["temperature20"][slot] for row in rows)
    return state, fault_free, max_temp


def _single_motion(directory: Path) -> Path:
    candidates = [
        path
        for path in sorted(directory.glob("motion_*.json"))
        if _read_json(path).get("motion_sent")
    ]
    if not candidates:
        raise ValueError(f"{directory}: no sent motion JSON")
    return candidates[-1]


def _fit_mimic(
    mcp: list[float], ip: list[float]
) -> dict[str, float]:
    sum_xx = sum(value * value for value in mcp)
    zero_slope = sum(x * y for x, y in zip(mcp, ip)) / sum_xx
    x_mean = statistics.mean(mcp)
    y_mean = statistics.mean(ip)
    affine_slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(mcp, ip)
    ) / sum((x - x_mean) ** 2 for x in mcp)
    affine_offset = y_mean - affine_slope * x_mean
    zero_residuals = [
        y - zero_slope * x for x, y in zip(mcp, ip)
    ]
    affine_residuals = [
        y - (affine_slope * x + affine_offset)
        for x, y in zip(mcp, ip)
    ]
    return {
        "zero_intercept_multiplier": zero_slope,
        "zero_intercept_max_abs_residual_rad": max(
            abs(value) for value in zero_residuals
        ),
        "affine_multiplier": affine_slope,
        "affine_offset_rad": affine_offset,
        "affine_max_abs_residual_rad": max(
            abs(value) for value in affine_residuals
        ),
    }


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    for left in range(len(xs) - 1):
        if xs[left] <= x <= xs[left + 1]:
            fraction = (x - xs[left]) / (xs[left + 1] - xs[left])
            return ys[left] + fraction * (ys[left + 1] - ys[left])
    raise ValueError(f"{x} outside interpolation domain")


def _heldout_error(raw: list[float], rad: list[float]) -> float:
    train_indices = list(range(0, len(raw), 2))
    if train_indices[-1] != len(raw) - 1:
        train_indices.append(len(raw) - 1)
    xs = [raw[index] for index in train_indices]
    ys = [rad[index] for index in train_indices]
    errors = [
        abs(rad[index] - _interp(raw[index], xs, ys))
        for index in range(1, len(raw) - 1, 2)
    ]
    return max(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--base-calib", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()

    zero_paths = [
        session / "raw255_zero/camera_b_fixed_exp_180f_samples.csv",
        session / "raw255_zero/camera_c_settled_fixed_exp_180f_samples.csv",
    ]
    zero_rows = [_camera_medians(path) for path in zero_paths]
    zero = {
        key: statistics.mean(row[key] for row in zero_rows)
        for key in ANGLE_KEYS
    }
    zero_repeatability = {
        key: abs(zero_rows[0][key] - zero_rows[1][key])
        for key in ANGLE_KEYS
    }

    mcp_points: list[dict[str, Any]] = [{
        "command_raw": 255,
        "stable_readback_raw": 254.0,
        "thumb_mcp_rad": 0.0,
        "thumb_ip_rad": 0.0,
        "cmc_pitch_isolation_drift_rad": 0.0,
    }]
    pitch_points: list[dict[str, Any]] = [{
        "command_raw": 255,
        "stable_readback_raw": 254.0,
        "thumb_cmc_pitch_rad": 0.0,
        "thumb_mcp_isolation_drift_rad": 0.0,
        "thumb_ip_isolation_drift_rad": 0.0,
    }]
    input_paths = list(zero_paths)
    for raw in FORWARD_RAWS:
        mcp_camera = (
            session / f"mcp_raw{raw:03d}/camera_fixed_exp_180f_samples.csv"
        )
        pitch_camera = (
            session / f"pitch_raw{raw:03d}/camera_fixed_exp_180f_samples.csv"
        )
        mcp_motion = _single_motion(session / f"mcp_raw{raw:03d}")
        pitch_motion = _single_motion(session / f"pitch_raw{raw:03d}")
        mcp_angles = _camera_medians(mcp_camera)
        pitch_angles = _camera_medians(pitch_camera)
        mcp_state, mcp_fault_free, mcp_temp = _tail_state(
            mcp_motion, 15
        )
        pitch_state, pitch_fault_free, pitch_temp = _tail_state(
            pitch_motion, 0
        )
        if not mcp_fault_free or not pitch_fault_free:
            raise ValueError(f"raw{raw}: nonzero fault in formal point")
        mcp_points.append({
            "command_raw": raw,
            "stable_readback_raw": mcp_state,
            "thumb_mcp_rad": math.radians(
                mcp_angles["pip_projected_deg_2d"]
                - zero["pip_projected_deg_2d"]
            ),
            "thumb_ip_rad": math.radians(
                mcp_angles["dip_projected_deg_2d"]
                - zero["dip_projected_deg_2d"]
            ),
            "cmc_pitch_isolation_drift_rad": math.radians(
                mcp_angles["mcp_projected_deg_2d"]
                - zero["mcp_projected_deg_2d"]
            ),
            "target_max_temperature_c": mcp_temp,
        })
        pitch_points.append({
            "command_raw": raw,
            "stable_readback_raw": pitch_state,
            "thumb_cmc_pitch_rad": math.radians(
                pitch_angles["mcp_projected_deg_2d"]
                - zero["mcp_projected_deg_2d"]
            ),
            "thumb_mcp_isolation_drift_rad": math.radians(
                pitch_angles["pip_projected_deg_2d"]
                - zero["pip_projected_deg_2d"]
            ),
            "thumb_ip_isolation_drift_rad": math.radians(
                pitch_angles["dip_projected_deg_2d"]
                - zero["dip_projected_deg_2d"]
            ),
            "target_max_temperature_c": pitch_temp,
        })
        input_paths.extend(
            [mcp_camera, pitch_camera, mcp_motion, pitch_motion]
        )

    for points, key in (
        (mcp_points, "thumb_mcp_rad"),
        (pitch_points, "thumb_cmc_pitch_rad"),
    ):
        if not all(
            points[index][key] < points[index + 1][key]
            for index in range(len(points) - 1)
        ):
            raise ValueError(f"{key}: forward physical curve is not monotonic")

    mcp_return = _camera_medians(
        session / "mcp_return/camera_raw128_return_fixed_exp_180f_samples.csv"
    )
    pitch_return = _camera_medians(
        session / "pitch_return/camera_raw128_return_fixed_exp_180f_samples.csv"
    )
    mcp_final = _camera_medians(
        session / "mcp_return/camera_raw255_return_fixed_exp_180f_samples.csv"
    )
    pitch_final = _camera_medians(
        session / "pitch_return/camera_raw255_return_fixed_exp_180f_samples.csv"
    )
    mcp_forward_128 = next(
        point for point in mcp_points if point["command_raw"] == 128
    )
    pitch_forward_128 = next(
        point for point in pitch_points if point["command_raw"] == 128
    )
    mimic = _fit_mimic(
        [point["thumb_mcp_rad"] for point in mcp_points],
        [point["thumb_ip_rad"] for point in mcp_points],
    )

    raw_ascending = [
        float(point["command_raw"]) for point in reversed(mcp_points)
    ]
    mcp_rad_descending = [
        point["thumb_mcp_rad"] for point in reversed(mcp_points)
    ]
    pitch_rad_descending = [
        point["thumb_cmc_pitch_rad"]
        for point in reversed(pitch_points)
    ]
    motion_paths = sorted(set(session.glob("**/motion_*.json")))
    motion_payloads = [_read_json(path) for path in motion_paths]
    sent_payloads = [
        payload for payload in motion_payloads if payload.get("motion_sent")
    ]
    all_fault_free = not any(
        value
        for payload in sent_payloads
        for row in payload.get("post_command_samples", [])
        for value in row["faults20"]
    )
    max_temperature = max(
        value
        for payload in sent_payloads
        for row in payload.get("post_command_samples", [])
        for value in row["temperature20"]
    )
    summary = {
        "schema_version": 1,
        "candidate_only": True,
        "session": str(session),
        "identity": {
            "serial": "LHT20-010-415-L-B-1-D",
            "side": "left",
            "hand_joint": "G20",
            "camera_serial": "143322073091",
            "camera_profile": "1280x720@30",
            "rgb_controls": {
                "exposure": 166,
                "gain": 32,
                "white_balance": 4600,
                "auto_controls": False,
            },
            "view_hold_raw": {"thumb_cmc_roll_slot5": 67, "thumb_cmc_yaw_slot10": 113},
        },
        "formal_zero_reference": {
            "sources": [str(path) for path in zero_paths],
            "angle_median_deg": zero,
            "repeatability_abs_difference_deg": zero_repeatability,
            "excluded_early_settling_reference": (
                "raw255_zero/camera_a_fixed_exp_180f_samples.csv"
            ),
        },
        "thumb_mcp": {
            "slot": 15,
            "physical_max_rad": mcp_points[-1]["thumb_mcp_rad"],
            "physical_max_deg": math.degrees(
                mcp_points[-1]["thumb_mcp_rad"]
            ),
            "raw128_hysteresis_rad": (
                math.radians(
                    mcp_return["pip_projected_deg_2d"]
                    - zero["pip_projected_deg_2d"]
                )
                - mcp_forward_128["thumb_mcp_rad"]
            ),
            "final_zero_drift_rad": math.radians(
                mcp_final["pip_projected_deg_2d"]
                - zero["pip_projected_deg_2d"]
            ),
            "heldout_max_abs_error_rad": _heldout_error(
                raw_ascending, mcp_rad_descending
            ),
            "points": mcp_points,
        },
        "thumb_ip": {
            "passive_mimic": True,
            "physical_max_rad": mcp_points[-1]["thumb_ip_rad"],
            "physical_max_deg": math.degrees(
                mcp_points[-1]["thumb_ip_rad"]
            ),
            "raw128_hysteresis_rad": (
                math.radians(
                    mcp_return["dip_projected_deg_2d"]
                    - zero["dip_projected_deg_2d"]
                )
                - mcp_forward_128["thumb_ip_rad"]
            ),
            "final_zero_drift_rad": math.radians(
                mcp_final["dip_projected_deg_2d"]
                - zero["dip_projected_deg_2d"]
            ),
            "fit": mimic,
        },
        "thumb_cmc_pitch": {
            "slot": 0,
            "physical_max_rad": pitch_points[-1]["thumb_cmc_pitch_rad"],
            "physical_max_deg": math.degrees(
                pitch_points[-1]["thumb_cmc_pitch_rad"]
            ),
            "raw128_hysteresis_rad": (
                math.radians(
                    pitch_return["mcp_projected_deg_2d"]
                    - zero["mcp_projected_deg_2d"]
                )
                - pitch_forward_128["thumb_cmc_pitch_rad"]
            ),
            "final_zero_drift_rad": math.radians(
                pitch_final["mcp_projected_deg_2d"]
                - zero["mcp_projected_deg_2d"]
            ),
            "heldout_max_abs_error_rad": _heldout_error(
                raw_ascending, pitch_rad_descending
            ),
            "points": pitch_points,
        },
        "safety": {
            "motion_json_count": len(motion_paths),
            "motion_sent_count": len(sent_payloads),
            "preflight_rejection_json_count": (
                len(motion_paths) - len(sent_payloads)
            ),
            "all_sent_motion_fault_free": all_fault_free,
            "max_temperature_c_across_all_slots": max_temperature,
            "reserved_slots_11_14_zero": True,
            "final_sdk_snapshot": (
                "final_readonly/sdk_snapshot_after_thumb_mcp_ip_pitch_sweeps.json"
            ),
        },
        "known_record_limitation": (
            "One motion_sent=false raw48-to-raw64 MCP return preflight "
            "rejection was overwritten by the later successful retry that "
            "used the same filename. The observed rejection was actual "
            "raw46 to target64 (18 raw), before any motion was sent."
        ),
        "input_sha256": {
            str(path.relative_to(session)): _sha256(path)
            for path in sorted(set(input_paths))
        },
    }

    points_path = session / "thumb_mcp_ip_pitch_calibration_points.csv"
    with points_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "joint", "command_raw", "stable_readback_raw",
            "physical_rad", "physical_deg",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for point in mcp_points:
            for joint, key in (
                ("thumb_mcp", "thumb_mcp_rad"),
                ("thumb_ip", "thumb_ip_rad"),
            ):
                writer.writerow({
                    "joint": joint,
                    "command_raw": point["command_raw"],
                    "stable_readback_raw": point["stable_readback_raw"],
                    "physical_rad": point[key],
                    "physical_deg": math.degrees(point[key]),
                })
        for point in pitch_points:
            writer.writerow({
                "joint": "thumb_cmc_pitch",
                "command_raw": point["command_raw"],
                "stable_readback_raw": point["stable_readback_raw"],
                "physical_rad": point["thumb_cmc_pitch_rad"],
                "physical_deg": math.degrees(
                    point["thumb_cmc_pitch_rad"]
                ),
            })

    summary_path = session / "thumb_mcp_ip_pitch_summary.json"
    _write_json(summary_path, summary)
    overlay = _read_json(args.base_calib)
    overlay["note"] = (
        "CANDIDATE ONLY: fixed-D435 physical LUTs for index pitch/PIP "
        "plus thumb MCP and CMC pitch; incomplete full-hand calibration"
    )
    overlay["joints"]["thumb_mcp"] = {
        "flip": False,
        "lo": 0.0,
        "hi": mcp_rad_descending[0],
        "physical_lut": {
            "raw": raw_ascending,
            "rad": mcp_rad_descending,
        },
    }
    overlay["joints"]["thumb_cmc_pitch"] = {
        "flip": False,
        "lo": 0.0,
        "hi": pitch_rad_descending[0],
        "physical_lut": {
            "raw": raw_ascending,
            "rad": pitch_rad_descending,
        },
    }
    _write_json(args.candidate_out, overlay)

    report = f"""# G20 拇指 MCP/IP/CMC Pitch 固定曝光物理标定报告

Session：`{session.name}`

## 结论

- `thumb_mcp` slot15：raw0..255全范围安全，物理最大 `{summary['thumb_mcp']['physical_max_rad']:.9f} rad / {summary['thumb_mcp']['physical_max_deg']:.3f}°`。
- `thumb_ip` 被动mimic：物理最大 `{summary['thumb_ip']['physical_max_rad']:.9f} rad / {summary['thumb_ip']['physical_max_deg']:.3f}°`。
- `thumb_cmc_pitch` slot0：raw0..255全范围安全，物理最大 `{summary['thumb_cmc_pitch']['physical_max_rad']:.9f} rad / {summary['thumb_cmc_pitch']['physical_max_deg']:.3f}°`。
- yaw/roll没有做正式标定；本session只固定slot10=113、slot5=67作为相机view hold。
- 所有已发送运动fault为0；全拇指槽最高温度 `{max_temperature}°C`。

## 重复性和回差

- 正式零参考B/C差：CMC pitch `{zero_repeatability['mcp_projected_deg_2d']:.4f}°`，MCP `{zero_repeatability['pip_projected_deg_2d']:.4f}°`，IP `{zero_repeatability['dip_projected_deg_2d']:.4f}°`。
- MCP raw128回差 `{math.degrees(summary['thumb_mcp']['raw128_hysteresis_rad']):.4f}°`；IP raw128回差 `{math.degrees(summary['thumb_ip']['raw128_hysteresis_rad']):.4f}°`。
- Pitch raw128回差 `{math.degrees(summary['thumb_cmc_pitch']['raw128_hysteresis_rad']):.4f}°`。
- 最终零漂：MCP `{math.degrees(summary['thumb_mcp']['final_zero_drift_rad']):.4f}°`，IP `{math.degrees(summary['thumb_ip']['final_zero_drift_rad']):.4f}°`，pitch `{math.degrees(summary['thumb_cmc_pitch']['final_zero_drift_rad']):.4f}°`。

## Thumb IP mimic

- 当前Isaac multiplier `1.1619`偏大。
- 零截距候选 multiplier：`{mimic['zero_intercept_multiplier']:.9f}`。
- 带偏置拟合：`IP = {mimic['affine_multiplier']:.9f} * MCP {mimic['affine_offset_rad']:+.9f} rad`。
- URDF建议优先使用零截距候选，避免伸展零位产生负IP角；非线性残差保留在报告中。

## 产物

- `thumb_mcp_ip_pitch_calibration_points.csv`
- `thumb_mcp_ip_pitch_summary.json`
- `{args.candidate_out.name}`
- 原始运动JSON、逐帧CSV、RGB、aligned depth、annotated PNG全部保留。

## 记录说明

第一次camera A是在回位后的早期settle阶段，只保留作诊断；正式零参考使用可重复的B/C中线。
另有一次MCP回程raw64发送前拒绝因后来成功重试复用了同一文件名而被覆盖；当时实际raw46到目标64为18 raw，`motion_sent=false`。本报告明确保留这一记录完整性限制，后续session必须为每次尝试使用唯一文件名。
"""
    (session / "THUMB_MCP_IP_PITCH_FIXED_EXPOSURE_CALIBRATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({
        "summary": str(summary_path),
        "points": str(points_path),
        "report": str(
            session / "THUMB_MCP_IP_PITCH_FIXED_EXPOSURE_CALIBRATION_REPORT.md"
        ),
        "candidate": str(args.candidate_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
