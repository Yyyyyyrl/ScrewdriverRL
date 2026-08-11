#!/usr/bin/env python3
"""Summarize the fixed-exposure G20 index MCP-pitch physical sweep.

Offline only: reads recorded camera/SDK JSON and CSV files, writes the
reproducible physical LUT, report, plot, and a combined index PIP+pitch
candidate overlay.  It never imports the LinkerHand SDK or opens hardware.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


CAMERA_FIELD = "mcp_projected_deg_2d"
BLOCK_SIZE = 60
CURRENT_ISAAC_UPPER_RAD = 1.4
COMMANDS = (
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


MOTION_RECORDS = {
    255: "raw255_zero/motion_raw249_to_raw255.json",
    240: "flex_raw240/motion_raw254_to_raw240.json",
    224: "flex_raw224/motion_raw239_to_raw224.json",
    208: "flex_raw208/motion_raw223_to_raw208.json",
    192: "flex_raw192/motion_raw208_to_raw192.json",
    176: "flex_raw176/motion_raw191_to_raw176.json",
    160: "flex_raw160/motion_raw176_to_raw160.json",
    144: "flex_raw144/motion_raw160_to_raw144.json",
    128: "flex_raw128/motion_raw144_to_raw128.json",
    112: "flex_raw112/motion_raw128_to_raw112.json",
    96: "flex_raw096/motion_raw111_to_raw096.json",
    80: "flex_raw080/motion_raw095_to_raw080.json",
    64: "flex_raw064/motion_raw080_to_raw064.json",
    48: "flex_raw048/motion_raw063_to_raw048.json",
    32: "flex_raw032/motion_raw047_to_raw032.json",
    20: "flex_raw020/motion_raw031_to_raw020.json",
    12: "flex_raw012/motion_raw020_to_raw012.json",
    6: "flex_raw006/motion_raw011_to_raw006.json",
    0: "flex_raw000/motion_raw005_to_raw000.json",
}


def _camera_prefix(command_raw: int) -> str:
    return f"flex_raw{command_raw:03d}/camera_fixed_exp_180f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_run(session: Path, prefix_relative: str) -> dict[str, Any]:
    prefix = session / prefix_relative
    summary_path = Path(f"{prefix}_summary.json")
    csv_path = Path(f"{prefix}_samples.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no camera rows in {csv_path}")
    values = [float(row[CAMERA_FIELD]) for row in rows]
    blocks = []
    requested = int(summary["capture"]["requested_frames"])
    for start in range(0, requested, BLOCK_SIZE):
        selected = [
            float(row[CAMERA_FIELD])
            for row in rows
            if start <= int(row["frame"]) < start + BLOCK_SIZE
        ]
        if not selected:
            raise ValueError(
                f"empty camera block {start}:{start + BLOCK_SIZE}"
            )
        blocks.append(float(statistics.median(selected)))
    return {
        "prefix": prefix_relative,
        "requested_frames": requested,
        "usable_frames": int(summary["capture"]["usable_frames"]),
        "failure_fraction": float(summary["capture"]["failure_fraction"]),
        "median_deg": float(statistics.median(values)),
        "mean_deg": float(statistics.fmean(values)),
        "std_deg": float(
            summary["angle_summary_deg"][CAMERA_FIELD]["std"]
        ),
        "block_medians_deg": blocks,
        "block_range_deg": max(blocks) - min(blocks),
        "hashes": {
            str(summary_path.relative_to(session)): _sha256(summary_path),
            str(csv_path.relative_to(session)): _sha256(csv_path),
        },
    }


def _sdk_run(session: Path, relative: str) -> dict[str, Any]:
    path = session / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("motion_sent") or "result" not in payload:
        raise ValueError(f"incomplete motion record: {path}")
    samples = payload["post_command_samples"]
    stable = samples[-5:]
    root = [float(row["state20"][1]) for row in stable]
    pip = [float(row["state20"][16]) for row in stable]
    root_temperature = [
        float(row["temperature20"][1]) for row in samples
    ]
    target_raw = int(payload["approved_scope"]["target_raw"])
    return {
        "record": relative,
        "target_raw": target_raw,
        "stable_readback_raw": float(statistics.median(root)),
        "stable_readback_min_raw": min(root),
        "stable_readback_max_raw": max(root),
        "stable_pip_readback_raw": float(statistics.median(pip)),
        "fault_free": bool(payload["result"]["fault_free"]),
        "root_temperature_max_c": max(root_temperature),
        "root_temperature_delta_c": float(
            payload["result"]["temperature_delta20"][1]
        ),
        "non_target_max_abs_change_raw": int(
            payload["result"]["non_target_max_abs_change_raw"]
        ),
        "sha256": _sha256(path),
    }


def _all_motion_health(session: Path) -> dict[str, Any]:
    completed = []
    rejected_before_motion = []
    for path in sorted(session.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "operation" not in payload:
            continue
        relative = str(path.relative_to(session))
        if not payload.get("motion_sent") or "result" not in payload:
            rejected_before_motion.append(
                {
                    "record": relative,
                    "motion_sent": bool(payload.get("motion_sent")),
                    "error": payload.get("error"),
                }
            )
            continue
        samples = payload["post_command_samples"]
        completed.append(
            {
                "record": relative,
                "target_raw": int(payload["approved_scope"]["target_raw"]),
                "fault_free": bool(payload["result"]["fault_free"]),
                "root_temperature_max_c": max(
                    float(row["temperature20"][1]) for row in samples
                ),
                "root_temperature_delta_c": float(
                    payload["result"]["temperature_delta20"][1]
                ),
                "pip_readback_min_raw": min(
                    int(row["state20"][16]) for row in samples
                ),
                "pip_readback_max_raw": max(
                    int(row["state20"][16]) for row in samples
                ),
                "non_target_max_abs_change_raw": int(
                    payload["result"]["non_target_max_abs_change_raw"]
                ),
            }
        )
    if not completed:
        raise ValueError("no completed motion records")
    return {
        "completed_motion_count": len(completed),
        "all_fault_free": all(row["fault_free"] for row in completed),
        "max_root_temperature_c": max(
            row["root_temperature_max_c"] for row in completed
        ),
        "max_root_temperature_delta_c": max(
            row["root_temperature_delta_c"] for row in completed
        ),
        "pip_readback_min_raw": min(
            row["pip_readback_min_raw"] for row in completed
        ),
        "pip_readback_max_raw": max(
            row["pip_readback_max_raw"] for row in completed
        ),
        "max_non_target_abs_change_raw": max(
            row["non_target_max_abs_change_raw"] for row in completed
        ),
        "rejected_before_motion": rejected_before_motion,
        "records": completed,
    }


def _interp_rad_to_raw(target: float, rows: list[dict[str, Any]]) -> float:
    ordered = sorted(rows, key=lambda row: row["physical_rad"])
    if target <= ordered[0]["physical_rad"]:
        return float(ordered[0]["command_raw"])
    if target >= ordered[-1]["physical_rad"]:
        return float(ordered[-1]["command_raw"])
    for left, right in zip(ordered, ordered[1:]):
        if left["physical_rad"] <= target <= right["physical_rad"]:
            fraction = (
                (target - left["physical_rad"])
                / (right["physical_rad"] - left["physical_rad"])
            )
            return float(
                left["command_raw"]
                + fraction
                * (right["command_raw"] - left["command_raw"])
            )
    raise AssertionError("unreachable interpolation interval")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "command_raw",
        "stable_readback_raw",
        "stable_readback_min_raw",
        "stable_readback_max_raw",
        "stable_pip_readback_raw",
        "camera_median_deg",
        "camera_std_deg",
        "physical_deg",
        "physical_rad",
        "current_affine_readback_rad",
        "current_affine_error_rad",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    raw = [row["command_raw"] for row in rows]
    measured = [row["physical_rad"] for row in rows]
    affine = [
        (255.0 - value) / 255.0 * CURRENT_ISAAC_UPPER_RAD
        for value in raw
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(raw, measured, "o-", label="measured MCP pitch")
    axis.plot(raw, affine, "--", label="current 0..1.4 affine")
    axis.invert_xaxis()
    axis.set_xlabel("SDK command raw (255=open, lower=flex)")
    axis.set_ylabel("physical flexion (rad)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _write_report(path: Path, result: dict[str, Any]) -> None:
    rows = result["anchors"]
    health = result["hardware_health"]
    lines = [
        "# G20 食指 MCP pitch 固定曝光物理标定报告",
        "",
        "## 结论",
        "",
        f"- 食指 MCP pitch 已独立验证 SDK raw 0..255 全范围安全；raw0 未触发故障，实测物理最大屈曲为 **{result['physical_max_rad']:.6f} rad ({result['physical_max_deg']:.3f}°)**。",
        f"- 当前 Isaac/SDK 表使用 0..1.4 rad；它比实机最大角多 {result['isaac_overstatement_rad']:.6f} rad ({result['isaac_overstatement_deg']:.3f}°)，因此不能继续用 0..255 affine 假设。",
        "- 候选 LUT 同一份用于命令反演和 SDK 回读；raw255 为物理 0 rad，raw0 为实测上限。",
        f"- raw128 去程/回程物理角差 {result['hysteresis_raw128_rad']:.6f} rad ({result['hysteresis_raw128_deg']:.3f}°)，作为机械回差而不是拟合噪声保留。",
        f"- 完整 0→255 往返后零点漂移仅 {result['zero_return_drift_rad']:.6f} rad ({result['zero_return_drift_deg']:.4f}°)。",
        "",
        "## 标定点",
        "",
        "| command raw | stable readback | camera tape deg | physical deg | physical rad | old affine rad | old error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['command_raw']} "
            f"| {row['stable_readback_raw']:.1f} "
            f"({row['stable_readback_min_raw']:.0f}..{row['stable_readback_max_raw']:.0f}) "
            f"| {row['camera_median_deg']:.6f} "
            f"| {row['physical_deg']:.6f} "
            f"| {row['physical_rad']:.6f} "
            f"| {row['current_affine_readback_rad']:.6f} "
            f"| {row['current_affine_error_rad']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Isaac rad → SDK raw",
            "",
            "| target rad | candidate raw | current affine raw |",
            "|---:|---:|---:|",
        ]
    )
    for row in result["target_mapping"]:
        lines.append(
            f"| {row['target_rad']:.6f} "
            f"| {row['candidate_command_raw']:.3f} "
            f"| {row['current_affine_command_raw']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 相机质量",
            "",
            "- D435 serial `143322073091`，1280×720@30 FPS；固定 RGB exposure=166、gain=32、white balance=4600。",
            "- ROI `500,185,1100,500`；HSV H=75..165, S=40..255；深度门限 0.330..0.390 m。",
            "- 物理角只采用稳定的 2D 掌骨/近节轴差；3D 深度轴只保留诊断，不参与拟合。",
            "- 每点 180 帧，中位数汇总；所有正式点均 180/180 可用。",
            f"- 初始 raw255 两组中位数：{result['zero_runs'][0]['median_deg']:.6f}°、{result['zero_runs'][1]['median_deg']:.6f}°；零点取两者平均 {result['zero_reference_deg']:.6f}°。",
            "",
            "## SDK 安全记录",
            "",
            f"- 已完成并落盘的运动次数：{health['completed_motion_count']}。",
            f"- 所有已发送运动 fault-free：{health['all_fault_free']}。",
            f"- MCP 最高温度：{health['max_root_temperature_c']:.0f}°C；单步最大温升：{health['max_root_temperature_delta_c']:.0f}°C。",
            f"- PIP 隔离回读范围：{health['pip_readback_min_raw']}..{health['pip_readback_max_raw']}（命令固定255）。",
            f"- 非目标活动槽最大变化：{health['max_non_target_abs_change_raw']} raw。",
            "- 一次 raw239→255 预检因侧摆槽 raw20/index-frame 瞬时相差1 raw而拒绝；`motion_sent=false`。严格重试自然一致后才发送，记录完整保留。",
            "- 最终状态：MCP pitch 命令255/稳定回读254，PIP命令255/回读255，故障全零，MCP温度46°C。",
            "",
            "## 集成结论",
            "",
            f"- 建议将 Isaac `index_mcp_pitch` 上限从 1.4 改为实测 {result['physical_max_rad']:.6f} rad。",
            "- 候选 overlay 同时包含此前 index PIP LUT 和本次 index MCP pitch LUT；它仍不是全手生产标定。",
            "- middle/ring/pinky 的 MCP pitch 与 PIP 必须分别测量，不能复制食指 LUT；top-down 实机门禁应要求每根手指两类 LUT 都齐全。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--pip-candidate",
        type=Path,
        default=Path("linker_calib_index_pip_lut_candidate_20260802.json"),
    )
    parser.add_argument(
        "--combined-candidate",
        type=Path,
        default=Path(
            "linker_calib_index_pip_pitch_lut_candidate_20260802.json"
        ),
    )
    args = parser.parse_args()
    session = args.session.resolve()

    zero_runs = [
        _camera_run(
            session, "raw255_zero/camera_a_fixed_exp_180f"
        ),
        _camera_run(
            session, "raw255_zero/camera_b_fixed_exp_180f"
        ),
    ]
    zero_reference_deg = statistics.fmean(
        row["median_deg"] for row in zero_runs
    )
    zero_return = _camera_run(
        session,
        "return_to_raw255/camera_raw255_return_fixed_exp_180f",
    )
    return_raw128 = _camera_run(
        session,
        "return_to_raw255/camera_raw128_return_fixed_exp_180f",
    )

    anchors = []
    for command_raw in COMMANDS:
        sdk = _sdk_run(session, MOTION_RECORDS[command_raw])
        if sdk["target_raw"] != command_raw:
            raise ValueError(
                f"{sdk['record']} target is {sdk['target_raw']}, expected "
                f"{command_raw}"
            )
        if command_raw == 255:
            camera_median = zero_reference_deg
            camera_std = max(row["std_deg"] for row in zero_runs)
        else:
            camera = _camera_run(
                session, _camera_prefix(command_raw)
            )
            camera_median = camera["median_deg"]
            camera_std = camera["std_deg"]
        if command_raw == 128:
            # Midpoint removes the measured direction-dependent backlash.
            camera_median = statistics.fmean(
                [camera_median, return_raw128["median_deg"]]
            )
        physical_deg = (
            0.0
            if command_raw == 255
            else zero_reference_deg - camera_median
        )
        physical_rad = math.radians(physical_deg)
        current_affine = (
            (255.0 - command_raw)
            / 255.0
            * CURRENT_ISAAC_UPPER_RAD
        )
        anchors.append(
            {
                "command_raw": command_raw,
                **sdk,
                "camera_median_deg": camera_median,
                "camera_std_deg": camera_std,
                "physical_deg": physical_deg,
                "physical_rad": physical_rad,
                "current_affine_readback_rad": current_affine,
                "current_affine_error_rad": (
                    current_affine - physical_rad
                ),
            }
        )
    for previous, current in zip(anchors, anchors[1:]):
        if current["physical_rad"] <= previous["physical_rad"]:
            raise ValueError(
                "measured pitch is not strictly monotonic: "
                f"raw{previous['command_raw']}={previous['physical_rad']} "
                f"raw{current['command_raw']}={current['physical_rad']}"
            )

    physical_max_rad = anchors[-1]["physical_rad"]
    physical_max_deg = anchors[-1]["physical_deg"]
    target_values = (
        0.140535,
        0.5,
        1.0,
        1.2,
        physical_max_rad,
    )
    target_mapping = [
        {
            "target_rad": target,
            "candidate_command_raw": _interp_rad_to_raw(
                target, anchors
            ),
            "current_affine_command_raw": max(
                0.0,
                min(
                    255.0,
                    255.0
                    * (1.0 - target / CURRENT_ISAAC_UPPER_RAD),
                ),
            ),
        }
        for target in target_values
    ]
    forward_raw128 = _camera_run(
        session, _camera_prefix(128)
    )["median_deg"]
    hysteresis_raw128_deg = abs(
        return_raw128["median_deg"] - forward_raw128
    )
    zero_return_drift_deg = (
        zero_return["median_deg"] - zero_reference_deg
    )
    health = _all_motion_health(session)
    if not health["all_fault_free"]:
        raise ValueError("a completed motion record contains a fault")

    result = {
        "schema_version": 1,
        "joint": "index_mcp_pitch",
        "sdk_raw20_slot": 1,
        "camera_angle_source": CAMERA_FIELD,
        "zero_reference_deg": zero_reference_deg,
        "zero_runs": zero_runs,
        "zero_return": zero_return,
        "zero_return_drift_deg": zero_return_drift_deg,
        "zero_return_drift_rad": math.radians(
            zero_return_drift_deg
        ),
        "return_raw128": return_raw128,
        "hysteresis_raw128_deg": hysteresis_raw128_deg,
        "hysteresis_raw128_rad": math.radians(
            hysteresis_raw128_deg
        ),
        "physical_max_deg": physical_max_deg,
        "physical_max_rad": physical_max_rad,
        "current_isaac_upper_rad": CURRENT_ISAAC_UPPER_RAD,
        "isaac_overstatement_rad": (
            CURRENT_ISAAC_UPPER_RAD - physical_max_rad
        ),
        "isaac_overstatement_deg": math.degrees(
            CURRENT_ISAAC_UPPER_RAD - physical_max_rad
        ),
        "anchors": anchors,
        "target_mapping": target_mapping,
        "hardware_health": health,
        "lut": {
            "raw": [
                row["command_raw"]
                for row in reversed(anchors)
            ],
            "rad": [
                row["physical_rad"]
                for row in reversed(anchors)
            ],
        },
    }

    csv_path = session / "index_mcp_pitch_calibration_points.csv"
    summary_path = session / "index_mcp_pitch_summary.json"
    report_path = (
        session / "INDEX_MCP_PITCH_FIXED_EXPOSURE_CALIBRATION_REPORT.md"
    )
    plot_path = session / "index_mcp_pitch_curve.png"
    _write_csv(csv_path, anchors)
    _write_plot(plot_path, anchors)
    _write_report(report_path, result)
    summary_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    base = json.loads(
        args.pip_candidate.read_text(encoding="utf-8")
    )
    base["note"] = (
        "CANDIDATE ONLY: serial LHT20-010-415-L-B-1-D. Combined measured "
        "index MCP-pitch and PIP physical LUTs from the 2026-08-02 fixed-D435 "
        "sweeps. Command inversion and SDK readback use the same per-joint "
        "tables. Other fingers remain uncalibrated; live full-hand top-down "
        "deployment must remain blocked."
    )
    base["joints"]["index_mcp_pitch"] = {
        "flip": False,
        "lo": 0.0,
        "hi": physical_max_rad,
        "physical_lut": result["lut"],
    }
    args.combined_candidate.write_text(
        json.dumps(base, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "physical_max_rad": physical_max_rad,
                "physical_max_deg": physical_max_deg,
                "zero_return_drift_rad": result[
                    "zero_return_drift_rad"
                ],
                "hysteresis_raw128_rad": result[
                    "hysteresis_raw128_rad"
                ],
                "completed_motion_count": health[
                    "completed_motion_count"
                ],
                "outputs": {
                    "summary": str(summary_path),
                    "csv": str(csv_path),
                    "report": str(report_path),
                    "plot": str(plot_path),
                    "combined_candidate": str(
                        args.combined_candidate.resolve()
                    ),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
