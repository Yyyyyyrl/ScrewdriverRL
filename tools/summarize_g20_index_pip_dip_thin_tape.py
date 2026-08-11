#!/usr/bin/env python3
"""Summarize the thin-tape G20 index PIP/DIP validation session.

This tool is offline-only: it reads recorded camera CSV/JSON and SDK motion
JSON files, then writes a reproducible candidate calibration.  It never imports
the hand SDK, opens CAN, opens the camera, or commands hardware.
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
from typing import Iterable


PIP_FIELD = "pip_projected_deg_2d"
DIP_FIELD = "dip_projected_deg_2d"
PIP_URDF_UPPER_RAD = 1.08
DIP_URDF_MIMIC_MULTIPLIER = 0.8917
DIP_MIN_AREA_PX = 80.0
DIP_MIN_ELONGATION = 10.0
BLOCK_SIZE = 60


@dataclass(frozen=True)
class CameraRun:
    label: str
    prefix: str


@dataclass(frozen=True)
class Anchor:
    command_raw: int
    sdk_record: str
    camera_runs: tuple[CameraRun, ...]


ZERO_RUNS = (
    CameraRun(
        "zero_a",
        "final_zero_raw255/camera_s130_min40_warm180_a_180f",
    ),
    CameraRun(
        "zero_b",
        "final_zero_raw255/camera_s130_min40_warm180_b_180f",
    ),
)

ANCHORS = (
    Anchor(
        184,
        "anchor_raw184/sdk_motion_flexing.json",
        (
            CameraRun(
                "raw184",
                "anchor_raw184/camera_s130_repeat_180f",
            ),
        ),
    ),
    Anchor(
        117,
        "anchor_raw117/sdk_motion_flexing.json",
        (
            CameraRun(
                "raw117_a",
                "anchor_raw117/camera_s130_flexing_180f",
            ),
            CameraRun(
                "raw117_b",
                "anchor_raw117/camera_s130_min40_repeat_180f",
            ),
        ),
    ),
    Anchor(
        101,
        "anchor_raw101/sdk_motion_flexing.json",
        (
            CameraRun(
                "raw101_a",
                "anchor_raw101/camera_s130_min40_flexing_180f",
            ),
            CameraRun(
                "raw101_b",
                "anchor_raw101/camera_s130_min40_warm180_repeat_180f",
            ),
        ),
    ),
)

ZERO_SDK_RECORD = "return_to_raw255/step_10_240_to255.json"
SIM_TARGETS_RAD = (0.5281, 0.97, 1.08)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def _camera_summary(session: Path, run: CameraRun) -> dict:
    prefix = session / run.prefix
    csv_path = Path(f"{prefix}_samples.csv")
    summary_path = Path(f"{prefix}_summary.json")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"no camera rows in {csv_path}")

    pip_rows = rows
    dip_rows = [
        row
        for row in rows
        if float(row["distal_area_px"]) >= DIP_MIN_AREA_PX
        and float(row["distal_elongation_2d"]) >= DIP_MIN_ELONGATION
    ]
    if not dip_rows:
        raise ValueError(f"no DIP rows pass the fixed geometry gate in {csv_path}")

    def reduce(field: str, selected: list[dict[str, str]]) -> dict:
        block_medians = []
        block_counts = []
        for start in range(0, int(summary["capture"]["requested_frames"]), BLOCK_SIZE):
            values = [
                float(row[field])
                for row in selected
                if start <= int(row["frame"]) < start + BLOCK_SIZE
            ]
            if not values:
                raise ValueError(
                    f"empty block {start}:{start + BLOCK_SIZE} in {csv_path}"
                )
            block_counts.append(len(values))
            block_medians.append(_median(values))
        values = [float(row[field]) for row in selected]
        return {
            "kept_frames": len(values),
            "median_deg": _median(values),
            "block_counts": block_counts,
            "block_medians_deg": block_medians,
            "block_range_deg": max(block_medians) - min(block_medians),
        }

    return {
        "label": run.label,
        "prefix": run.prefix,
        "requested_frames": int(summary["capture"]["requested_frames"]),
        "usable_frames": int(summary["capture"]["usable_frames"]),
        "failure_fraction": float(summary["capture"]["failure_fraction"]),
        "failure_gate_pass": float(summary["capture"]["failure_fraction"]) <= 0.05,
        "pip": reduce(PIP_FIELD, pip_rows),
        "dip": reduce(DIP_FIELD, dip_rows),
        "hashes": {
            str(csv_path.relative_to(session)): _sha256(csv_path),
            str(summary_path.relative_to(session)): _sha256(summary_path),
        },
    }


def _sdk_summary(session: Path, relative: str) -> dict:
    path = session / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("motion_sent") or "result" not in payload:
        raise ValueError(f"selected SDK record has no completed motion: {path}")
    samples = payload["post_command_samples"]
    # The first-step records contain a deliberate slow transition.  The final
    # five 5 Hz samples are the window immediately preceding camera capture.
    stable = samples[-5:]
    raw = [float(row["state20"][16]) for row in stable]
    temperature = [
        float(row["temperature20"][16])
        for row in samples
    ]
    return {
        "record": relative,
        "target_raw": int(payload["approved_scope"]["target_raw"]),
        "stable_readback_raw": _median(raw),
        "stable_readback_min_raw": min(raw),
        "stable_readback_max_raw": max(raw),
        "fault_free": bool(payload["result"]["fault_free"]),
        "tip_temperature_max_c": max(temperature),
        "tip_temperature_delta_c": float(
            payload["result"]["temperature_delta20"][16]
        ),
        "non_target_max_abs_change_raw": int(
            payload["result"]["non_target_max_abs_change_raw"]
        ),
        "sha256": _sha256(path),
    }


def _interp(q: float, rows: list[dict], field: str) -> float:
    if q <= rows[0]["pip_rad"]:
        return float(rows[0][field])
    if q >= rows[-1]["pip_rad"]:
        return float(rows[-1][field])
    for left, right in zip(rows, rows[1:]):
        if left["pip_rad"] <= q <= right["pip_rad"]:
            fraction = (
                (q - left["pip_rad"])
                / (right["pip_rad"] - left["pip_rad"])
            )
            return float(
                left[field] + fraction * (right[field] - left[field])
            )
    raise AssertionError("unreachable interpolation interval")


def _all_motion_health(session: Path) -> dict:
    records = []
    for path in sorted(session.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not payload.get("motion_sent") or "result" not in payload:
            continue
        samples = payload["post_command_samples"]
        records.append(
            {
                "record": str(path.relative_to(session)),
                "target_raw": int(payload["approved_scope"]["target_raw"]),
                "fault_free": bool(payload["result"]["fault_free"]),
                "tip_temperature_max_c": max(
                    float(row["temperature20"][16])
                    for row in samples
                ),
                "tip_temperature_delta_c": float(
                    payload["result"]["temperature_delta20"][16]
                ),
                "non_target_max_abs_change_raw": int(
                    payload["result"]["non_target_max_abs_change_raw"]
                ),
            }
        )
    if not records:
        raise ValueError("no completed motion records found")
    return {
        "completed_motion_count": len(records),
        "all_fault_free": all(record["fault_free"] for record in records),
        "max_tip_temperature_c": max(
            record["tip_temperature_max_c"] for record in records
        ),
        "max_tip_temperature_delta_c": max(
            record["tip_temperature_delta_c"] for record in records
        ),
        "max_non_target_abs_change_raw": max(
            record["non_target_max_abs_change_raw"] for record in records
        ),
        "records": records,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = (
        "command_raw",
        "stable_readback_raw",
        "stable_readback_min_raw",
        "stable_readback_max_raw",
        "pip_tape_deg",
        "dip_tape_deg",
        "pip_rad",
        "dip_rad",
        "dip_over_pip",
        "isaac_mimic_dip_rad",
        "dip_mimic_error_rad",
        "current_affine_readback_rad",
        "current_affine_readback_error_rad",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_plot(path: Path, rows: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    command = [row["command_raw"] for row in rows]
    pip = [row["pip_rad"] for row in rows]
    dip = [row["dip_rad"] for row in rows]
    old = [(255.0 - raw) / 255.0 * PIP_URDF_UPPER_RAD for raw in command]
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(command, pip, "o-", label="measured PIP")
    axis.plot(command, dip, "o-", label="measured DIP")
    axis.plot(command, old, "--", label="current affine PIP interpretation")
    axis.invert_xaxis()
    axis.set_xlabel("SDK command raw (255=open, lower=flex)")
    axis.set_ylabel("physical flexion (rad)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _write_report(path: Path, result: dict) -> None:
    rows = result["anchors"]
    targets = result["sim_target_mapping"]
    health = result["hardware_health"]
    lines = [
        "# G20 食指 PIP/DIP 细胶带物理标定复核",
        "",
        "## 结论",
        "",
        "- 实机 PIP 的完整机械范围接近此前宽胶带测得的 1.56 rad；1.08 rad 是当前 Isaac 工作上限，不是硬件满量程。",
        "- 本次细胶带复核测得：PIP 物理 1.083 rad 对应 SDK 命令 raw=101（稳定回读约 102），而不是当前 affine 映射的 raw=0。",
        "- 当前 `linker_calib_thumbfit.json` 的 `lo=0, hi=1.08` 仍通过 0..255 全幅 affine 缩放，因此命令和回读都会产生很大的 sim-real 角度偏差。",
        "- 生产校准文件未覆盖；候选结果必须先由 SDK 映射器支持分段 knots，并且其余三根手指应分别标定后才能全手部署。",
        "",
        "## 物理锚点",
        "",
        "| command raw | stable readback | PIP rad | DIP rad | DIP/PIP | current readback rad | readback error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = "" if row["dip_over_pip"] is None else f"{row['dip_over_pip']:.3f}"
        lines.append(
            f"| {row['command_raw']} | {row['stable_readback_raw']:.1f} "
            f"({row['stable_readback_min_raw']:.0f}..{row['stable_readback_max_raw']:.0f}) "
            f"| {row['pip_rad']:.6f} | {row['dip_rad']:.6f} | {ratio} "
            f"| {row['current_affine_readback_rad']:.6f} "
            f"| {row['current_affine_readback_error_rad']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Isaac rad 到 SDK raw",
            "",
            "| Isaac PIP rad | candidate command raw | candidate readback raw | current command raw |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in targets:
        lines.append(
            f"| {row['sim_pip_rad']:.4f} "
            f"| {row['candidate_command_raw']:.2f} "
            f"| {row['candidate_readback_raw']:.2f} "
            f"| {row['current_affine_command_raw']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## DIP 耦合",
            "",
            f"- Isaac 当前固定 mimic multiplier: `{DIP_URDF_MIMIC_MULTIPLIER}`。",
            f"- 本次三个非零锚点的零截距最小二乘 DIP/PIP 比例: `{result['dip_coupling']['zero_intercept_fit']:.6f}`。",
            "- 实测耦合有非线性：低弯曲点约 0.624，高弯曲点约 0.752；仅把 mimic 常数改成 0.731 只能改善总体误差，不能完美匹配所有角度。",
            "",
            "## 相机与质量门",
            "",
            "- D435 serial: `143322073091`，1280×720@30 FPS，固定 ROI `550,185,1045,550`。",
            "- 固定 HSV: H=85..145, S=130..255, V=25..255；chain-distance 标记关联；关闭 3×3 morphology；最小连通区 40 px。",
            "- 正式采集使用 180 帧自动曝光预热和 180 帧测量；失败率门限 5%。",
            "- PIP 使用全部有效帧。DIP 只保留 distal area>=80 px 且 elongation>=10 的帧，再用 60 帧分块中位数。",
            f"- 最终零位参考：PIP tape={result['zero_reference_deg']['pip']:.6f} deg，DIP tape={result['zero_reference_deg']['dip']:.6f} deg。",
            "",
            "## SDK 安全记录",
            "",
            f"- 已完成并落盘的运动次数：{health['completed_motion_count']}。",
            f"- 全部 fault-free：{health['all_fault_free']}。",
            f"- 食指 tip 最高温度：{health['max_tip_temperature_c']:.0f} °C；最大温升：{health['max_tip_temperature_delta_c']:.0f} °C。",
            f"- 非目标活动关节最大回读变化：{health['max_non_target_abs_change_raw']} raw。",
            "- 最终状态：食指 PIP 命令 raw=255，稳定回读约 raw=254；根关节保持命令 raw=250，回读约 raw=249。",
            "",
            "## 下一步",
            "",
            "1. 给 `linker_sdk_map.py` 增加严格校验的 command/readback piecewise knots，保持未标定关节继续使用旧 affine。",
            "2. 在空手低速下做 PIP 0、0.528、0.97、1.08 rad 的 sim→真机闭环回归，要求 SDK 回读换算和相机物理角均在容差内。",
            "3. 分别标定 middle/ring/pinky；不能把食指 knots 默认复制给其他手指。",
            "4. 再修正 Isaac 的 DIP coupling；当前 0.8917 会比真实 DIP 多弯约 0.14..0.15 rad。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()

    zero_camera = [_camera_summary(session, run) for run in ZERO_RUNS]
    zero_sdk = _sdk_summary(session, ZERO_SDK_RECORD)
    zero_pip_deg = statistics.fmean(
        run["pip"]["median_deg"] for run in zero_camera
    )
    zero_dip_deg = statistics.fmean(
        run["dip"]["median_deg"] for run in zero_camera
    )

    rows = [
        {
            "command_raw": 255,
            **zero_sdk,
            "camera_runs": zero_camera,
            "pip_tape_deg": zero_pip_deg,
            "dip_tape_deg": zero_dip_deg,
            "pip_rad": 0.0,
            "dip_rad": 0.0,
            "dip_over_pip": None,
            "isaac_mimic_dip_rad": 0.0,
            "dip_mimic_error_rad": 0.0,
        }
    ]
    for anchor in ANCHORS:
        camera = [
            _camera_summary(session, run)
            for run in anchor.camera_runs
        ]
        sdk = _sdk_summary(session, anchor.sdk_record)
        pip_deg = statistics.fmean(
            run["pip"]["median_deg"] for run in camera
        )
        dip_deg = statistics.fmean(
            run["dip"]["median_deg"] for run in camera
        )
        pip_rad = math.radians(zero_pip_deg - pip_deg)
        dip_rad = math.radians(zero_dip_deg - dip_deg)
        rows.append(
            {
                "command_raw": anchor.command_raw,
                **sdk,
                "camera_runs": camera,
                "pip_tape_deg": pip_deg,
                "dip_tape_deg": dip_deg,
                "pip_rad": pip_rad,
                "dip_rad": dip_rad,
                "dip_over_pip": dip_rad / pip_rad,
                "isaac_mimic_dip_rad": (
                    DIP_URDF_MIMIC_MULTIPLIER * pip_rad
                ),
                "dip_mimic_error_rad": (
                    dip_rad - DIP_URDF_MIMIC_MULTIPLIER * pip_rad
                ),
            }
        )

    rows.sort(key=lambda row: row["pip_rad"])
    for row in rows:
        readback = float(row["stable_readback_raw"])
        current = (255.0 - readback) / 255.0 * PIP_URDF_UPPER_RAD
        row["current_affine_readback_rad"] = current
        row["current_affine_readback_error_rad"] = (
            current - float(row["pip_rad"])
        )

    target_mapping = []
    for q in SIM_TARGETS_RAD:
        target_mapping.append(
            {
                "sim_pip_rad": q,
                "candidate_command_raw": _interp(q, rows, "command_raw"),
                "candidate_readback_raw": _interp(
                    q, rows, "stable_readback_raw"
                ),
                "current_affine_command_raw": 255.0 * (
                    1.0 - q / PIP_URDF_UPPER_RAD
                ),
            }
        )

    nonzero = rows[1:]
    dip_fit = sum(
        row["pip_rad"] * row["dip_rad"] for row in nonzero
    ) / sum(row["pip_rad"] ** 2 for row in nonzero)
    health = _all_motion_health(session)
    result = {
        "schema_version": 1,
        "status": "candidate_not_promoted",
        "scope": "left G20 index PIP and mechanically coupled DIP only",
        "serial": "LHT20-010-415-L-B-1-D",
        "camera_serial": "143322073091",
        "production_calibration_modified": False,
        "zero_reference_deg": {
            "pip": zero_pip_deg,
            "dip": zero_dip_deg,
        },
        "camera_detector": {
            "roi_xyxy": [550, 185, 1045, 550],
            "hsv_lower": [85, 130, 25],
            "hsv_upper": [145, 255, 255],
            "minimum_component_area_px": 40,
            "marker_association": "chain-distance",
            "morph_open_kernel": 0,
            "warmup_frames_formal": 180,
            "measurement_frames": 180,
            "failure_fraction_limit": 0.05,
            "dip_geometry_gate": {
                "minimum_distal_area_px": DIP_MIN_AREA_PX,
                "minimum_distal_elongation": DIP_MIN_ELONGATION,
            },
        },
        "anchors": rows,
        "piecewise_candidate": {
            "command_knots": [
                {
                    "sim_pip_rad": row["pip_rad"],
                    "sdk_command_raw": row["command_raw"],
                }
                for row in rows
            ],
            "readback_knots": [
                {
                    "physical_pip_rad": row["pip_rad"],
                    "sdk_readback_raw": row["stable_readback_raw"],
                    "observed_raw_interval": [
                        row["stable_readback_min_raw"],
                        row["stable_readback_max_raw"],
                    ],
                }
                for row in rows
            ],
        },
        "sim_target_mapping": target_mapping,
        "dip_coupling": {
            "current_isaac_mimic_multiplier": DIP_URDF_MIMIC_MULTIPLIER,
            "zero_intercept_fit": dip_fit,
            "warning": "physical coupling is nonlinear; constant fit is approximate",
        },
        "hardware_health": health,
    }

    output_json = session / "index_pip_dip_thin_tape_candidate.json"
    output_csv = session / "index_pip_dip_thin_tape_points.csv"
    output_plot = session / "index_pip_dip_thin_tape_curve.png"
    output_report = session / "INDEX_PIP_DIP_THIN_TAPE_REPORT.md"
    output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_csv, rows)
    plot_written = _write_plot(output_plot, rows)
    _write_report(output_report, result)
    print(
        json.dumps(
            {
                "candidate_json": str(output_json),
                "points_csv": str(output_csv),
                "plot": str(output_plot) if plot_written else None,
                "report": str(output_report),
                "production_calibration_modified": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
