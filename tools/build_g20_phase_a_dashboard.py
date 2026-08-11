#!/usr/bin/env python3
"""Build the source-backed Phase A G20 calibration dashboard artifact."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RECORD_ROOT = ROOT / "records/g20_physical_joint_calibration_20260801"
SESSION_ROOT = RECORD_ROOT / "sessions"

INDEX_PIP = SESSION_ROOT / (
    "20260802T060815Z_index_pip_dip_thin_tape_full_raw_sweep/"
    "index_pip_dip_fixed_exposure_validation.json"
)
INDEX_PITCH = SESSION_ROOT / (
    "20260802T191709Z_index_mcp_pitch_thin_tape_full_raw_sweep/"
    "index_mcp_pitch_summary.json"
)
THUMB = SESSION_ROOT / (
    "20260802T202449Z_thumb_mcp_ip_pitch_thin_tape_full_raw_sweep/"
    "thumb_mcp_ip_pitch_summary.json"
)
MIDDLE = SESSION_ROOT / (
    "20260802T_middle_pip_dip_pitch_thin_tape_full_raw_sweep/"
    "middle_pip_dip_pitch_summary.json"
)
RING = SESSION_ROOT / (
    "20260802T_ring_pip_dip_pitch_thin_tape_full_raw_sweep/"
    "ring_pip_dip_pitch_summary.json"
)
PINKY = SESSION_ROOT / (
    "20260802T_pinky_pip_dip_pitch_thin_tape_full_raw_sweep/"
    "pinky_pip_dip_pitch_summary.json"
)
MAPPING_VALIDATION = PINKY.parent / "phase_a_candidate_mapping_validation.json"
CANDIDATE = ROOT / (
    "linker_calib_index_thumb_middle_ring_pinky_mcp_pitch_lut_candidate_20260802.json"
)
URDF = ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"

FINGER_ZH = {
    "index": "食指 Index",
    "middle": "中指 Middle",
    "ring": "无名指 Ring",
    "pinky": "小指 Pinky",
    "thumb": "拇指 Thumb",
}
JOINT_ZH = {
    "index_mcp_pitch": "食指 MCP pitch",
    "index_pip": "食指 PIP",
    "middle_mcp_pitch": "中指 MCP pitch",
    "middle_pip": "中指 PIP",
    "ring_mcp_pitch": "无名指 MCP pitch",
    "ring_pip": "无名指 PIP",
    "pinky_mcp_pitch": "小指 MCP pitch",
    "pinky_pip": "小指 PIP",
    "thumb_cmc_pitch": "拇指 CMC pitch",
    "thumb_mcp": "拇指 MCP",
    "index_dip": "食指 DIP",
    "middle_dip": "中指 DIP",
    "ring_dip": "无名指 DIP",
    "pinky_dip": "小指 DIP",
    "thumb_ip": "拇指 IP",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def rad_to_deg(value: float) -> float:
    return math.degrees(value)


def joint_finger(name: str) -> str:
    return name.split("_", 1)[0]


def urdf_contract() -> tuple[dict[str, float], dict[str, float]]:
    root = ET.parse(URDF).getroot()
    limits: dict[str, float] = {}
    mimics: dict[str, float] = {}
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        if limit is not None and name in JOINT_ZH:
            limits[name] = float(limit.get("upper"))
        if mimic is not None and name in JOINT_ZH:
            mimics[name] = float(mimic.get("multiplier"))
    return limits, mimics


def build_data() -> tuple[dict[str, list[dict]], list[dict], dict]:
    candidate = read_json(CANDIDATE)
    validation = read_json(MAPPING_VALIDATION)
    index_pip = read_json(INDEX_PIP)
    index_pitch = read_json(INDEX_PITCH)
    thumb = read_json(THUMB)
    middle = read_json(MIDDLE)
    ring = read_json(RING)
    pinky = read_json(PINKY)
    urdf_limits, urdf_mimics = urdf_contract()

    active_curves: list[dict] = []
    range_detail: list[dict] = []
    range_long: list[dict] = []
    for name, entry in candidate["joints"].items():
        lut = entry.get("physical_lut")
        if not lut:
            continue
        finger = joint_finger(name)
        measured = float(entry["hi"])
        current = float(urdf_limits[name])
        raw_min = float(lut["raw"][0])
        mismatch = measured - current
        status = (
            "当前limit偏小"
            if mismatch > 0.01
            else "当前limit偏大"
            if mismatch < -0.01
            else "基本一致"
        )
        row = {
            "finger": FINGER_ZH[finger],
            "finger_key": finger,
            "joint": name,
            "joint_label": JOINT_ZH[name],
            "joint_family": "PIP" if name.endswith("_pip") else "MCP/CMC pitch",
            "current_urdf_rad": current,
            "measured_safe_rad": measured,
            "mismatch_rad": mismatch,
            "mismatch_deg": rad_to_deg(mismatch),
            "measured_to_current_ratio": measured / current,
            "min_valid_command_raw": raw_min,
            "status": status,
        }
        range_detail.append(row)
        for series, value in (
            ("标定前：当前URDF limit", current),
            ("标定后：实测安全端点", measured),
        ):
            range_long.append(
                {
                    **row,
                    "series": series,
                    "limit_rad": value,
                    "limit_deg": rad_to_deg(value),
                }
            )
        raw0 = float(lut["raw"][0])
        for raw, rad in zip(lut["raw"], lut["rad"]):
            raw_f = float(raw)
            rad_f = float(rad)
            active_curves.append(
                {
                    "finger": FINGER_ZH[finger],
                    "finger_key": finger,
                    "joint": name,
                    "joint_label": JOINT_ZH[name],
                    "joint_family": row["joint_family"],
                    "command_raw": raw_f,
                    "measured_rad": rad_f,
                    "measured_deg": rad_to_deg(rad_f),
                    "normalized_flex": (255.0 - raw_f) / (255.0 - raw0),
                    "normalized_angle": rad_f / measured,
                    "current_urdf_rad": current,
                    "min_valid_command_raw": raw_min,
                }
            )

    mimic_detail = [
        {
            "finger": FINGER_ZH["index"],
            "passive_joint": "index_dip",
            "passive_joint_label": JOINT_ZH["index_dip"],
            "current_multiplier": urdf_mimics["index_dip"],
            "measured_multiplier": index_pip["dip_coupling_candidate"][
                "zero_offset_multiplier"
            ],
            "affine_slope": index_pip["dip_coupling_candidate"]["affine_multiplier"],
            "affine_offset_rad": index_pip["dip_coupling_candidate"][
                "affine_offset_rad"
            ],
            "max_residual_rad": index_pip["dip_coupling_candidate"][
                "zero_offset_max_abs_residual_rad"
            ],
        },
        {
            "finger": FINGER_ZH["middle"],
            "passive_joint": "middle_dip",
            "passive_joint_label": JOINT_ZH["middle_dip"],
            "current_multiplier": urdf_mimics["middle_dip"],
            "measured_multiplier": middle["mimic_fit"][
                "zero_intercept_multiplier"
            ],
            "affine_slope": middle["mimic_fit"]["affine_multiplier"],
            "affine_offset_rad": middle["mimic_fit"]["affine_offset_rad"],
            "max_residual_rad": middle["mimic_fit"][
                "zero_intercept_max_abs_residual_rad"
            ],
        },
        {
            "finger": FINGER_ZH["ring"],
            "passive_joint": "ring_dip",
            "passive_joint_label": JOINT_ZH["ring_dip"],
            "current_multiplier": urdf_mimics["ring_dip"],
            "measured_multiplier": ring["mimic_fit"]["zero_intercept_multiplier"],
            "affine_slope": ring["mimic_fit"]["affine_multiplier"],
            "affine_offset_rad": ring["mimic_fit"]["affine_offset_rad"],
            "max_residual_rad": ring["mimic_fit"][
                "zero_intercept_max_abs_residual_rad"
            ],
        },
        {
            "finger": FINGER_ZH["pinky"],
            "passive_joint": "pinky_dip",
            "passive_joint_label": JOINT_ZH["pinky_dip"],
            "current_multiplier": urdf_mimics["pinky_dip"],
            "measured_multiplier": pinky["mimic_fit"]["zero_intercept_multiplier"],
            "affine_slope": pinky["mimic_fit"]["affine_multiplier"],
            "affine_offset_rad": pinky["mimic_fit"]["affine_offset_rad"],
            "max_residual_rad": pinky["mimic_fit"][
                "zero_intercept_max_abs_residual_rad"
            ],
        },
        {
            "finger": FINGER_ZH["thumb"],
            "passive_joint": "thumb_ip",
            "passive_joint_label": JOINT_ZH["thumb_ip"],
            "current_multiplier": urdf_mimics["thumb_ip"],
            "measured_multiplier": thumb["thumb_ip"]["fit"][
                "zero_intercept_multiplier"
            ],
            "affine_slope": thumb["thumb_ip"]["fit"]["affine_multiplier"],
            "affine_offset_rad": thumb["thumb_ip"]["fit"]["affine_offset_rad"],
            "max_residual_rad": thumb["thumb_ip"]["fit"][
                "zero_intercept_max_abs_residual_rad"
            ],
        },
    ]
    mimic_long: list[dict] = []
    for row in mimic_detail:
        row["multiplier_delta"] = (
            row["measured_multiplier"] - row["current_multiplier"]
        )
        for series, value in (
            ("标定前：URDF mimic", row["current_multiplier"]),
            ("标定后：零截距拟合", row["measured_multiplier"]),
        ):
            mimic_long.append({**row, "series": series, "multiplier": value})

    hysteresis = [
        {
            "finger": FINGER_ZH["index"],
            "joint_label": JOINT_ZH["index_mcp_pitch"],
            "joint_role": "主动",
            "signed_rad": index_pitch["hysteresis_raw128_rad"],
        },
        {
            "finger": FINGER_ZH["thumb"],
            "joint_label": JOINT_ZH["thumb_mcp"],
            "joint_role": "主动",
            "signed_rad": thumb["thumb_mcp"]["raw128_hysteresis_rad"],
        },
        {
            "finger": FINGER_ZH["thumb"],
            "joint_label": JOINT_ZH["thumb_ip"],
            "joint_role": "被动",
            "signed_rad": thumb["thumb_ip"]["raw128_hysteresis_rad"],
        },
        {
            "finger": FINGER_ZH["thumb"],
            "joint_label": JOINT_ZH["thumb_cmc_pitch"],
            "joint_role": "主动",
            "signed_rad": thumb["thumb_cmc_pitch"]["raw128_hysteresis_rad"],
        },
    ]
    for finger, payload in (
        ("middle", middle),
        ("ring", ring),
        ("pinky", pinky),
    ):
        hyst = payload["hysteresis_raw128"]
        for suffix, role in (
            ("pip", "主动"),
            ("dip", "被动"),
            ("mcp_pitch", "主动"),
        ):
            hysteresis.append(
                {
                    "finger": FINGER_ZH[finger],
                    "joint_label": JOINT_ZH[f"{finger}_{suffix}"],
                    "joint_role": role,
                    "signed_rad": hyst[f"{finger}_{suffix}_rad"],
                }
            )
    for row in hysteresis:
        row["signed_deg"] = rad_to_deg(row["signed_rad"])
        row["abs_hysteresis_deg"] = abs(row["signed_deg"])

    error_rows = [
        {
            "finger": FINGER_ZH["index"],
            "joint_label": JOINT_ZH["index_pip"],
            "metric": "固定曝光 vs 双向中线最大差",
            "error_rad": index_pip["pip_cross_validation"]["max_abs_delta_rad"],
        },
        {
            "finger": FINGER_ZH["thumb"],
            "joint_label": JOINT_ZH["thumb_mcp"],
            "metric": "held-out分段线性最大误差",
            "error_rad": thumb["thumb_mcp"]["heldout_max_abs_error_rad"],
        },
        {
            "finger": FINGER_ZH["thumb"],
            "joint_label": JOINT_ZH["thumb_cmc_pitch"],
            "metric": "held-out分段线性最大误差",
            "error_rad": thumb["thumb_cmc_pitch"]["heldout_max_abs_error_rad"],
        },
    ]
    for finger, payload in (
        ("middle", middle),
        ("ring", ring),
        ("pinky", pinky),
    ):
        heldout = payload["heldout_piecewise_linear_max_abs_error_rad"]
        for suffix in ("pip", "dip", "mcp_pitch"):
            error_rows.append(
                {
                    "finger": FINGER_ZH[finger],
                    "joint_label": JOINT_ZH[f"{finger}_{suffix}"],
                    "metric": "held-out分段线性最大误差",
                    "error_rad": heldout[f"{finger}_{suffix}"],
                }
            )
    for row in error_rows:
        row["error_deg"] = rad_to_deg(row["error_rad"])

    safety = [
        {
            "finger": FINGER_ZH["index"],
            "accepted_motion_records": 57,
            "post_samples": None,
            "fault_samples_accepted": 0,
            "max_temperature_c": max(
                index_pip["hardware_return"]["max_tip_temperature_c"],
                index_pitch["hardware_health"]["max_root_temperature_c"],
            ),
            "excluded_or_rejected": "PIP raw0触发fault64；pitch一次发送前状态不同步",
            "final_state": "安全回到raw255，最终fault=0",
        },
        {
            "finger": FINGER_ZH["thumb"],
            "accepted_motion_records": thumb["safety"]["motion_sent_count"],
            "post_samples": None,
            "fault_samples_accepted": 0,
            "max_temperature_c": thumb["safety"][
                "max_temperature_c_across_all_slots"
            ],
            "excluded_or_rejected": "2次preflight拒绝，均未发送",
            "final_state": "最终只读快照正常",
        },
        {
            "finger": FINGER_ZH["middle"],
            "accepted_motion_records": middle["motion_health"][
                "sent_middle_motion_files"
            ],
            "post_samples": middle["motion_health"][
                "recorded_post_command_samples"
            ],
            "fault_samples_accepted": middle["motion_health"][
                "middle_fault_samples"
            ],
            "max_temperature_c": middle["motion_health"][
                "max_middle_temperature_c"
            ],
            "excluded_or_rejected": "2份运动记录缺post samples；后续重试有完整证据",
            "final_state": "PIP/pitch回到raw255，最终fault=0",
        },
        {
            "finger": FINGER_ZH["ring"],
            "accepted_motion_records": ring["motion_health"][
                "sent_ring_motion_files"
            ],
            "post_samples": ring["motion_health"]["recorded_post_command_samples"],
            "fault_samples_accepted": ring["motion_health"]["ring_fault_samples"],
            "max_temperature_c": ring["motion_health"][
                "max_ring_temperature_c"
            ],
            "excluded_or_rejected": "deep-pitch遮挡批次被拒绝并重拍",
            "final_state": "最终20帧fault=0",
        },
        {
            "finger": FINGER_ZH["pinky"],
            "accepted_motion_records": pinky["motion_health"][
                "sent_pinky_motion_files"
            ],
            "post_samples": pinky["motion_health"][
                "recorded_post_command_samples"
            ],
            "fault_samples_accepted": pinky["motion_health"][
                "pinky_fault_samples"
            ],
            "max_temperature_c": pinky["motion_health"][
                "max_pinky_temperature_c"
            ],
            "excluded_or_rejected": "pitch raw0发生side耦合，LUT从raw6开始",
            "final_state": "最终20帧fault=0，视觉复核180/180",
        },
    ]

    exceptions = [
        {
            "severity": "阻断部署",
            "finger": "全手",
            "item": "旧Topdown门禁",
            "evidence": "四根PIP仍硬编码hi=1.08 rad",
            "handling": "后续按实测端点修改Isaac limits与门禁；当前保持fail-closed",
        },
        {
            "severity": "安全端点",
            "finger": FINGER_ZH["index"],
            "item": "PIP raw0",
            "evidence": "稳定回读约raw4时触发fault64",
            "handling": "候选LUT最低raw20，故障点只留作极限证据",
        },
        {
            "severity": "安全端点",
            "finger": FINGER_ZH["pinky"],
            "item": "MCP pitch raw0",
            "evidence": "pitch回读raw3，side从raw153偏到raw140",
            "handling": "raw0拒绝；候选LUT最低raw6并向下夹紧",
        },
        {
            "severity": "测量质量",
            "finger": FINGER_ZH["middle"],
            "item": "PIP深弯低raw",
            "evidence": "部分点可用帧低于180；报告已降置信度",
            "handling": "保留逐帧证据与held-out误差，不隐藏不确定性",
        },
        {
            "severity": "测量质量",
            "finger": FINGER_ZH["ring"],
            "item": "pitch深弯遮挡",
            "evidence": "首批数据不单调且标记受遮挡",
            "handling": "整批拒绝；移开遮挡手指后重拍进入LUT",
        },
        {
            "severity": "测量质量",
            "finger": FINGER_ZH["pinky"],
            "item": "pitch raw112标记合并",
            "evidence": "两批次与其他蓝标记瞬时合并",
            "handling": "两批拒绝；避让后180/180重拍进入LUT",
        },
    ]

    total_motions = sum(row["accepted_motion_records"] for row in safety)
    hero = [
        {
            "phase_complete": 15,
            "phase_total": 15,
            "active_lut_joints": len(
                [row for row in candidate["joints"].values() if row.get("physical_lut")]
            ),
            "lut_knots": validation["knots_checked"],
            "max_command_raw_error": validation["max_command_raw_error"],
            "accepted_motion_records": total_motions,
            "max_temperature_c": max(row["max_temperature_c"] for row in safety),
            "excluded_physical_endpoints": 2,
        }
    ]

    datasets = {
        "hero": hero,
        "range_detail": sorted(range_detail, key=lambda row: row["joint_label"]),
        "range_long": sorted(
            range_long, key=lambda row: (row["joint_label"], row["series"])
        ),
        "active_curves": sorted(
            active_curves, key=lambda row: (row["joint_label"], row["command_raw"])
        ),
        "mimic_detail": mimic_detail,
        "mimic_long": mimic_long,
        "hysteresis": sorted(
            hysteresis, key=lambda row: row["abs_hysteresis_deg"], reverse=True
        ),
        "error": sorted(error_rows, key=lambda row: row["error_deg"], reverse=True),
        "safety": safety,
        "exceptions": exceptions,
    }

    source_files = [
        CANDIDATE,
        URDF,
        INDEX_PIP,
        INDEX_PITCH,
        THUMB,
        MIDDLE,
        RING,
        PINKY,
        MAPPING_VALIDATION,
        RECORD_ROOT / "CALIBRATION_RECORD_INDEX.md",
        ROOT / "docs/g20-physical-joint-calibration-runbook.md",
    ]
    sources = [
        {
            "id": "phase-a-derived",
            "label": "Phase A校准汇总与可复现转换",
            "path": rel(Path(__file__)),
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    "SELECT * FROM read_json_auto("
                    "'deliverables/g20_phase_a_calibration_dashboard/data/hero.json', "
                    "format='array')"
                ),
                "description": (
                    "读取五根手指最终summary、聚合候选LUT、当前运行URDF和映射验证，"
                    "统一生成端点、曲线、mimic、回差、误差与安全数据集。"
                ),
                "tables_used": [rel(path) for path in source_files],
                "filters": [
                    "仅Phase A已完成的PIP/DIP或IP/MCP/CMC pitch",
                    "拒绝批次和异常端点不进入正式LUT",
                    "标定后端点取候选physical_lut的安全首节点",
                ],
                "metric_definitions": [
                    "端点差 = 实测安全端点 - 当前URDF upper limit",
                    "回差图使用同机位raw128去程/回程差的绝对角度",
                    "误差图使用各session报告的held-out最大误差；食指PIP使用固定曝光对双向中线最大差",
                    "映射误差在候选LUT全部186个实测节点上计算",
                ],
            },
        },
        {
            "id": "candidate-lut",
            "label": "Phase A聚合候选标定",
            "path": rel(CANDIDATE),
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    "SELECT * FROM read_json_auto("
                    "'linker_calib_index_thumb_middle_ring_pinky_mcp_pitch_lut_candidate_20260802.json')"
                ),
                "description": "10个主动关节的raw→物理rad候选LUT。",
                "tables_used": [rel(CANDIDATE)],
            },
        },
        {
            "id": "mapping-validation",
            "label": "Phase A候选映射验证",
            "path": rel(MAPPING_VALIDATION),
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    "SELECT * FROM read_json_auto("
                    "'records/g20_physical_joint_calibration_20260801/sessions/"
                    "20260802T_pinky_pip_dip_pitch_thin_tape_full_raw_sweep/"
                    "phase_a_candidate_mapping_validation.json')"
                ),
                "description": "候选LUT全部节点的命令与SDK回读映射验证。",
                "tables_used": [rel(MAPPING_VALIDATION)],
            },
        },
    ]
    for dataset_id in datasets:
        dataset_path = (
            "deliverables/g20_phase_a_calibration_dashboard/data/"
            f"{dataset_id}.json"
        )
        sources.append(
            {
                "id": f"dataset-{dataset_id}",
                "label": f"Dashboard reviewed dataset: {dataset_id}",
                "path": dataset_path,
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": (
                        "SELECT * FROM read_json_auto("
                        f"'{dataset_path}', format='array')"
                    ),
                    "description": (
                        f"读取生成器审查并落盘的{dataset_id}数据集；"
                        "上游文件与转换规则见phase-a-derived来源。"
                    ),
                    "tables_used": [dataset_path, rel(Path(__file__))],
                },
            }
        )
    return datasets, sources, validation


def build_artifact() -> dict:
    datasets, sources, validation = build_data()
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    cards = [
        {
            "id": "phase-complete",
            "dataset": "hero",
            "sourceId": "phase-a-derived",
            "description": "已完成真实运动、相机测量、LUT、回差、健康记录的Phase A合同。",
            "metrics": [
                {"label": "Phase A完成", "field": "phase_complete", "format": "number"},
                {"label": "总项", "field": "phase_total", "format": "number"},
            ],
        },
        {
            "id": "active-luts",
            "dataset": "hero",
            "sourceId": "candidate-lut",
            "description": "四根长指PIP/pitch与拇指MCP/CMC pitch。",
            "metrics": [
                {"label": "主动关节LUT", "field": "active_lut_joints", "format": "number"}
            ],
        },
        {
            "id": "lut-knots",
            "dataset": "hero",
            "sourceId": "mapping-validation",
            "description": "聚合候选中经过SDK正反映射检查的所有实测节点。",
            "metrics": [
                {"label": "已验证LUT节点", "field": "lut_knots", "format": "number"}
            ],
        },
        {
            "id": "mapping-error",
            "dataset": "hero",
            "sourceId": "mapping-validation",
            "description": "186个实测节点上的最大指令raw误差。",
            "metrics": [
                {
                    "label": "最大指令误差",
                    "field": "max_command_raw_error",
                    "format": "number",
                }
            ],
        },
        {
            "id": "motions",
            "dataset": "hero",
            "sourceId": "phase-a-derived",
            "description": "五根手指Phase A中已发送且被记录的运动文件总数。",
            "metrics": [
                {
                    "label": "已记录安全运动",
                    "field": "accepted_motion_records",
                    "format": "number",
                }
            ],
        },
        {
            "id": "temperature",
            "dataset": "hero",
            "sourceId": "phase-a-derived",
            "description": "全部Phase A记录中观察到的最高槽位温度。",
            "metrics": [
                {
                    "label": "最高温度 °C",
                    "field": "max_temperature_c",
                    "format": "number",
                }
            ],
        },
        {
            "id": "excluded-endpoints",
            "dataset": "hero",
            "sourceId": "phase-a-derived",
            "description": "因fault或轴间耦合而明确排除的物理端点。",
            "metrics": [
                {
                    "label": "排除的异常端点",
                    "field": "excluded_physical_endpoints",
                    "format": "number",
                }
            ],
        },
    ]

    charts = [
        {
            "id": "range-compare",
            "title": "主动关节：当前URDF limit与实测安全端点",
            "subtitle": "单位rad；标定后端点来自每关节physical LUT的安全首节点",
            "type": "bar",
            "intent": "comparison",
            "dataset": "range_long",
            "sourceId": "phase-a-derived",
            "encodings": {
                "x": {"field": "joint_label", "type": "nominal", "label": "关节"},
                "y": {
                    "field": "limit_rad",
                    "type": "quantitative",
                    "label": "角度",
                    "unit": "rad",
                },
                "color": {"field": "series", "type": "nominal", "label": "合同"},
                "tooltip": [
                    {"field": "limit_deg", "type": "quantitative", "label": "角度", "unit": "deg"},
                    {"field": "min_valid_command_raw", "type": "quantitative", "label": "最低有效raw"},
                ],
            },
            "palette": {"kind": "semantic"},
            "legend": {"position": "bottom", "sort": "spec"},
            "settings": {
                "groupMode": "grouped",
                "categoryLabelPolicy": "rotate",
                "showValues": True,
            },
            "layout": "full",
            "surface": {"viewMode": "both", "interactiveLegend": True},
        },
        {
            "id": "range-delta",
            "title": "实测安全端点相对当前URDF limit的差值",
            "subtitle": "正值=当前软件低估物理范围；负值=当前软件高估物理范围",
            "type": "horizontalBar",
            "intent": "comparison",
            "dataset": "range_detail",
            "sourceId": "phase-a-derived",
            "encodings": {
                "x": {"field": "joint_label", "type": "nominal", "label": "关节"},
                "y": {
                    "field": "mismatch_rad",
                    "type": "quantitative",
                    "label": "实测 - 当前",
                    "unit": "rad",
                },
                "tooltip": [
                    {"field": "mismatch_deg", "type": "quantitative", "label": "差值", "unit": "deg"},
                    {"field": "status", "type": "text", "label": "判断"},
                ],
            },
            "palette": {"kind": "diverging", "midpoint": 0},
            "referenceLines": [
                {"axis": "y", "value": 0, "label": "一致", "color": "neutral"}
            ],
            "settings": {"showValues": True, "sort": "descending"},
            "layout": "full",
            "surface": {"viewMode": "both"},
        },
        {
            "id": "lut-curves",
            "title": "主动关节physical LUT：SDK raw→实测物理角",
            "subtitle": "使用上方手指筛选器；raw255为伸展零位，低raw为屈曲端",
            "type": "line",
            "intent": "relationship",
            "dataset": "active_curves",
            "sourceId": "candidate-lut",
            "encodings": {
                "x": {
                    "field": "command_raw",
                    "type": "quantitative",
                    "label": "SDK command raw",
                },
                "y": {
                    "field": "measured_rad",
                    "type": "quantitative",
                    "label": "实测物理角",
                    "unit": "rad",
                },
                "color": {"field": "joint_label", "type": "nominal", "label": "关节"},
                "tooltip": [
                    {"field": "measured_deg", "type": "quantitative", "label": "角度", "unit": "deg"},
                    {"field": "min_valid_command_raw", "type": "quantitative", "label": "最低有效raw"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "labelAsc"},
            "settings": {"showPoints": "always"},
            "layout": "full",
            "surface": {"viewMode": "both", "interactiveLegend": True},
        },
        {
            "id": "normalized-curves",
            "title": "归一化LUT形状：线性映射偏差",
            "subtitle": "理想线性映射接近y=x；曲线弯曲表示不能只靠0–255仿射映射",
            "type": "line",
            "intent": "relationship",
            "dataset": "active_curves",
            "sourceId": "candidate-lut",
            "encodings": {
                "x": {
                    "field": "normalized_flex",
                    "type": "quantitative",
                    "label": "归一化屈曲raw",
                },
                "y": {
                    "field": "normalized_angle",
                    "type": "quantitative",
                    "label": "归一化物理角",
                },
                "color": {"field": "joint_label", "type": "nominal", "label": "关节"},
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "labelAsc"},
            "referenceLines": [
                {"axis": "y", "value": 0.5, "label": "50%角度", "lineStyle": "dotted", "color": "neutral"}
            ],
            "settings": {"showPoints": "always"},
            "layout": "full",
            "surface": {"viewMode": "both", "interactiveLegend": True},
        },
        {
            "id": "mimic-compare",
            "title": "被动DIP/IP：当前URDF mimic与实测拟合",
            "subtitle": "零截距multiplier；详细表同时给出affine slope、offset与最大残差",
            "type": "bar",
            "intent": "comparison",
            "dataset": "mimic_long",
            "sourceId": "phase-a-derived",
            "encodings": {
                "x": {"field": "passive_joint_label", "type": "nominal", "label": "被动关节"},
                "y": {
                    "field": "multiplier",
                    "type": "quantitative",
                    "label": "mimic multiplier",
                },
                "color": {"field": "series", "type": "nominal", "label": "合同"},
            },
            "palette": {"kind": "semantic"},
            "legend": {"position": "bottom", "sort": "spec"},
            "settings": {"groupMode": "grouped", "showValues": True},
            "layout": "full",
            "surface": {"viewMode": "both", "interactiveLegend": True},
        },
        {
            "id": "hysteresis",
            "title": "raw128同机位去程/回程回差",
            "subtitle": "显示绝对角度deg；方向符号保留在展开数据表中",
            "type": "horizontalBar",
            "intent": "comparison",
            "dataset": "hysteresis",
            "sourceId": "phase-a-derived",
            "encodings": {
                "x": {"field": "joint_label", "type": "nominal", "label": "关节"},
                "y": {
                    "field": "abs_hysteresis_deg",
                    "type": "quantitative",
                    "label": "绝对回差",
                    "unit": "deg",
                },
                "tooltip": [
                    {"field": "signed_deg", "type": "quantitative", "label": "有符号回差", "unit": "deg"},
                    {"field": "joint_role", "type": "nominal", "label": "角色"},
                ],
            },
            "palette": {"kind": "sequential"},
            "settings": {"showValues": True, "sort": "descending"},
            "layout": "full",
            "surface": {"viewMode": "both"},
        },
        {
            "id": "error",
            "title": "LUT held-out / 交叉验证最大误差",
            "subtitle": "单位deg；食指PIP为固定曝光相对双向中线最大差，其余为held-out分段线性误差",
            "type": "horizontalBar",
            "intent": "comparison",
            "dataset": "error",
            "sourceId": "phase-a-derived",
            "encodings": {
                "x": {"field": "joint_label", "type": "nominal", "label": "关节"},
                "y": {
                    "field": "error_deg",
                    "type": "quantitative",
                    "label": "最大绝对误差",
                    "unit": "deg",
                },
                "tooltip": [
                    {"field": "error_rad", "type": "quantitative", "label": "误差", "unit": "rad"},
                    {"field": "metric", "type": "text", "label": "定义"},
                ],
            },
            "palette": {"kind": "sequential"},
            "settings": {"showValues": True, "sort": "descending"},
            "layout": "full",
            "surface": {"viewMode": "both"},
        },
        {
            "id": "temperature-chart",
            "title": "各手指Phase A最高记录温度",
            "subtitle": "单位°C；不同session的温度口径以各自报告为准",
            "type": "bar",
            "intent": "comparison",
            "dataset": "safety",
            "sourceId": "phase-a-derived",
            "encodings": {
                "x": {"field": "finger", "type": "nominal", "label": "手指"},
                "y": {
                    "field": "max_temperature_c",
                    "type": "quantitative",
                    "label": "最高温度",
                    "unit": "°C",
                },
            },
            "palette": {"kind": "sequential"},
            "settings": {"showValues": True},
            "layout": "full",
            "surface": {"viewMode": "both"},
        },
    ]

    tables = [
        {
            "id": "range-table",
            "title": "主动关节端点前后对比明细",
            "dataset": "range_detail",
            "sourceId": "phase-a-derived",
            "defaultSort": {"field": "mismatch_rad", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "joint_label", "label": "关节", "type": "text"},
                {"field": "current_urdf_rad", "label": "当前URDF rad", "format": "number"},
                {"field": "measured_safe_rad", "label": "实测安全rad", "format": "number"},
                {
                    "field": "mismatch_rad",
                    "label": "差值rad",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "mismatch_deg",
                    "label": "差值deg",
                    "format": "number",
                    "movement": True,
                },
                {"field": "min_valid_command_raw", "label": "最低有效raw", "format": "number"},
                {"field": "status", "label": "判断", "type": "text"},
            ],
        },
        {
            "id": "mimic-table",
            "title": "被动DIP/IP拟合明细",
            "dataset": "mimic_detail",
            "sourceId": "phase-a-derived",
            "defaultSort": {"field": "multiplier_delta", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "passive_joint_label", "label": "关节", "type": "text"},
                {"field": "current_multiplier", "label": "当前mimic", "format": "number"},
                {"field": "measured_multiplier", "label": "实测零截距", "format": "number"},
                {
                    "field": "multiplier_delta",
                    "label": "差值",
                    "format": "number",
                    "movement": True,
                },
                {"field": "affine_slope", "label": "Affine slope", "format": "number"},
                {"field": "affine_offset_rad", "label": "Offset rad", "format": "number"},
                {"field": "max_residual_rad", "label": "最大残差rad", "format": "number"},
            ],
        },
        {
            "id": "safety-table",
            "title": "安全、记录覆盖与最终状态",
            "dataset": "safety",
            "sourceId": "phase-a-derived",
            "defaultSort": {"field": "max_temperature_c", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "finger", "label": "手指", "type": "text"},
                {"field": "accepted_motion_records", "label": "已发送运动", "format": "number"},
                {"field": "post_samples", "label": "post样本", "format": "number"},
                {"field": "fault_samples_accepted", "label": "正式fault样本", "format": "number"},
                {"field": "max_temperature_c", "label": "最高°C", "format": "number"},
                {"field": "excluded_or_rejected", "label": "排除/拒绝证据", "type": "text"},
                {"field": "final_state", "label": "最终状态", "type": "text"},
            ],
        },
        {
            "id": "exception-table",
            "title": "必须保留的异常、局限与处理",
            "dataset": "exceptions",
            "sourceId": "phase-a-derived",
            "defaultSort": {"field": "severity", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "severity", "label": "级别", "type": "text"},
                {"field": "finger", "label": "范围", "type": "text"},
                {"field": "item", "label": "项目", "type": "text"},
                {"field": "evidence", "label": "证据", "type": "text"},
                {"field": "handling", "label": "处理", "type": "text"},
            ],
        },
        {
            "id": "lut-table",
            "title": "主动关节LUT全部节点",
            "subtitle": "受上方手指筛选器控制；精确值用于审计，不用图形反推",
            "dataset": "active_curves",
            "sourceId": "candidate-lut",
            "defaultSort": {"field": "command_raw", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "finger", "label": "手指", "type": "text"},
                {"field": "joint_label", "label": "关节", "type": "text"},
                {"field": "command_raw", "label": "命令raw", "format": "number"},
                {"field": "measured_rad", "label": "实测rad", "format": "number"},
                {"field": "measured_deg", "label": "实测deg", "format": "number"},
                {"field": "normalized_flex", "label": "归一化raw", "format": "number"},
                {"field": "normalized_angle", "label": "归一化角", "format": "number"},
            ],
        },
    ]

    for item in [*cards, *charts, *tables]:
        item["sourceId"] = f"dataset-{item['dataset']}"

    manifest = {
        "version": 1,
        "surface": "dashboard",
        "title": "LinkerHand G20 Phase A 物理关节标定",
        "description": "Isaac/URDF rad、SDK raw0–255与D435实测物理角的Phase A交互式审计面板。",
        "generatedAt": generated_at,
        "filters": [
            {
                "id": "finger-filter",
                "label": "LUT手指",
                "dataset": "active_curves",
                "field": "finger",
                "defaultValue": FINGER_ZH["index"],
                "includeAll": True,
                "targets": [
                    {"dataset": "active_curves", "field": "finger"},
                ],
            }
        ],
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": [
            {
                "id": "status-note",
                "type": "markdown",
                "body": (
                    "## 当前结论\n\n"
                    "Phase A已完成 **15/15**：四根长指的PIP、被动DIP、MCP pitch，"
                    "以及拇指MCP、被动IP、CMC pitch。物理标定以实机安全端点为准；"
                    "当前Topdown门禁仍硬编码PIP `1.08 rad`，因此在Isaac limits和门禁"
                    "按实测值同步更新前，候选保持 **CANDIDATE ONLY / fail-closed**。"
                ),
                "sourceId": "mapping-validation",
                "layout": "full",
            },
            {
                "id": "hero-strip",
                "type": "metric-strip",
                "cardIds": [
                    "phase-complete",
                    "active-luts",
                    "lut-knots",
                    "mapping-error",
                    "motions",
                    "temperature",
                    "excluded-endpoints",
                ],
            },
            {"id": "range-compare-block", "type": "chart", "chartId": "range-compare", "layout": "full"},
            {"id": "range-delta-block", "type": "chart", "chartId": "range-delta", "layout": "full"},
            {"id": "range-table-block", "type": "table", "tableId": "range-table", "layout": "full"},
            {
                "id": "lut-note",
                "type": "markdown",
                "body": (
                    "## Raw→rad映射\n\n"
                    "下面两张图使用同一份候选physical LUT。第一张显示真实rad，第二张把raw和"
                    "角度都归一化，用来观察非线性。筛选器默认显示食指；选择“全部”可比较10个主动关节。"
                ),
                "sourceId": "candidate-lut",
                "layout": "full",
            },
            {"id": "lut-curves-block", "type": "chart", "chartId": "lut-curves", "layout": "full"},
            {"id": "normalized-curves-block", "type": "chart", "chartId": "normalized-curves", "layout": "full"},
            {"id": "lut-table-block", "type": "table", "tableId": "lut-table", "layout": "full"},
            {
                "id": "mimic-note",
                "type": "markdown",
                "body": (
                    "## 被动DIP/IP联动\n\n"
                    "DIP/IP没有独立SDK槽位。柱状图比较当前URDF multiplier与相机实测的"
                    "零截距拟合；表格保留affine slope、offset和最大残差，避免把带偏置关系强行压成单一比例。"
                ),
                "sourceId": "phase-a-derived",
                "layout": "full",
            },
            {"id": "mimic-compare-block", "type": "chart", "chartId": "mimic-compare", "layout": "full"},
            {"id": "mimic-table-block", "type": "table", "tableId": "mimic-table", "layout": "full"},
            {
                "id": "quality-note",
                "type": "markdown",
                "body": (
                    "## 回差、拟合误差与安全质量\n\n"
                    "回差是同机位raw128去程/回程差；误差图混合两种已明确标注的口径："
                    "多数关节为held-out分段线性最大误差，食指PIP为固定曝光对双向中线最大差。"
                    "这些数值描述测量与插值质量，不等同于186个LUT节点的软件往返误差。"
                ),
                "sourceId": "phase-a-derived",
                "layout": "full",
            },
            {"id": "hysteresis-block", "type": "chart", "chartId": "hysteresis", "layout": "full"},
            {"id": "error-block", "type": "chart", "chartId": "error", "layout": "full"},
            {"id": "temperature-block", "type": "chart", "chartId": "temperature-chart", "layout": "full"},
            {"id": "safety-table-block", "type": "table", "tableId": "safety-table", "layout": "full"},
            {"id": "exception-table-block", "type": "table", "tableId": "exception-table", "layout": "full"},
            {
                "id": "source-note",
                "type": "markdown",
                "body": (
                    "## 数据质量与使用边界\n\n"
                    "- 权威数值来自各session最终summary/CSV和聚合候选JSON；中间诊断批次不进入正式曲线。\n"
                    "- 当前记录是2026-08-02的离线快照，不会自动刷新。\n"
                    "- Index PIP的raw0 fault64和Pinky pitch的raw0侧轴耦合都保留为拒绝证据。\n"
                    "- Phase B的四指MCP roll以及Phase C的拇指yaw/roll尚未标定，不在本面板范围内。"
                ),
                "sourceId": "phase-a-derived",
                "layout": "full",
            },
        ],
    }
    return {
        "surface": "dashboard",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
        "package_info": {
            "root": ".",
            "manifestPath": "deliverables/g20_phase_a_calibration_dashboard/artifact.json",
            "snapshotPath": "embedded",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "deliverables/g20_phase_a_calibration_dashboard/artifact.json",
    )
    args = parser.parse_args()
    artifact = build_artifact()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_dir = args.output.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id, rows in artifact["snapshot"]["datasets"].items():
        (data_dir / f"{dataset_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output.resolve()),
                "datasets": {
                    key: len(rows)
                    for key, rows in artifact["snapshot"]["datasets"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
