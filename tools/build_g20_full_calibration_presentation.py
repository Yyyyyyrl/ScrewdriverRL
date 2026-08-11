#!/usr/bin/env python3
"""Build a presentation-ready, source-backed G20 full calibration report."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RECORD_ROOT = ROOT / "records/g20_physical_joint_calibration_20260801"
SESSION_ROOT = RECORD_ROOT / "sessions"
OUTPUT_ROOT = ROOT / "deliverables/g20_full_calibration_presentation_20260803"

CANDIDATE = ROOT / "linker_calib_phase_c_thumb_yaw_roll_candidate_20260803.json"
URDF = ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
FULL_SUMMARY = RECORD_ROOT / "FULL_CALIBRATION_SUMMARY.md"
RECORD_INDEX = RECORD_ROOT / "CALIBRATION_RECORD_INDEX.md"
PHASE_A_ARTIFACT = (
    ROOT / "deliverables/g20_phase_a_calibration_dashboard/artifact.json"
)

ROLL_SUMMARIES = {
    finger: SESSION_ROOT
    / f"20260803T_{finger}_mcp_roll_roll_view_sweep"
    / f"{finger}_mcp_roll_mid_summary.json"
    for finger in ("index", "middle", "ring", "pinky")
}
YAW_SUMMARY = (
    SESSION_ROOT
    / "20260803T190607Z_thumb_cmc_yaw_lmarker_full_sweep"
    / "thumb_cmc_yaw_robust_l_mid_summary.json"
)
THUMB_ROLL_SUMMARY = (
    SESSION_ROOT
    / "20260803T201650Z_thumb_cmc_roll_new_view_full_sweep"
    / "white_palm_marker_recalibration"
    / "thumb_cmc_roll_summary.json"
)

FINGER_LABEL = {
    "thumb": "拇指 Thumb",
    "index": "食指 Index",
    "middle": "中指 Middle",
    "ring": "无名指 Ring",
    "pinky": "小指 Pinky",
}

JOINT_LABEL = {
    "thumb_cmc_pitch": "拇指 CMC pitch",
    "thumb_cmc_roll": "拇指 CMC roll",
    "thumb_cmc_yaw": "拇指 CMC yaw",
    "thumb_mcp": "拇指 MCP",
    "thumb_ip": "拇指 IP",
    "index_mcp_roll": "食指 MCP roll",
    "index_mcp_pitch": "食指 MCP pitch",
    "index_pip": "食指 PIP",
    "index_dip": "食指 DIP",
    "middle_mcp_roll": "中指 MCP roll",
    "middle_mcp_pitch": "中指 MCP pitch",
    "middle_pip": "中指 PIP",
    "middle_dip": "中指 DIP",
    "ring_mcp_roll": "无名指 MCP roll",
    "ring_mcp_pitch": "无名指 MCP pitch",
    "ring_pip": "无名指 PIP",
    "ring_dip": "无名指 DIP",
    "pinky_mcp_roll": "小指 MCP roll",
    "pinky_mcp_pitch": "小指 MCP pitch",
    "pinky_pip": "小指 PIP",
    "pinky_dip": "小指 DIP",
}

SLOTS = {
    "thumb_cmc_pitch": 0,
    "index_mcp_pitch": 1,
    "middle_mcp_pitch": 2,
    "ring_mcp_pitch": 3,
    "pinky_mcp_pitch": 4,
    "thumb_cmc_roll": 5,
    "index_mcp_roll": 6,
    "middle_mcp_roll": 7,
    "ring_mcp_roll": 8,
    "pinky_mcp_roll": 9,
    "thumb_cmc_yaw": 10,
    "thumb_mcp": 15,
    "index_pip": 16,
    "middle_pip": 17,
    "ring_pip": 18,
    "pinky_pip": 19,
}

ACTIVE_ORDER = [
    "middle_pip",
    "pinky_pip",
    "ring_pip",
    "index_pip",
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_mcp",
    "index_mcp_pitch",
    "pinky_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "thumb_cmc_pitch",
    "middle_mcp_roll",
    "index_mcp_roll",
    "pinky_mcp_roll",
    "ring_mcp_roll",
]

PASSIVE = [
    ("thumb_ip", "thumb_mcp"),
    ("index_dip", "index_pip"),
    ("middle_dip", "middle_pip"),
    ("ring_dip", "ring_pip"),
    ("pinky_dip", "pinky_pip"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def joint_finger(name: str) -> str:
    return name.split("_", 1)[0]


def joint_family(name: str) -> str:
    if name.endswith("_pip"):
        return "PIP flexion"
    if name.endswith("_dip") or name == "thumb_ip":
        return "Passive mimic"
    if name.endswith("_mcp_roll"):
        return "MCP roll"
    if name == "thumb_cmc_roll":
        return "CMC roll"
    if name == "thumb_cmc_yaw":
        return "CMC yaw"
    return "MCP/CMC pitch"


def urdf_contract() -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    root = ET.parse(URDF).getroot()
    limits: dict[str, tuple[float, float]] = {}
    mimics: dict[str, float] = {}
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        if limit is not None and name in JOINT_LABEL:
            limits[name] = (
                float(limit.get("lower", "0")),
                float(limit.get("upper", "0")),
            )
        if mimic is not None and name in JOINT_LABEL:
            mimics[name] = float(mimic.get("multiplier", "1"))
    return limits, mimics


def attention_for(name: str) -> str:
    if name == "index_mcp_pitch":
        return "当前 URDF 已单独改；golden/manifest 尚未同步"
    if name.endswith("_pip"):
        return "旧 top-down 门禁仍硬编码 hi=1.08 rad"
    if name in {"index_mcp_roll", "ring_mcp_roll"}:
        return "零锚漂移超过 0.5°；已保留为候选不确定度"
    if name == "pinky_mcp_pitch":
        return "伸展端零锚漂移 −1.023°；已保留不确定度"
    if name == "thumb_cmc_yaw":
        return "raw2 被 80° 防卷绕门拒绝；raw0 未尝试"
    if name == "thumb_cmc_roll":
        return "raw4..248 为稳定范围；未声称 raw0/255 可达"
    return "候选已完成；待整体生产化"


def source(
    source_id: str,
    label: str,
    path: Path,
    description: str,
    tables: list[Path],
    dataset_files: list[str],
    definitions: list[str] | None = None,
) -> dict:
    sql = "\n".join(
        (
            "SELECT * FROM read_json_auto("
            f"'deliverables/g20_full_calibration_presentation_20260803/data/{name}.json', "
            "format='array');"
        )
        for name in dataset_files
    )
    query = {
        "engine": "duckdb",
        "language": "sql",
        "sql": sql,
        "description": description,
        "tables_used": [
            *[rel(item) for item in tables],
            *[
                (
                    "deliverables/g20_full_calibration_presentation_20260803/"
                    f"data/{name}.json"
                )
                for name in dataset_files
            ],
        ],
    }
    if definitions:
        query["metric_definitions"] = definitions
    return {
        "id": source_id,
        "label": label,
        "path": rel(path),
        "query": query,
    }


def build_datasets() -> dict[str, list[dict]]:
    candidate = read_json(CANDIDATE)
    phase_a = read_json(PHASE_A_ARTIFACT)["snapshot"]["datasets"]
    urdf_limits, _ = urdf_contract()

    active_ranges: list[dict] = []
    active_range_long: list[dict] = []
    joint_inventory: list[dict] = []
    total_knots = 0

    for rank, name in enumerate(ACTIVE_ORDER, start=1):
        entry = candidate["joints"][name]
        lut = entry["physical_lut"]
        raw_values = lut["raw"]
        rad_values = lut["rad"]
        measured_lo = min(rad_values)
        measured_hi = max(rad_values)
        measured_span = measured_hi - measured_lo
        current_lo, current_hi = urdf_limits[name]
        current_span = current_hi - current_lo
        knots = len(raw_values)
        total_knots += knots
        finger = joint_finger(name)
        row = {
            "rank": rank,
            "finger": FINGER_LABEL[finger],
            "finger_key": finger,
            "joint": name,
            "joint_label": JOINT_LABEL[name],
            "family": joint_family(name),
            "slot": SLOTS[name],
            "knots": knots,
            "raw_min": min(raw_values),
            "raw_max": max(raw_values),
            "measured_lo_rad": measured_lo,
            "measured_hi_rad": measured_hi,
            "measured_span_rad": measured_span,
            "measured_span_deg": math.degrees(measured_span),
            "current_lo_rad": current_lo,
            "current_hi_rad": current_hi,
            "current_span_rad": current_span,
            "current_span_deg": math.degrees(current_span),
            "span_delta_deg": math.degrees(measured_span - current_span),
            "current_contract": f"{current_lo:.3f}..{current_hi:.3f} rad",
            "measured_contract": f"{measured_lo:.3f}..{measured_hi:.3f} rad",
            "status": "实测 LUT 完成",
            "production_state": "候选，未整体启用",
            "attention": attention_for(name),
        }
        active_ranges.append(row)
        joint_inventory.append(
            {
                "finger": row["finger"],
                "joint_label": row["joint_label"],
                "role": "主动 Active",
                "slot": str(row["slot"]),
                "evidence": f"{knots} 结点 LUT",
                "physical_span_deg": row["measured_span_deg"],
                "current_contract": row["current_contract"],
                "status": row["production_state"],
                "attention": row["attention"],
            }
        )
        for series, span in (
            ("实测稳定物理行程", row["measured_span_deg"]),
            ("当前 URDF 行程", row["current_span_deg"]),
        ):
            active_range_long.append(
                {
                    **row,
                    "series": series,
                    "span_deg": span,
                }
            )

    mimic_detail = phase_a["mimic_detail"]
    mimic_long: list[dict] = []
    mimic_by_name = {row["passive_joint"]: row for row in mimic_detail}
    for name, driver in PASSIVE:
        row = mimic_by_name[name]
        for series, value in (
            ("当前 URDF multiplier", row["current_multiplier"]),
            ("实测零截距候选", row["measured_multiplier"]),
        ):
            mimic_long.append(
                {
                    **row,
                    "driver": driver,
                    "series": series,
                    "multiplier": value,
                }
            )
        joint_inventory.append(
            {
                "finger": row["finger"],
                "joint_label": row["passive_joint_label"],
                "role": "被动 Mimic",
                "slot": "—",
                "evidence": f"倍率 {row['measured_multiplier']:.4f}",
                "physical_span_deg": None,
                "current_contract": f"multiplier {row['current_multiplier']:.4f}",
                "status": "候选，未整体启用",
                "attention": (
                    f"从动于 {JOINT_LABEL[driver]}；现行倍率偏大，待整体更新"
                ),
            }
        )

    phase_a_error = [
        row for row in phase_a["error"] if "DIP" not in row["joint_label"]
    ]
    phase_a_hysteresis = [
        row
        for row in phase_a["hysteresis"]
        if row["joint_role"] == "主动"
    ]
    validation_by_joint = {
        row["joint_label"]: {
            "joint_label": row["joint_label"],
            "finger": row["finger"],
            "error_deg": row["error_deg"],
            "error_basis": row["metric"],
        }
        for row in phase_a_error
    }
    hysteresis_by_joint = {
        row["joint_label"]: row["abs_hysteresis_deg"]
        for row in phase_a_hysteresis
    }

    summary_paths = [*ROLL_SUMMARIES.values(), YAW_SUMMARY, THUMB_ROLL_SUMMARY]
    for path in summary_paths:
        summary = read_json(path)
        name = summary["joint"]
        label = JOINT_LABEL[name]
        validation_by_joint[label] = {
            "joint_label": label,
            "finger": FINGER_LABEL[joint_finger(name)],
            "error_deg": math.degrees(
                summary["heldout_piecewise_linear_max_abs_error_rad"]
            ),
            "error_basis": "held-out 分段线性最大误差",
        }
        hysteresis_by_joint[label] = summary["hysteresis_readback_domain"][
            "max_abs_deg"
        ]

    quality_rows: list[dict] = []
    quality_long: list[dict] = []
    for name in ACTIVE_ORDER:
        label = JOINT_LABEL[name]
        validation = validation_by_joint.get(label, {})
        error = validation.get("error_deg")
        hysteresis = hysteresis_by_joint.get(label)
        row = {
            "finger": FINGER_LABEL[joint_finger(name)],
            "joint": name,
            "joint_label": label,
            "family": joint_family(name),
            "validation_error_deg": error,
            "validation_basis": validation.get(
                "error_basis", "当前汇总无同口径 held-out 数值"
            ),
            "hysteresis_deg": hysteresis,
            "validation_gate_deg": 3.0,
            "within_validation_gate": None if error is None else error <= 3.0,
        }
        quality_rows.append(row)
        if error is not None:
            quality_long.append(
                {
                    **row,
                    "metric": "最大验证误差",
                    "value_deg": error,
                }
            )
        if hysteresis is not None:
            quality_long.append(
                {
                    **row,
                    "metric": "记录回差",
                    "value_deg": hysteresis,
                }
            )

    phases = [
        {
            "phase": "Phase A",
            "scope": "4×PIP、4×被动 DIP/IP、4×MCP/CMC pitch、thumb MCP",
            "completed": 15,
            "total": 15,
            "status": "完成",
            "measurement_view": "正视 flexion / pitch",
        },
        {
            "phase": "Phase B",
            "scope": "4×MCP roll",
            "completed": 4,
            "total": 4,
            "status": "完成",
            "measurement_view": "roll 专用机位",
        },
        {
            "phase": "Phase C",
            "scope": "thumb CMC yaw + roll",
            "completed": 2,
            "total": 2,
            "status": "完成",
            "measurement_view": "换机位 + L 形双标",
        },
    ]

    hero = [
        {
            "joint_contracts_complete": 21,
            "joint_contracts_target": 21,
            "active_lut_joints": 16,
            "passive_mimic_contracts": 5,
            "lut_knots": total_knots,
            "worst_validation_error_deg": max(
                row["error_deg"] for row in validation_by_joint.values()
            ),
            "validation_gate_deg": 3.0,
        }
    ]

    return {
        "hero": hero,
        "phases": phases,
        "active_ranges": active_ranges,
        "active_range_long": active_range_long,
        "quality": quality_rows,
        "quality_long": quality_long,
        "mimic_detail": mimic_detail,
        "mimic_long": mimic_long,
        "joint_inventory": joint_inventory,
    }


def build_artifact() -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    datasets = build_datasets()

    corpus_paths = [
        CANDIDATE,
        URDF,
        FULL_SUMMARY,
        RECORD_INDEX,
        PHASE_A_ARTIFACT,
        *ROLL_SUMMARIES.values(),
        YAW_SUMMARY,
        THUMB_ROLL_SUMMARY,
    ]
    sources = [
        source(
            "full-summary",
            "21 关节标定收口汇总",
            FULL_SUMMARY,
            "Phase A/B/C 的完成口径、主动关节范围、mimic 候选、复核结果与部署阻断项。",
            [FULL_SUMMARY, RECORD_INDEX],
            ["hero", "phases", "joint_inventory"],
            [
                "完成率 = 已完成关节合同 / 21",
                "主动关节 = 具有实测 physical_lut 的关节",
                "被动关节 = 具有实测零截距 mimic multiplier 候选的关节",
            ],
        ),
        source(
            "candidate-lut",
            "Phase C 聚合候选标定",
            CANDIDATE,
            "读取 16 个主动关节的 raw→rad LUT，计算稳定物理行程、结点数与 raw 覆盖。",
            [CANDIDATE],
            ["active_ranges"],
            [
                "实测物理行程 = max(physical_lut.rad) - min(physical_lut.rad)",
                "LUT 结点数 = len(physical_lut.raw)",
            ],
        ),
        source(
            "quality-corpus",
            "逐阶段质量摘要",
            RECORD_INDEX,
            "合并 Phase A 的已验证数据快照与 Phase B/C 最终 summary，保留验证误差、回差和测量口径。",
            corpus_paths,
            ["quality", "quality_long", "joint_inventory"],
            [
                "最大验证误差 = session summary 的 held-out 分段线性最大绝对误差；食指 PIP 使用固定曝光 vs 双向中线最大差",
                "记录回差 = Phase B/C 的 readback-domain max abs；Phase A 使用原 session 记录的 raw128 双向差",
                "图中 3° 参考线仅用于静态验证误差验收，不表示动态控制误差",
            ],
        ),
        source(
            "current-urdf",
            "当前 LinkerHand L20 左手 URDF",
            URDF,
            "读取当前软件关节 lower/upper limits 与 mimic multiplier，和实测候选做差。",
            [URDF, CANDIDATE, PHASE_A_ARTIFACT],
            ["active_range_long", "mimic_detail", "mimic_long"],
            [
                "当前 URDF 行程 = upper - lower",
                "mimic 差值 = 实测零截距候选 - 当前 URDF multiplier",
            ],
        ),
    ]

    cards = [
        {
            "id": "joint-completion",
            "dataset": "hero",
            "sourceId": "full-summary",
            "description": "主动关节 LUT 与被动 mimic 合同合计完成数。",
            "metrics": [
                {
                    "label": "关节合同完成",
                    "field": "joint_contracts_complete",
                    "format": "number",
                },
                {
                    "label": "目标",
                    "field": "joint_contracts_target",
                    "format": "number",
                },
            ],
        },
        {
            "id": "active-luts",
            "dataset": "hero",
            "sourceId": "candidate-lut",
            "description": "具有实测 raw→rad physical LUT 的主动关节。",
            "metrics": [
                {
                    "label": "主动关节 LUT",
                    "field": "active_lut_joints",
                    "format": "number",
                },
                {
                    "label": "被动 mimic",
                    "field": "passive_mimic_contracts",
                    "format": "number",
                },
            ],
        },
        {
            "id": "lut-knots",
            "dataset": "hero",
            "sourceId": "candidate-lut",
            "description": "16 条候选物理 LUT 的结点总数。",
            "metrics": [
                {
                    "label": "LUT 结点",
                    "field": "lut_knots",
                    "format": "number",
                }
            ],
        },
        {
            "id": "worst-validation",
            "dataset": "hero",
            "sourceId": "quality-corpus",
            "description": "已记录的最大静态验证误差；小指 PIP 为当前较弱项。",
            "metrics": [
                {
                    "label": "最差验证误差 °",
                    "field": "worst_validation_error_deg",
                    "format": "number",
                },
                {
                    "label": "验收门 °",
                    "field": "validation_gate_deg",
                    "format": "number",
                },
            ],
        },
    ]

    charts = [
        {
            "id": "measured-span",
            "title": "16 个主动关节的实测稳定物理行程",
            "subtitle": "按实测跨度降序；单位为度，roll 为双向总行程",
            "type": "horizontalBar",
            "intent": "comparison",
            "dataset": "active_ranges",
            "sourceId": "candidate-lut",
            "encodings": {
                "x": {
                    "field": "joint_label",
                    "type": "nominal",
                    "label": "主动关节",
                },
                "y": {
                    "field": "measured_span_deg",
                    "type": "quantitative",
                    "label": "实测行程",
                    "unit": "deg",
                },
                "color": {
                    "field": "family",
                    "type": "nominal",
                    "label": "关节族",
                },
                "tooltip": [
                    {
                        "field": "slot",
                        "type": "quantitative",
                        "label": "SDK slot",
                    },
                    {
                        "field": "knots",
                        "type": "quantitative",
                        "label": "LUT 结点",
                    },
                    {
                        "field": "raw_min",
                        "type": "quantitative",
                        "label": "稳定 raw 下界",
                    },
                    {
                        "field": "raw_max",
                        "type": "quantitative",
                        "label": "稳定 raw 上界",
                    },
                ],
            },
            "palette": {"kind": "categorical"},
            "settings": {
                "showValues": True,
                "sort": "descending",
                "categoryLabelPolicy": "wrap",
            },
            "layout": "full",
            "surface": {"viewMode": "both", "interactiveLegend": True},
        },
        {
            "id": "urdf-vs-measured",
            "title": "当前 URDF 行程与实测稳定行程",
            "subtitle": "实测更宽与更窄的关节同时存在，因此不能用一个全局缩放修正",
            "type": "horizontalBar",
            "intent": "comparison",
            "dataset": "active_range_long",
            "sourceId": "current-urdf",
            "encodings": {
                "x": {
                    "field": "joint_label",
                    "type": "nominal",
                    "label": "主动关节",
                },
                "y": {
                    "field": "span_deg",
                    "type": "quantitative",
                    "label": "行程",
                    "unit": "deg",
                },
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "label": "合同",
                },
                "tooltip": [
                    {
                        "field": "span_delta_deg",
                        "type": "quantitative",
                        "label": "实测−当前",
                        "unit": "deg",
                    },
                    {
                        "field": "attention",
                        "type": "text",
                        "label": "注意事项",
                    },
                ],
            },
            "palette": {"kind": "semantic"},
            "legend": {"position": "bottom", "sort": "spec"},
            "settings": {
                "groupMode": "grouped",
                "showValues": False,
                "sort": "custom",
                "categoryLabelPolicy": "wrap",
            },
            "layout": "full",
            "surface": {"viewMode": "both", "interactiveLegend": True},
        },
        {
            "id": "quality-overview",
            "title": "静态验证误差与记录回差",
            "subtitle": "单位为度；3° 参考线只对应静态验证误差门，回差按各 session 已记录口径展示",
            "type": "horizontalBar",
            "intent": "comparison",
            "dataset": "quality_long",
            "sourceId": "quality-corpus",
            "encodings": {
                "x": {
                    "field": "joint_label",
                    "type": "nominal",
                    "label": "主动关节",
                },
                "y": {
                    "field": "value_deg",
                    "type": "quantitative",
                    "label": "角度",
                    "unit": "deg",
                },
                "color": {
                    "field": "metric",
                    "type": "nominal",
                    "label": "指标",
                },
                "tooltip": [
                    {
                        "field": "validation_basis",
                        "type": "text",
                        "label": "验证口径",
                    },
                    {
                        "field": "family",
                        "type": "text",
                        "label": "关节族",
                    },
                ],
            },
            "palette": {"kind": "semantic"},
            "legend": {"position": "bottom", "sort": "spec"},
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 3.0,
                    "label": "静态验证门 3°",
                    "color": "red",
                    "lineStyle": "dashed",
                }
            ],
            "settings": {
                "groupMode": "grouped",
                "showValues": True,
                "sort": "descending",
                "categoryLabelPolicy": "wrap",
            },
            "layout": "full",
            "surface": {"viewMode": "both", "interactiveLegend": True},
        },
        {
            "id": "mimic-compare",
            "title": "5 个被动关节：当前与实测 mimic multiplier",
            "subtitle": "实测为零截距候选；所有当前 URDF 倍率均偏大",
            "type": "bar",
            "intent": "comparison",
            "dataset": "mimic_long",
            "sourceId": "current-urdf",
            "encodings": {
                "x": {
                    "field": "passive_joint_label",
                    "type": "nominal",
                    "label": "被动关节",
                },
                "y": {
                    "field": "multiplier",
                    "type": "quantitative",
                    "label": "mimic multiplier",
                },
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "label": "合同",
                },
                "tooltip": [
                    {
                        "field": "affine_slope",
                        "type": "quantitative",
                        "label": "带偏置 slope",
                    },
                    {
                        "field": "affine_offset_rad",
                        "type": "quantitative",
                        "label": "带偏置 offset",
                        "unit": "rad",
                    },
                ],
            },
            "palette": {"kind": "semantic"},
            "legend": {"position": "bottom", "sort": "spec"},
            "settings": {
                "groupMode": "grouped",
                "showValues": True,
                "categoryLabelPolicy": "wrap",
            },
            "layout": "full",
            "surface": {"viewMode": "both", "interactiveLegend": True},
        },
    ]

    tables = [
        {
            "id": "phase-table",
            "title": "三阶段完成矩阵",
            "subtitle": "2026 年 8 月 2–3 日；每阶段均保留最终 summary 与被拒证据",
            "dataset": "phases",
            "density": "spacious",
            "sourceId": "full-summary",
            "layout": "full",
            "columns": [
                {"field": "phase", "label": "阶段", "type": "text"},
                {"field": "scope", "label": "范围", "type": "text"},
                {"field": "completed", "label": "完成", "format": "number"},
                {"field": "total", "label": "总项", "format": "number"},
                {"field": "measurement_view", "label": "测量机位", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
        },
        {
            "id": "joint-table",
            "title": "21 关节证据与生产状态",
            "subtitle": "16 个主动 LUT + 5 个被动 mimic；空白物理行程表示被动关节不单独定义主动行程",
            "dataset": "joint_inventory",
            "density": "dense",
            "sourceId": "quality-corpus",
            "layout": "full",
            "columns": [
                {"field": "finger", "label": "手指", "type": "text"},
                {"field": "joint_label", "label": "关节", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "slot", "label": "slot", "type": "text"},
                {"field": "evidence", "label": "标定证据", "type": "text"},
                {
                    "field": "physical_span_deg",
                    "label": "实测行程 °",
                    "format": "number",
                },
                {"field": "current_contract", "label": "当前合同", "type": "text"},
                {"field": "status", "label": "生产状态", "type": "text"},
                {"field": "attention", "label": "注意事项", "type": "text"},
            ],
        },
    ]

    title = "LinkerHand G20 左手：21 关节物理标定全景"
    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": (
                f"# {title}\n\n"
                "**对象：** G20 左手 `LHT20-010-415-L-B-1-D`  ·  "
                "**相机：** RealSense D435 `143322073091`  ·  "
                "**测量日期：** 2026-08-02 至 2026-08-03"
            ),
            "layout": "full",
        },
        {
            "id": "executive-summary",
            "type": "markdown",
            "body": (
                "## Executive Summary\n\n"
                "全手物理标定已经完成：**21/21 个关节合同闭环**，其中 16 个主动关节形成 "
                "**382 个结点**的实测 `raw→rad` LUT，5 个被动 DIP/IP 形成独立的实测 "
                "mimic 倍率候选。所有已记录的静态验证误差均低于 `3°` 门，最弱项是小指 "
                "PIP 的 `2.80°`。\n\n"
                "但“标定完成”不等于“可以直接上线”。当前结果仍是 **candidate-only**："
                "四根 PIP 的 top-down 安全门仍硬编码 `hi=1.08 rad`，5 个 mimic 倍率尚未更新，"
                "并且 `index_mcp_pitch` 处于单关节先改、golden/manifest 未同步的中间态。"
                "正确决策是先做一次原子化生产迁移，再重新验证、重训并分级 commissioning。"
            ),
            "sourceId": "full-summary",
            "layout": "full",
        },
        {
            "id": "hero-strip",
            "type": "metric-strip",
            "cardIds": [
                "joint-completion",
                "active-luts",
                "lut-knots",
                "worst-validation",
            ],
            "layout": "full",
        },
        {
            "id": "completion-heading",
            "type": "markdown",
            "body": (
                "## 1. 完成结构\n\n"
                "三阶段覆盖了所有主动自由度与被动联动合同：Phase A 完成正视 flexion/pitch，"
                "Phase B 用专用机位补齐四指 roll，Phase C 用换机位和 L 形双标完成拇指 "
                "yaw/roll。下表用于 presentation 中快速解释“21”是如何构成的。"
            ),
            "sourceId": "full-summary",
            "layout": "full",
        },
        {
            "id": "phase-table-block",
            "type": "table",
            "tableId": "phase-table",
            "layout": "full",
        },
        {
            "id": "span-heading",
            "type": "markdown",
            "body": (
                "## 2. 真实物理包络\n\n"
                "实测行程不是同一种尺度：四根 PIP 约 `88–100°`，拇指 yaw/roll 约 "
                "`75–83°`，四根 MCP roll 的双向总行程约 `27–28°`。图中按跨度排序，"
                "便于听众先形成全手量级直觉；这些端点是已接受的稳定安全端点，不是机械极限。"
            ),
            "sourceId": "candidate-lut",
            "layout": "full",
        },
        {
            "id": "measured-span-block",
            "type": "chart",
            "chartId": "measured-span",
            "layout": "full",
        },
        {
            "id": "contract-heading",
            "type": "markdown",
            "body": (
                "### 当前软件合同与实测并非统一偏差\n\n"
                "旧合同既有低估也有高估：四根 PIP 明显低估，部分 MCP pitch 高估，roll "
                "则把双向总行程压窄。因而不能靠一个比例因子修复，必须逐关节更新 limits、"
                "门禁与语义映射，并重新检查依赖这些范围的姿态和策略。"
            ),
            "sourceId": "current-urdf",
            "layout": "full",
        },
        {
            "id": "urdf-vs-measured-block",
            "type": "chart",
            "chartId": "urdf-vs-measured",
            "layout": "full",
        },
        {
            "id": "quality-heading",
            "type": "markdown",
            "body": (
                "## 3. 静态标定质量\n\n"
                "最大静态验证误差的最弱项是小指 PIP `2.80°`，仍在 `3°` 验收门内；"
                "其余已记录项更低。回差用于表达方向相关不确定度，不与验证误差相加，且 "
                "Phase A 与 Phase B/C 的回差记录域不同，因此图表保留原 session 口径，"
                "不把它包装成单一全手评分。食指 MCP pitch 当前没有同口径 held-out 数值，"
                "食指 PIP 当前没有同口径回差数值，缺口在明细数据中保留为空。"
            ),
            "sourceId": "quality-corpus",
            "layout": "full",
        },
        {
            "id": "quality-overview-block",
            "type": "chart",
            "chartId": "quality-overview",
            "layout": "full",
        },
        {
            "id": "mimic-heading",
            "type": "markdown",
            "body": (
                "## 4. 被动联动不是一个通用倍率\n\n"
                "5 个现行 URDF mimic multiplier 全部偏大；四根长指的实测候选落在 "
                "`0.699–0.790`，而不是共同的 `0.8917`。拇指 IP 也从 `1.1619` 修正到 "
                "`0.9594` 候选。图中使用零截距候选以匹配 URDF mimic 形式；带偏置 affine "
                "诊断仍保留在源数据和 tooltip 中。"
            ),
            "sourceId": "current-urdf",
            "layout": "full",
        },
        {
            "id": "mimic-compare-block",
            "type": "chart",
            "chartId": "mimic-compare",
            "layout": "full",
        },
        {
            "id": "inventory-heading",
            "type": "markdown",
            "body": (
                "## 5. 逐关节证据矩阵\n\n"
                "下表是演示后的审计入口：每个主动关节给出 slot、LUT 结点数、实测行程与当前 "
                "软件合同；每个被动关节给出实测倍率。生产状态统一标为候选，同时单列影响迁移 "
                "顺序的注意事项。"
            ),
            "sourceId": "quality-corpus",
            "layout": "full",
        },
        {
            "id": "joint-table-block",
            "type": "table",
            "tableId": "joint-table",
            "layout": "full",
        },
        {
            "id": "next-steps",
            "type": "markdown",
            "body": (
                "## 6. Recommended Next Steps\n\n"
                "1. **原子化更新软件合同。** 同一次变更内同步 16 个主动关节 limits/LUT、"
                "5 个 mimic multiplier、top-down 安全门与 semantic limits；不要保留单关节先改的中间态。\n"
                "2. **重签 golden 与模型清单。** 更新 URDF digest、映射 golden、序列号绑定的"
                "生产标定 artifact，并让 fail-closed 门只接受完整版本。\n"
                "3. **做静态 sim↔real 复核。** 覆盖边界、中位和旧策略常用姿态，核查自碰撞、"
                "夹持几何与 raw 往返误差。\n"
                "4. **重训受范围变化影响的策略。** 尤其是 PIP、roll 与 mimic 的动作/观测分布；"
                "旧 checkpoint 不能默认继承新物理合同。\n"
                "5. **分级 commissioning。** 按 no-send → startup → 1–20 tick → 低速完整任务推进，"
                "每级记录故障、温度、零漂和接触异常后再放行。"
            ),
            "sourceId": "full-summary",
            "layout": "full",
        },
        {
            "id": "further-questions",
            "type": "markdown",
            "body": (
                "## Further Questions\n\n"
                "- 生产语义范围是否采用完整实测安全包络，还是为策略保留更保守的 operating envelope？\n"
                "- `index_mcp_roll`、`ring_mcp_roll` 与 `pinky_mcp_pitch` 的零锚不确定度，"
                "是以 metadata 传播，还是在上线前追加一次独立复测？\n"
                "- 这套标定是否只绑定当前序列号，还是要形成多台 G20 的批次标定与漂移监测流程？"
            ),
            "sourceId": "full-summary",
            "layout": "full",
        },
        {
            "id": "caveats",
            "type": "markdown",
            "body": (
                "## Caveats\n\n"
                "- 本报告描述的是单台左手 G20 在固定曝光、静态缓慢 sweep 下的候选物理标定；"
                "不代表动态负载、温升、磨损或多设备分布。\n"
                "- “稳定安全端点”不是机械极限。被防卷绕、接触、死区或复位不重复性拒绝的端点"
                "没有进入正式 LUT，但仍保留在原始证据中。\n"
                "- 全部 21 个关节的测量合同已经闭环；生产映射仍未整体启用。当前仓库还存在 "
                "`index_mcp_pitch` 单独变更与 golden/manifest 不一致，必须在上线前消除。\n"
                "- 质量图中的回差口径按各阶段原记录保留；不得将不同测量域的数值直接求和或"
                "解释为动态闭环误差。"
            ),
            "sourceId": "quality-corpus",
            "layout": "full",
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "LinkerHand G20 左手 21 关节全量物理标定的 presentation-ready "
            "证据报告：完成度、物理包络、质量、mimic 与生产迁移。"
        ),
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }


def write_supporting_notes() -> None:
    chart_map = """# Chart map

| Report segment | Question | Family/type | Dataset / fields | Supported claim |
|---|---|---|---|---|
| 真实物理包络 | 16 个主动关节各有多大稳定行程？ | ranked horizontal bar | `active_ranges`: `joint_label`, `measured_span_deg`, `family` | PIP 最大，thumb yaw/roll 次之，MCP roll 最小 |
| 软件合同差异 | 当前 URDF 与实测差在哪里？ | grouped horizontal bar | `active_range_long`: `joint_label`, `series`, `span_deg` | 偏差方向不一致，必须逐关节迁移 |
| 静态质量 | 验证误差和回差是否可接受？ | grouped horizontal bar + 3° reference | `quality_long`: `joint_label`, `metric`, `value_deg` | 已记录静态验证误差全部低于 3°；口径差异保留 |
| 被动联动 | 现行 mimic 与实测候选差多少？ | grouped bar | `mimic_long`: `passive_joint_label`, `series`, `multiplier` | 5 个现行倍率均偏大，四指不能共用单一倍率 |

Repeated horizontal bars are deliberate: the first is a ranked status view, the
second is a paired contract comparison, and the third compares two quality
measures against a decision threshold. The mimic comparison uses vertical bars
because it has only five compact categories.
"""
    readme = """# G20 full calibration presentation

Primary presentation surface:

- `g20_full_calibration_presentation.html`

Supporting audit artifacts:

- `artifact.json` — canonical report manifest and immutable snapshot datasets
- `CHART_MAP.md` — chart selection and claim map

Regenerate:

```bash
python3 tools/build_g20_full_calibration_presentation.py
```

Then package from the Data Analytics plugin root with its `report:deliver`
command. The HTML is self-contained and read-only.
"""
    (OUTPUT_ROOT / "CHART_MAP.md").write_text(chart_map, encoding="utf-8")
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    data_root = OUTPUT_ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for name, rows in artifact["snapshot"]["datasets"].items():
        (data_root / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (OUTPUT_ROOT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_supporting_notes()
    print(OUTPUT_ROOT / "artifact.json")


if __name__ == "__main__":
    main()
