#!/usr/bin/env python3
"""Build an offline sim-rad / SDK-range / physical-angle calibration packet.

This program is deliberately incapable of opening CAN or importing the vendor
hardware API.  It freezes the exact policy, URDF, provisional mapping, and SDK
source patch, then emits a human-reviewable sequence of raw 20-slot commands.
The generated sequence is a *plan*, not an authorization to move the hand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
JOINT_ORDER = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
)
RESERVED_SLOTS = (11, 12, 13, 14)
FIT_POINTS = (0.20, 0.50, 0.80)
HOLDOUT_POINTS = (0.35, 0.65)
LOW_PRECONDITION = 0.10
HIGH_PRECONDITION = 0.90
# Direct actuator candidates for discovering the *physical* PIP range.  These
# are not radian targets and are not labelled safe.  A future guarded executor
# must release them one at a time and may stop before either endpoint.
PIP_RANGE_RAW_DESCENDING = (255, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 20, 12, 6, 0)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": stat.st_size,
    }


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_record(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--short")
    diff = _git(root, "diff", "--no-ext-diff", "--binary")
    return {
        "root": str(root.resolve()),
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
        "status_line_count": len(status.splitlines()),
        "tracked_diff_sha256": _sha256_bytes(diff.encode("utf-8")),
        "tracked_diff_bytes": len(diff.encode("utf-8")),
    }


def _numbers(text: str | None, count: int, default: Sequence[float]) -> tuple[float, ...]:
    if text is None:
        return tuple(default)
    values = tuple(float(value) for value in text.split())
    if len(values) != count:
        raise ValueError(f"expected {count} numbers, got {text!r}")
    return values


def _load_urdf(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    root = ET.parse(path).getroot()
    joints: dict[str, dict[str, Any]] = {}
    mimic: dict[str, dict[str, Any]] = {}
    for node in root.findall("joint"):
        name = node.attrib["name"]
        parent = node.find("parent")
        child = node.find("child")
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        mimic_node = node.find("mimic")
        entry = {
            "name": name,
            "type": node.attrib.get("type"),
            "parent": None if parent is None else parent.attrib.get("link"),
            "child": None if child is None else child.attrib.get("link"),
            "origin_xyz": _numbers(None if origin is None else origin.attrib.get("xyz"), 3, (0, 0, 0)),
            "origin_rpy": _numbers(None if origin is None else origin.attrib.get("rpy"), 3, (0, 0, 0)),
            "axis_xyz": _numbers(None if axis is None else axis.attrib.get("xyz"), 3, (1, 0, 0)),
            "lower": None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"]),
            "upper": None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"]),
        }
        joints[name] = entry
        if mimic_node is not None:
            mimic[name] = {
                "source": mimic_node.attrib["joint"],
                "multiplier": float(mimic_node.attrib.get("multiplier", "1")),
                "offset": float(mimic_node.attrib.get("offset", "0")),
            }
    return joints, mimic


def _load_bundle(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"{path}: expected deploy bundle with a config object")
    config = payload["config"]
    required = (
        "task", "n_finger", "finger_lower", "finger_upper", "home_targets",
        "startup_reset_targets", "startup_reset_hardware_lower",
        "startup_reset_hardware_upper",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"bundle is missing config fields: {missing}")
    metadata = payload.get("deployment_metadata_repackage", {})
    return config, metadata if isinstance(metadata, dict) else {}


def _schema_limits(path: Path) -> dict[str, tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry["name"]): (float(entry["position_limit"][0]), float(entry["position_limit"][1]))
        for entry in data.get("ordered_joints", [])
    }


def _fraction_value(lower: float, upper: float, fraction: float) -> float:
    return lower + fraction * (upper - lower)


def _encode(q16: Sequence[float], table: Sequence[Any], sdkmap: Any) -> list[int]:
    if len(q16) != 16:
        raise ValueError("q16 must contain exactly 16 values")
    arc = [0.0] * 20
    for q, joint in zip(q16, table):
        fraction = min(max((float(q) - joint.lo) / (joint.hi - joint.lo), 0.0), 1.0)
        if joint.flip:
            fraction = 1.0 - fraction
        arc[joint.slot] = (
            sdkmap.L20_L_MIN[joint.slot]
            + fraction * (sdkmap.L20_L_MAX[joint.slot] - sdkmap.L20_L_MIN[joint.slot])
        )
    raw = sdkmap.arc_to_range_left(arc)
    return [int(round(min(max(value, 0.0), 255.0))) for value in raw]


def _decode(raw20: Sequence[float], table: Sequence[Any], sdkmap: Any) -> list[float]:
    arc = sdkmap.range_to_arc_left(raw20)
    values: list[float] = []
    for joint in table:
        span = sdkmap.L20_L_MAX[joint.slot] - sdkmap.L20_L_MIN[joint.slot]
        fraction = (arc[joint.slot] - sdkmap.L20_L_MIN[joint.slot]) / span
        if joint.flip:
            fraction = 1.0 - fraction
        values.append(joint.lo + min(max(fraction, 0.0), 1.0) * (joint.hi - joint.lo))
    return values


def _marker_contract(name: str, urdf: dict[str, Any]) -> dict[str, str]:
    axis = " ".join(f"{value:g}" for value in urdf["axis_xyz"])
    common = (
        f"固定标记分别贴在 {urdf['parent']} 与 {urdf['child']} 的刚性表面；"
        "不要跨关节缝、软胶或可移动外壳。"
    )
    if name.endswith("_pip"):
        return {
            "method": "manual_2d_coarse_then_video_review",
            "marker": common + "两条细直线沿相邻指骨纵向中心线。",
            "view": f"侧视，镜头光轴尽量平行 URDF 局部轴 [{axis}]；量两条中心线的屈曲夹角。",
            "zero_rule": "两指骨纵向中心线共线定义为约 0 rad；同时记录正方向。",
        }
    if name.endswith("_mcp_roll"):
        return {
            "method": "manual_2d_coarse_two_view",
            "marker": common + "掌骨与近节指骨各贴纵向细线。",
            "view": f"掌面/背面视图，镜头沿局部 roll 轴 [{axis}]；同指 pitch、PIP 固定在隔离姿态。",
            "zero_rule": "近节指骨位于该指掌骨中心平面时定义 roll 约 0 rad。",
        }
    if name.endswith("_mcp_pitch"):
        return {
            "method": "manual_2d_coarse_two_view",
            "marker": common + "掌骨与近节指骨各贴纵向细线。",
            "view": f"侧视，镜头沿局部 pitch 轴 [{axis}]；同指 roll 固定为 0，PIP 固定在隔离姿态。",
            "zero_rule": "按 URDF/仿真同视角判断伸直参考；任意贴条角只能证明角度增量，不能证明绝对零偏。",
        }
    if name == "thumb_mcp":
        return {
            "method": "manual_2d_coarse_then_video_review",
            "marker": common + "拇指近节与远节的刚性纵向面各贴细线。",
            "view": f"侧视，镜头沿局部轴 [{axis}]，其余拇指轴保持固定。",
            "zero_rule": "纵向参考线的伸直姿态定义约 0 rad，并拍清楚正方向。",
        }
    return {
        "method": "dual_view_or_apriltag_required_for_final",
        "marker": common + "最终精标建议掌部一枚固定标签、运动连杆一枚固定标签；手工细线仅用于粗筛。",
        "view": f"至少两个近似正交固定视角；目标关节轴为 URDF 局部 [{axis}]，其余拇指轴保持固定。",
        "zero_rule": "拇指 CMC 为串联多轴；单张 2D 图不能独立证明绝对角。需双视角/AprilTag 相对姿态或等价 3D 测量。",
    }


def _isolation_pose(
    joint_index: int,
    baseline: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> list[float]:
    pose = list(baseline)
    name = JOINT_ORDER[joint_index]
    if joint_index < 12:
        group = (joint_index // 3) * 3
        roll, pitch, pip = group, group + 1, group + 2
        if name.endswith("_mcp_roll"):
            pose[pitch] = _fraction_value(lower[pitch], upper[pitch], 0.20)
            pose[pip] = _fraction_value(lower[pip], upper[pip], 0.20)
        elif name.endswith("_mcp_pitch"):
            pose[roll] = 0.0
            pose[pip] = _fraction_value(lower[pip], upper[pip], 0.20)
        else:
            pose[roll] = 0.0
            pose[pitch] = _fraction_value(lower[pitch], upper[pitch], 0.20)
    elif name.startswith("thumb_cmc_"):
        pose[15] = _fraction_value(lower[15], upper[15], 0.20)
    return pose


def _validate_q(q16: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> None:
    for name, q, lo, hi in zip(JOINT_ORDER, q16, lower, upper):
        if not math.isfinite(q) or q < lo - 1e-9 or q > hi + 1e-9:
            raise ValueError(f"{name}: planned q={q} is outside bundle [{lo}, {hi}]")


def _build_rows(
    lower: Sequence[float],
    upper: Sequence[float],
    baseline: Sequence[float],
    table: Sequence[Any],
    sdkmap: Any,
) -> list[dict[str, Any]]:
    planned_lower = tuple(min(lo, start) for lo, start in zip(lower, baseline))
    planned_upper = tuple(max(hi, start) for hi, start in zip(upper, baseline))
    raw_envelopes: dict[str, tuple[int, int]] = {}
    for index, joint in enumerate(table):
        low_pose = list(baseline)
        high_pose = list(baseline)
        low_pose[index], high_pose[index] = lower[index], upper[index]
        values = (_encode(low_pose, table, sdkmap)[joint.slot], _encode(high_pose, table, sdkmap)[joint.slot])
        raw_envelopes[joint.name] = (min(values), max(values))

    rows: list[dict[str, Any]] = []
    sequence = 0

    def add(
        *, joint_index: int | None, phase: str, role: str, approach: str,
        q_fraction: float | None, q16: Sequence[float], measure: bool,
        note: str,
    ) -> None:
        nonlocal sequence
        _validate_q(q16, planned_lower, planned_upper)
        raw = _encode(q16, table, sdkmap)
        if any(raw[slot] != 0 for slot in RESERVED_SLOTS):
            raise ValueError(f"reserved SDK slot populated in sequence {sequence}")
        name = "all" if joint_index is None else JOINT_ORDER[joint_index]
        slot = "" if joint_index is None else table[joint_index].slot
        target_q = "" if joint_index is None else q16[joint_index]
        target_raw = "" if joint_index is None else raw[int(slot)]
        envelope = ("", "") if joint_index is None else raw_envelopes[name]
        if joint_index is not None and q_fraction is not None and not envelope[0] <= int(target_raw) <= envelope[1]:
            raise ValueError(f"{name}: target raw {target_raw} outside operational envelope {envelope}")
        rows.append({
            "sequence": sequence,
            "step_id": f"S{sequence:04d}",
            "joint_index": "" if joint_index is None else joint_index,
            "joint": name,
            "sdk_slot": slot,
            "phase": phase,
            "fit_role": role,
            "approach": approach,
            "q_fraction": "" if q_fraction is None else q_fraction,
            "planned_sim_q_rad": target_q,
            "planned_command_raw": target_raw,
            "configured_policy_raw_min": envelope[0],
            "configured_policy_raw_max": envelope[1],
            "requires_physical_measurement": "YES" if measure else "NO",
            "execution_gate": (
                "REGENERATE_AFTER_PHYSICAL_PIP_RANGE"
                if joint_index is not None and name.endswith("_pip")
                else "RANGE_DISCOVERY_AND_HUMAN_REVIEW"
            ),
            "hold_seconds": 2.0 if measure else 1.0,
            "raw20_json": json.dumps(raw, separators=(",", ":")),
            "q16_json": json.dumps([round(value, 9) for value in q16], separators=(",", ":")),
            "sdk_state_median_raw": "",
            "sdk_state_min_raw": "",
            "sdk_state_max_raw": "",
            "physical_angle_deg": "",
            "measurement_uncertainty_deg": "",
            "evidence_file": "",
            "operator_notes": note,
        })
        sequence += 1

    add(
        joint_index=None, phase="session", role="baseline", approach="none",
        q_fraction=None, q16=baseline, measure=False,
        note="Known previously exercised startup pose; visually confirm empty-hand clearance before any later motion.",
    )
    for index, name in enumerate(JOINT_ORDER):
        isolation = _isolation_pose(index, baseline, lower, upper)
        add(
            joint_index=index, phase="joint_setup", role="isolation_pose", approach="none",
            q_fraction=None, q16=isolation, measure=False,
            note="Pause for marker, camera-axis, and collision-clearance confirmation.",
        )
        for point in FIT_POINTS:
            pre = list(isolation)
            pre[index] = _fraction_value(lower[index], upper[index], LOW_PRECONDITION)
            add(
                joint_index=index, phase="fit", role="precondition_low", approach="increasing_q",
                q_fraction=LOW_PRECONDITION, q16=pre, measure=False,
                note=f"Precondition before fit point {point:.2f}; do not measure this row.",
            )
            target = list(isolation)
            target[index] = _fraction_value(lower[index], upper[index], point)
            add(
                joint_index=index, phase="fit", role=f"fit_{int(point * 100)}", approach="increasing_q",
                q_fraction=point, q16=target, measure=True,
                note="Record 20 SDK state samples plus physical angle and uncertainty.",
            )
        for point in HOLDOUT_POINTS:
            pre = list(isolation)
            pre[index] = _fraction_value(lower[index], upper[index], HIGH_PRECONDITION)
            add(
                joint_index=index, phase="holdout", role="precondition_high", approach="decreasing_q",
                q_fraction=HIGH_PRECONDITION, q16=pre, measure=False,
                note=f"Precondition before held-out point {point:.2f}; do not measure this row.",
            )
            target = list(isolation)
            target[index] = _fraction_value(lower[index], upper[index], point)
            add(
                joint_index=index, phase="holdout", role=f"holdout_{int(point * 100)}", approach="decreasing_q",
                q_fraction=point, q16=target, measure=True,
                note="Held out from fitting; record state and angle without changing the model.",
            )
        if name == "index_pip":
            for repetition in range(1, 6):
                pre = list(isolation)
                pre[index] = _fraction_value(lower[index], upper[index], LOW_PRECONDITION)
                add(
                    joint_index=index, phase="repeatability", role=f"repeat_pre_{repetition}",
                    approach="increasing_q", q_fraction=LOW_PRECONDITION, q16=pre,
                    measure=False, note="Manual-measurement repeatability precondition.",
                )
                target = list(isolation)
                target[index] = _fraction_value(lower[index], upper[index], 0.50)
                add(
                    joint_index=index, phase="repeatability", role=f"repeat_50_{repetition}",
                    approach="increasing_q", q_fraction=0.50, q16=target,
                    measure=True, note="Repeat the same physical measurement without reusing the previous reading.",
                )
        add(
            joint_index=index, phase="joint_end", role="return_baseline", approach="none",
            q_fraction=None, q16=baseline, measure=False,
            note="Return to the known startup pose before changing camera/markers.",
        )
    return rows


def _build_pip_range_rows(
    lower: Sequence[float],
    upper: Sequence[float],
    baseline: Sequence[float],
    table: Sequence[Any],
    sdkmap: Any,
) -> list[dict[str, Any]]:
    """Emit candidate raw sweeps without assigning either radian hypothesis.

    Only the selected PIP slot changes during each sweep.  The same finger's
    MCP roll/pitch are placed in the ordinary bundle isolation pose first.
    Endpoint rows remain conditional candidates; this function does not imply
    that raw 0 or 255 is mechanically safe for unattended execution.
    """

    rows: list[dict[str, Any]] = []
    sequence = 0
    for joint_index in (2, 5, 8, 11):
        name = JOINT_ORDER[joint_index]
        joint = table[joint_index]
        isolation = _isolation_pose(joint_index, baseline, lower, upper)
        isolation_raw = _encode(isolation, table, sdkmap)

        def add(raw_value: int, direction: str, phase: str, conditional: bool) -> None:
            nonlocal sequence
            raw20 = list(isolation_raw)
            raw20[joint.slot] = raw_value
            if any(raw20[slot] != 0 for slot in RESERVED_SLOTS):
                raise ValueError(f"reserved SDK slot populated in PIP range sequence {sequence}")
            rows.append({
                "sequence": sequence,
                "step_id": f"PIPR{sequence:03d}",
                "joint": name,
                "sdk_slot": joint.slot,
                "phase": phase,
                "approach": direction,
                "direct_command_raw": raw_value,
                "raw20_json": json.dumps(raw20, separators=(",", ":")),
                "isolation_reference_q16_json": json.dumps([round(value, 9) for value in isolation], separators=(",", ":")),
                "candidate_only": "YES",
                "requires_operator_release": "YES",
                "conditional_endpoint": "YES" if conditional else "NO",
                "screening_speed5_json": "[20,20,20,20,20]",
                "screening_torque5_json": "[40,40,40,40,40]",
                "operating_40_80_confirmation_required": "YES",
                "sdk_state_median_raw": "",
                "sdk_state_min_raw": "",
                "sdk_state_max_raw": "",
                "physical_pip_angle_deg": "",
                "physical_dip_angle_deg": "",
                "measurement_uncertainty_deg": "",
                "faults20_json": "",
                "temperature_before20_json": "",
                "temperature_after20_json": "",
                "sdk_current_feedback": "UNAVAILABLE_RETURNS_MINUS_ONE",
                "non_target_slot_max_delta_raw": "",
                "evidence_file": "",
                "operator_clearance": "",
                "operator_notes": (
                    "Do not execute unless the previous point is fault-free, clear of self-contact, and still moving monotonically. "
                    "Stop on plateau, unexpected sound/heat, fault, state jump, or contact."
                ),
            })
            sequence += 1

        # Extension-to-flexion pass identifies the high-angle plateau without
        # assuming that SDK nominal 1.08 or legacy schema 1.57 is correct.
        for raw_value in PIP_RANGE_RAW_DESCENDING:
            add(
                raw_value,
                direction="toward_flexion_raw_decreasing",
                phase="coarse_to_fine_upper_range_discovery",
                conditional=raw_value in (255, 6, 0),
            )
        # Sparse return pass exposes hysteresis.  It is conditional on the
        # descending pass stopping normally at a released point.
        for raw_value in (6, 12, 20, 32, 48, 64, 96, 128, 160, 192, 224, 240):
            add(
                raw_value,
                direction="return_toward_extension_raw_increasing",
                phase="upper_range_hysteresis_return",
                conditional=True,
            )
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = args.bundle.resolve()
    calib = args.calib.resolve()
    urdf_path = args.urdf.resolve()
    schema_path = args.schema.resolve()
    sdk_root = args.sdk_root.resolve()
    out_dir = args.out_dir.resolve()

    from screwdriver_rl.deploy import linker_sdk_map as sdkmap

    config, bundle_metadata = _load_bundle(bundle)
    if int(config["n_finger"]) != 16:
        raise ValueError(f"expected 16 bundle joints, got {config['n_finger']}")
    lower = tuple(float(value) for value in config["finger_lower"])
    upper = tuple(float(value) for value in config["finger_upper"])
    baseline = tuple(float(value) for value in config["startup_reset_targets"])
    if not (len(lower) == len(upper) == len(baseline) == 16):
        raise ValueError("bundle finger arrays must all have length 16")

    overlay = sdkmap.load_calibration_file(str(calib))
    table = tuple(sdkmap.build_joint_table(overlay))
    table_names = tuple(joint.name for joint in table)
    if table_names != JOINT_ORDER:
        raise ValueError(f"mapping order differs from runtime contract: {table_names}")

    urdf, mimic = _load_urdf(urdf_path)
    missing = [name for name in JOINT_ORDER if name not in urdf]
    if missing:
        raise ValueError(f"URDF is missing active joints: {missing}")
    urdf_limits = [(float(urdf[name]["lower"]), float(urdf[name]["upper"])) for name in JOINT_ORDER]
    hardware_lower = tuple(float(value) for value in config["startup_reset_hardware_lower"])
    hardware_upper = tuple(float(value) for value in config["startup_reset_hardware_upper"])
    for index, name in enumerate(JOINT_ORDER):
        lo, hi = urdf_limits[index]
        if not math.isclose(hardware_lower[index], lo, abs_tol=1e-9) or not math.isclose(hardware_upper[index], hi, abs_tol=1e-9):
            raise ValueError(f"{name}: bundle configured hardware limits differ from the URDF used by this software checkout")
        if lower[index] < lo - 1e-9 or upper[index] > hi + 1e-9:
            raise ValueError(f"{name}: bundle action range exceeds its current software URDF configuration")
    _validate_q(baseline, hardware_lower, hardware_upper)
    startup_outside_action_range = [
        {
            "joint": name,
            "startup_reset_q_rad": baseline[index],
            "action_lower_rad": lower[index],
            "action_upper_rad": upper[index],
        }
        for index, name in enumerate(JOINT_ORDER)
        if baseline[index] < lower[index] - 1e-9 or baseline[index] > upper[index] + 1e-9
    ]

    g20_path = sdk_root / "linker_hand_sdk_ros/scripts/LinkerHand/core/can/linker_hand_g20_can.py"
    api_path = sdk_root / "linker_hand_sdk_ros/scripts/LinkerHand/linker_hand_api.py"
    setting_path = sdk_root / "linker_hand_sdk_ros/scripts/LinkerHand/config/setting.yaml"
    mapping_path = sdk_root / "linker_hand_sdk_ros/scripts/LinkerHand/utils/mapping.py"
    g20_text = g20_path.read_text(encoding="utf-8")
    setting_text = setting_path.read_text(encoding="utf-8")
    sdk_checks = {
        "thumb_command_slots_match_readback": "'拇指': [5, 10, 0, 11, 12, 15]" in g20_text,
        "reserved_state_slots_guarded": "if 11 <= slot <= 14" in g20_text,
        "tip_state_written_after_reserved": "original[i + 15] = finger_data[5]" in g20_text,
        "left_hand_setting_is_g20": bool(re.search(r"LEFT_HAND:.*?JOINT:\s*G20\b", setting_text, re.S)),
    }
    if not all(sdk_checks.values()):
        raise RuntimeError(f"SDK patch/config checks failed: {sdk_checks}")
    sdk_telemetry = {
        "position_state20_available": "def get_current_status(self):" in g20_text,
        "fault20_available": "def get_fault(self):" in g20_text,
        "temperature20_available": "def get_temperature(self):" in g20_text,
        "actual_current_feedback_available": False,
        "get_current_implementation": "returns [-1] * 20",
        "get_torque_semantics": "configured maximum torque, not measured joint load",
        "range_stop_must_not_depend_on_current": True,
    }

    schema = _schema_limits(schema_path)
    schema_conflicts: list[dict[str, Any]] = []
    for name, (urdf_lo, urdf_hi) in zip(JOINT_ORDER, urdf_limits):
        if name in schema:
            schema_lo, schema_hi = schema[name]
            if not (math.isclose(schema_lo, urdf_lo, abs_tol=1e-9) and math.isclose(schema_hi, urdf_hi, abs_tol=1e-9)):
                schema_conflicts.append({
                    "joint": name,
                    "current_urdf_limit_rad": [urdf_lo, urdf_hi],
                    "semantic_schema_limit_rad": [schema_lo, schema_hi],
                    "blocks_new_mapper_integration": True,
                })

    rows = _build_rows(lower, upper, baseline, table, sdkmap)
    pip_range_rows = _build_pip_range_rows(lower, upper, baseline, table, sdkmap)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "measurement_plan.csv", rows)
    _write_csv(out_dir / "pip_physical_range_discovery.csv", pip_range_rows)

    followers = {entry["source"]: (name, entry) for name, entry in mimic.items()}
    contract_rows: list[dict[str, Any]] = []
    for index, (name, joint, limits) in enumerate(zip(JOINT_ORDER, table, urdf_limits)):
        low_pose, high_pose = list(baseline), list(baseline)
        low_pose[index], high_pose[index] = lower[index], upper[index]
        low_raw = _encode(low_pose, table, sdkmap)[joint.slot]
        high_raw = _encode(high_pose, table, sdkmap)[joint.slot]
        marker = _marker_contract(name, urdf[name])
        follower = followers.get(name)
        contract_rows.append({
            "sim_index": index,
            "joint": name,
            "sdk_slot": joint.slot,
            "urdf_parent": urdf[name]["parent"],
            "urdf_child": urdf[name]["child"],
            "urdf_axis_xyz": " ".join(str(value) for value in urdf[name]["axis_xyz"]),
            "urdf_lower_rad": limits[0],
            "urdf_upper_rad": limits[1],
            "bundle_lower_rad": lower[index],
            "bundle_upper_rad": upper[index],
            "provisional_map_lower_rad": joint.lo,
            "provisional_map_upper_rad": joint.hi,
            "provisional_flip": joint.flip,
            "bundle_lower_raw": low_raw,
            "bundle_upper_raw": high_raw,
            "configured_policy_raw_min": min(low_raw, high_raw),
            "configured_policy_raw_max": max(low_raw, high_raw),
            "provisional_rad_per_raw_count": abs(joint.hi - joint.lo) / 255.0,
            "mimic_follower": "" if follower is None else follower[0],
            "mimic_multiplier": "" if follower is None else follower[1]["multiplier"],
            "physical_limit_status": (
                "UNKNOWN_MUST_MEASURE_DIRECT_RAW_RANGE"
                if name.endswith("_pip") else "UNVERIFIED"
            ),
            "measurement_method": marker["method"],
            "marker_definition": marker["marker"],
            "camera_definition": marker["view"],
            "zero_definition": marker["zero_rule"],
        })
    _write_csv(out_dir / "joint_contract.csv", contract_rows)

    dry_run: list[dict[str, Any]] = []
    for row in rows:
        if row["requires_physical_measurement"] != "YES":
            continue
        q16 = json.loads(row["q16_json"])
        raw = json.loads(row["raw20_json"])
        decoded = _decode(raw, table, sdkmap)
        index = int(row["joint_index"])
        dry_run.append({
            "step_id": row["step_id"],
            "joint": row["joint"],
            "planned_sim_q_rad": q16[index],
            "command_raw": raw[int(row["sdk_slot"])],
            "same_table_decoded_q_rad": decoded[index],
            "same_table_roundtrip_error_rad": decoded[index] - q16[index],
            "warning": "same-table arithmetic only; this is not physical validation",
        })
    (out_dir / "software_dry_run.json").write_text(json.dumps(dry_run, indent=2) + "\n", encoding="utf-8")

    sdk_rel_paths = [
        "linker_hand_sdk_ros/scripts/LinkerHand/core/can/linker_hand_g20_can.py",
        "linker_hand_sdk_ros/scripts/LinkerHand/config/setting.yaml",
    ]
    sdk_patch = _git(sdk_root, "diff", "--no-ext-diff", "--", *sdk_rel_paths)
    (out_dir / "sdk_mapping_patch.diff").write_text(sdk_patch, encoding="utf-8")

    frozen_files = {
        "policy_bundle": _file_record(bundle),
        "current_urdf": _file_record(urdf_path),
        "provisional_overlay": _file_record(calib),
        "live_mapping_module": _file_record(REPO_ROOT / "screwdriver_rl/deploy/linker_sdk_map.py"),
        "live_deploy_entrypoint": _file_record(REPO_ROOT / "screwdriver_rl/deploy/deploy_linker.py"),
        "task_joint_contract": _file_record(REPO_ROOT / "screwdriver_rl/tasks/linker_l20/screwdriver_rotation_env.py"),
        "semantic_schema_not_live": _file_record(schema_path),
        "sdk_g20_mapping": _file_record(g20_path),
        "sdk_api": _file_record(api_path),
        "sdk_setting": _file_record(setting_path),
        "sdk_native_mapping": _file_record(mapping_path),
    }
    manifest = {
        "schema_version": 1,
        "packet_type": "offline-g20-physical-joint-calibration-plan",
        "hardware_io_performed": False,
        "hardware_motion_authorized": False,
        "expected_hand": {
            "serial": args.expected_serial,
            "side": "left",
            "model": "G20",
            "can": "can0",
            "must_be_verified_live_before_motion": True,
        },
        "runtime_source_of_truth": {
            "joint_order": list(JOINT_ORDER),
            "task": config["task"],
            "policy_control_period_ns": config.get("proprio_codec", {}).get("control_period_ns"),
            "policy_action_delta_scale": config.get("action_delta_scale"),
            "urdf_limits_are_current_software_configuration": True,
            "urdf_limits_are_physical_truth": False,
            "pip_physical_upper_limit_status": "UNKNOWN_PENDING_DIRECT_RAW_MEASUREMENT",
            "pip_1p08_and_1p57_are_unverified_software_hypotheses": True,
            "startup_reset_targets_outside_policy_action_range": startup_outside_action_range,
            "provisional_overlay_is_not_independent_physical_calibration": True,
        },
        "bundle_repackage_metadata": bundle_metadata,
        "sdk_source_checks": sdk_checks,
        "sdk_telemetry_contract": sdk_telemetry,
        "known_schema_conflicts": schema_conflicts,
        "schema_conflict_disposition": (
            "Do not choose URDF 1.08 or semantic-schema 1.57 as physical truth. Measure all four PIP raw-to-angle ranges first, then update both artifacts and regenerate digests."
            if schema_conflicts else "none"
        ),
        "git": {
            "dex_forge": _git_record(REPO_ROOT),
            "linkerhand_ros_sdk": _git_record(sdk_root),
        },
        "frozen_files": frozen_files,
        "plan_counts": {
            "all_rows": len(rows),
            "measurement_rows": sum(row["requires_physical_measurement"] == "YES" for row in rows),
            "joint_count": len(JOINT_ORDER),
            "index_pip_repeatability_measurements": sum(
                row["joint"] == "index_pip" and row["phase"] == "repeatability"
                and row["requires_physical_measurement"] == "YES" for row in rows
            ),
            "pip_range_candidate_rows": len(pip_range_rows),
            "pip_range_joint_count": len({row["joint"] for row in pip_range_rows}),
        },
    }
    (out_dir / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    preview_lines = [
        "# Raw 20-slot calibration preview",
        "",
        "> OFFLINE PLAN ONLY. No command in this file has been sent to hardware.",
        "",
        "| joint | slot | fit q/raw (20%, 50%, 80%) | holdout q/raw (35%, 65%) | configured policy raw |",
        "|---|---:|---|---|---|",
    ]
    for name in JOINT_ORDER:
        measured = [row for row in rows if row["joint"] == name and row["requires_physical_measurement"] == "YES" and row["phase"] in ("fit", "holdout")]
        fit = [row for row in measured if row["phase"] == "fit"]
        hold = [row for row in measured if row["phase"] == "holdout"]
        fmt = lambda values: ", ".join(f"{float(row['planned_sim_q_rad']):.4f}/{row['planned_command_raw']}" for row in values)
        preview_lines.append(
            f"| {name} | {fit[0]['sdk_slot']} | {fmt(fit)} | {fmt(hold)} | {fit[0]['configured_policy_raw_min']}..{fit[0]['configured_policy_raw_max']} |"
        )
    (out_dir / "raw20_preview.md").write_text("\n".join(preview_lines) + "\n", encoding="utf-8")

    conflict_text = "\n".join(
        f"- `{item['joint']}`: URDF {item['current_urdf_limit_rad']} vs schema {item['semantic_schema_limit_rad']}"
        for item in schema_conflicts
    ) or "- 无"
    instructions = f"""# G20 物理关节角校准操作定义（尚未授权运动）

## 这次要独立验证什么

每个关节分别拟合两条曲线，不能继续拿同一张表正算再反算来证明正确：

1. `sim q 目标(rad) -> 实际发送 raw(0..255) -> 外部测得物理 q`
2. `SDK 回读 raw(0..255) -> 外部测得物理 q`

第一条用于指令编码，第二条用于 observation 解码。两者允许不同，以显式保留零偏、回差和负载变形。

## 固定与贴标

- 手掌刚性固定；整个单关节视频期间不能挪相机、支架或手掌。
- 每条标记只贴在一个刚性连杆上，不能跨关节缝、软胶、线缆或松动外壳。
- 镜头光轴应尽量沿 `joint_contract.csv` 的关节轴；透视不正会把 3D 转动误读成 2D 角。
- 长指 PIP、拇指 MCP：相邻指骨纵向中心线可先做手工量角。
- 长指 MCP roll/pitch：同一关节是串联双轴，按表内隔离姿态与正交视角分别测，不能从任意单视图同时读两个角。
- 拇指 CMC yaw/roll/pitch：单张 2D 手工图只用于发现大错；最终合格必须用固定双视角或掌部+运动连杆 AprilTag 的相对 3D 姿态。

贴条可以有常数角偏，但此时只能可靠测 `delta q`。若要证明绝对 q，贴条必须与指定连杆中心线/参考面平行，或另外记录可信的物理零位；否则不能把 arbitrary marker angle 直接叫作 sim rad。

## 第零阶段：先独立测四个 PIP 的物理范围

`pip_physical_range_discovery.csv` 直接给 slot 16/17/18/19 候选 raw，不把 raw 换算成 1.08 或 1.57 中的任何一个 rad。raw 从伸展端逐步向屈曲端扫描，并有稀疏反向返回点测回差。

- 第一遍只是低参数危险筛查：建议 speed=20、torque limit=40；这不是最终物理上限。筛查无异常后，只对平台附近候选点用此前已实机运行过的 speed=40、torque=80 复核，最终结果必须注明扭矩条件。
- 每个候选点都必须人工逐点放行；raw 255、6、0 等端点是条件候选，不是“已证明安全”。
- 每步记录 PIP 本体角、DIP 跟随角、20 次状态 median/min/max、故障、非目标槽变化和视频。
- 看到角度平台、异常声响/发热、故障、状态跳变或自碰撞立即停止，不能为了碰到 raw 0 强行继续。
- 物理上限定义为低速低扭矩条件下、无故障/无自碰撞时可重复到达的角度平台；不是配置文件里的数字。
- 当前 SDK 没有真实电流反馈：`get_current()` 固定返回 -1，`get_torque()` 只回读最大扭矩设定。因此不得声称“看电流确认到机械上限”；判停依赖外部角度平台、位置回读、故障、温度、声音与接触检查。
- 测完四指后，先更新 URDF/semantic schema/映射并重生成本包，原 `measurement_plan.csv` 内所有 PIP 行均标为 `REGENERATE_AFTER_PHYSICAL_PIP_RANGE`，不能直接执行。

## 后续分阶段门禁

1. PIP 范围落盘并重生成软件定义后，先做 `index_pip` 的 50% 点五次重复测量，由五次极差估计手工量角噪声。
2. 再做各关节 20/50/80% 拟合点；35/65% 是保留验证点，拟合时不能用。
3. 每个测量点采 20 个 SDK 回读，表中填 median/min/max，并记录视频文件与时间。
4. 每次先到 10% 或 90% 预处理点，再单向接近目标，以测出回差。
5. 每换一个关节，回到冻结的 startup baseline，重新检查空手碰撞间隙。

## 暂定验收线

- 槽位、符号、单调性必须 100% 正确；保留槽 11--14 始终为 0。
- 手工重复性若不差于 2 deg：held-out 平均绝对误差 <= 2 deg、最大 <= 4 deg、方向回差 <= 3 deg。
- 任一反向、非单调、明显串扰或外部误差 > 7 deg：立即停止，policy 部署不放行。
- 最终阈值会按第一步测得的人工量角重复性调整；手工噪声本身不能被误判成硬件误差。
- 同时检查 DIP/IP 实际跟随比：长指约 0.8917，拇指约 1.1619；偏差需单独记录，不能塞进主动关节 offset。

## 当前已知架构冲突

{conflict_text}

这里的 URDF 1.08 与 schema 1.57 都只是互相冲突的软件假设，不代表实机物理上限。本包冻结它们是为了可追溯，不是判定谁正确。新 semantic mapper 在四个 PIP 实测、统一限位并重算 digest 前禁止接入。

## 安全说明

`measurement_plan.csv` 只是待审指令表。本规划器没有导入厂商硬件 API、没有打开 CAN、没有给真手发送任何数据。实际执行器必须另做串口号核对、故障核对、低速/低扭矩、逐步人工确认与急停。
"""
    (out_dir / "measurement_instructions.md").write_text(instructions, encoding="utf-8")

    pip_checklist = """# PIP 物理范围首轮检查表（不含运动授权）

## 机械与视觉准备

- [ ] 手上没有螺丝刀或其他物体；四指完整运动空间内无障碍物。
- [ ] 手掌刚性固定，急停/断电手段在手边。
- [ ] 目标指的 proximal 与 middle 刚性指骨各贴一条纵向细线，标记没有跨关节缝。
- [ ] 同时给 middle 与 distal 贴线，以独立测 DIP 跟随角。
- [ ] 手机固定侧视，镜头尽量沿该 PIP 转轴；画面包含两条 PIP 标记、两条 DIP 标记和步骤编号。
- [ ] 先连续量同一静态姿态五次，写下人工测量极差。

## 软件预检

- [ ] live serial 必须等于 `LHT20-010-415-L-B-1-D`。
- [ ] 左手型号 G20、CAN can0；位置/故障/温度查询全部有效。
- [ ] 所有故障为 0；保留槽 11--14 为 0。
- [ ] 当前 G20 SDK patch hash 与 `freeze_manifest.json` 一致。
- [ ] 明确知道 SDK 没有实际电流反馈；不能用 `get_current()` 判机械止挡。

## 初筛与判停

- [ ] 第一遍仅用候选 speed=20、torque=40，逐行人工放行；不连续自动扫。
- [ ] 每点稳定后采 20 个 state，填 median/min/max，再手工量 PIP 与 DIP 角并记录视频时间。
- [ ] 任一故障、异常声响/温升、状态跳变、非目标关节变化、自碰撞或连续 raw 变化却角度不再安全增加：立即停止并保持/安全释放。
- [ ] raw 255、6、0 均为条件候选；不要求一定到 0。
- [ ] 初筛结束后不直接改 URDF；先画 raw-command/raw-state/外部角三条曲线并确定平台与回差。
- [ ] 最终上限只在平台附近用已验证的 speed=40、torque=80 少量点复核，并把扭矩条件写入校准 artifact。

## 四指记录

| PIP | slot | 初筛完成 | 复核完成 | 可重复物理上限 deg | 不确定度 deg | 备注 |
|---|---:|---|---|---:|---:|---|
| index_pip | 16 | [ ] | [ ] | | | |
| middle_pip | 17 | [ ] | [ ] | | | |
| ring_pip | 18 | [ ] | [ ] | | | |
| pinky_pip | 19 | [ ] | [ ] | | | |
"""
    (out_dir / "PIP_FIRST_RUN_CHECKLIST.md").write_text(pip_checklist, encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path,
        default=REPO_ROOT / "deliverables/linker_g20_topdown_d64_action008_stage2_commissioning_20260727/runtime/deploy.pth",
    )
    parser.add_argument("--calib", type=Path, default=REPO_ROOT / "linker_calib_thumbfit.json")
    parser.add_argument("--urdf", type=Path, default=REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf")
    parser.add_argument(
        "--schema", type=Path,
        default=REPO_ROOT / "assets/calibrations/linker_g20_left_semantic_schema_v1.json",
    )
    parser.add_argument("--sdk-root", type=Path, default=Path("/home/user/linkerhand-ros-sdk"))
    parser.add_argument("--expected-serial", default="LHT20-010-415-L-B-1-D")
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "records/g20_physical_joint_calibration_20260801/planning",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate(args)
    print(f"[offline-plan] wrote {args.out_dir.resolve()}")
    print(f"[offline-plan] hardware_io_performed={manifest['hardware_io_performed']}")
    print(f"[offline-plan] measurement_rows={manifest['plan_counts']['measurement_rows']}")
    print(f"[offline-plan] pip_range_candidate_rows={manifest['plan_counts']['pip_range_candidate_rows']}")
    print(f"[offline-plan] schema_conflicts={len(manifest['known_schema_conflicts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
