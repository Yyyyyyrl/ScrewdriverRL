"""CPU/XML contracts for the 60/64/68 mm top-down geometry distribution."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from screwdriver_rl.utils.linker_topdown_diameter_postures import (
    TOPDOWN_HANDLE_DIAMETERS_M,
    TOPDOWN_HANDLE_LENGTH_M,
    TOPDOWN_NOMINAL_BUCKET_INDEX,
    TOPDOWN_PREGRASP_POSITIONS_BUCKETS,
    TOPDOWN_RESET_POSITIONS_BUCKETS,
    TOPDOWN_ROOT_POS_OFFSETS_BUCKETS,
    TOPDOWN_ROOT_POS_W,
    TOPDOWN_ROOT_QUAT_WXYZ,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "assets/screwdriver/screwdriver_isaaclab.urdf"
NOMINAL = ROOT / "assets/screwdriver/screwdriver_64mm_handle.urdf"
VARIANT_DIR = ROOT / "assets/screwdriver/topdown_variants"
MANIFEST = VARIANT_DIR / "manifest.json"
HAND = ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
REGISTRATION = (
    ROOT
    / "screwdriver_rl/tasks/linker_l20/screwdriver_rotation_topdown_registration.py"
)
DR_CFG = (
    ROOT
    / "screwdriver_rl/tasks/linker_l20/screwdriver_rotation_topdown_diameter_env_cfg.py"
)
BASE_CFG = ROOT / "screwdriver_rl/tasks/base/screwdriver_rotation_env_cfg.py"
BASE_ENV = ROOT / "screwdriver_rl/tasks/base/screwdriver_rotation_env.py"
LINKER_ENV = ROOT / "screwdriver_rl/tasks/linker_l20/screwdriver_rotation_env.py"
NOMINAL_VALIDATOR = ROOT / "tools/validate_linker_l20_screwdriver_topdown.py"
PHYSICS_VALIDATOR = (
    ROOT / "tools/validate_linker_l20_screwdriver_topdown_physics.py"
)
BANK_VALIDATOR = ROOT / "tools/validate_linker_l20_screwdriver_topdown_diameter_bank.py"
ORIGINAL_SHA256 = "aa6b2a38f82845e7c0665d2d480114dead56f4457930f90fc765a3d937b6a949"


def _find_link(root: ET.Element, name: str) -> ET.Element:
    return next(link for link in root.findall("link") if link.get("name") == name)


def _signature(element: ET.Element) -> tuple:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        tuple(_signature(child) for child in element),
    )


def _canonical_asset(path: Path) -> tuple:
    root = ET.parse(path).getroot()
    root.set("name", "canonical")
    for link_name in ("screwdriver_body", "screwdriver_cap"):
        link = _find_link(root, link_name)
        for role in ("visual", "collision"):
            link.find(f"{role}/geometry/cylinder").set("radius", "canonical")
    return _signature(root)


def _inertials(path: Path) -> dict[str, tuple]:
    root = ET.parse(path).getroot()
    return {
        link.get("name"): _signature(link.find("inertial"))
        for link in root.findall("link")
        if link.find("inertial") is not None
    }


def _independent_positions(bucket: dict[str, tuple[float, ...]]) -> dict[str, float]:
    result = {
        name: value
        for finger in ("index", "middle", "ring", "pinky")
        for name, value in zip(
            (f"{finger}_mcp_roll", f"{finger}_mcp_pitch", f"{finger}_pip"),
            bucket[finger],
        )
    }
    result.update(
        zip(
            ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
            bucket["thumb"],
        )
    )
    return result


def _rotate_wxyz(quat, vector) -> tuple[float, float, float]:
    w, x, y, z = quat
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def test_manifest_is_fixed_length_equal_weight_three_diameter_bank():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["assignment_mode"] == "cyclic"
    assert manifest["preserve_source_mass_inertia"] is True
    assert manifest["num_diameter_buckets"] == 3
    assert manifest["num_length_buckets"] == 1
    assert manifest["num_buckets"] == 3
    variants = manifest["variants"]
    assert tuple(row["diameter"] for row in variants) == pytest.approx(
        TOPDOWN_HANDLE_DIAMETERS_M, abs=1.0e-12
    )
    assert all(row["length"] == pytest.approx(TOPDOWN_HANDLE_LENGTH_M) for row in variants)
    assert [row["bucket"] for row in variants] == [0, 1, 2]
    assert [row["index"] for row in variants] == [0, 1, 2]
    assert all(row["length_scale"] == pytest.approx(1.0) for row in variants)


def test_all_variant_assets_change_only_body_cap_visual_collision_radius():
    manifest = json.loads(MANIFEST.read_text())
    nominal_signature = _canonical_asset(NOMINAL)
    nominal_inertials = _inertials(NOMINAL)
    for row in manifest["variants"]:
        asset = VARIANT_DIR.parent / row["file"]
        assert asset.is_file()
        root = ET.parse(asset).getroot()
        expected_radius = 0.5 * row["diameter"]
        for link_name in ("screwdriver_body", "screwdriver_cap"):
            link = _find_link(root, link_name)
            for role in ("visual", "collision"):
                cylinder = link.find(f"{role}/geometry/cylinder")
                assert float(cylinder.get("radius")) == pytest.approx(
                    expected_radius, abs=1.0e-12
                )
            if link_name == "screwdriver_body":
                assert float(link.find("collision/geometry/cylinder").get("length")) == pytest.approx(
                    0.100, abs=1.0e-12
                )
        assert _canonical_asset(asset) == nominal_signature
        assert _inertials(asset) == nominal_inertials


def test_original_and_nominal_assets_remain_unchanged():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA256
    assert _canonical_asset(NOMINAL) == _canonical_asset(ORIGINAL)


def test_all_bucket_postures_keep_joint_and_mimic_margin():
    joints = {
        joint.get("name"): joint
        for joint in ET.parse(HAND).getroot().findall("joint")
        if joint.get("type") == "revolute"
    }
    for table in (TOPDOWN_RESET_POSITIONS_BUCKETS, TOPDOWN_PREGRASP_POSITIONS_BUCKETS):
        assert len(table) == 3
        for bucket in table:
            expanded = _independent_positions(bucket)
            unresolved = {
                name for name, joint in joints.items() if joint.find("mimic") is not None
            }
            while unresolved:
                for name in tuple(unresolved):
                    mimic = joints[name].find("mimic")
                    if mimic.get("joint") not in expanded:
                        continue
                    expanded[name] = (
                        expanded[mimic.get("joint")] * float(mimic.get("multiplier", "1"))
                        + float(mimic.get("offset", "0"))
                    )
                    unresolved.remove(name)
            assert set(expanded) == set(joints)
            minimum = math.inf
            for name, value in expanded.items():
                limit = joints[name].find("limit")
                lo, hi = float(limit.get("lower")), float(limit.get("upper"))
                minimum = min(minimum, value - lo, hi - value)
            assert minimum >= 0.10


def test_all_buckets_keep_palm_down_and_hand_above_fixed_length_handle():
    normal = _rotate_wxyz(TOPDOWN_ROOT_QUAT_WXYZ, (1.0, 0.0, 0.0))
    assert normal == pytest.approx((0.0, 0.0, -1.0), abs=1.0e-9)
    handle_top_z = 1.205 + 0.100 + TOPDOWN_HANDLE_LENGTH_M + 0.001
    for offset in TOPDOWN_ROOT_POS_OFFSETS_BUCKETS:
        assert TOPDOWN_ROOT_POS_W[2] + offset[2] > handle_top_z


def test_nominal_bucket_is_index_one_and_geometry_runtime_is_explicit():
    assert TOPDOWN_NOMINAL_BUCKET_INDEX == 1
    registration = REGISTRATION.read_text()
    assert "screwdriver_rotation_topdown_diameter_env_cfg" in registration
    assert "LinkerL20ScrewdriverRotationTopdownDiameterRandEnvCfg" in registration

    cfg_source = DR_CFG.read_text()
    assert "LinkerL20ScrewdriverRotationTopdownEnvCfg" in cfg_source
    assert 'self.geometry_variant_assignment = "cyclic"' in cfg_source
    assert "self.domain_rand.randomize_geometry = True" in cfg_source
    assert '"topdown_variants"' in cfg_source
    assert "self.privileged_obs_dim += 2" in cfg_source
    assert "TOPDOWN_PREGRASP_POSITIONS_BUCKETS" in cfg_source
    assert "TOPDOWN_RESET_POSITIONS_BUCKETS" in cfg_source

    base_cfg_source = BASE_CFG.read_text()
    assert 'geometry_variant_assignment: str = "signature"' in base_cfg_source
    assert 'random_choice=assignment == "signature"' in base_cfg_source
    base_env_source = BASE_ENV.read_text()
    assert 'if assignment == "cyclic"' in base_env_source
    assert "vidx % table.num_variants" in base_env_source
    assert "reset_joint_positions_buckets" in base_env_source
    assert "reset_posture[bucket_of]" in base_env_source
    linker_env_source = LINKER_ENV.read_text()
    assert 'getattr(cfg, "scale_drive_speed_with_geometry", False)' in linker_env_source
    assert "self._env_geom_scale[:, :1]" in linker_env_source


def test_dr_subclass_does_not_copy_reward_action_or_curriculum_contracts():
    tree = ast.parse(DR_CFG.read_text(), filename=str(DR_CFG))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LinkerL20ScrewdriverRotationTopdownDiameterRandEnvCfg"
    )
    assert [ast.unparse(base) for base in cls.bases] == [
        "LinkerL20ScrewdriverRotationTopdownEnvCfg"
    ]
    assigned = {
        target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        for target in (node.target,)
    }
    assert assigned == {"scale_drive_speed_with_geometry"}
    forbidden = {
        "observation_space",
        "action_space",
        "curriculum_phases",
        "domain_rand",
        "action_delta_scale",
        "contact_f_min",
        "contact_f_lo",
        "contact_f_hi",
        "contact_f_max",
        "screwdriver_load_torque",
        "joint_motion_range",
    }
    assert not (assigned & forbidden)


def test_diameter_bank_clearance_override_is_physics_backed():
    def constant(path: Path, name: str) -> float:
        tree = ast.parse(path.read_text(), filename=str(path))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        )
        return ast.literal_eval(node.value)

    nominal_clearance = constant(NOMINAL_VALIDATOR, "MIN_NON_DISTAL_CLEARANCE_M")
    bank_clearance = constant(BANK_VALIDATOR, "DIAMETER_BANK_MIN_NON_DISTAL_CLEARANCE_M")
    assert nominal_clearance == pytest.approx(0.0015)
    assert 0.0 < bank_clearance < nominal_clearance

    nominal_tree = ast.parse(NOMINAL_VALIDATOR.read_text(), filename=str(NOMINAL_VALIDATOR))
    checker = next(
        node
        for node in nominal_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_non_distal_checks"
    )
    assert [arg.arg for arg in checker.args.kwonlyargs] == ["min_clearance_m"]
    assert ast.unparse(checker.args.kw_defaults[0]) == (
        "MIN_NON_DISTAL_CLEARANCE_M"
    )

    bank_source = BANK_VALIDATOR.read_text()
    for contract in (
        "min_clearance_m=DIAMETER_BANK_MIN_NON_DISTAL_CLEARANCE_M",
        '"physics_validation_pass"',
        '"non_fingertip_contact_below_threshold"',
        '"zero_action_raw_shaft_drift_within_limit"',
        '"training_turn_authorization_persistent"',
        'and physics_contact_gate["pass"]',
    ):
        assert contract in bank_source

    physics_source = PHYSICS_VALIDATOR.read_text()
    for contract in (
        "--max_zero_action_drift_rad_s",
        "zero_action_raw_shaft_drift_rad_s_max_abs_environment_mean",
        "zero_action_raw_shaft_drift_within_limit",
        "training_turn_authorization_persistent",
    ):
        assert contract in physics_source
