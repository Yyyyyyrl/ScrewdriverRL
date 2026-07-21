"""Pure-Python guards for the Linker L20 top-down in-hand task."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_URDF = _ROOT / "assets" / "linker_hand_l20" / "linkerhand_l20_left.urdf"
_CFG = (
    _ROOT
    / "screwdriver_rl"
    / "tasks"
    / "linker_l20"
    / "inhand_rotation_topdown_env_cfg.py"
)
_REGISTRY = _ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "__init__.py"
_GENERATOR = _ROOT / "tools" / "gen_inhand_grasp_cache.py"

_FINGER_JOINTS = {
    "index": ("index_mcp_roll", "index_mcp_pitch", "index_pip"),
    "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
    "ring": ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
    "pinky": ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
    "thumb": ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
}


@pytest.fixture(scope="module")
def cfg_tree() -> ast.Module:
    return ast.parse(_CFG.read_text())


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        target = None
        value = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target == name:
            return ast.literal_eval(value)
    raise AssertionError(f"{name} not found")


def _quat_rotate(q, v):
    w, x, y, z = q
    t = (
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    )
    return (
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    )


def test_topdown_root_and_palm_geometry(cfg_tree):
    root = _literal(cfg_tree, "TOPDOWN_HAND_POS")
    obj = _literal(cfg_tree, "TOPDOWN_OBJECT_INIT_POS")
    rot = _literal(cfg_tree, "TOPDOWN_HAND_ROT")

    assert sum(c * c for c in rot) == pytest.approx(1.0)
    # Linker base-local +X is the palm normal; +Z is the finger direction.
    assert _quat_rotate(rot, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, -1.0))
    assert _quat_rotate(rot, (0.0, 0.0, 1.0)) == pytest.approx((0.0, -1.0, 0.0))
    assert root[2] - obj[2] >= 0.06


def test_rotation_axis_is_gravity(cfg_tree):
    assert _literal(cfg_tree, "TOPDOWN_ROT_AXIS") == (0.0, 0.0, -1.0)


def test_each_shape_has_a_complete_distinct_seed(cfg_tree):
    seeds = _literal(cfg_tree, "TOPDOWN_PREGRASP_BY_SHAPE")
    assert set(seeds) == {"cylinder", "cuboid", "sphere"}
    for shape, seed in seeds.items():
        assert set(seed) == set(_FINGER_JOINTS), shape
        for finger, joint_names in _FINGER_JOINTS.items():
            assert len(seed[finger]) == len(joint_names), (shape, finger)
    assert seeds["cylinder"] != seeds["cuboid"]
    assert seeds["cylinder"] != seeds["sphere"]


def test_all_shape_seeds_respect_urdf_limits_with_margin(cfg_tree):
    seeds = _literal(cfg_tree, "TOPDOWN_PREGRASP_BY_SHAPE")
    root = ET.parse(_URDF).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}

    for shape, seed in seeds.items():
        independent = {
            joint_name: value
            for finger, joint_names in _FINGER_JOINTS.items()
            for joint_name, value in zip(joint_names, seed[finger], strict=True)
        }
        for joint_name, value in independent.items():
            limit = joints[joint_name].find("limit")
            lo = float(limit.get("lower"))
            hi = float(limit.get("upper"))
            assert lo <= value <= hi, (shape, joint_name, value, lo, hi)
            assert min(value - lo, hi - value) >= 0.0399, (
                shape,
                joint_name,
                value,
                lo,
                hi,
            )


def test_topdown_training_and_graspgen_tasks_are_registered():
    source = _REGISTRY.read_text()
    assert 'id="Isaac-LinkerL20-Inhand-Rotation-Topdown"' in source
    assert "inhand_rotation_topdown_env_cfg:" in source
    assert "LinkerL20InhandRotationTopdownEnvCfg" in source
    assert 'id="Isaac-LinkerL20-Inhand-GraspGen-Topdown"' in source
    assert "LinkerL20InhandRotationTopdownGraspGenEnvCfg" in source


def test_topdown_reuses_base_goal_and_requires_validated_caches(cfg_tree):
    training_class = next(
        node
        for node in cfg_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LinkerL20InhandRotationTopdownEnvCfg"
    )
    assert [base.id for base in training_class.bases if isinstance(base, ast.Name)] == [
        "LinkerL20InhandRotationEnvCfg"
    ]
    source = _CFG.read_text()
    assert 'grasp_cache_name: str = "linker_l20_topdown"' in source
    assert "require_complete_grasp_cache: bool = True" in source


def test_cache_tool_can_select_the_topdown_generator():
    source = _GENERATOR.read_text()
    assert '"Isaac-LinkerL20-Inhand-GraspGen-Topdown"' in source
    assert 'task = args.task' in source
    assert 'getattr(env_cfg, "configure_cache_shape", None)' in source
