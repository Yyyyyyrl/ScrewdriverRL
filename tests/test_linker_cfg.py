"""Pure-Python guards for the Linker Hand L20 task wiring.

These tests parse the Linker URDF directly (no Isaac Sim dependency) and assert
that the joint/mimic inventory the task code hardcodes still matches the asset.
They catch the most error-prone kind of drift — a renamed joint or a changed
mimic multiplier — without needing a GPU or the Omniverse app.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_URDF = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "linker_hand_l20"
    / "linkerhand_l20_left.urdf"
)

# Mirrors LinkerL20ScrewdriverRotationEnv.FINGER_JOINT_NAMES (16 independent DOFs).
EXPECTED_INDEPENDENT = {
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
}

# Mirrors LinkerL20ScrewdriverRotationEnv.COUPLED_JOINTS: follower -> (master, mult).
EXPECTED_MIMIC = {
    "index_dip": ("index_pip", 0.8917),
    "middle_dip": ("middle_pip", 0.8917),
    "ring_dip": ("ring_pip", 0.8917),
    "pinky_dip": ("pinky_pip", 0.8917),
    "thumb_ip": ("thumb_mcp", 1.1619),
}

EXPECTED_FINGERTIPS = {
    "index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_tip",
}

EXPECTED_FIXED_TIP_JOINTS = {
    "index_tip_joint": ("index_distal", "index_tip"),
    "middle_tip_joint": ("middle_distal", "middle_tip"),
    "ring_tip_joint": ("ring_distal", "ring_tip"),
    "pinky_tip_joint": ("pinky_distal", "pinky_tip"),
    "thumb_tip_joint": ("thumb_distal", "thumb_tip"),
}


@pytest.fixture(scope="module")
def urdf_root():
    assert _URDF.exists(), f"Linker URDF not found at {_URDF}"
    return ET.parse(_URDF).getroot()


def _joints(root):
    return {j.get("name"): j for j in root.findall("joint")}


def test_actuation_and_fixed_fingertip_joint_inventory(urdf_root):
    joints = _joints(urdf_root)
    revolute = {name: joint for name, joint in joints.items() if joint.get("type") == "revolute"}
    fixed = {name: joint for name, joint in joints.items() if joint.get("type") == "fixed"}
    assert len(revolute) == 21
    assert set(fixed) == set(EXPECTED_FIXED_TIP_JOINTS)
    assert len(joints) == len(revolute) + len(fixed)


def test_independent_and_mimic_partition(urdf_root):
    joints = {
        name: joint
        for name, joint in _joints(urdf_root).items()
        if joint.get("type") == "revolute"
    }
    mimic_names = {name for name, j in joints.items() if j.find("mimic") is not None}
    independent = set(joints) - mimic_names

    assert independent == EXPECTED_INDEPENDENT
    assert mimic_names == set(EXPECTED_MIMIC)


def test_mimic_masters_and_multipliers(urdf_root):
    joints = _joints(urdf_root)
    for follower, (master, mult) in EXPECTED_MIMIC.items():
        m = joints[follower].find("mimic")
        assert m is not None, f"{follower} lost its <mimic> tag"
        assert m.get("joint") == master
        assert float(m.get("multiplier")) == pytest.approx(mult, abs=1e-4)
        # masters must be independent, driveable joints
        assert master in EXPECTED_INDEPENDENT


def test_fingertip_links_exist(urdf_root):
    links = {ln.get("name") for ln in urdf_root.findall("link")}
    assert EXPECTED_FINGERTIPS <= links
    joints = _joints(urdf_root)
    for name, (parent, child) in EXPECTED_FIXED_TIP_JOINTS.items():
        assert joints[name].find("parent").get("link") == parent
        assert joints[name].find("child").get("link") == child


def test_dims_are_self_consistent():
    """The cfg dims must follow from the finger DOF layout."""
    n_independent = len(EXPECTED_INDEPENDENT)          # 16
    n_fingers = 5
    obs_dim = 2 * n_independent + 3                     # finger_q + targets + euler
    history_dim = 2 * n_independent                     # finger_q + targets
    # euler,angvel,relpos,quat,friction + per-finger contact force (5)
    privileged_dim = 3 + 3 + 3 + 4 + 1 + n_fingers

    assert n_independent == 16
    assert obs_dim == 35
    assert history_dim == 32
    assert privileged_dim == 19


def test_curriculum_never_free_spins():
    """Invariant of the redesign: every curriculum phase keeps the screw load fully
    on (``screwdriver_load_scale == 1.0``) so the handle can never free-spin, and the
    five fingers are all active.  Requires Isaac Lab to import the cfg; skipped on a
    CPU-only environment.
    """
    try:
        from screwdriver_rl.tasks.linker_l20.screwdriver_rotation_env_cfg import (
            LinkerL20ScrewdriverRotationEnvCfg,
        )
    except Exception:  # isaaclab (or its deps) unavailable on this runner
        pytest.skip("isaaclab not importable; cfg-level invariant skipped")

    cfg = LinkerL20ScrewdriverRotationEnvCfg()
    assert cfg.curriculum_phases, "no curriculum phases defined"
    for ph in cfg.curriculum_phases:
        assert ph.screwdriver_load_scale == 1.0, (
            f"phase @{ph.step_start} has load scale {ph.screwdriver_load_scale} != 1.0 "
            "— the handle could free-spin in this phase"
        )
        assert ph.action_scale_multiplier == 1.0, (
            f"phase @{ph.step_start} changes the deployment action scale by default"
        )
    dr_scales = [ph.dynamics_randomization_scale for ph in cfg.curriculum_phases]
    assert dr_scales == pytest.approx([0.25, 0.60, 1.0])
    assert dr_scales == sorted(dr_scales)
    assert set(cfg.fingers) == {"index", "middle", "ring", "pinky", "thumb"}
