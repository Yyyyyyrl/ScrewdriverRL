"""CPU/pure-Python contract tests for the 64 mm Linker top-down task."""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import pytest
import yaml

import screwdriver_rl.tasks  # noqa: F401 - populates Gym's registry
from screwdriver_rl.tasks.linker_l20.screwdriver_rotation_topdown_posture import (
    TOPDOWN_HANDLE_RADIUS_M,
    TOPDOWN_JOINT_POSITIONS,
    TOPDOWN_PREGRASP_POSITIONS,
    TOPDOWN_RESET_JOINT_POSITIONS,
    TOPDOWN_RESET_POSITIONS,
    TOPDOWN_ROOT_POS_W,
    TOPDOWN_ROOT_QUAT_WXYZ,
    TOPDOWN_TARGET_JOINT_POSITIONS,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ASSET = ROOT / "assets/screwdriver/screwdriver_isaaclab.urdf"
TOPDOWN_ASSET = ROOT / "assets/screwdriver/screwdriver_64mm_handle.urdf"
HAND_ASSET = ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
BASE_CFG = ROOT / "screwdriver_rl/tasks/linker_l20/screwdriver_rotation_env_cfg.py"
TOPDOWN_CFG = ROOT / "screwdriver_rl/tasks/linker_l20/screwdriver_rotation_topdown_env_cfg.py"
TOPDOWN_DIAMETER_CFG = ROOT / "screwdriver_rl/tasks/linker_l20/screwdriver_rotation_topdown_diameter_env_cfg.py"
BASE_ENV = ROOT / "screwdriver_rl/tasks/base/screwdriver_rotation_env.py"
SHARED_BASE_CFG = ROOT / "screwdriver_rl/tasks/base/screwdriver_rotation_env_cfg.py"
LINKER_ENV = ROOT / "screwdriver_rl/tasks/linker_l20/screwdriver_rotation_env.py"
EVAL_SCRIPT = ROOT / "eval.py"
TRAIN_SCRIPT = ROOT / "train.py"
PPO_CFG = ROOT / "screwdriver_rl/tasks/linker_l20/agents/rl_games_ppo_cfg.yaml"

TASK_ID = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
BASE_TASK_ID = "Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0"
ORIGINAL_ASSET_SHA256 = "aa6b2a38f82845e7c0665d2d480114dead56f4457930f90fc765a3d937b6a949"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_link(root: ET.Element, name: str) -> ET.Element:
    return next(link for link in root.findall("link") if link.get("name") == name)


def _radii(root: ET.Element, name: str) -> tuple[float, float]:
    link = _find_link(root, name)
    return tuple(
        float(link.find(f"{role}/geometry/cylinder").get("radius"))
        for role in ("visual", "collision")
    )


def _signature(element: ET.Element) -> tuple:
    """XML structure/attributes, intentionally ignoring formatting and comments."""

    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        tuple(_signature(child) for child in element),
    )


def _normalised_screwdriver_signature(path: Path) -> tuple:
    root = ET.parse(path).getroot()
    root.set("name", "canonical")
    for link_name in ("screwdriver_body", "screwdriver_cap"):
        link = _find_link(root, link_name)
        for role in ("visual", "collision"):
            link.find(f"{role}/geometry/cylinder").set("radius", "canonical")
    return _signature(root)


def _class_node(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(), filename=str(path))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assigned_names(class_node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in class_node.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text()
    class_node = _class_node(path, class_name)
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def _rotate_wxyz(quat, vector) -> tuple[float, float, float]:
    w, x, y, z = quat
    vx, vy, vz = vector
    # q * (0,v) * q^-1, expanded to avoid a numpy/scipy test dependency.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def test_exact_task_registration_and_shared_runtime_entries():
    spec = gym.spec(TASK_ID)
    base = gym.spec(BASE_TASK_ID)

    assert spec.id == TASK_ID
    assert "Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Direct-v0" not in gym.registry
    assert spec.entry_point == base.entry_point
    assert spec.kwargs["rl_games_cfg_entry_point"] == base.kwargs["rl_games_cfg_entry_point"]
    assert spec.kwargs["env_cfg_entry_point"].endswith(
        "screwdriver_rotation_topdown_diameter_env_cfg:"
        "LinkerL20ScrewdriverRotationTopdownDiameterRandEnvCfg"
    )


def test_new_asset_path_is_task_specific():
    source = TOPDOWN_CFG.read_text()
    baseline = BASE_CFG.read_text()
    assert "screwdriver/screwdriver_64mm_handle.urdf" in source
    assert "screwdriver/screwdriver_64mm_handle.urdf" not in baseline
    assert "screwdriver/screwdriver_isaaclab.urdf" in baseline
    assert TOPDOWN_ASSET.is_file()


def test_body_and_cap_visual_collision_diameter_is_64_mm():
    original = ET.parse(ORIGINAL_ASSET).getroot()
    topdown = ET.parse(TOPDOWN_ASSET).getroot()
    assert TOPDOWN_HANDLE_RADIUS_M == pytest.approx(0.032, abs=1.0e-12)
    for link_name in ("screwdriver_body", "screwdriver_cap"):
        assert _radii(topdown, link_name) == pytest.approx((0.032, 0.032), abs=1.0e-12)
        assert _radii(original, link_name) == pytest.approx((0.020, 0.020), abs=1.0e-12)


def test_original_asset_unchanged_and_all_other_xml_values_match():
    assert _sha256(ORIGINAL_ASSET) == ORIGINAL_ASSET_SHA256
    assert _normalised_screwdriver_signature(TOPDOWN_ASSET) == (
        _normalised_screwdriver_signature(ORIGINAL_ASSET)
    )


def test_palm_normal_is_world_minus_z_and_hand_root_is_above_handle():
    norm = math.sqrt(sum(value * value for value in TOPDOWN_ROOT_QUAT_WXYZ))
    assert norm == pytest.approx(1.0, abs=1.0e-9)
    # Linker L20's local palm normal is +X.
    palm_normal = _rotate_wxyz(TOPDOWN_ROOT_QUAT_WXYZ, (1.0, 0.0, 0.0))
    assert palm_normal == pytest.approx((0.0, 0.0, -1.0), abs=1.0e-9)
    handle_top_z = 1.205 + 0.100 + 0.100 + 0.001
    assert TOPDOWN_ROOT_POS_W[2] > handle_top_z


def test_posture_joint_limits_and_mimics_have_margin():
    root = ET.parse(HAND_ASSET).getroot()
    joints = {
        joint.get("name"): joint
        for joint in root.findall("joint")
        if joint.get("type") == "revolute"
    }
    assert TOPDOWN_JOINT_POSITIONS is TOPDOWN_RESET_JOINT_POSITIONS
    independent = {name for name, joint in joints.items() if joint.find("mimic") is None}
    assert len(independent) == 16

    def check_full_posture(posture: dict[str, float], required_margin: float) -> None:
        assert set(posture) == set(joints)
        minimum_margin = math.inf
        for name, joint in joints.items():
            value = posture[name]
            limit = joint.find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            minimum_margin = min(minimum_margin, value - lower, upper - value)
            mimic = joint.find("mimic")
            if mimic is not None:
                expected = (
                    posture[mimic.get("joint")] * float(mimic.get("multiplier", "1"))
                    + float(mimic.get("offset", "0"))
                )
                assert value == pytest.approx(expected, abs=1.0e-9)
        assert minimum_margin >= required_margin

    def flatten(per_finger: dict[str, tuple[float, ...]]) -> dict[str, float]:
        result = {
            name: value
            for finger in ("index", "middle", "ring", "pinky")
            for name, value in zip(
                (f"{finger}_mcp_roll", f"{finger}_mcp_pitch", f"{finger}_pip"),
                per_finger[finger],
            )
        }
        result.update(
            zip(
                ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
                per_finger["thumb"],
            )
        )
        return result

    check_full_posture(TOPDOWN_RESET_JOINT_POSITIONS, 0.109999)
    check_full_posture(TOPDOWN_TARGET_JOINT_POSITIONS, 0.104999)
    assert flatten(TOPDOWN_RESET_POSITIONS) == pytest.approx(
        {name: TOPDOWN_RESET_JOINT_POSITIONS[name] for name in independent}, abs=1.0e-12
    )
    assert flatten(TOPDOWN_PREGRASP_POSITIONS) == pytest.approx(
        {name: TOPDOWN_TARGET_JOINT_POSITIONS[name] for name in independent}, abs=1.0e-12
    )


def test_topdown_cfg_inherits_baseline_observation_action_reward_and_curriculum():
    cls = _class_node(TOPDOWN_CFG, "LinkerL20ScrewdriverRotationTopdownEnvCfg")
    assert [ast.unparse(base) for base in cls.bases] == ["LinkerL20ScrewdriverRotationEnvCfg"]

    # No shadow copies of baseline policy/task contracts are allowed.  The
    # post-init changes only geometry-derived values and the reset posture.
    assert _assigned_names(cls) == {"screwdriver_handle_radius"}
    forbidden = {
        "observation_space",
        "action_space",
        "curriculum_phases",
        "domain_rand",
        "action_delta",
        "action_delta_scale",
        "contact_f_min",
        "contact_f_lo",
        "contact_f_hi",
        "contact_f_max",
        "screwdriver_load_torque",
        "joint_motion_range",
    }
    assert not (_assigned_names(cls) & forbidden)

    post_init = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    calls_parent = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__post_init__"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        for node in ast.walk(post_init)
    )
    assert calls_parent

    source = TOPDOWN_CFG.read_text()
    assert "self.drive_full_tangential_speed = TOPDOWN_HANDLE_RADIUS_M" in source
    assert "self.screwdriver_cfg.spawn.asset_path" in source
    assert "self.robot_cfg.init_state.pos" in source
    assert "self.robot_cfg.init_state.rot" in source
    assert "self.robot_cfg.init_state.joint_pos" in source
    assert "self.reset_joint_positions" in source
    assert "self.pregrasp_positions" in source


def test_topdown_uses_role_neutral_contact_without_changing_lateral_default():
    topdown_source = TOPDOWN_CFG.read_text()
    base_cfg_source = BASE_CFG.read_text()
    reward_source = _method_source(
        LINKER_ENV, "LinkerL20ScrewdriverRotationEnv", "_get_rewards"
    )
    privileged_source = _method_source(
        LINKER_ENV,
        "LinkerL20ScrewdriverRotationEnv",
        "_compute_privileged_obs",
    )

    assert "self.role_neutral_fingertip_contact = True" in topdown_source
    assert "self.role_neutral_min_contact_fingers = 3" in topdown_source
    assert (
        "TOPDOWN_WRONG_SURFACE_WEIGHTS: tuple[float, float, float] = "
        "(30.0, 60.0, 90.0)"
    ) in topdown_source
    assert "phase.w_wrong = weight" in topdown_source
    assert "role_neutral_fingertip_contact: bool = False" in base_cfg_source
    assert "if cfg.role_neutral_fingertip_contact:" in reward_source
    assert "drive_score = distance_score" in reward_source
    assert "index_cap_reward = torch.zeros_like(turn_reward)" in reward_source
    assert "contact_d_margin: float = 0.008" in base_cfg_source
    assert "contact_d_far_margin: float = 0.020" in base_cfg_source
    for margin in ("0.0070", "0.0060", "0.0095", "0.0290"):
        assert margin in topdown_source

    # Force amplitudes are diagnostic-only: no force-window, excess-force, or
    # force variable may shape reward/progress, and privileged state is purely
    # kinematic. Wrong-surface retains only a binary sensor predicate.
    assert "_compute_distance_contact()" in reward_source
    assert "_read_contact_forces()" in reward_source
    assert "diagnostic_tip_force" in reward_source
    assert "force_window" not in reward_source
    assert "excess_force" not in reward_source
    for name in ("F_total", "F_body", "F_cap"):
        assert name not in reward_source
    assert "wrong_force > cfg.wrong_surface_force_threshold" in reward_source
    assert "rewards.target_penetration(" in reward_source
    assert "torch.relu(" in reward_source
    assert "drive_count" in reward_source
    assert "_read_contact_forces" not in privileged_source
    assert "distance_score" in privileged_source


def test_shared_surface_clearance_has_radius_override_without_changing_baseline_default():
    source = BASE_ENV.read_text()
    assert 'getattr(self.cfg, "screwdriver_handle_radius", BASE_RADIUS)' in source
    assert "self._env_geom_scale[:, 0] * base_radius" in source
    assert "if self._reset_joint_pos is None:" in source
    assert "target_jpos[:, jids]" in source


def test_force_free_final_observation_contract_and_deployability_dr():
    base_cfg_source = BASE_CFG.read_text()
    shared_cfg_source = SHARED_BASE_CFG.read_text()
    topdown_source = TOPDOWN_CFG.read_text()
    diameter_source = TOPDOWN_DIAMETER_CFG.read_text()

    assert "privileged_obs_dim: int = 20" in base_cfg_source
    assert "shape=(52,)" in base_cfg_source
    assert 'observation_semantics_version: str = "linker-l20-force-free-reset-dr-v1"' in (
        base_cfg_source
    )
    assert "self.privileged_obs_dim += 2" in diameter_source
    assert "obs_dim = self.history_obs_dim + self.privileged_obs_dim" in diameter_source

    for assignment in (
        "self.domain_rand.randomize_contact_friction = True",
        "self.domain_rand.contact_friction_range = (0.6, 2.5)",
        "self.domain_rand.rotation_damping_range = (0.5, 3.0)",
        "self.domain_rand.randomize_tilt_damping = True",
        "self.domain_rand.tilt_damping_range = (0.5, 2.0)",
        "self.domain_rand.screwdriver_load_torque_range = (0.5, 13.1)",
        "self.domain_rand.reset_root_pos_noise_m = 0.008",
        "self.domain_rand.reset_root_z_noise_m = 0.002",
        "self.domain_rand.reset_root_tilt_noise_rad = 0.025",
        "self.domain_rand.reset_root_yaw_noise_rad = 0.09",
        "self.domain_rand.reset_screwdriver_tilt_noise_rad = 0.03",
        "self.domain_rand.joint_zero_bias_rad = 0.015",
        "self.reset_root_pose_ramp = True",
        "self.reset_contact_guard_min_fingers = 3",
        "self.reset_contact_guard_max_resamples = 64",
        "self.reset_action_hold_steps = 0",
        "self.reset_action_ramp_steps = 3",
        "self.home_deviation_deadband = 0.15",
    ):
        assert assignment in topdown_source
    for declaration in (
        "reset_root_pos_noise_m: float = 0.0",
        "reset_root_z_noise_m: float = 0.0",
        "reset_root_tilt_noise_rad: float = 0.0",
        "reset_root_yaw_noise_rad: float = 0.0",
        "reset_screwdriver_tilt_noise_rad: float = 0.0",
        "joint_zero_bias_rad: float = 0.0",
    ):
        assert declaration in shared_cfg_source
    for declaration in (
        "reset_root_pose_ramp: bool = False",
        "reset_contact_guard_min_fingers: int = 0",
        "reset_contact_guard_max_resamples: int = 0",
    ):
        assert declaration in shared_cfg_source

    reset_source = _method_source(
        LINKER_ENV,
        "LinkerL20ScrewdriverRotationEnv",
        "_enforce_initial_contact_guard",
    )
    assert "self._compute_distance_contact()" in reset_source
    assert "self._restore_contact_states_from_donors" in reset_source
    assert "super()._reset_idx(unresolved)" in reset_source
    assert "raise RuntimeError" in reset_source

    privileged_source = _method_source(
        LINKER_ENV,
        "LinkerL20ScrewdriverRotationEnv",
        "_compute_privileged_obs",
    )
    assert "load_proxy" in privileged_source
    assert "contact_friction" in privileged_source



def test_final_phase_fall_penalty_matches_done_predicate_and_prices_one_drop_at_minus_150():
    source = BASE_CFG.read_text()
    assert "reward_fall_weight=5_000.0" in source
    assert "reward_fall_weight=15_000.0" in source
    assert "reward_fall_weight=30_000.0" in source
    fall_weight = 30_000.0
    reward_scale = float(
        yaml.safe_load(PPO_CFG.read_text())["params"]["config"]["reward_shaper"][
            "scale_value"
        ]
    )
    assert fall_weight * reward_scale == pytest.approx(150.0)

    reward_source = LINKER_ENV.read_text()
    done_source = BASE_ENV.read_text()
    assert "phase.reward_fall_weight" in reward_source
    assert "tilt_norm > phase.upright_termination_threshold" in reward_source
    assert "- fall_cost" in reward_source
    assert (
        "threshold = float(self._curriculum_phase.upright_termination_threshold)"
        in done_source
    )
    assert "terminated = tilt_norm > threshold" in done_source


def test_contact_authority_shaping_does_not_relax_turn_progress_gate():
    cfg_source = BASE_CFG.read_text()
    reward_source = LINKER_ENV.read_text()

    # Every phase keeps a positive sustained-contact incentive, strongest while
    # the policy is learning the initial role assignment.
    assert "w_contact_authority=8.0" in cfg_source
    assert "w_contact_authority=5.0" in cfg_source
    assert "w_contact_authority=3.0" in cfg_source

    # Shaping uses the same hard+sustained+upright gate, but progress accounting
    # remains binary-authorized and cannot be reopened by a soft count.
    assert (
        "contact_authority_reward = phase.w_contact_authority * combined_gate"
        in reward_source
    )
    assert "+ contact_authority_reward" in reward_source
    assert "qualified_delta_z = delta_z * sustained_gate" in reward_source
    assert "self._net_turn += qualified_delta_z.detach()" in reward_source


def test_topdown_reverse_motion_cannot_hide_behind_contact_dropout():
    cfg_source = TOPDOWN_CFG.read_text()
    reward_source = LINKER_ENV.read_text()

    assert "self.penalize_reverse_outside_contact = True" in cfg_source
    assert "self.match_reverse_weight_to_turn_weight = True" in cfg_source
    assert "self.reward_physical_progress_outside_contact = True" in cfg_source
    assert "reverse_gate = (" in reward_source
    assert "upright_gate" in reward_source
    assert "phase.reward_turn_weight" in reward_source
    assert "reverse_cost = reverse_weight * rev_signal * reverse_gate" in reward_source


def test_topdown_reward_has_no_fixed_speed_target():
    cfg_source = TOPDOWN_CFG.read_text()
    base_cfg_source = BASE_CFG.read_text()
    reward_source = LINKER_ENV.read_text()
    train_source = TRAIN_SCRIPT.read_text()

    assert "turn_reward_speed_cap" not in base_cfg_source
    assert "turn_reward_speed_cap" not in cfg_source
    assert "turn_reward_gate = (" in reward_source
    assert "reward_physical_progress_outside_contact" in reward_source
    assert "fwd_signal = fwd_vel.pow(speed_power)" in reward_source
    assert "rev_signal = rev_vel.pow(speed_power)" in reward_source
    assert (
        "turn_reward = phase.reward_turn_weight * fwd_signal * turn_reward_gate"
        in reward_source
    )
    assert '"--phase_turn_weights"' in train_source
    assert chr(34) + "--phase_load_scales" + chr(34) in train_source
    assert '"--ppo_sigma_override"' in train_source
    assert '"sigma": args.ppo_sigma_override' in train_source
    assert chr(34) + "--target_bound_weight" + chr(34) in train_source
    assert chr(34) + "--reverse_to_turn_ratio" + chr(34) in train_source
    assert chr(34) + "--turn_reward_power" + chr(34) in train_source
    assert chr(34) + "--joint_motion_range" + chr(34) in train_source
    assert chr(34) + "--joint_motion_range" + chr(34) in EVAL_SCRIPT.read_text()
    assert chr(34) + "--action_delta_scale" + chr(34) in train_source
    assert chr(34) + "--action_delta_scale" + chr(34) in EVAL_SCRIPT.read_text()
    assert chr(34) + "--absolute_action_targets" + chr(34) in train_source
    assert chr(34) + "--absolute_action_targets" + chr(34) in EVAL_SCRIPT.read_text()
    assert "absolute_action_targets: bool = False" in base_cfg_source
    assert "self._home_targets + self._joint_range * self.actions" in reward_source
    assert chr(34) + "--screwdriver_load_scale" + chr(34) in EVAL_SCRIPT.read_text()
    assert chr(34) + "--topdown_posture_search" + chr(34) in train_source
    assert chr(34) + "--topdown_posture_search" + chr(34) in EVAL_SCRIPT.read_text()
    assert chr(34) + "--phase_excess_weights" + chr(34) in train_source
    assert chr(34) + "--phase_wrong_weights" + chr(34) in train_source
    assert "target_edge_fraction" in reward_source
    assert "cfg.w_target_bound" in reward_source


def test_three_turn_success_uses_physical_not_contact_masked_progress():
    source = EVAL_SCRIPT.read_text()

    assert 'physical_net = _safe(ex, "eval_raw_net_turns")' in source
    assert 'net = torch.cat(ep_net_turns).float()' in source
    assert 'success = upright & (net >= args.success_turns)' in source
    assert '"authorized_net_turns_mean"' in source


def test_creep_exploit_fixes_are_wired():
    """Lock the 2026-07-28 frozen-finger/creep-exploit fixes (plan doc §7).

    Guards the three structural fixes: co-motion authorization on the turn
    reward and qualified progress, the zero-tension target snap at settle end,
    and the recalibrated squeeze-proxy scale.  Source-level assertions follow
    this file's convention (instantiating the cfg needs Isaac Sim).
    """
    topdown_source = TOPDOWN_CFG.read_text()
    base_cfg_source = SHARED_BASE_CFG.read_text()
    linker_cfg_source = BASE_CFG.read_text()
    base_env_source = BASE_ENV.read_text()
    reward_source = LINKER_ENV.read_text()
    eval_source = EVAL_SCRIPT.read_text()

    # Top-down opts into both fixes.
    assert "self.turn_motion_authorized = True" in topdown_source
    assert "self.reset_zero_tension_targets = True" in topdown_source

    # Defaults stay off so lateral/other tasks are unchanged.
    assert "turn_motion_authorized: bool = False" in linker_cfg_source
    assert "reset_zero_tension_targets: bool = False" in base_cfg_source

    # Turn reward, milestone and checkpoint-selection progress all pass
    # through the motion authorization.
    assert "rewards.surface_co_motion" in reward_source
    assert ") * motion_auth" in reward_source
    assert "delta_z * sustained_gate * motion_auth" in reward_source
    assert "gate=combined_gate * motion_auth" in reward_source

    # Zero-tension snap runs at the end of the compliant settle.
    assert "_snap_targets_to_settled_state" in base_env_source
    # Joint limits are per-env tensors; asynchronous resets must select the
    # same rows as the settled joint batch instead of broadcasting all envs.
    assert "self._finger_lower[env_ids]" in base_env_source
    assert "self._finger_upper[env_ids]" in base_env_source

    # Squeeze proxy priced for exploration (100 blocked the gradient path).
    assert "target_penetration_scale: float = 10.0" in linker_cfg_source

    # The physics-artifact probe exists for M2/M4 acceptance.
    assert chr(34) + "--zero_action" + chr(34) in eval_source
    assert '"zero_action_probe"' in eval_source
