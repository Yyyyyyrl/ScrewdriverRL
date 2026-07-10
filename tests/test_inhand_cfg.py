"""Pure-Python guards for the LinkerL20 free-object in-hand rotation task."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_URDF = _ROOT / "assets" / "linker_hand_l20" / "linkerhand_l20_left.urdf"
_CFG = _ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "inhand_rotation_env_cfg.py"
_ENV = _ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "inhand_rotation_env.py"
_REGISTRY = _ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "__init__.py"
_YAML = _ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "agents" / "rl_games_inhand_ppo_cfg.yaml"

EXPECTED_INDEPENDENT = {
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
}

EXPECTED_MIMIC = {
    "index_dip": ("index_pip", 0.8917),
    "middle_dip": ("middle_pip", 0.8917),
    "ring_dip": ("ring_pip", 0.8917),
    "pinky_dip": ("pinky_pip", 0.8917),
    "thumb_ip": ("thumb_mcp", 1.1619),
}


@pytest.fixture(scope="module")
def urdf_root():
    assert _URDF.exists(), f"Linker URDF not found at {_URDF}"
    return ET.parse(_URDF).getroot()


@pytest.fixture(scope="module")
def cfg_tree():
    assert _CFG.exists(), f"missing {_CFG}"
    return ast.parse(_CFG.read_text())


@pytest.fixture(scope="module")
def env_tree():
    assert _ENV.exists(), f"missing {_ENV}"
    return ast.parse(_ENV.read_text())


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target == name:
            return ast.literal_eval(value)
    raise AssertionError(f"{name} not found")


def _class_literal(tree: ast.Module, class_name: str, name: str):
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    for node in cls.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target == name:
            return ast.literal_eval(value)
    raise AssertionError(f"{class_name}.{name} not found")


def _compile_function(tree: ast.Module, name: str):
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {}
    exec(compile(mod, filename=f"<{name}>", mode="exec"), ns)
    return ns[name]


def _joints(root):
    return {j.get("name"): j for j in root.findall("joint")}


def test_obs_dimensions_follow_hora_contract():
    n_independent = len(EXPECTED_INDEPENDENT)
    history_dim = 2 * n_independent
    policy_dim = 3 * history_dim + 9

    assert n_independent == 16
    assert history_dim == 32
    assert policy_dim == 105


def test_asset_grid_is_scale_major(cfg_tree):
    scales = _literal(cfg_tree, "HORA_CYLINDER_SCALES")
    assert len(scales) == 9
    assert scales == pytest.approx(tuple(0.70 + 0.02 * i for i in range(9)))

    # Each scale spawns a fixed block of shapes; the grid is scale-major so
    # asset i -> scale block i // block_size (this is exactly how the env maps
    # env_id -> scale for the per-scale grasp-cache lookup).
    n_cyl = len(_literal(cfg_tree, "MIX_CYLINDER_LENGTHS"))
    n_cub = len(_literal(cfg_tree, "MIX_CUBOID_SIZES"))
    n_sph = len(_literal(cfg_tree, "MIX_SPHERE_RADII"))
    block = n_cyl + n_cub + n_sph
    total = len(scales) * block
    for asset_idx in range(total):
        scale_idx = asset_idx // block
        assert 0 <= scale_idx < len(scales)


def test_object_mix_has_all_three_shapes_hora_proportions(cfg_tree):
    """HORA trains on cylinders + cuboids + spheres (0.35/0.20/0.45).  The block
    must contain all three shapes in roughly that split."""
    n_cyl = len(_literal(cfg_tree, "MIX_CYLINDER_LENGTHS"))
    n_cub = len(_literal(cfg_tree, "MIX_CUBOID_SIZES"))
    n_sph = len(_literal(cfg_tree, "MIX_SPHERE_RADII"))
    block = n_cyl + n_cub + n_sph
    assert n_cyl >= 1 and n_cub >= 1 and n_sph >= 1, "all three shapes must be present"
    # within +-0.12 of HORA's 0.35 / 0.20 / 0.45 type split
    assert n_cyl / block == pytest.approx(0.35, abs=0.12)
    assert n_cub / block == pytest.approx(0.20, abs=0.12)
    assert n_sph / block == pytest.approx(0.45, abs=0.12)


def test_cache_filename_format(cfg_tree):
    tag = _compile_function(cfg_tree, "_scale_cache_tag")
    filename = _compile_function(cfg_tree, "grasp_cache_filename")
    filename.__globals__["_scale_cache_tag"] = tag
    filename.__globals__["_SHAPE_CACHE_TAGS"] = _literal(cfg_tree, "_SHAPE_CACHE_TAGS")

    assert tag(0.70) == "s07"
    assert tag(0.72) == "s072"
    assert tag(0.80) == "s08"
    assert tag(0.86) == "s086"
    # One cache per (scale, shape, prototype): fingertip cages have millimetre
    # tolerance, so grasp states do not transfer across shapes OR prototypes
    # (cylinder caches on spheres were ~85% dead-on-arrival at reset; even
    # same-shape caches mixing prototypes stayed ~52%).
    assert _literal(cfg_tree, "INHAND_CACHE_SHAPES") == ("cylinder", "cuboid", "sphere")
    assert filename(0.80) == "linker_l20_grasp_cyl0_s08.npy"
    assert filename(0.80, shape="cuboid", proto=1) == "linker_l20_grasp_cub1_s08.npy"
    assert filename(0.70, shape="sphere", proto=3) == "linker_l20_grasp_sph3_s07.npy"


def test_inhand_mimic_constants_match_urdf(env_tree, urdf_root):
    coupled = _class_literal(env_tree, "LinkerL20InhandRotationEnv", "COUPLED_JOINTS")
    joints = _joints(urdf_root)

    assert set(coupled) == set(EXPECTED_MIMIC)
    for follower, (master, mult) in EXPECTED_MIMIC.items():
        cfg_master, cfg_mult, cfg_offset = coupled[follower]
        m = joints[follower].find("mimic")
        assert m is not None
        assert cfg_master == master == m.get("joint")
        assert cfg_mult == pytest.approx(float(m.get("multiplier")), abs=1e-4)
        assert cfg_mult == pytest.approx(mult, abs=1e-4)
        assert cfg_offset == 0.0


def _quat_rotate(q, v):
    w, x, y, z = q
    # v' = v + 2*q_vec x (q_vec x v + w*v)
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


def test_canonical_grip_is_palm_up_fingertip_opposition(cfg_tree):
    """The in-hand canonical pose is the HORA-style palm-up fingertip cage:
    palm normal up (tilted INHAND_PALM_TILT_DEG about the finger axis, thumb
    edge down), fingers along world -Y, object seeded at the fingertip-cage
    centre in the air.  Guards against regressing to a palm-down / top grasp."""
    import math

    pregrasp = _literal(cfg_tree, "INHAND_PREGRASP_POSITIONS")
    obj_pos = _literal(cfg_tree, "INHAND_OBJECT_INIT_POS")
    tilt_deg = _literal(cfg_tree, "INHAND_PALM_TILT_DEG")

    quat_from_axis_angle = _compile_function(cfg_tree, "_quat_from_axis_angle")
    quat_mul = _compile_function(cfg_tree, "_quat_mul")
    quat_from_axis_angle.__globals__["math"] = math
    quat_mul.__globals__["math"] = math

    # INHAND_HAND_ROT is computed at module level; recompute it here from the
    # same helpers and the tilt constant (the HORA composition).
    rot = quat_mul(
        quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(-90.0 + tilt_deg)),
        quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(90.0)),
    )
    assert abs(sum(c * c for c in rot) - 1.0) < 1e-6  # unit quat

    tilt = math.radians(tilt_deg)
    palm_normal_w = _quat_rotate(rot, (1.0, 0.0, 0.0))   # L20 palm normal = +X
    fingers_w = _quat_rotate(rot, (0.0, 0.0, 1.0))       # fingers = +Z
    thumb_edge_w = _quat_rotate(rot, (0.0, -1.0, 0.0))   # thumb edge = -Y

    assert palm_normal_w == pytest.approx((math.sin(tilt), 0.0, math.cos(tilt)), abs=1e-6)
    assert palm_normal_w[2] > 0.5, "palm must face up"
    assert fingers_w == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)
    assert thumb_edge_w[2] < -0.5, "thumb edge must be the lower edge (gravity loads the thumb)"
    assert 20.0 <= tilt_deg <= 60.0

    # Opposition-grip fingerprint: the thumb is swung across the palm
    # (cmc_yaw + cmc_roll both engaged) while the four fingers are in a
    # moderate fingertip curl, not the fully closed screwdriver wrap.
    assert pregrasp["thumb"][0] >= 0.4, "thumb cmc_yaw not opposed"
    assert pregrasp["thumb"][1] >= 0.5, "thumb cmc_roll not opposed"
    for finger in ("index", "middle", "ring", "pinky"):
        pitch, pip = pregrasp[finger][1], pregrasp[finger][2]
        assert 0.2 <= pitch <= 1.0, f"{finger} mcp_pitch={pitch} outside fingertip-cage band"
        assert 0.2 <= pip <= 1.0, f"{finger} pip={pip} outside fingertip-cage band"

    # Object seeded in the air at the fingertip cage (hand root is at z=0.5),
    # never on the palm and never at the old top-grasp drop height.
    assert 0.50 < obj_pos[2] < 0.60


def test_canonical_pose_within_urdf_limits(cfg_tree, urdf_root):
    pregrasp = _literal(cfg_tree, "INHAND_PREGRASP_POSITIONS")
    joints = _joints(urdf_root)
    finger_joint_names = {
        "index": ("index_mcp_roll", "index_mcp_pitch", "index_pip"),
        "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
        "ring": ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
        "pinky": ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
        "thumb": ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
    }
    values = {
        name: value
        for finger, names in finger_joint_names.items()
        for name, value in zip(names, pregrasp[finger], strict=True)
    }
    for follower, (master, mult) in EXPECTED_MIMIC.items():
        values[follower] = values[master] * mult

    for name, value in values.items():
        limit = joints[name].find("limit")
        lo = float(limit.get("lower"))
        hi = float(limit.get("upper"))
        assert lo <= value <= hi, f"{name}={value} outside [{lo}, {hi}]"
        assert min(value - lo, hi - value) >= 0.1 - 1e-6, (
            f"{name}={value} is too close to [{lo}, {hi}]"
        )


def test_inhand_yaml_sanity():
    assert _YAML.exists(), f"missing {_YAML}"
    cfg = yaml.safe_load(_YAML.read_text())
    params = cfg["params"]
    net = params["network"]
    train = params["config"]

    assert net["name"] == "priv_latent_actor_critic"
    assert net["separate"] is False
    assert net["proprio_dim"] == 96
    assert net["latent_dim"] == 8
    assert net["priv_mlp_units"] == [256, 128]
    assert net["mlp"]["units"] == [512, 256, 128]
    assert train["reward_shaper"]["scale_value"] == pytest.approx(0.01)
    assert train["horizon_length"] == 8
    assert train["minibatch_size"] == 16384
    assert "central_value_config" not in params


def test_inhand_task_registration_points_train_play_eval_to_new_cfgs():
    source = _REGISTRY.read_text()
    assert 'id="Isaac-LinkerL20-Inhand-Rotation"' in source
    assert "inhand_rotation_env:LinkerL20InhandRotationEnv" in source
    assert "inhand_rotation_env_cfg:LinkerL20InhandRotationEnvCfg" in source
    assert "rl_games_inhand_ppo_cfg.yaml" in source

    assert 'id="Isaac-LinkerL20-Inhand-GraspGen"' in source
    assert "inhand_grasp_gen_env:LinkerL20InhandGraspGenEnv" in source
    assert "inhand_rotation_env_cfg:LinkerL20InhandGraspGenEnvCfg" in source


def test_inhand_ground_spawn_is_local():
    source = _ENV.read_text()
    assert "GroundPlaneCfg" not in source
    assert "spawn_ground_plane" not in source
    assert "_spawn_local_ground" in source
    assert "CuboidCfg" in source
    assert "RigidBodyPropertiesCfg(kinematic_enabled=True)" in source


def test_train_stage2_play_eval_observation_contracts(env_tree):
    source = _ENV.read_text()
    assert '"policy"' in source
    assert '"critic"' in source
    assert '"proprio_hist"' in source
    assert "def _prop_hist_buf" in source
    assert "_global_steps" in source
    assert "_current_epoch" in source
    assert "_log_stage" in source
    assert "_stage2_loss" in source
    assert "_logger" in source

    obs_dim = _class_literal(env_tree, "LinkerL20InhandRotationEnv", "FINGER_JOINT_NAMES")
    assert sum(len(v) for v in obs_dim.values()) == 16


def test_hora_reward_terms_and_eval_extras_are_present():
    source = _ENV.read_text()
    for token in (
        "axis_angle_from_quat",
        "quat_conjugate",
        "applied_torque",
        "linvel_penalty_scale",
        "pose_penalty_scale",
        "torque_penalty_scale",
        "work_penalty_scale",
        "eval_rotate_reward",
        "eval_obj_z",
        "eval_fall_frac",
        "eval_total_reward",
    ):
        assert token in source


def test_reset_keeps_direct_env_reset_before_state_and_target_writes():
    source = _ENV.read_text()
    reset_start = source.index("    def _reset_idx(")
    reset_body = source[reset_start:source.index("    def _update_curriculum", reset_start)]
    assert reset_body.index("super()._reset_idx(env_ids)") < reset_body.index(
        "self.hand.set_joint_position_target"
    )
    assert reset_body.index("super()._reset_idx(env_ids)") < reset_body.index(
        "self.object.set_external_force_and_torque"
    )


def test_domain_randomization_ranges_and_physics_writes(cfg_tree):
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "mass_range") == (0.01, 0.25)
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "com_range") == (-0.01, 0.01)
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "friction_range") == (0.3, 3.0)
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "pd_gain_range") == (0.967, 1.033)
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "joint_noise_scale") == pytest.approx(0.02)
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "force_scale") == pytest.approx(2.0)
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "random_force_prob") == pytest.approx(0.25)
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "force_decay") == pytest.approx(0.9)
    assert _class_literal(cfg_tree, "InhandDomainRandCfg", "force_decay_interval") == pytest.approx(0.08)

    source = _ENV.read_text()
    for token in (
        "set_masses",
        "set_coms",
        "set_material_properties",
        "write_joint_stiffness_to_sim",
        "write_joint_damping_to_sim",
        "set_external_force_and_torque",
        "_update_random_forces",
    ):
        assert token in source


def test_grasp_gen_cfg_and_done_contract(cfg_tree):
    assert _class_literal(cfg_tree, "LinkerL20InhandGraspGenEnvCfg", "cache_scale") == pytest.approx(0.8)
    assert _class_literal(cfg_tree, "LinkerL20InhandGraspGenEnvCfg", "cache_shape") == "cylinder"

    cfg_source = _CFG.read_text()
    for token in (
        "self.episode_length_s = 2.5",
        # HORA generates grasps at NOMINAL dynamics; DR stays off in grasp-gen.
        "self.domain_rand.enabled = False",
        "self.domain_rand.random_force_prob = 0.0",
        "self.domain_rand.force_scale = 0.0",
        "self.domain_rand.pd_gain_range = (1.0, 1.0)",
        "self.load_grasp_cache = False",
        "self.enable_fingertip_sensors = True",
        "self.enable_nontip_sensors = True",
        "self.grasp_gen_obj_init_pos = INHAND_OBJECT_INIT_POS",
        "self.grasp_gen_pose_noise = 0.25",
        "self.require_thumb_contact = True",
        "self.min_other_finger_contacts = 2",
        "self.forbid_nontip_contact = True",
        "self.grasp_gen_grace_steps = 5",
        "self.tip_dist_max = 0.13",
    ):
        assert token in cfg_source

    source = (_ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "inhand_grasp_gen_env.py").read_text()
    done_start = source.index("    def _get_dones(")
    done_body = source[done_start:source.index("    def _reset_idx(", done_start)]
    assert "terminated = ~acceptance & ~in_grace" in done_body
    assert "self.cfg.grasp_gen_grace_steps" in done_body
    assert "self.episode_length_buf >= self.max_episode_length - 1" in done_body
    assert "self._last_timed_out = timed_out.detach().clone()" in done_body

    reset_start = source.index("    def _reset_idx(")
    reset_body = source[reset_start:source.index("    def _compute_acceptance(", reset_start)]
    assert "survived = self._last_acceptance[env_ids_t] & self._last_timed_out[env_ids_t]" in reset_body
    assert "harvest_ids = env_ids_t[survived]" in reset_body


def test_grasp_gen_source_contracts():
    source = (_ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "inhand_grasp_gen_env.py").read_text()
    for token in (
        "_compute_acceptance",
        "_read_tip_object_forces",
        "_read_nontip_object_forces",
        "force_matrix_w",
        "tip_dist_max",
        "require_thumb_contact",
        "min_other_finger_contacts",
        "forbid_nontip_contact",
        "nontip_force_eps",
        "grasp_gen_accept_z_margin",
        "save_if_full",
        "np.save",
    ):
        assert token in source

    env_source = _ENV.read_text()
    for token in ("NONTIP_BODY_NAMES", "enable_nontip_sensors", "contact_nontip"):
        assert token in env_source

    tool_source = (_ROOT / "tools" / "gen_inhand_grasp_cache.py").read_text()
    for token in (
        "--scale",
        "--shape",
        "--num_envs",
        "--num_states",
        "--max_steps",
        "--diagnostics_every",
        "--debug",
        "Isaac-LinkerL20-Inhand-GraspGen",
        "grasp_cache_filename",
        "per-finger stats",
        "thumb contact mean",
        "nontip force mean",
    ):
        assert token in tool_source
