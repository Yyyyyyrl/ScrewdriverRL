"""Configuration for Linker G20/L20 64 mm free-cube in-hand rotation.

It intentionally does not subclass the mounted-screwdriver task config: the
object is a free rigid cube, resets come from versioned physics-harvested grasp
caches, and the observation contract supports the full two-stage RMA pipeline.
"""

from __future__ import annotations

import copy
import math
from dataclasses import field

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from screwdriver_rl.tasks.base.screwdriver_rotation_env_cfg import ASSET_ROOT


HORA_CYLINDER_RADIUS: float = 0.04
HORA_CYLINDER_LENGTHS: tuple[float, ...] = (
    0.064,
    0.068,
    0.072,
    0.076,
    0.080,
    0.084,
    0.088,
    0.092,
    0.096,
)
# Deployment target: a nominal 64 mm PLA cube.  The three geometry buckets
# mirror the mounted 60/64/68 mm campaign: broad enough to prevent a policy from
# keying on one exact contact geometry, while retaining an explicit nominal row
# for deployment and bench evaluation.  Every bucket requires its own
# physics-harvested cache; changing this tuple invalidates those caches.
INHAND_CUBE_EDGE_M: float = 0.064
INHAND_OBJECT_SCALES: tuple[float, ...] = (0.9375, 1.0, 1.0625)
INHAND_CACHE_CERTIFICATION: dict[str, object] = {
    "schema": "dex-forge-inhand-cache-replay-cert-v1",
    "generation_episode_s": 20.0,
    "replays_per_row": 4,
    "domain_rand_enabled": True,
    "support_threshold_n": 0.05,
    "require_no_fall": True,
}

# Historical name retained as an import-compatible alias.  New code and cache
# manifests use ``INHAND_OBJECT_SCALES`` because the production task is cube-only.
HORA_CYLINDER_SCALES: tuple[float, ...] = INHAND_OBJECT_SCALES

# Canonical pregrasp = palm-up fingertip cage with the palm longitudinal axis
# raised INHAND_LONG_AXIS_TILT_DEG above the ground plane (fingertips above
# the wrist, like a human presenting a cup on their fingertips).  The object
# is held in the air by the fingertip PADS only — four fingers curled with the
# flexion led by the MCP joints (no hook grasp: not MCP-extended/PIP-flexed;
# no flat fingers; no fist), thumb opposing with its tip pad from the side
# (not swept across the palm centre, no adduction, no excessive IP flexion) —
# and never rests on the palm.  This is only the operator-load search seed: the
# deployable startup/home posture is selected from the versioned physics cache.
# Each production row survives a released 20 s fingertip-only hold and then four
# independent full-domain-randomisation replays from zero velocity.  The grasp
# generator perturbs this pose (+-0.15 rad) and rejects all non-tip support.
INHAND_PREGRASP_POSITIONS: dict[str, tuple[float, ...]] = {
    "index": (-0.0130, 0.5800, 0.5700),
    "middle": (-0.0132, 0.5300, 0.5200),
    "ring": (-0.0137, 0.5700, 0.5500),
    "pinky": (0.0249, 0.5800, 0.5700),
    "thumb": (0.7000, 0.9400, 0.5400, 0.4700),
}

# Inclination (deg) of the palm longitudinal axis — the vector from the wrist
# centre to the middle-finger MCP, lying within the palm surface — above the
# horizontal ground plane.  This is the spec'd 45 deg constraint; it is NOT a
# tilt of the palm surface normal.
INHAND_LONG_AXIS_TILT_DEG: float = 45.0

# Thumb-edge-down roll (deg) about the finger axis.  0 = palm facing straight
# up across its width.  History: an earlier pose used 45 deg roll (with a flat
# longitudinal axis) so gravity loaded the thumb pad against the object; that
# misread the 45 deg spec as a palm-normal tilt.  Kept as a knob — re-add a
# small roll here if the thumb loses the object (spin-and-drop regressions).
INHAND_PALM_ROLL_DEG: float = 0.0


def _quat_from_axis_angle(
    axis: tuple[float, float, float], angle_rad: float
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(a * a for a in axis))
    half = 0.5 * angle_rad
    s = math.sin(half) / norm
    return (math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)


def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


# Hand base orientation (wxyz), composition ``Quat(X, -long_tilt) *
# Quat(Y, -90deg+roll) * Quat(X, +90deg)``.  In the L20 base frame the fingers
# extend along +Z, the palm normal is +X and the thumb edge is -Y.  The two
# right factors are the HORA flat palm-up pose (palm normal up, fingers along
# world -Y, thumb edge +X) plus the optional thumb-edge-down roll; the leading
# world-X pitch then raises the palm longitudinal axis (wrist -> middle MCP)
# by INHAND_LONG_AXIS_TILT_DEG above the horizontal.  At tilt=45/roll=0 the
# longitudinal axis maps to (0, -0.707, +0.707) (fingertips above the wrist),
# the palm normal to (0, +0.707, +0.707) (still up-facing), and the thumb
# edge stays horizontal at +X.
INHAND_HAND_ROT: tuple[float, float, float, float] = _quat_mul(
    _quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(-INHAND_LONG_AXIS_TILT_DEG)),
    _quat_mul(
        _quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(-90.0 + INHAND_PALM_ROLL_DEG)),
        _quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(90.0)),
    ),
)

# Object reset position (env-local) = the pregrasp's fingertip-cage centre with
# a small upward margin so the dropped object settles down into the fingertip
# cage (never onto the palm).  Both the grasp-gen drop and the main-env
# canonical fallback use this.  Seeded by rotating the old cage centre into
# the new hand orientation; refined from grasp-gen settle diagnostics.
INHAND_OBJECT_INIT_POS: tuple[float, float, float] = (-0.004, -0.090, 0.675)

# Rotation axis (world frame) = world -z, gravity-aligned, exactly HORA's.
# Deliberately DECOUPLED from the palm tilt (design decision 2026-07-16): with
# the axis parallel to gravity the load direction is invariant under the
# rotation, so every phase of the finger gait sees the same gravity loading —
# the object spins in place instead of tumbling.  The earlier tilted-axis
# variant (-(palm normal), 45 deg off gravity) forced the object axis to
# precess and cycled the load direction through every revolution.
# History note: with the OLD roll-tilted palm, world -z produced a downhill
# spin-and-drop exploit (rotate-reward 0.35/step, 100% falls at ~40 steps);
# the current pitch orientation + thumb-shelf pregrasp blocks the downhill
# path, but watch for that signature (RotateReward up, EpLen collapsing) when
# retraining.  If learning stalls with rotation reward pinned at/below zero,
# flip the sign: the natural gait direction can mirror on a left hand.
INHAND_ROT_AXIS: tuple[float, float, float] = (0.0, 0.0, -1.0)

INHAND_MIMIC_JOINTS: dict[str, tuple[str, float, float]] = {
    "index_dip": ("index_pip", 0.8917, 0.0),
    "middle_dip": ("middle_pip", 0.8917, 0.0),
    "ring_dip": ("ring_pip", 0.8917, 0.0),
    "pinky_dip": ("pinky_pip", 0.8917, 0.0),
    "thumb_ip": ("thumb_mcp", 1.1619, 0.0),
}


def _scale_cache_tag(scale: float) -> str:
    body = f"{scale:.3f}".rstrip("0").rstrip(".").replace(".", "")
    return f"s{body}"


# Grasp caches are keyed by (scale, SHAPE, PROTOTYPE).  Cylinder-only caches
# replayed onto spheres/cubes gave ~76% dead-on-arrival resets, and even
# same-shape caches mixing prototypes (3 cylinder lengths / 2 cuboid sizes /
# 4 sphere radii) stayed ~52% DOA — a fingertip cage has millimetre tolerance,
# so a grasp settled on one prototype rarely transfers to another.  Every
# training prototype therefore gets its own cache file.
INHAND_CACHE_SHAPES: tuple[str, ...] = ("cylinder", "cuboid", "sphere")
_SHAPE_CACHE_TAGS: dict[str, str] = {"cylinder": "cyl", "cuboid": "cub", "sphere": "sph"}


def grasp_cache_filename(
    scale: float, name: str = "linker_l20", shape: str = "cylinder", proto: int = 0
) -> str:
    return (
        f"{name}_grasp_{_SHAPE_CACHE_TAGS[shape]}{int(proto)}_{_scale_cache_tag(scale)}.npy"
    )


def _joint_pos_with_mimics(
    pregrasp: dict[str, tuple[float, ...]],
) -> dict[str, float]:
    joint_pos: dict[str, float] = {
        "index_mcp_roll": pregrasp["index"][0],
        "index_mcp_pitch": pregrasp["index"][1],
        "index_pip": pregrasp["index"][2],
        "middle_mcp_roll": pregrasp["middle"][0],
        "middle_mcp_pitch": pregrasp["middle"][1],
        "middle_pip": pregrasp["middle"][2],
        "ring_mcp_roll": pregrasp["ring"][0],
        "ring_mcp_pitch": pregrasp["ring"][1],
        "ring_pip": pregrasp["ring"][2],
        "pinky_mcp_roll": pregrasp["pinky"][0],
        "pinky_mcp_pitch": pregrasp["pinky"][1],
        "pinky_pip": pregrasp["pinky"][2],
        "thumb_cmc_yaw": pregrasp["thumb"][0],
        "thumb_cmc_roll": pregrasp["thumb"][1],
        "thumb_cmc_pitch": pregrasp["thumb"][2],
        "thumb_mcp": pregrasp["thumb"][3],
    }
    for follower, (master, mult, off) in INHAND_MIMIC_JOINTS.items():
        joint_pos[follower] = joint_pos[master] * mult + off
    return joint_pos


def _make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/LinkerHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "linker_hand_l20/linkerhand_l20_left.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=True,
            make_instanceable=False,
            activate_contact_sensors=True,
            collider_type="convex_hull",
            self_collision=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None, damping=None
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=INHAND_HAND_ROT,
            joint_pos=_joint_pos_with_mimics(INHAND_PREGRASP_POSITIONS),
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # The exact HORA/Allegro P=3, |tau|<=0.5 N m controller failed
                # this task's physical-palm reset gate (5.54% > 1%): it cannot
                # hold the much larger 64 mm G20 cube clear of the palm.  Keep
                # the G20's certified P/effort scale, but remove the 10x excess
                # damping relative to HORA so cyclic finger motion is not
                # needlessly resisted.
                effort_limit_sim=1.0,
                stiffness=6.0,
                damping=0.1,
                armature=0.001,
            )
        },
    )


# ---------------------------------------------------------------------------
# Object prototypes.  The local Wonik HORA fork defaults to a
# cylinder/cuboid/sphere mix, while the original paper primarily reports
# cylinders.  This deployment task deliberately trains cuboid-only because the
# user's physical object is a 60/64/68 mm cube; it is not a claim of shape-level
# parity with HORA.  The full shape
# machinery is retained: switch object_kind back to "mixed" to restore the mix.
# The deterministic MultiAssetSpawner (random_choice=False) gives every env a
# fixed, known (scale, shape, prototype).
MIX_CYLINDER_LENGTHS: tuple[float, ...] = (0.064, 0.080, 0.096)            # 3
MIX_CUBOID_SIZES: tuple[tuple[float, float, float], ...] = (              # 1
    (INHAND_CUBE_EDGE_M, INHAND_CUBE_EDGE_M, INHAND_CUBE_EDGE_M),
)
MIX_SPHERE_RADII: tuple[float, ...] = (0.034, 0.038, 0.042, 0.046)         # 4
MIX_BLOCK_SIZE: int = len(MIX_CYLINDER_LENGTHS) + len(MIX_CUBOID_SIZES) + len(MIX_SPHERE_RADII)
OBJECT_MASS: float = 0.10


def _shape_common() -> dict:
    return dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0
        ),
    )


def _cylinder_block(scale: float, lengths: tuple[float, ...]) -> list:
    return [
        sim_utils.CylinderCfg(
            radius=HORA_CYLINDER_RADIUS * scale, height=length * scale, axis="Z",
            **copy.deepcopy(_shape_common()),
        )
        for length in lengths
    ]


def _mixed_block(scale: float) -> list:
    """One scale's 9-asset block: 3 cylinders + 2 cuboids + 4 spheres."""
    block = _cylinder_block(scale, MIX_CYLINDER_LENGTHS)
    block += [
        sim_utils.CuboidCfg(
            size=(bx * scale, by * scale, bz * scale), **copy.deepcopy(_shape_common())
        )
        for (bx, by, bz) in MIX_CUBOID_SIZES
    ]
    block += [
        sim_utils.SphereCfg(radius=r * scale, **copy.deepcopy(_shape_common()))
        for r in MIX_SPHERE_RADII
    ]
    return block


def _cuboid_block(scale: float) -> list:
    return [
        sim_utils.CuboidCfg(
            size=(bx * scale, by * scale, bz * scale), **copy.deepcopy(_shape_common())
        )
        for (bx, by, bz) in MIX_CUBOID_SIZES
    ]


def _sphere_block(scale: float) -> list:
    return [
        sim_utils.SphereCfg(radius=r * scale, **copy.deepcopy(_shape_common()))
        for r in MIX_SPHERE_RADII
    ]


def _build_object(
    scales: tuple[float, ...], kind: str
) -> tuple[RigidObjectCfg, list[int], list[int], list[int]]:
    """Build the multi-asset object grid and the parallel per-asset scale,
    shape and within-shape prototype indices.

    ``kind='mixed'`` -> cylinders+cuboids+spheres (main task); a single-shape
    kind (``'cylinder'``/``'cuboid'``/``'sphere'``) spawns only that shape's
    training prototypes (grasp-cache generation).  For asset ``i`` in spawn
    order, ``scale_idx[i]`` indexes ``scales``, ``shape_idx[i]`` indexes
    ``INHAND_CACHE_SHAPES`` and ``proto_idx[i]`` is the prototype within that
    shape (e.g. which cylinder length), so the env can recover each env's
    (scale, shape, prototype) from ``asset_idx = env_id % n_assets``.
    """
    single = {
        "cylinder": lambda s: _cylinder_block(s, MIX_CYLINDER_LENGTHS),
        "cuboid": _cuboid_block,
        "sphere": _sphere_block,
    }
    n_cyl, n_cub, n_sph = len(MIX_CYLINDER_LENGTHS), len(MIX_CUBOID_SIZES), len(MIX_SPHERE_RADII)
    assets: list = []
    scale_idx: list[int] = []
    shape_idx: list[int] = []
    proto_idx: list[int] = []
    for si, s in enumerate(scales):
        if kind == "mixed":
            block = _mixed_block(s)
            block_shapes = [0] * n_cyl + [1] * n_cub + [2] * n_sph
            block_protos = list(range(n_cyl)) + list(range(n_cub)) + list(range(n_sph))
        else:
            block = single[kind](s)
            block_shapes = [INHAND_CACHE_SHAPES.index(kind)] * len(block)
            block_protos = list(range(len(block)))
        assets.extend(block)
        scale_idx.extend([si] * len(block))
        shape_idx.extend(block_shapes)
        proto_idx.extend(block_protos)
    cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=assets, random_choice=False, activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=INHAND_OBJECT_INIT_POS, rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    return cfg, scale_idx, shape_idx, proto_idx


def _make_object_cfg(
    scales: tuple[float, ...] = HORA_CYLINDER_SCALES,
    lengths: tuple[float, ...] = HORA_CYLINDER_LENGTHS,  # kept for signature compat
) -> RigidObjectCfg:
    """Cylinder-only object grid (used by the grasp-cache generator)."""
    return _build_object(scales, "cylinder")[0]


@configclass
class InhandCurriculumPhaseCfg:
    step_start: int = 0
    reward_turn_weight: float = 1.0
    strict_contact_bonus: float = 0.4
    partial_contact_bonus: float = 0.1
    upright_termination_threshold: float = 0.0
    episode_length_s: float = 20.0


@configclass
class InhandDomainRandCfg:
    enabled: bool = True
    # The physical 64 mm PLA cube has not been weighed yet.  Cover the user's
    # stated tens-of-grams through roughly 200 g envelope without extending to
    # the old 10 g/250 g tails that describe a different object family.
    mass_range: tuple[float, float] = (0.03, 0.20)
    com_range: tuple[float, float] = (-0.008, 0.008)
    friction_range: tuple[float, float] = (0.4, 2.5)
    pd_gain_range: tuple[float, float] = (0.967, 1.033)
    joint_noise_scale: float = 0.02
    force_scale: float = 2.0
    random_force_prob: float = 0.25
    force_decay: float = 0.9
    force_decay_interval: float = 0.08


@configclass
class LinkerL20InhandRotationEnvCfg(DirectRLEnvCfg):
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(105,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
    state_space = 0

    latent_conditioned: bool = True
    asymmetric_obs: bool = False
    # Actor = 3 proprio frames (3 x [scaled q16, target16]) plus the original
    # HORA 9-D teacher tail: object position xyz and six episode-constant
    # extrinsics.  Object orientation and linear/angular velocity remain out of
    # the actor latent.  The asymmetric critic independently gets the full 19-D
    # free-object state.
    privileged_obs_dim: int = 19
    history_obs_dim: int = 32
    prop_hist_len: int = 30
    actor_frame_count: int = 3
    actor_extrinsics_dim: int = 9
    slow_extrinsics_only: bool = True
    observation_semantics_version: str = (
        "linker-g20-inhand-cube-stack3-hora-pos-extrinsics-v2"
    )
    # Compatibility alias used by the original in-hand env.  It is checked
    # against ``actor_frame_count`` in __post_init__; neither may silently drift.
    num_obs_frames: int = 3

    decimation: int = 6
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            gpu_max_rigid_patch_count=2**22,
        ),
    )
    episode_length_s: float = 20.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=0.75,
        replicate_physics=False,
    )

    action_delta_scale: float = 1.0 / 24.0
    action_clip: float = 1.0
    joint_target_margin: float = 0.0
    # Keep integrated targets inside a reset-grasp-relative working window so
    # the policy must learn a drive/release/return cycle instead of making one
    # irreversible push and parking at the URDF limit.
    joint_motion_range: float = 0.0
    w_target_bound: float = 0.0
    absolute_action_targets: bool = False
    # Preserve the certified cache grasp while the policy history fills, then
    # blend in actions rather than applying a discontinuous command at reset.
    reset_action_hold_steps: int = 5
    reset_action_ramp_steps: int = 10

    rot_axis: tuple[float, float, float] = INHAND_ROT_AXIS
    # HORA-safe objective: preserve HORA's signed, clipped rotation signal and
    # exact reward scale while keeping this task's explicit fall/palm constraints.
    # Rotation is deliberately NOT multiplied by contact/upright/height gates:
    # those gates taught the 40M pilot stable contact and posture but suppressed
    # the early rotation gradient and produced an oscillatory local optimum.
    rotate_reward_scale: float = 1.0
    rotate_reward_scale_final: float = 1.0
    reverse_reward_ratio: float = 1.0
    angvel_clip: tuple[float, float] = (-0.5, 0.5)
    gate_rotation_reward: bool = False
    # Retained as diagnostics and for explicit ablations.  The HORA-safe
    # default does not multiply true rotation by these gates.
    turn_upright_gate_std: float = 0.20
    turn_height_margin_m: float = 0.020
    # Dense, direction-aware finger-drive shaping.  A contacting fingertip is
    # rewarded for moving with the cube surface in the requested rotation
    # direction; it can return without penalty after releasing contact.
    # Disabled in the HORA baseline.  The bounded pilot overrode this to 50,
    # making local fingertip motion worth up to five times the maximum true
    # forward-rotation reward without improving net turns.
    drive_reward_scale: float = 0.0
    drive_speed_ref: float = 0.020
    # One-off reward at the fall-termination step.  HORA has no fall penalty
    # (its flat-palm grasps rarely fall); with the tilted fingertip cage,
    # three training runs converged to a spin-fast-and-drop equilibrium
    # (rotate-reward ~0.37/step, 100% of episodes ending in falls at ~50
    # steps) because dropping cost nothing.  Standard IsaacGymEnvs in-hand
    # practice (AllegroHand fallPenalty).
    # Re-sized -10 -> -25 (2026-07-16): the 1.64B-step rotation-first run
    # converged to rotate +0.32/step with 54% falls at median ep 140 — at -10
    # a drop cost barely 30 steps of rotation income, so spin-and-drop stayed
    # mildly profitable.  -25 prices a fall at roughly half a typical
    # episode's rotation income.  Total reward is NOT comparable across this
    # change; judge runs by the probe's rotate-reward/step and fall%.
    fall_penalty: float = -1000.0
    # Deployment permits finger-side contact but rejects physical palm support.
    # Penalize the exact evaluation threshold densely during training.
    palm_support_force_threshold: float = 0.05
    palm_support_penalty_scale: float = -10.0
    # Diagnostic margin retained for evaluation; disabled in the HORA-safe
    # objective because the terminal fall cost already prices a drop.
    drop_margin_m: float = 0.020
    drop_margin_penalty_scale: float = 0.0
    # Downward-motion and tilt diagnostics remain logged but are not rewarded.
    downward_velocity_penalty_scale: float = 0.0
    upright_tilt_penalty_scale: float = 0.0
    tilt_velocity_penalty_scale: float = 0.0
    # Match HORA's four regularizers.  The fall and physical-palm penalties
    # below remain task-specific safety terms.
    linvel_penalty_scale: float = -0.3
    pose_penalty_scale: float = -0.3
    torque_penalty_scale: float = -0.1
    work_penalty_scale: float = -2.0
    # Fall-termination / grasp-acceptance height: ~3 cm below the settled cage
    # centre (zero-action settle at the 45 deg longitudinal-axis pose parks the
    # object at z ~ 0.6765 across prototypes; DR-off probe 2026-07-14).
    reset_height_threshold: float = 0.645

    fingers: tuple[str, ...] = ("index", "middle", "ring", "pinky", "thumb")
    pregrasp_positions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: copy.deepcopy(INHAND_PREGRASP_POSITIONS)
    )
    robot_cfg: ArticulationCfg = field(default_factory=_make_robot_cfg)

    object_scales: tuple[float, ...] = INHAND_OBJECT_SCALES
    object_lengths: tuple[float, ...] = HORA_CYLINDER_LENGTHS
    # "mixed" = cylinders+cuboids+spheres (main task); grasp-gen overrides to a
    # single shape.  object_cfg + object_asset_{scale,shape}_idx are (re)built
    # in __post_init__.
    # "cuboid" = the user's physical cube family.  "mixed" restores the local
    # Wonik HORA fork's cylinder/cuboid/sphere grid.
    object_kind: str = "cuboid"
    object_cfg: RigidObjectCfg = None
    object_asset_scale_idx: list[int] = None
    object_asset_shape_idx: list[int] = None
    object_asset_proto_idx: list[int] = None

    grasp_cache_dir: str = str(ASSET_ROOT / "grasp_cache")
    grasp_cache_name: str = "linker_l20_bottomup_cube64_v2"
    grasp_orientation: str = "bottom-up-tilted-45deg-v2-replay-certified"
    grasp_cache_certification: dict[str, object] = field(
        default_factory=lambda: copy.deepcopy(INHAND_CACHE_CERTIFICATION)
    )
    # This is the identity of the complete production cache bank.  Evaluation
    # may pin ``object_scales`` to only the nominal bucket, but that must not
    # create a second, deceptively compatible manifest identity.
    grasp_manifest_scales: tuple[float, ...] = INHAND_OBJECT_SCALES
    load_grasp_cache: bool = True
    # Both production orientations require a complete, version-matched cache.
    # A canonical fallback made the old bottom-up task appear runnable while it
    # was actually replaying no validated reset distribution at all.
    require_complete_grasp_cache: bool = True

    domain_rand: InhandDomainRandCfg = field(default_factory=InhandDomainRandCfg)
    # One phase, with true signed rotation present from the first rollout.  The
    # previous hold-first schedule spent 8M steps explicitly optimizing a
    # static grasp and gave a 40M pilot only 12M steps at its final objective.
    curriculum_phases: list[InhandCurriculumPhaseCfg] = field(
        default_factory=lambda: [
            InhandCurriculumPhaseCfg(
                step_start=0,
                reward_turn_weight=1.0,
                strict_contact_bonus=0.0,
                partial_contact_bonus=0.0,
            )
        ]
    )

    # Reward authorization must use the same physical object-force signal as
    # oracle evaluation.  A geometric distal-origin proxy was explicitly
    # rejected after a 40M pilot learned to coast with only 5.82% true contact.
    enable_fingertip_sensors: bool = True
    enable_palm_sensor: bool = True
    # Exact sensors over every non-distal link.  Training recognizes legal
    # finger-side/proximal/middle contact when authorizing manipulation reward,
    # while the physical palm remains a separate hard rejection signal.
    enable_nontip_sensors: bool = True
    grasp_gen_obj_init_pos: tuple[float, float, float] = INHAND_OBJECT_INIT_POS
    grasp_gen_obj_init_rot: tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    grasp_gen_pose_noise: float = 0.15
    # Human-like opposition grip: the thumb fingertip must press the object and
    # at least this many of the other four fingertips must also be in contact.
    require_thumb_contact: bool = True
    min_other_finger_contacts: int = 2
    # Fingertip-ONLY hold: reject states where the object touches the palm or
    # any non-distal finger link (the object must float in the fingertip cage).
    forbid_nontip_contact: bool = True
    nontip_force_eps: float = 1.0e-4
    # Ignore the acceptance criteria for the first few grasp-gen steps so the
    # dropped object may transiently graze non-distal links while settling.
    grasp_gen_grace_steps: int = 10
    # During cache generation only, keep the operator-loaded object at its seed
    # pose for a few control steps so the position drives can establish squeeze
    # preload.  It is then fully released; a row is harvested only if the free
    # object passes the strict fingertip-only gate through episode timeout.
    # Without this closing phase a 100 g, 64 mm cube falls several centimetres
    # before the first acceptance check even for a valid geometric cage.
    grasp_gen_object_hold_steps: int = 10
    grasp_gen_object_min_hold_steps: int = 2
    # Cache rows must remain fingertip-only for a complete production-length
    # zero-action episode before harvest.  The old 2.5 s dwell certified only
    # the initial settling transient; replaying those rows for the 20 s main
    # episode exposed delayed proximal/palm support.
    grasp_gen_validation_episode_s: float = 20.0
    # Start flexion DOFs this far more open than their randomized squeeze
    # targets.  Roll/yaw opposition joints stay at the fitted cage values.
    grasp_gen_initial_opening_rad: float = 0.06
    # Grasp-gen acceptance needs the object this far ABOVE the fall threshold:
    # without it the harvest keeps mid-slide states sitting barely above the
    # threshold, which then fall immediately when used as training resets.
    grasp_gen_accept_z_margin: float = 0.02
    # Harvested measured joints and their squeeze targets must both retain
    # room from the current hand's semantic/URDF limits.  Random-noise samples
    # that get clipped to a stop can look stable in simulation but are not a
    # deployable real-hand pre-grasp.
    grasp_gen_joint_limit_margin: float = 0.04
    # Fingertip-to-object-CENTRE sanity bound for grasp acceptance.  The L20
    # ``*_distal`` body origins sit ~0.10 m from the object centre at contact (the
    # origin is at the proximal end of the distal link, not the pad), so HORA's
    # 0.1 m (tuned to the Allegro tip origin) is too tight here; 0.13 leaves the
    # +-0.25 rad grasp-gen noise room while still rejecting flown-off fingers.
    tip_dist_max: float = 0.13

    def _rebuild_object(self) -> None:
        """(Re)build ``object_cfg`` + ``object_asset_{scale,shape}_idx`` from the
        current ``object_scales`` / ``object_kind``.  Called by ``__post_init__``
        and by the grasp-cache tool after it pins a single scale/shape."""
        (
            self.object_cfg,
            self.object_asset_scale_idx,
            self.object_asset_shape_idx,
            self.object_asset_proto_idx,
        ) = _build_object(self.object_scales, self.object_kind)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_obs_frames != self.actor_frame_count:
            raise ValueError(
                "num_obs_frames and actor_frame_count must describe the same "
                "in-hand actor observation contract"
            )
        if self.object_cfg is None or self.object_asset_scale_idx is None:
            self._rebuild_object()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.actor_frame_count * self.history_obs_dim
                + self.actor_extrinsics_dim,
            ),
            dtype=np.float32,
        )


@configclass
class LinkerL20InhandGraspGenEnvCfg(LinkerL20InhandRotationEnvCfg):
    cache_scale: float = 1.0
    cache_shape: str = "cuboid"
    # Optional batched physics-search candidates.  Production harvesting leaves
    # these empty and perturbs the configured canonical seed.  Search tools can
    # assign one absolute q16/object-position pair per stable env-id bucket.
    grasp_gen_joint_candidates: tuple[tuple[float, ...], ...] = ()
    grasp_gen_object_pos_candidates: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        # Grasp-cache generation runs on ONE shape's training prototypes at a
        # single pinned scale (one cache file per (scale, shape) pair).
        self.object_kind = str(self.cache_shape)
        self.object_scales = (float(self.cache_scale),)
        self.object_cfg = None
        self.object_asset_scale_idx = None
        self.object_asset_shape_idx = None
        self.object_asset_proto_idx = None
        super().__post_init__()  # builds the single-shape grid + indices
        self.episode_length_s = float(self.grasp_gen_validation_episode_s)
        self.curriculum_phases = [
            InhandCurriculumPhaseCfg(
                step_start=0,
                reward_turn_weight=1.0,
                upright_termination_threshold=0.0,
                episode_length_s=float(self.grasp_gen_validation_episode_s),
            )
        ]
        # HORA generates grasps at NOMINAL dynamics (here mass 0.10, friction 1.0,
        # no randomisation) and replays them under training DR.  Generating
        # under randomised mass/friction skews the cache toward states that
        # only hold for that draw (e.g. feather-weight objects).
        self.domain_rand.enabled = False
        self.domain_rand.random_force_prob = 0.0
        self.domain_rand.force_scale = 0.0
        self.domain_rand.pd_gain_range = (1.0, 1.0)
        self.load_grasp_cache = False
        self.enable_fingertip_sensors = True
        self.enable_nontip_sensors = True
        self.grasp_gen_obj_init_pos = INHAND_OBJECT_INIT_POS
        self.grasp_gen_pose_noise = 0.15
        self.require_thumb_contact = True
        self.min_other_finger_contacts = 2
        self.forbid_nontip_contact = True
        self.grasp_gen_grace_steps = 10
        self.grasp_gen_object_hold_steps = 10
        self.grasp_gen_object_min_hold_steps = 2
        self.grasp_gen_initial_opening_rad = 0.06
        self.grasp_gen_joint_limit_margin = 0.04
        self.tip_dist_max = 0.13
        if bool(self.grasp_gen_joint_candidates) != bool(
            self.grasp_gen_object_pos_candidates
        ):
            raise ValueError(
                "grasp search requires both joint and object-position candidates"
            )
        if self.grasp_gen_joint_candidates:
            if len(self.grasp_gen_joint_candidates) != len(
                self.grasp_gen_object_pos_candidates
            ):
                raise ValueError("grasp search candidate arrays must have equal length")
            if any(len(row) != 16 for row in self.grasp_gen_joint_candidates):
                raise ValueError("every grasp search joint candidate must be q16")
