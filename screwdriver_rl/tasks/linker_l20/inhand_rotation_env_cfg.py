"""Configuration for Linker Hand L20 free-object in-hand rotation.

This task is the HORA in-hand cylinder rotation setup ported onto the LinkerHand
L20.  It intentionally does not subclass the mounted-screwdriver task config:
the object is a free rigid cylinder, the reset comes from per-scale grasp caches,
and the observation/reward contracts match the existing two-stage RMA pipeline.
"""

from __future__ import annotations

import copy
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
HORA_CYLINDER_SCALES: tuple[float, ...] = (
    0.70,
    0.72,
    0.74,
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
)

# Canonical pregrasp = the LinkerL20's VALIDATED five-finger vertical-cylinder
# grip (reused from the screwdriver task's proven grasp).  A palm-up cup guess
# left the cylinder resting on the finger backs (no grip); this closed grip holds
# a vertical cylinder in 100% of envs with thumb + finger opposition, which the
# grasp-gen then perturbs (+-0.25 rad) and filters into the cache.  Paired with
# INHAND_HAND_ROT (the same grasp's hand orientation) and INHAND_OBJECT_INIT_POS
# (the grip's fingertip centroid); see the calibration in the plan/verification.
INHAND_PREGRASP_POSITIONS: dict[str, tuple[float, ...]] = {
    "index": (0.070000, 0.140535, 1.379901),
    "middle": (-0.070000, 0.184823, 1.354472),
    "ring": (-0.070000, 0.449762, 1.207458),
    "pinky": (0.045669, 0.654302, 1.027913),
    "thumb": (0.673745, 1.120000, 0.100000, 0.876434),
}

# Hand base orientation (wxyz): the validated grip's palm-toward-object pose.
INHAND_HAND_ROT: tuple[float, float, float, float] = (
    0.44578500, -0.47244983, -0.22820989, 0.72524971,
)

# Object reset position (env-local), = the grip's measured fingertip centroid
# with a small upward margin so the cylinder settles down into the grip.  Both
# the grasp-gen drop and the main-env canonical fallback use this.
INHAND_OBJECT_INIT_POS: tuple[float, float, float] = (-0.170, 0.050, 0.570)

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


def grasp_cache_filename(scale: float, name: str = "linker_l20") -> str:
    return f"{name}_grasp_50k_{_scale_cache_tag(scale)}.npy"


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
                stiffness=6.0,
                damping=1.0,
                armature=0.001,
            )
        },
    )


# ---------------------------------------------------------------------------
# Object mix.  HORA trains on cylinders + cuboids + spheres (sampleProb
# 0.35 / 0.20 / 0.45).  Per scale we spawn a fixed 9-asset block
# (3 cylinders + 2 cuboids + 4 spheres ~= that 0.33 / 0.22 / 0.44 split), so the
# deterministic MultiAssetSpawner (random_choice=False) gives every env a fixed,
# known (scale, shape).  The grasp cache is generated on CYLINDERS only and
# reused across shapes at reset (HORA keys the cache by scale, not shape).
MIX_CYLINDER_LENGTHS: tuple[float, ...] = (0.064, 0.080, 0.096)            # 3
MIX_CUBOID_SIZES: tuple[tuple[float, float, float], ...] = (              # 2
    (0.064, 0.064, 0.080),
    (0.072, 0.072, 0.064),
)
MIX_SPHERE_RADII: tuple[float, ...] = (0.034, 0.038, 0.042, 0.046)         # 4
MIX_BLOCK_SIZE: int = len(MIX_CYLINDER_LENGTHS) + len(MIX_CUBOID_SIZES) + len(MIX_SPHERE_RADII)
OBJECT_MASS: float = 0.05


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


def _build_object(scales: tuple[float, ...], kind: str) -> tuple[RigidObjectCfg, list[int]]:
    """Build the multi-asset object grid and the parallel per-asset scale index.

    ``kind='mixed'`` -> cylinders+cuboids+spheres (main task); ``kind='cylinder'``
    -> cylinders only (grasp-cache generation).  The returned ``scale_idx[i]`` is
    the index into ``scales`` of asset ``i`` (in spawn order), so the env can
    recover each env's scale from ``asset_idx = env_id % n_assets`` regardless of
    the block composition.
    """
    assets: list = []
    scale_idx: list[int] = []
    for si, s in enumerate(scales):
        block = _mixed_block(s) if kind == "mixed" else _cylinder_block(s, HORA_CYLINDER_LENGTHS)
        assets.extend(block)
        scale_idx.extend([si] * len(block))
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
    return cfg, scale_idx


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
    upright_termination_threshold: float = 0.0
    episode_length_s: float = 20.0


@configclass
class InhandDomainRandCfg:
    enabled: bool = True
    mass_range: tuple[float, float] = (0.01, 0.25)
    com_range: tuple[float, float] = (-0.01, 0.01)
    friction_range: tuple[float, float] = (0.3, 3.0)
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
    privileged_obs_dim: int = 9
    history_obs_dim: int = 32
    prop_hist_len: int = 30
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

    rot_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    rotate_reward_scale: float = 1.0
    angvel_clip: tuple[float, float] = (-0.5, 0.5)
    linvel_penalty_scale: float = -0.3
    pose_penalty_scale: float = -0.3
    torque_penalty_scale: float = -0.1
    work_penalty_scale: float = -2.0
    reset_height_threshold: float = 0.52

    fingers: tuple[str, ...] = ("index", "middle", "ring", "pinky", "thumb")
    pregrasp_positions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: copy.deepcopy(INHAND_PREGRASP_POSITIONS)
    )
    robot_cfg: ArticulationCfg = field(default_factory=_make_robot_cfg)

    object_scales: tuple[float, ...] = HORA_CYLINDER_SCALES
    object_lengths: tuple[float, ...] = HORA_CYLINDER_LENGTHS
    # "mixed" = cylinders+cuboids+spheres (main task); grasp-gen overrides to
    # "cylinder".  object_cfg + object_asset_scale_idx are (re)built in __post_init__.
    object_kind: str = "mixed"
    object_cfg: RigidObjectCfg = None
    object_asset_scale_idx: list[int] = None

    grasp_cache_dir: str = str(ASSET_ROOT / "grasp_cache")
    grasp_cache_name: str = "linker_l20"
    load_grasp_cache: bool = True

    domain_rand: InhandDomainRandCfg = field(default_factory=InhandDomainRandCfg)
    curriculum_phases: list[InhandCurriculumPhaseCfg] = field(
        default_factory=lambda: [InhandCurriculumPhaseCfg()]
    )

    enable_fingertip_sensors: bool = False
    grasp_gen_obj_init_pos: tuple[float, float, float] = INHAND_OBJECT_INIT_POS
    grasp_gen_pose_noise: float = 0.25
    min_contact_fingers: int = 2
    # Fingertip-to-object-CENTRE sanity bound for grasp acceptance.  The L20
    # ``*_distal`` body origins sit ~0.10 m from the object centre at contact (the
    # origin is at the proximal end of the distal link, not the pad), so HORA's
    # 0.1 m (tuned to the Allegro tip origin) is too tight here; 0.13 leaves the
    # +-0.25 rad grasp-gen noise room while still rejecting flown-off fingers.
    tip_dist_max: float = 0.13

    def _rebuild_object(self) -> None:
        """(Re)build ``object_cfg`` + ``object_asset_scale_idx`` from the current
        ``object_scales`` / ``object_kind``.  Called by ``__post_init__`` and by the
        grasp-cache tool after it pins a single scale."""
        self.object_cfg, self.object_asset_scale_idx = _build_object(
            self.object_scales, self.object_kind
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.object_cfg is None or self.object_asset_scale_idx is None:
            self._rebuild_object()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs_frames * self.history_obs_dim + self.privileged_obs_dim,),
            dtype=np.float32,
        )


@configclass
class LinkerL20InhandGraspGenEnvCfg(LinkerL20InhandRotationEnvCfg):
    cache_scale: float = 0.8

    def __post_init__(self) -> None:
        # Grasp-cache generation runs on CYLINDERS at a single pinned scale.
        self.object_kind = "cylinder"
        self.object_scales = (float(self.cache_scale),)
        self.object_cfg = None
        self.object_asset_scale_idx = None
        super().__post_init__()  # builds the cylinder-only grid + scale index
        self.episode_length_s = 2.5
        self.curriculum_phases = [
            InhandCurriculumPhaseCfg(
                step_start=0,
                reward_turn_weight=1.0,
                upright_termination_threshold=0.0,
                episode_length_s=2.5,
            )
        ]
        self.domain_rand.enabled = True
        self.domain_rand.random_force_prob = 0.0
        self.domain_rand.force_scale = 0.0
        self.domain_rand.pd_gain_range = (1.0, 1.0)
        self.load_grasp_cache = False
        self.enable_fingertip_sensors = True
        self.grasp_gen_obj_init_pos = INHAND_OBJECT_INIT_POS
        self.grasp_gen_pose_noise = 0.25
        self.min_contact_fingers = 2
        self.tip_dist_max = 0.13
