"""Config for the continuous screwdriver turning task.

Reward / termination defaults are the *final* (strictest) values; training
normally starts from much looser settings applied by the curriculum
(``screwdriver_rl/configs/curricula.py``) which overrides these fields on the
live config each phase. Rationale for every term: docs/reward_design.md.

The config is hand-agnostic: ``hand_name`` selects a HandSpec from the
registry and :meth:`ContinuousTurningEnvCfg.resolve_for_hand` derives the
action/observation/privileged dimensions from it at env construction time.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ... import ASSETS_DIR

_SCREWDRIVER_URDF = ASSETS_DIR / "objects" / "screwdriver" / "screwdriver.urdf"

# Privileged DR feature count: friction(1) + mass_scale(1) + com_xy(2)
# + stiffness_scale(1) + damping_scale(1) + z_damping(1)
_NUM_DR_PRIV = 7


@configclass
class DomainRandCfg:
    """Domain randomization ranges + per-step noise.

    Sampled values are exposed to the privileged observation, which is why DR
    is applied manually in ``_reset_idx`` instead of via the Events manager.
    Enables/widths are scheduled by the curriculum (start narrow, end here).
    """

    # ---- per-reset randomization enables ----
    randomize_friction: bool = True
    randomize_mass: bool = True
    randomize_gains: bool = True
    randomize_com: bool = True
    randomize_z_damping: bool = True
    randomize_init_state: bool = True

    # ---- per-reset ranges ----
    friction_range: tuple[float, float] = (0.5, 1.5)
    mass_scale_range: tuple[float, float] = (0.8, 1.25)
    com_offset_max: float = 0.002  # meters, xy offset on screwdriver bodies
    stiffness_scale_range: tuple[float, float] = (0.85, 1.15)
    damping_scale_range: tuple[float, float] = (0.85, 1.15)
    z_damping_range: tuple[float, float] = (0.02, 0.1)
    init_z_angle_range: tuple[float, float] = (-math.pi, math.pi)
    init_tilt_max: float = 0.05  # rad, initial screwdriver tilt
    pregrasp_noise: float = 0.03  # rad, on controlled finger joints

    # ---- per-step noise (0 disables) ----
    obs_noise_std: float = 0.01  # on finger joint positions in the policy obs
    action_noise_std: float = 0.01  # on the raw policy action


@configclass
class ContinuousTurningEnvCfg(DirectRLEnvCfg):
    """Continuous screwdriver turning with a dexterous hand."""

    # ---- env timing ----
    decimation = 6  # 60 Hz physics -> 10 Hz control
    episode_length_s = 60.0  # 600 policy steps

    # Spaces are placeholders; resolve_for_hand() rewrites them from the HandSpec.
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
    state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)

    # ---- simulation ----
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=1.5, replicate_physics=True)

    # ---- hand selection (resolved via robots registry) ----
    hand_name: str = "allegro"
    controlled_fingers: tuple[str, ...] = ("index", "middle", "thumb")
    # Set by resolve_for_hand(); ArticulationCfg of the selected hand.
    robot_cfg: ArticulationCfg | None = None

    # ---- actions: HORA-style integrated delta position targets ----
    action_clip: float = 1.0
    action_delta_scale: float = 0.05  # rad/step @ 10 Hz => 0.5 rad/s max joint speed
    joint_target_margin: float = 0.02  # rad kept away from soft limits

    # ---- observations ----
    # Policy obs is proprio-only ([finger_q, cur_targets]) so the stage-2
    # student is deployable from joint encoders alone. Object state lives in
    # the privileged obs. Flag below re-adds screwdriver euler to the policy
    # obs as an A/B fallback if teacher discovery stalls (not deployable).
    include_object_in_policy_obs: bool = False
    asymmetric_obs: bool = True
    prop_hist_len: int = 30
    # Derived by resolve_for_hand():
    privileged_obs_dim: int = 22
    history_obs_dim: int = 24

    # ---- reset ----
    reset_contact_steps: int = 32  # settle physics after each reset

    # ---- turn reward ----
    turn_direction: float = -1.0  # clockwise (negative z), like driving a screw
    reward_turn_weight: float = 200.0
    turn_velocity_clip: float = 0.5  # rad/s
    # Strictly above the turn weight in EVERY curriculum phase: backward
    # rotation is always net-negative so oscillation cannot farm reward. Safe
    # against "never touch" because the reverse cost carries the same
    # contact/motion/upright gates as the turn reward.
    reward_reverse_weight: float = 220.0
    # Measure spin about the shaft's own axis (quaternion delta projected on
    # body z) instead of the Euler-z mount coordinate, which also moves under
    # precession of a tilted shaft (wobble would count as turning).
    use_shaft_spin_measure: bool = True

    # ---- uprightness ----
    # Multiplicative gate exp(-(tilt/std)^2) on turn/reverse/milestone terms.
    turn_upright_gate_std: float = 0.15
    reward_upright_weight: float = 400.0
    reward_tilt_velocity_weight: float = 5.0

    # ---- regularization ----
    reward_action_weight: float = 0.25  # sum(a^2)
    reward_action_rate_weight: float = 0.1  # mean(da^2)
    reward_finger_pose_weight: float = 0.02  # sum((q - pregrasp)^2)
    reward_finger_velocity_weight: float = 0.001  # mean(qdot^2)
    reward_joint_limit_weight: float = 200.0  # linear barrier near soft limits
    joint_limit_margin: float = 0.05  # rad
    reward_work_weight: float = 0.01  # sum(|tau * qdot|), energy shaping

    # ---- milestones (sparse net-progress bonus) ----
    milestone_angle: float = 0.5 * math.pi
    milestone_bonus: float = 0.25

    # ---- near-contact shaping (discovery only; decays to 0 in curriculum) ----
    near_reward_weight: float = 0.0
    near_reward_std: float = 0.12
    near_reward_top_k: int = 2

    # ---- contact/motion gate on spin rewards ----
    # Fingertip distance to the handle axis segment (handle radius 0.02 m =>
    # pad contact at ~0.03 m axis distance). <= 0 disables the gate.
    turn_reward_contact_distance: float = 0.035
    turn_reward_min_contact_fingers: int = 2
    turn_reward_min_fingertip_speed: float = 0.003  # m/s, gate starts opening
    turn_reward_full_fingertip_speed: float = 0.015  # m/s, gate fully open

    # ---- terminations ----
    upright_termination_threshold: float = 0.4  # rad tilt norm; <= 0 disables
    lost_contact_termination_distance: float = 0.06  # m; <= 0 disables
    lost_contact_min_fingers: int = 1
    lost_contact_grace_steps: int = 5  # consecutive policy steps out of contact
    # Variance of the z angle over the stagnation window; <= 0 disables.
    stagnation_variance_eps: float = 0.003  # rad^2
    stagnation_window: int = 60  # policy steps (6 s)
    stagnation_grace_steps: int = 100  # min episode length before it can fire

    # ---- domain randomization ----
    dr: DomainRandCfg = DomainRandCfg()

    # ---- screwdriver object (hand-independent) ----
    # Passive articulation: a mounted screwdriver with x/y tilt joints, a free
    # z spin joint, and a free cap joint. Damping values reproduce the MFR
    # benchmark dynamics and are load-bearing for stability: tilt 1e-4
    # (nearly free wobble), z spin 0.05 (light resistance like a real screw).
    screwdriver_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Screwdriver",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(_SCREWDRIVER_URDF),
            fix_base=True,
            merge_fixed_joints=False,
            # Keep true cylinders: capsule end-caps change contact geometry on
            # the shaft/handle.
            replace_cylinders_with_capsules=False,
            make_instanceable=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.205),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "tilt": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_1", "table_screwdriver_joint_2"],
                stiffness=0.0,
                damping=0.0001,
            ),
            "rotation": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_3"],
                stiffness=0.0,
                damping=0.05,
            ),
            "cap": ImplicitActuatorCfg(
                joint_names_expr=["screwdriver_body_cap_joint"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    def resolve_for_hand(self, hand_spec) -> None:
        """Derive robot cfg + space dims from a HandSpec. Idempotent."""
        self.robot_cfg = hand_spec.articulation_cfg
        act_dim = hand_spec.num_action_dofs
        obs_dim = 2 * act_dim + (3 if self.include_object_in_policy_obs else 0)
        priv_dim = 2 + 2 + 3 + 1 + len(hand_spec.controlled_fingers) + 3 + _NUM_DR_PRIV + 1
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(priv_dim,), dtype=np.float32)
        self.privileged_obs_dim = priv_dim
        self.history_obs_dim = 2 * act_dim
