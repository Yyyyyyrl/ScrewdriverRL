"""Configuration for Allegro hand continuous screwdriver rotation.

All shared task config (curriculum, domain randomisation, reward weights,
screwdriver asset, simulation) lives in
:class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnvCfg`.  This module only
overrides the Allegro-specific fields: the gym spaces, active fingers, pregrasp,
RMA dims, and the Allegro articulation.
"""

from __future__ import annotations

from dataclasses import field

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from screwdriver_rl.tasks.base.screwdriver_rotation_env_cfg import (
    ASSET_ROOT,
    ScrewdriverRotationEnvCfg,
)

# Re-export shared phase/DR configs so existing imports of this module keep working.
from screwdriver_rl.tasks.base.screwdriver_rotation_env_cfg import (  # noqa: F401
    CurriculumPhaseCfg,
    DomainRandCfg,
)


@configclass
class AllegroScrewdriverRotationEnvCfg(ScrewdriverRotationEnvCfg):
    """Allegro continuous screwdriver rotation task.

    Observation space (27-D): [finger_q(12), cur_targets(12), euler(3)]
    Action space (12-D): HORA-style delta targets for index(4)+middle(4)+thumb(4)
    Privileged obs (17-D): euler(3)+angvel(3)+rel_pos(3)+quat(4)+friction(1)+tip_dist(3)
    """

    # ---- Gym spaces ----
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    state_space = 0

    # ---- Active fingers ----
    fingers: tuple[str, ...] = ("index", "middle", "thumb")
    """3-finger configuration: 1–2 fingers stabilise, 1–2 push/reposition."""

    # ---- Reward (044e558 clean design; computed in the shared base _get_rewards) ----
    # turn_reward × (distance+motion contact gate) × upright_gate + near + milestone
    # − reverse − proximal − action/finger regularizers.  No contact sensor, no
    # pad-facing, no screw load.  Most weights are inherited base defaults; these two
    # restore the Allegro-tuned 044e558 values WITHOUT changing the shared base
    # defaults that LinkerL20 also reads (reward_tilt_velocity_weight, gate std).
    reward_tilt_velocity_weight: float = 5.0
    turn_upright_gate_std: float = 0.25

    # ---- Free-spinning handle (clean Allegro) ----
    # Zero screw load: the base load mechanism stays in place but early-returns, so
    # the handle turns freely, exactly like the original 044e558 task.
    screwdriver_load_torque: float = 0.0

    # episode_length_s must match curriculum_phases[0] for the initial stagger.
    episode_length_s: float = 20.0

    # ---- Curriculum (Allegro — 044e558 clean) ----
    # Three phases ramp the distance contact gate (off → 0.10 m → 0.05 m) and the
    # proximal penalty (0 → 2 → 5) while the near-reward tapers (0.8 → 0.3 → 0.15)
    # and the upright termination tightens (2.0 → 1.5 → 1.0).  Free-spinning handle
    # throughout (no screw load).
    curriculum_phases: list[CurriculumPhaseCfg] = field(
        default_factory=lambda: [
            CurriculumPhaseCfg(
                # P0: contact gate OFF — first learn to approach/hold the handle
                # (near-reward dominates); even accidental rotation gives signal.
                step_start=0,
                reward_turn_weight=30.0,
                turn_reward_contact_distance=0.0,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.0,
                reward_proximal_penalty_weight=0.0,
                near_reward_weight=0.8,
                episode_length_s=20.0,
                upright_termination_threshold=2.0,
            ),
            CurriculumPhaseCfg(
                # P1: contact gate ON, generous (0.10 m); mild proximal penalty.
                step_start=15_000_000,
                reward_turn_weight=150.0,
                turn_reward_contact_distance=0.10,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.003,
                reward_proximal_penalty_weight=2.0,
                near_reward_weight=0.3,
                episode_length_s=40.0,
                upright_termination_threshold=1.5,
            ),
            CurriculumPhaseCfg(
                # P2: tighten gate to 0.05 m, strong proximal penalty, strict upright.
                step_start=60_000_000,
                reward_turn_weight=200.0,
                turn_reward_contact_distance=0.05,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.003,
                reward_proximal_penalty_weight=5.0,
                near_reward_weight=0.15,
                episode_length_s=60.0,
                upright_termination_threshold=1.0,
            ),
        ]
    )

    # ---- RMA dims (hand-specific) ----
    privileged_obs_dim: int = 17
    """3 euler + 3 angvel + 3 rel-pos + 4 quat + 1 friction + 3 fingertip-dist."""
    history_obs_dim: int = 24
    """[finger_q(12), cur_targets(12)] per frame."""

    # ---- Robot (Allegro hand) ----
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Allegro",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "xela_models/allegro_hand_right_isaaclab.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=True,
            make_instanceable=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None, damping=None
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # x compensated 0.0 -> -0.009 to track the shared screwdriver mount's
            # -0.009 m x shift (see base screwdriver_cfg), keeping Allegro's
            # hand-to-handle geometry identical.
            pos=(-0.009, -0.095, 1.33),
            rot=(0.664463, 0.2418448, 0.2418448, 0.664463),
            joint_pos={
                "allegro_hand_hitosashi_finger_finger_joint_0": 0.1,
                "allegro_hand_hitosashi_finger_finger_joint_1": 0.6,
                "allegro_hand_hitosashi_finger_finger_joint_2": 0.6,
                "allegro_hand_hitosashi_finger_finger_joint_3": 0.6,
                "allegro_hand_naka_finger_finger_joint_4": -0.1,
                "allegro_hand_naka_finger_finger_joint_5": 0.5,
                "allegro_hand_naka_finger_finger_joint_6": 0.9,
                "allegro_hand_naka_finger_finger_joint_7": 0.9,
                "allegro_hand_oya_finger_joint_12": 1.2,
                "allegro_hand_oya_finger_joint_13": 0.3,
                "allegro_hand_oya_finger_joint_14": 0.3,
                "allegro_hand_oya_finger_joint_15": 1.2,
            },
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=6.0,
                # Damping 1.0: resists joint velocity, prevents finger oscillation.
                # Together with stiffness=6, this is a PD controller tuned for
                # compliant grasping (not high-impedance position tracking).
                damping=1.0,
                armature=0.001,
            )
        },
    )

    # ---- Pregrasp joint positions (per finger, 4 joints each) ----
    pregrasp_positions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "index":  (0.1, 0.6, 0.6, 0.6),
            "middle": (-0.1, 0.5, 0.9, 0.9),
            "ring":   (0.0, 0.5, 0.65, 0.65),
            "thumb":  (1.2, 0.3, 0.3, 1.2),
        }
    )
    """Finger joint positions at episode reset.  index/middle/thumb wrap around
    the handle body from above; the thumb opposes from the side.  This creates a
    3-point pinch from which the policy discovers the repositioning strategy."""
