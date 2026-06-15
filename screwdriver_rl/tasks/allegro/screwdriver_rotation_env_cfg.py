"""Configuration for Allegro hand continuous screwdriver rotation.

All shared task config (curriculum, domain randomisation, reward weights,
screwdriver asset, simulation) lives in
:class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnvCfg`.  This module only
overrides the Allegro-specific fields: the gym spaces, active fingers, pregrasp,
RMA dims, fingertip pad axis, and the Allegro articulation.
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

    # ---- Curriculum (Allegro-tuned; see CurriculumPhaseCfg for field docs) ----
    # The Allegro hand drives the task with 3 active fingers (index/middle/thumb),
    # so a 2-of-N contact gate is appropriate.  These were the original base
    # values; they now live here so each hand owns its curriculum independently.
    curriculum_phases: list[CurriculumPhaseCfg] = field(
        default_factory=lambda: [
            CurriculumPhaseCfg(
                # --- P0: learn PAD contact from the start ---
                # The ungated ``near_reward`` (0.8) provides the "approach the
                # handle" gradient, so we start with the contact gate ON (generous
                # 0.10 m), a lenient SOFT pad factor, and a free-spinning handle.
                step_start=0,
                reward_turn_weight=120.0,
                turn_reward_contact_distance=0.10,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.0,
                pad_facing_cos_threshold=0.0,
                screwdriver_load_scale=0.0,
                reward_proximal_penalty_weight=0.0,
                near_reward_weight=0.8,
                episode_length_s=30.0,
                upright_termination_threshold=2.0,
            ),
            CurriculumPhaseCfg(
                # --- P1: introduce the screw load and tighten the pad ---
                step_start=40_000_000,
                reward_turn_weight=180.0,
                turn_reward_contact_distance=0.07,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.003,
                pad_facing_cos_threshold=0.3,
                screwdriver_load_scale=0.5,
                reward_proximal_penalty_weight=3.0,
                near_reward_weight=0.3,
                episode_length_s=50.0,
                upright_termination_threshold=1.3,
            ),
            CurriculumPhaseCfg(
                # --- P2: final — strict pad + full load ---
                step_start=90_000_000,
                reward_turn_weight=200.0,
                turn_reward_contact_distance=0.05,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.003,
                pad_facing_cos_threshold=0.5,
                screwdriver_load_scale=1.0,
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

    # ---- Fingertip pad axis ----
    fingertip_pad_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Outward pad normal in the fingertip (_ee) link local frame.  The Allegro
    distal phalanx pad faces +x of the phalanx; the fixed tip-frame rotation
    (rpy = π/2,0,π/2) maps that to +z of _ee.  Verified by calibrate_pad.py."""

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
