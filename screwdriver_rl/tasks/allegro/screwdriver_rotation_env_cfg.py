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

    # ---- Reward / upright-gate overrides (restore the validated 3bc6e70 "good run") ----
    # CORRECTION (2026-06-24): the 96k June-13 run launched at commit 3bc6e70, which used
    # reward_tilt_velocity_weight=5.0 and turn_upright_gate_std=0.25 — the LOOSER upright
    # gate that tolerates the transient tilt of genuine finger-rolling.  The base defaults
    # (7 / 0.2) are STRICTER and bias the policy toward minimal-disturbance finger
    # "flicker" (handle still rotates, fingers don't properly roll it), so they are
    # overridden back to the good-run values here (Allegro-only; base + LinkerL20 keep
    # 7 / 0.2).  The shared base _get_rewards is byte-identical to the good run — only
    # these knobs + the curriculum below had regressed.
    reward_tilt_velocity_weight: float = 5.0
    turn_upright_gate_std: float = 0.25

    # ---- Screw load (Allegro-specific) ----
    # Coulomb screw load 0.02 N·m, ramped via screwdriver_load_scale in the curriculum.
    # NOTE: the 3bc6e70 good run had NO load (free, lightly-damped handle, rotation
    # damping 0.15); the load + the base damping 0.5 are deliberately KEPT here for
    # screwdriver realism (user recovery choice = "reward-shaping only").  If flicker
    # persists after this reward-shaping revert, the free handle (load→0, damping→0.15
    # Allegro-only) is the next lever.  Kept as an override so the shared base default
    # (0.045, LinkerL20) is untouched.
    screwdriver_load_torque: float = 0.02

    # episode_length_s must match curriculum_phases[0] for the initial stagger.
    episode_length_s: float = 20.0

    # ---- Curriculum (Allegro — restored to the validated 3bc6e70 "good run") ----
    # The 96k run used this exact schedule.  The KEY anti-flicker property is P0:
    # the contact gate is OFF (turn_reward_contact_distance=0.0 → the gate returns
    # all-ones), so turn reward = 30·fwd_vel·upright_gate is UNGATED.  The only way
    # to earn it is to actually rotate the handle, so the policy learns a genuine
    # rolling gait FIRST — before the raw-fingertip-speed motion gate (P1+) exists to
    # be hacked by flicking.  The later regression (P0 gate ON @0.10 + turn weight 120
    # + boundaries 0/40M/90M + proximal 0/3/5) let the policy hack the motion gate with
    # flicker from step 0; that is reverted here.  Boundaries 0/15M/60M; episodes
    # 20/40/60 s.  screwdriver_load_scale 0/0.5/1.0 ramps the (kept) screw load in.
    curriculum_phases: list[CurriculumPhaseCfg] = field(
        default_factory=lambda: [
            CurriculumPhaseCfg(
                # P0 "learn to rotate at all": contact gate OFF (UNGATED turn reward),
                # low turn weight, free handle.  This ungated bootstrap is what makes
                # the policy learn genuine rotation instead of flicker.
                step_start=0,
                reward_turn_weight=30.0,
                turn_reward_contact_distance=0.0,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.0,
                screwdriver_load_scale=0.0,
                reward_proximal_penalty_weight=0.0,
                near_reward_weight=0.8,
                episode_length_s=20.0,
                upright_termination_threshold=2.0,
            ),
            CurriculumPhaseCfg(
                # P1: turn the contact gate ON (0.10 m) + half screw load.
                step_start=15_000_000,
                reward_turn_weight=150.0,
                turn_reward_contact_distance=0.10,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.003,
                screwdriver_load_scale=0.5,
                reward_proximal_penalty_weight=0.0,
                near_reward_weight=0.3,
                episode_length_s=40.0,
                upright_termination_threshold=1.5,
            ),
            CurriculumPhaseCfg(
                # P2: full screw load, strict gate (0.05 m), strict upright.
                step_start=60_000_000,
                reward_turn_weight=200.0,
                turn_reward_contact_distance=0.05,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.003,
                screwdriver_load_scale=1.0,
                reward_proximal_penalty_weight=0.0,
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
