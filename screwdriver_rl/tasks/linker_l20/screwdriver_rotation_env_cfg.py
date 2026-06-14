"""Configuration for Linker Hand L20 (Left) continuous screwdriver rotation.

All shared task config (curriculum, domain randomisation, reward weights,
screwdriver asset, simulation) lives in
:class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnvCfg`.  This module only
overrides the Linker-specific fields: the gym spaces, active fingers, pregrasp,
RMA dims, fingertip pad axis, and the Linker articulation.

Bring-up TODO (see the project plan):
  * ``fingertip_pad_axis_local`` is a PLACEHOLDER — recover the true value with
    ``calibrate_pad.py --task Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0``.
  * The hand pose (``init_state.pos/rot``) and ``pregrasp_positions`` are seeded
    from the MFR reference and must be tuned in play.py (zero action) so the five
    fingertip pads contact the handle with the screwdriver upright.
  * ``turn_direction`` is inherited as -1.0; confirm the sign for this LEFT hand
    (a natural grip may drive +z) and override here if the reverse/oscillation
    metrics show the grasp turns the other way.
  * The per-phase contact distances are inherited from the Allegro curriculum and
    may need re-deriving for the Linker tip geometry.
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


@configclass
class LinkerL20ScrewdriverRotationEnvCfg(ScrewdriverRotationEnvCfg):
    """Linker Hand L20 (left) continuous screwdriver rotation task.

    Observation space (35-D): [finger_q(16), cur_targets(16), euler(3)]
    Action space (16-D): HORA-style delta targets for the 16 independent finger
      DOFs (index/middle/ring/pinky × 3 + thumb × 4); 5 mimic distal joints
      follow via COUPLED_JOINTS.
    Privileged obs (19-D): euler(3)+angvel(3)+rel_pos(3)+quat(4)+friction(1)+tip_dist(5)
    """

    # ---- Gym spaces ----
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
    state_space = 0

    # ---- Active fingers (full five-finger grasp) ----
    fingers: tuple[str, ...] = ("index", "middle", "ring", "pinky", "thumb")

    # ---- Turn direction ----
    # The Linker is a LEFT hand (mirror of the right-handed Allegro), so the
    # natural grip drives the screwdriver the opposite way: +1 (vs Allegro -1).
    turn_direction: float = 1.0

    # ---- RMA dims (hand-specific) ----
    privileged_obs_dim: int = 19
    """3 euler + 3 angvel + 3 rel-pos + 4 quat + 1 friction + 5 fingertip-dist."""
    history_obs_dim: int = 32
    """[finger_q(16), cur_targets(16)] per frame."""

    # ---- Fingertip pad axis (PLACEHOLDER — calibrate before training) ----
    fingertip_pad_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Outward pad normal in the ``*_distal`` link local frame.  This is a
    PLACEHOLDER copied from the Allegro convention; the Linker distal frames are
    raw phalanx links (not rotated _ee tips), so this WILL differ.  Recover the
    true axis with calibrate_pad.py and replace this value."""

    # ---- Robot (Linker Hand L20, left) ----
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LinkerHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "linker_hand_l20/linkerhand_l20_left.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=True,
            make_instanceable=False,
            # Cheap per-link convex_hull collision (1 shape/link) + importer-level
            # self-collision.  Hull shapes are slightly inflated vs the real
            # geometry, so non-adjacent links phantom-overlap near the palm; those
            # pairs are excluded by SELF_COLLISION_FILTER_PAIRS in the env class.
            # (convex_decomposition gives a tighter fit but is far more expensive
            # in memory + compute; with pair-filtering it is unnecessary.)
            collider_type="convex_hull",
            self_collision=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # Self-collision ON for sim-to-real fidelity: the policy must not
                # learn finger configurations that pass through each other / the
                # palm.  Phantom hull overlaps that would otherwise destabilise
                # the light joints are filtered out (see SELF_COLLISION_FILTER_PAIRS).
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
            # Hand x kept at the reference 0.13: at this distance the screwdriver
            # stays upright (the four fingers sit ~0.06-0.07 m off the handle
            # axis, close enough for the phase-0 near-reward to pull them in,
            # without pushing the unopposed handle over).  See the measured
            # geometry notes in the project memory; finer opposition tuning of
            # the hand ORIENTATION is best done visually in play.py.
            pos=(0.13, -0.045, 1.36),
            rot=(0.5, -0.5, -0.5, 0.5),
            joint_pos={
                # index / middle / ring / pinky: roll, pitch, pip, dip(=0.8917*pip)
                "index_mcp_roll": 0.0, "index_mcp_pitch": 0.55, "index_pip": 0.9, "index_dip": 0.8025,
                "middle_mcp_roll": 0.0, "middle_mcp_pitch": 0.55, "middle_pip": 0.9, "middle_dip": 0.8025,
                "ring_mcp_roll": 0.0, "ring_mcp_pitch": 0.55, "ring_pip": 0.9, "ring_dip": 0.8025,
                "pinky_mcp_roll": 0.0, "pinky_mcp_pitch": 0.55, "pinky_pip": 0.9, "pinky_dip": 0.8025,
                # thumb: cmc_yaw, cmc_roll, cmc_pitch, mcp, ip(=1.1619*mcp).
                # cmc_pitch and mcp backed off (0.62->0.50, 0.65->0.50) so the
                # thumb is NOT jammed into the cap: this frees those joints (they
                # had near-zero authority while pressed against the cap) and
                # opens up room toward their upper limits (0.79 / 1.05).
                "thumb_cmc_yaw": 0.24, "thumb_cmc_roll": 0.6, "thumb_cmc_pitch": 0.5,
                "thumb_mcp": 0.5, "thumb_ip": 0.581,
            },
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

    # ---- Pregrasp joint positions (INDEPENDENT joints only, semantic order) ----
    # Followers (*_dip, thumb_ip) are set at reset from these via COUPLED_JOINTS.
    pregrasp_positions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "index":  (0.0, 0.55, 0.9),
            "middle": (0.0, 0.55, 0.9),
            "ring":   (0.0, 0.55, 0.9),
            "pinky":  (0.0, 0.55, 0.9),
            "thumb":  (0.24, 0.6, 0.5, 0.5),
        }
    )
