"""Configuration for Linker Hand L20 (Left) continuous screwdriver rotation.

All shared task config (curriculum, domain randomisation, reward weights,
screwdriver asset, simulation) lives in
:class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnvCfg`.  This module only
overrides the Linker-specific fields: the gym spaces, active fingers, pregrasp,
RMA dims, fingertip pad axis, and the Linker articulation.

Bring-up TODO (see the project plan):
  * ``fingertip_pad_axis_local`` is calibrated to ``(+1, 0, 0)`` via
    ``calibrate_pad.py --task Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0``;
    re-run it if the pregrasp changes.
  * The hand pose (``init_state.pos/rot``) and ``pregrasp_positions`` should be
    confirmed in play.py (zero action) so the five fingertip pads contact the
    handle with the screwdriver upright; re-render with ``render_posture.py``
    after any change.

This hand owns its own ``curriculum_phases`` (LinkerL20-tuned for the 5-finger,
non-uniform geometry — see the curriculum section below) and overrides
``turn_direction`` to +1.0 for the LEFT hand.
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
    CurriculumPhaseCfg,
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

    # ---- Near-reward finger count (override the 3-finger Allegro default) ----
    # The base default (2) only credits the 2 closest NON-thumb fingertips, which
    # is right for the 3-finger Allegro (2 non-thumb) but lets the 5-finger Linker
    # park its 2 "extra" non-thumb fingers idle at an extreme (they earn nothing
    # from proximity and a frozen finger costs no action/velocity penalty).  Credit
    # ALL FOUR non-thumb fingertips so every finger is incentivised to engage.
    near_reward_top_k: int = 4

    # ---- Curriculum (LinkerL20-tuned; see CurriculumPhaseCfg for field docs) ----
    # FOUR phases.  The phase COUNT is not what prevents collapse — continuity at
    # each boundary is.  The earlier 3-phase schedule stacked every tightening at
    # one boundary; spreading it over four + reordering (full screw load lands
    # BEFORE the strictest pad/finger/upright gates) helps, but run 16-08-11 still
    # collapsed at the FIRST boundary (P0->P1) because the contact gate's finger-
    # count test used to be a HARD step: raising turn_reward_min_contact_fingers
    # 3->4 instantly zeroed the turn reward on the working 3-finger grasp, so the
    # policy released into a free-spin basin with no gradient back (phase2.pth:
    # ContactCount~2, AvgRollSpeed 0, TurnRew 0).
    #
    # Two fixes, applied together:
    #   1) The count gate is now SOFT (base cfg `contact_count_soft_width`, see
    #      _compute_contact_gate): one finger short of the target keeps partial turn
    #      reward, so raising min_fingers DE-RATES the grasp instead of zeroing it.
    #   2) This schedule is gentle + continuous.  P0 is UNCHANGED (eval proved it
    #      reaches a genuine ~3-finger rolling grasp: NetTurns/ep +1.14, force in
    #      target, survives a 4x rotation-damping stress test).  P1 only NUDGES it —
    #      min_fingers stays 3 (recruit the 4th/5th via the ramped abandon penalty,
    #      which has a gradient, not via a gate-count step), load +0.15, a soft pad
    #      threshold 0.15, light proximal 0.5, abandon x2 (not x4), and the dense
    #      contact/near bootstrap reward only TAPERS (not halved the instant the
    #      penalties switch on).  min_fingers reaches 4 at P2 and 5 at P3, where the
    #      soft gate turns each into a smooth pull toward the next finger.
    #
    # Per-axis ramps (P0 / P1 / P2 / P3):
    #   screwdriver_load_scale          0.6 / 0.75 / 1.0  / 1.0   (full load at P2)
    #   turn_reward_min_contact_fingers 3   / 3    / 4    / 4     (soft-gated; capped at 4)
    #   pad_facing_cos_threshold        0.0 / 0.15 / 0.35 / 0.5
    #   turn_reward_min_fingertip_speed 0.0 / 0.004/ 0.008/ 0.01  (anti-noise floor)
    #   reward_proximal_penalty_weight  0.0 / 0.5  / 2.0  / 3.5
    #   reward_finger_abandon_weight    5   / 10   / 20   / 30    (drives recruitment)
    #   reward_contact_force_weight     0.3 / 0.5  / 0.7  / 0.8   (anti-crush)
    #   reward_contact_weight           0.6 / 0.45 / 0.25 / 0.12  (dense bootstrap)
    #   near_reward_weight              0.8 / 0.6  / 0.35 / 0.18
    #   reward_turn_weight              120 / 140  / 170  / 200
    #   turn_reward_contact_distance    0.07/ 0.07 / 0.07 / 0.07  (LOOSE+HELD; grasp sits 0.039-0.064)
    #   upright_termination_threshold   1.8 / 1.5  / 1.2  / 1.0
    #   episode_length_s                30  / 40   / 50   / 60
    curriculum_phases: list[CurriculumPhaseCfg] = field(
        default_factory=lambda: [
            CurriculumPhaseCfg(
                # --- P0: reach & grasp; contact gate ON, real load from step 0 ---
                step_start=0,
                reward_turn_weight=120.0,
                turn_reward_contact_distance=0.07,
                # 3 in P0 and P1 (ramps to 4 at P2, capped there): keep the bar low
                # while the policy discovers a grip + turn; the ramped abandon penalty
                # (not a gate-count step) recruits the spare finger, and the SOFT count
                # gate makes the 3->4 increase a smooth de-rating, not a cliff.
                turn_reward_min_contact_fingers=3,
                turn_reward_min_fingertip_speed=0.0,
                turn_reward_full_fingertip_speed=0.03,
                pad_facing_cos_threshold=0.0,
                screwdriver_load_scale=0.6,
                reward_proximal_penalty_weight=0.0,
                # Grip-force GENTLE in P0 (only bites above the 2.5 N target anyway):
                # ramps up later.  Abandon gentle too (a strong abandon penalty traps
                # the policy in a "hover, don't grip" optimum during early exploration).
                reward_contact_force_weight=0.3,
                reward_finger_abandon_weight=5.0,
                # Contact-engagement reward HIGH in P0: this is the key bootstrap
                # signal that bridges "hover at the surface" -> "press".  The
                # distance-based near-reward saturates at the surface, so without this
                # the policy parked hovering (ContactForce ~0) and never opened the
                # contact gate.  near_reward kept at 0.8 (its job is only the approach;
                # boosting it to 1.5 made the hover optimum MORE comfortable).
                reward_contact_weight=0.6,
                near_reward_weight=0.8,
                episode_length_s=30.0,
                upright_termination_threshold=1.8,
            ),
            CurriculumPhaseCfg(
                # --- P1: GENTLE nudge of the working P0 grasp (was the cliff) ---
                # min_fingers stays 3; recruit via the abandon penalty.  Every change
                # here is small so the +14.5k P0 grasp is perturbed, not destroyed.
                step_start=40_000_000,
                reward_turn_weight=140.0,
                turn_reward_contact_distance=0.07,
                turn_reward_min_contact_fingers=3,
                turn_reward_min_fingertip_speed=0.004,
                turn_reward_full_fingertip_speed=0.04,
                pad_facing_cos_threshold=0.15,
                screwdriver_load_scale=0.75,
                reward_proximal_penalty_weight=0.5,
                reward_contact_force_weight=0.5,
                reward_finger_abandon_weight=10.0,
                # Dense bootstrap reward only tapers (was halved 0.6->0.3 here, which
                # removed the gradient back to grasping just as penalties switched on).
                reward_contact_weight=0.45,
                near_reward_weight=0.6,
                episode_length_s=40.0,
                upright_termination_threshold=1.5,
            ),
            CurriculumPhaseCfg(
                # --- P2: full screw load; now demand the 4th finger (soft-gated) ---
                step_start=90_000_000,
                reward_turn_weight=170.0,
                # 0.07 (LOOSENED from the old 0.04).  diag_fingers.py showed the genuine
                # 4-finger grasp sits 0.039 (thumb) .. 0.064 (ring) from the axis, so a
                # tight gate EXCLUDES fingers that already grip (0.05->0.04 dropped the
                # count 3->1 and collapsed run 17-09-13).  0.07 counts all four; the
                # force + pad + rolling gates still enforce quality.
                turn_reward_contact_distance=0.07,
                turn_reward_min_contact_fingers=4,
                turn_reward_min_fingertip_speed=0.008,
                turn_reward_full_fingertip_speed=0.045,
                pad_facing_cos_threshold=0.35,
                screwdriver_load_scale=1.0,
                reward_proximal_penalty_weight=2.0,
                reward_contact_force_weight=0.7,
                reward_finger_abandon_weight=20.0,
                reward_contact_weight=0.25,
                near_reward_weight=0.35,
                episode_length_s=50.0,
                upright_termination_threshold=1.2,
            ),
            CurriculumPhaseCfg(
                # --- P3: final — strict pad; demand a 4th finger (capped at 4) ---
                # contact_distance 0.07 (loosened, same as P2 — counts the real grasp).
                # min_fingers capped at 4 (not 5): diag_fingers.py shows only the MIDDLE
                # finger truly parks (~0.106 m); the other four (thumb/index/ring/pinky)
                # grip with force.  Engaging middle too (->5) is a separate problem
                # (likely geometry: index already occupies that side of the handle).
                step_start=150_000_000,
                reward_turn_weight=200.0,
                turn_reward_contact_distance=0.07,
                turn_reward_min_contact_fingers=4,
                turn_reward_min_fingertip_speed=0.01,
                turn_reward_full_fingertip_speed=0.05,
                pad_facing_cos_threshold=0.5,
                screwdriver_load_scale=1.0,
                reward_proximal_penalty_weight=3.5,
                reward_contact_force_weight=0.8,
                reward_finger_abandon_weight=30.0,
                reward_contact_weight=0.12,
                near_reward_weight=0.18,
                episode_length_s=60.0,
                upright_termination_threshold=1.0,
            ),
        ]
    )

    # ---- RMA dims (hand-specific) ----
    privileged_obs_dim: int = 19
    """3 euler + 3 angvel + 3 rel-pos + 4 quat + 1 friction + 5 fingertip-dist."""
    history_obs_dim: int = 32
    """[finger_q(16), cur_targets(16)] per frame."""

    # ---- Fingertip pad axis (calibrated in sim with calibrate_pad.py) ----
    fingertip_pad_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Outward pad normal in the ``*_distal`` link local frame.  Calibrated with
    ``calibrate_pad.py`` at the zero-action pregrasp: ``+x`` is the axis whose
    pad-facing cosine is positive across all five fingers (index/middle/ring/pinky
    ~+0.79..+0.94, thumb ~+0.37), i.e. it points from each distal pad toward the
    handle.  ``-x`` faces away (all negative) and the old ``(0,0,1)`` placeholder
    pointed out the fingertip *end*, both of which made the pad-facing cosine
    meaningless.  NOTE: a single axis is shared by all fingers; the thumb_distal
    frame is rotated relative to the four fingers (it prefers ~``+z``, ~+0.93), so
    ``+x`` is the best whole-hand compromise and the thumb is mildly undervalued.
    Re-run calibrate_pad.py if the pregrasp changes; consider a per-finger axis if
    the thumb pad-facing proves limiting in training."""

    # ---- Robot (Linker Hand L20, left) ----
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LinkerHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "linker_hand_l20/linkerhand_l20_left.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=True,
            make_instanceable=False,
            # Required for the fingertip ContactSensor force gate (the env builds
            # a ContactSensor over the *_distal bodies; see use_contact_force_gate).
            activate_contact_sensors=True,
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
            # Hand x kept at the reference 0.13.  Measured pregrasp geometry (see
            # render_posture.py): the four non-thumb fingertips sit at x~=-0.08 and
            # the thumb at x~=+0.11, straddling the handle axis at x~=0 — an
            # intentionally OPEN grip the policy closes during phase 0.  Note x is
            # the thumb-vs-four-fingers opposition axis: translating the hand along
            # x just unbalances it (the screwdriver, on its free universal joint,
            # topples toward whichever side dominates), so DON'T retune reach via
            # this position.  Closing the grip so the four fingers reach with less
            # curl is a coupled finger+thumb re-tune best done visually in play.py.
            pos=(0.13, -0.03, 1.36),
            rot=(0.5, -0.5, -0.5, 0.5),
            joint_pos={
                # index / middle / ring / pinky: roll, pitch, pip, dip(=0.8917*pip)
                "index_mcp_roll": 0.0, "index_mcp_pitch": 0.85, "index_pip": 0.9, "index_dip": 0.8025,
                "middle_mcp_roll": 0.0, "middle_mcp_pitch": 0.8, "middle_pip": 0.9, "middle_dip": 0.8025,
                "ring_mcp_roll": 0.0, "ring_mcp_pitch": 0.8, "ring_pip": 0.9, "ring_dip": 0.8025,
                "pinky_mcp_roll": 0.0, "pinky_mcp_pitch": 0.8, "pinky_pip": 0.9, "pinky_dip": 0.8025,
                # thumb: cmc_yaw, cmc_roll, cmc_pitch, mcp, ip(=1.1619*mcp).
                # cmc_pitch and mcp backed off (0.62->0.50, 0.65->0.50) so the
                # thumb is NOT jammed into the cap: this frees those joints (they
                # had near-zero authority while pressed against the cap) and
                # opens up room toward their upper limits (0.79 / 1.05).
                "thumb_cmc_yaw": 0.24, "thumb_cmc_roll": 0.75, "thumb_cmc_pitch": 0.6,
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
    # Must stay in sync with init_state.joint_pos above: the (roll, pitch, pip)
    # pip here drives the *_dip follower at reset via COUPLED_JOINTS.
    pregrasp_positions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "index":  (0.0, 0.85, 0.9),
            "middle": (0.0, 0.8, 0.9),
            "ring":   (0.0, 0.8, 0.9),
            "pinky":  (0.0, 0.8, 0.9),
            "thumb":  (0.24, 0.75, 0.6, 0.5),
        }
    )
