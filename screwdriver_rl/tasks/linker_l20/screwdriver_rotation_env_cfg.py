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
    # FOUR phases (was 3) so the final tightening is spread out: the old 3-phase
    # schedule slammed full load (0.6->1.0) + tight contact (0.045->0.032) + strict
    # pad (0.3->0.5) + strict upright (1.3->1.0) ALL at the P2 boundary, which
    # collapsed the policy (reward went negative for ~200 epochs and never cleanly
    # recovered).  Now each axis ramps gently, and full load (P2) arrives BEFORE the
    # strictest contact/upright/finger-count gates (P3).
    #
    # Key changes vs. the old schedule (which produced the idle-finger + "handle
    # spins by itself" policy):
    #   * screwdriver_load_scale 0.6/0.8/1.0/1.0 (P0 was 0.25): a MEANINGFUL screw
    #     load from step 0, so the free-spin exploit never establishes.  Combined
    #     with the joint-physics fix (base cfg: Coulomb 0.10 dominant, damping 0.12)
    #     the handle no longer spins under a standing squeeze.
    #   * turn_reward_min_contact_fingers 3/4/4/5 (was 3/3/3): with near_reward_top_k
    #     now 4 (above), every non-thumb finger earns proximity reward, and the turn
    #     gate progressively demands nearly all fingers in genuine contact — no more
    #     "clamp with 3, park 2 idle".  P0 stays 3 so the grip can bootstrap.
    #   * reward_contact_force_weight 0.3/0.7/0.8/0.8 and reward_finger_abandon_weight
    #     5/20/30/30: the anti-crush and anti-idle penalties RAMP IN (like the
    #     proximal penalty) so they don't strangle the initial grasp — full-strength
    #     from step 0 trapped the policy in a "hover, don't grip" optimum.
    #   * reward_contact_weight 0.6/0.3/0.15/0.1: dense contact-engagement reward,
    #     HIGH in P0 to bootstrap pressing.  The distance-based near-reward saturates
    #     at the surface, so the policy hovered there (ContactForce ~0) and never
    #     opened the contact gate; this rewards force up to the 2.5 N target (then
    #     flat) to bridge "hover" -> "press", and tapers as the turn reward takes over.
    #   * turn_reward_min_fingertip_speed 0/0.008/0.01/0.01: an absolute anti-noise
    #     floor under the rolling-consistency factor (the gate now credits handle
    #     spin only to the extent the fingertips actually drive it — see
    #     _compute_contact_gate / rewards.rolling_consistency).  P0 keeps 0 so the
    #     approach is unpenalised.  turn_reward_full_fingertip_speed is no longer
    #     used by the gate (the rolling factor self-normalises to the handle speed).
    #   * turn_reward_contact_distance 0.06->0.035 and pad/upright ramps unchanged in
    #     spirit but stretched over four phases.
    curriculum_phases: list[CurriculumPhaseCfg] = field(
        default_factory=lambda: [
            CurriculumPhaseCfg(
                # --- P0: reach & grasp; contact gate ON, real load from step 0 ---
                step_start=0,
                reward_turn_weight=120.0,
                turn_reward_contact_distance=0.06,
                # 3 in P0 (ramps 4 -> 4 -> 5 in later phases): keep the bar low so the
                # policy can FIRST discover a grip + turn; the abandon penalty
                # (ramped) + the min-fingers ramp engage the spare fingers later.
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
                # --- P1: require genuine rolling; engage a 4th finger ---
                step_start=40_000_000,
                reward_turn_weight=160.0,
                turn_reward_contact_distance=0.05,
                turn_reward_min_contact_fingers=4,
                turn_reward_min_fingertip_speed=0.008,
                turn_reward_full_fingertip_speed=0.04,
                pad_facing_cos_threshold=0.3,
                screwdriver_load_scale=0.8,
                reward_proximal_penalty_weight=2.0,
                reward_contact_force_weight=0.7,
                reward_finger_abandon_weight=20.0,
                reward_contact_weight=0.3,
                near_reward_weight=0.4,
                episode_length_s=45.0,
                upright_termination_threshold=1.4,
            ),
            CurriculumPhaseCfg(
                # --- P2: full screw load (but contact/upright not yet strictest) ---
                step_start=90_000_000,
                reward_turn_weight=180.0,
                turn_reward_contact_distance=0.04,
                turn_reward_min_contact_fingers=4,
                turn_reward_min_fingertip_speed=0.01,
                turn_reward_full_fingertip_speed=0.045,
                pad_facing_cos_threshold=0.4,
                screwdriver_load_scale=1.0,
                reward_proximal_penalty_weight=3.0,
                reward_contact_force_weight=0.8,
                reward_finger_abandon_weight=30.0,
                reward_contact_weight=0.15,
                near_reward_weight=0.25,
                episode_length_s=55.0,
                upright_termination_threshold=1.2,
            ),
            CurriculumPhaseCfg(
                # --- P3: final — strict pad + tight contact; demand 5-of-5 ---
                step_start=150_000_000,
                reward_turn_weight=200.0,
                turn_reward_contact_distance=0.035,
                turn_reward_min_contact_fingers=5,
                turn_reward_min_fingertip_speed=0.01,
                turn_reward_full_fingertip_speed=0.05,
                pad_facing_cos_threshold=0.5,
                screwdriver_load_scale=1.0,
                reward_proximal_penalty_weight=4.0,
                reward_contact_force_weight=0.8,
                reward_finger_abandon_weight=30.0,
                reward_contact_weight=0.1,
                near_reward_weight=0.15,
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
