"""Configuration for Linker Hand L20 (Left) continuous screwdriver rotation.

Reworked from scratch.  The *shared* asset/sim plumbing still comes from
:class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnvCfg`, but this hand now
owns its reward design, curriculum, screwdriver physics, and contact model; the
matching :class:`LinkerL20ScrewdriverRotationEnv` overrides the corresponding base
methods so the Allegro task is left untouched.

Design goals
------------
* **No free-spin.**  The screwdriver carries its full rotational Coulomb load
  (breakaway "stiction") from the very first step — there is no zero-load warm-up
  phase — and the rotation bearing is strongly damped, so the handle only turns
  while a finger is actively driving it.
* **Contact judged by force, not geometry.**  Per-fingertip ``ContactSensor``s
  report the force each finger applies *to the screwdriver* (and to the cap
  specifically); a trapezoidal force window defines "good" contact (too soft and
  too hard both fail).  No distance gate, no pad-facing gate.
* **Prescribed-lite finger roles.**  The index holds the cap down (axial
  stabilise) while the thumb + middle/ring/pinky apply tangential turning force;
  an anti-idle term keeps every finger contributing.
* **Stay near the working grip.**  The initial posture is already a valid grasp,
  so each finger DOF is clamped to a small window around its home value and a soft
  deviation penalty discourages large, flailing motions.

The initial pose and pregrasp are intentionally LEFT UNCHANGED from the validated
five-contact grasp (``tools/render_linker_posture.py`` and the post_render grid).
"""

from __future__ import annotations

import json
from dataclasses import field
from pathlib import Path

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
from screwdriver_rl.utils.variants import seed_pregrasp_buckets


# Fraction of the radial flexion delta applied to each joint of a finger's
# pregrasp tuple (semantic order; abduction joint gets 0, flexion joints share
# it).  Used by ``utils.variants.seed_pregrasp_buckets`` (see plan §3d): the
# index finger stabilises the cap, so handle length is not grasp-neutral and
# postures are bucketed over both diameter and length.
_FLEX_WEIGHTS = {
    "index":  (0.0, 0.5, 0.5),
    "middle": (0.0, 0.5, 0.5),
    "ring":   (0.0, 0.5, 0.5),
    "pinky":  (0.0, 0.5, 0.5),
    "thumb":  (0.0, 0.5, 0.5, 0.0),
}


# ---------------------------------------------------------------------------
# Curriculum phase config (LinkerL20-specific)
# ---------------------------------------------------------------------------

@configclass
class LinkerCurriculumPhaseCfg:
    """Reward weights / gates for one LinkerL20 curriculum phase.

    The env selects the phase whose ``step_start`` is the largest value that does
    not exceed the global step counter.  The field names shared with the base
    ``CurriculumPhaseCfg`` (``step_start``, ``reward_turn_weight``,
    ``screwdriver_load_scale``, ``episode_length_s``,
    ``upright_termination_threshold``) are kept so ``play.py`` / ``eval.py``
    phase-pinning keeps working unchanged.
    """

    step_start: int = 0
    """Global step at which this phase activates."""

    reward_turn_weight: float = 150.0
    """Forward shaft-spin reward weight (rad/s x weight per policy step)."""

    screwdriver_load_scale: float = 1.0
    """Multiplier on ``cfg.screwdriver_load_torque`` for this phase.  PINNED to 1.0
    in EVERY phase — the handle never free-spins, not even in Phase 0.  This is the
    core of the redesign and the invariant ``tests/test_linker_cfg.py`` guards."""

    min_drive_fingers: float = 2.0
    """Soft target for how many non-index ("drive") fingers must be in the force
    window for the turn gate to fully open (see ``rewards.soft_count_gate``)."""

    # ---- Per-phase reward weights ----
    w_grip: float = 1.5
    """Dense "establish good all-finger contact" weight.  High early (the main
    positive signal before the handle turns), tapers in later phases."""

    w_index_cap: float = 0.5
    """Reward weight for the index holding the cap pressed (axial stabilise)."""

    w_drive: float = 0.5
    """Reward weight for each drive finger pressing AND moving tangentially
    (genuine turning work)."""

    w_excess: float = 0.5
    """Penalty weight on contact force above the safe ceiling (crush -> free-spin /
    flickering fingers)."""

    w_wrong: float = 1.0
    """Penalty weight on contact force on NON-fingertip links (palm/knuckle/back)."""

    w_idle: float = 0.3
    """Penalty weight per finger not touching the screwdriver (anti-hang)."""

    # ---- Episode / termination ----
    episode_length_s: float = 25.0
    """Episode length for this phase.  Short early; longer later so the policy can
    accumulate many turns."""

    upright_termination_threshold: float = 1.5
    """Tilt norm (rad) above which the episode terminates.  Lenient early, strict
    in the final phase."""


@configclass
class LinkerL20ScrewdriverRotationEnvCfg(ScrewdriverRotationEnvCfg):
    """Linker Hand L20 (left) continuous screwdriver rotation task.

    Observation space (51-D, latent-conditioned/deployable):
      [finger_q(16), cur_targets(16), privileged(19)].  A custom rl_games network
      encodes the privileged tail into a low-D latent; the raw euler is no longer
      a standalone actor input (it lives at privileged[0:3]).
    Action space (16-D): HORA-style delta targets for the 16 independent finger
      DOFs (index/middle/ring/pinky x 3 + thumb x 4); 5 mimic distal joints follow
      via COUPLED_JOINTS.
    Privileged obs (19-D): euler(3)+angvel(3)+rel_pos(3)+quat(4)+friction(1)+
      per_finger_contact_force(5).
    """

    # ---- Gym spaces ----
    # HORA-faithful deployable mode: the actor obs is [finger_q(16),
    # cur_targets(16), privileged(19)] = 51-D (the raw euler is no longer a
    # standalone obs; it lives inside the privileged tail the network encodes).
    # The shape is finalised in __post_init__ once privileged_obs_dim is known
    # (it bumps +2 under geometry DR).  Box(51) is the default (no geometry DR).
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(51,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
    state_space = 0

    # HORA-faithful latent-conditioned actor (deployable). See base cfg.
    latent_conditioned: bool = True

    # ---- Active fingers (full five-finger grasp) ----
    fingers: tuple[str, ...] = ("index", "middle", "ring", "pinky", "thumb")

    # ---- Turn direction ----
    # The Linker is a LEFT hand (mirror of the right-handed Allegro), so the
    # natural grip drives the screwdriver the opposite way: +1 (vs Allegro -1).
    turn_direction: float = 1.0

    # ---- Contact model (force-based; no distance / pad-facing gate) ----
    # The pad-facing gate is disabled outright: contact is judged purely by the
    # per-fingertip ContactSensor force (see the env class).  fingertip_pad_axis_local
    # is still set because the base __init__ builds a tensor from it, but it is
    # never consulted by the reward.
    require_pad_facing: bool = False

    # Trapezoidal "good contact pressure" window (Newtons) shared by the grip,
    # index-cap and drive force scores.  Below f_min = not touching; above f_max =
    # crushing.  FIRST-CUT values — calibrate from the logged per-finger forces at
    # the resting pregrasp (see docs/verification).
    contact_f_min: float = 0.1
    contact_f_lo: float = 0.5
    contact_f_hi: float = 4.0
    contact_f_max: float = 8.0

    # Drive-finger turning: tangential fingertip speed (m/s) at which the
    # turning-work factor saturates.  ~handle_radius (0.02 m) x target spin
    # (~1 rad/s) => ~0.02 m/s.
    drive_full_tangential_speed: float = 0.02

    # ---- Joint-range restriction (stay near the working grip) ----
    # Each finger DOF is hard-clamped to home +/- joint_motion_range (the env
    # tightens the target-clamp bounds), and a soft quadratic penalty discourages
    # drifting toward those bounds.  The window must stay wide enough for the drive
    # fingers to roll the 0.02 m handle.
    joint_motion_range: float = 0.35
    """Symmetric half-width (rad) of the per-DOF motion window around the home
    (pregrasp) value."""
    joint_motion_range_overrides: dict[str, float] = field(default_factory=dict)
    """Optional per-joint override of ``joint_motion_range`` (keyed by joint name)."""
    home_deviation_deadband: float = 0.1
    """Free play (rad) per joint before the stay-home penalty starts."""
    w_home_dev: float = 2.0
    """Weight on the quadratic stay-home deviation penalty."""

    # ---- Curriculum (3 phases; load scale PINNED to 1.0 throughout) ----
    curriculum_phases: list[LinkerCurriculumPhaseCfg] = field(
        default_factory=lambda: [
            LinkerCurriculumPhaseCfg(
                # --- P0: establish the five-finger grip; gentle turning ---
                step_start=0,
                reward_turn_weight=120.0,
                screwdriver_load_scale=1.0,
                min_drive_fingers=2.0,
                w_grip=1.5,
                w_index_cap=0.5,
                w_drive=0.5,
                w_excess=0.5,
                w_wrong=1.0,
                w_idle=0.3,
                episode_length_s=25.0,
                upright_termination_threshold=1.5,
            ),
            LinkerCurriculumPhaseCfg(
                # --- P1: steady multi-finger rotation; tighten contact quality ---
                step_start=40_000_000,
                reward_turn_weight=170.0,
                screwdriver_load_scale=1.0,
                min_drive_fingers=3.0,
                w_grip=0.6,
                w_index_cap=1.0,
                w_drive=1.0,
                w_excess=1.0,
                w_wrong=2.0,
                w_idle=0.5,
                episode_length_s=45.0,
                upright_termination_threshold=1.2,
            ),
            LinkerCurriculumPhaseCfg(
                # --- P2: refined steady rotation; strict upright + anti-crush ---
                step_start=90_000_000,
                reward_turn_weight=200.0,
                screwdriver_load_scale=1.0,
                min_drive_fingers=3.0,
                w_grip=0.3,
                w_index_cap=1.2,
                w_drive=1.2,
                w_excess=1.5,
                w_wrong=3.0,
                w_idle=0.6,
                episode_length_s=60.0,
                upright_termination_threshold=1.0,
            ),
        ]
    )

    # episode_length_s must match curriculum_phases[0] for the initial setup /
    # episode-start stagger (the env updates it at each curriculum transition).
    episode_length_s: float = 25.0

    # ---- RMA dims (hand-specific) ----
    privileged_obs_dim: int = 19
    """3 euler + 3 angvel + 3 rel-pos + 4 quat + 1 friction + 5 per-finger force.
    Bumped to 21 in ``__post_init__`` when ``domain_rand.randomize_geometry`` is
    on (+2 channels: handle diameter scale + length scale)."""
    history_obs_dim: int = 32
    """[finger_q(16), cur_targets(16)] per frame."""

    # ---- Fingertip pad axis (kept for base __init__; unused by the reward) ----
    fingertip_pad_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)

    # ---- Screwdriver physics override (isolate from Allegro/base) ----
    # Same asset and mount as the base, but the tilt joints are more strongly
    # damped to kill wobble/oscillation (NO restoring spring — the policy must
    # actively keep the screwdriver upright), and the rotation bearing keeps its
    # strong damping so the handle stops the instant a finger stops driving it.
    # Combined with the full Coulomb load (applied from step 0), the handle cannot
    # free-spin or coast for reward.
    screwdriver_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Screwdriver",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "screwdriver/screwdriver_isaaclab.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=False,
            make_instanceable=False,
            # The screwdriver must report contact forces for the per-fingertip
            # filtered ContactSensors (the sensor filters distal->screwdriver).
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Unchanged from the shared base mount (keeps the validated grasp
            # geometry): -0.009 m x shift, upright, joints zeroed.
            pos=(-0.009, 0.0, 1.205),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "tilt": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_1", "table_screwdriver_joint_2"],
                stiffness=0.0,
                # Raised 0.003 -> 0.05: damps the tilt DOFs so the screwdriver
                # cannot rock/oscillate freely, WITHOUT a restoring spring — the
                # hand still has to actively keep it upright (it can still be
                # knocked over by a bad push, which the reward/termination punish).
                damping=0.05,
            ),
            "rotation": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_3"],
                stiffness=0.0,
                # Strong bearing damping: the velocity time-constant tau = I/c is
                # far shorter than one physics step, so the handle stops the
                # instant the finger leaves — it cannot coast forward for reward.
                damping=0.5,
            ),
            "cap": ImplicitActuatorCfg(
                joint_names_expr=["screwdriver_body_cap_joint"],
                stiffness=0.0,
                # Light damping couples the free-spinning cap to the body so it
                # cannot keep rotating on its own bearing.
                damping=0.05,
            ),
        },
    )

    # Full Coulomb breakaway load (N.m), applied at scale 1.0 from step 0 (see
    # curriculum).  Breakaway finger force ~= load / handle_radius = 0.045 / 0.02
    # ~= 2.25 N, comfortably inside the contact force window.
    screwdriver_load_torque: float = 0.045

    # ---- Robot (Linker Hand L20, left) — POSTURE UNCHANGED ----
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LinkerHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "linker_hand_l20/linkerhand_l20_left.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=True,
            make_instanceable=False,
            # Required for the fingertip ContactSensor force gate (the env builds a
            # ContactSensor per *_distal body; see use_contact_force_gate).
            activate_contact_sensors=True,
            # Cheap per-link convex_hull collision (1 shape/link) + importer-level
            # self-collision.  Hull shapes are slightly inflated vs the real
            # geometry, so non-adjacent links phantom-overlap near the palm; those
            # pairs are excluded by SELF_COLLISION_FILTER_PAIRS in the env class.
            collider_type="convex_hull",
            self_collision=True,
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
            # Five-contact full-mesh grasp generated by
            # tools/render_linker_posture.py.  The wrist is translated farther
            # inward from the previous clean posture and slightly rolled back so
            # the palm sits closer while the index terminal/front pad still
            # presses near the cap center.  Middle/ring/pinky stack down one
            # handle side and the thumb opposes them.  Only *_distal meshes
            # touch the screwdriver.
            pos=(0.15107654, -0.06682857, 1.32253946),
            rot=(0.44578500, -0.47244983, -0.22820989, 0.72524971),
            joint_pos={
                # index / middle / ring / pinky: roll, pitch, pip, dip(=0.8917*pip)
                "index_mcp_roll": 0.130000, "index_mcp_pitch": 0.179608,
                "index_pip": 1.371321, "index_dip": 1.222807,
                "middle_mcp_roll": -0.130000, "middle_mcp_pitch": 0.205776,
                "middle_pip": 1.351135, "middle_dip": 1.204807,
                "ring_mcp_roll": -0.057778, "ring_mcp_pitch": 0.462712,
                "ring_pip": 1.204709, "ring_dip": 1.074239,
                "pinky_mcp_roll": 0.078981, "pinky_mcp_pitch": 0.665494,
                "pinky_pip": 1.024866, "pinky_dip": 0.913873,
                # All joints retain at least 0.04 rad of URDF-limit margin.
                "thumb_cmc_yaw": 0.672749, "thumb_cmc_roll": 1.180000,
                "thumb_cmc_pitch": 0.040000, "thumb_mcp": 0.941868,
                "thumb_ip": 1.094356,
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
    # Must stay in sync with init_state.joint_pos above.
    pregrasp_positions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "index":  (0.130000, 0.179608, 1.371321),
            "middle": (-0.130000, 0.205776, 1.351135),
            "ring":   (-0.057778, 0.462712, 1.204709),
            "pinky":  (0.078981, 0.665494, 1.024866),
            "thumb":  (0.672749, 1.180000, 0.040000, 0.941868),
        }
    )

    def __post_init__(self) -> None:
        # Base wires the geometry MultiAssetSpawner + replicate_physics when
        # randomize_geometry is on; then we add the LinkerL20-specific pieces:
        # the +2 privileged-obs channels and the per-(d,L)-bucket pregrasp table.
        super().__post_init__()
        if self.domain_rand.randomize_geometry:
            self.privileged_obs_dim += 2  # +diameter scale, +length scale → 21
            manifest_path = Path(self.screwdriver_variants_dir) / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.pregrasp_positions_buckets = seed_pregrasp_buckets(
                self.pregrasp_positions, manifest, _FLEX_WEIGHTS
            )

        # Finalise the actor obs space for the latent-conditioned (deployable)
        # mode: [proprio(history_obs_dim) , privileged(privileged_obs_dim)].
        # history_obs_dim = 2 * n_finger_dofs = 32; privileged_obs_dim is 19
        # (or 21 under geometry DR, bumped just above).
        if self.latent_conditioned:
            obs_dim = self.history_obs_dim + self.privileged_obs_dim
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
