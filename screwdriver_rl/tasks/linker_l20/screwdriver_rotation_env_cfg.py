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
# pregrasp tuple.  Used by ``utils.variants.seed_pregrasp_buckets`` (see plan
# §3d).  The index finger stabilises the cap top, so it should NOT follow handle
# diameter; handle length is handled by a per-bucket fixed-base root z-offset in
# ``__post_init__`` below.
_FLEX_WEIGHTS = {
    "index":  (0.0, 0.0, 0.0),
    "middle": (0.0, 0.5, 0.5),
    "ring":   (0.0, 0.5, 0.5),
    "pinky":  (0.0, 0.5, 0.5),
    # The refined palm-closer posture already puts thumb_cmc_roll near its upper
    # guard margin and thumb_cmc_pitch near its lower guard margin.  Do not use
    # those two joints for diameter bucket compensation; put the small reach bias
    # into yaw + MCP where the URDF limits still have useful authority.
    "thumb":  (0.2, 0.0, 0.0, 0.8),
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

    action_scale_multiplier: float = 1.0
    """Multiplier on the normal accumulated-delta action during this phase.

    A value below one is an early curriculum aid for remapping an existing gait
    to a changed grasp without immediately saturating the target integrator.
    Production phases remain at 1.0, preserving the deployment control contract.
    """

    dynamics_randomization_scale: float = 1.0
    """Interpolation from nominal dynamics (0) to the configured DR ranges (1).

    Geometry remains fully randomized throughout.  Only reset-time dynamics
    ranges widen with the curriculum so the teacher first learns contact control
    before being exposed to the full deployment robustness distribution.
    """

    min_drive_fingers: float = 2.0
    """Minimum non-index ("drive") fingers required by the distance gate.

    This is a hard authorization threshold for turn reward/progress. A smooth
    distance-contact score still scales reward after the threshold is satisfied.
    """

    # ---- Per-phase reward weights ----
    w_grip: float = 1.5
    """Dense all-finger distance-contact weight. High early (the main positive
    signal before the handle turns), then tapered in later phases."""

    w_index_cap: float = 0.5
    """Reward weight for geometric index contact (axial stabilisation role)."""

    w_drive: float = 0.5
    """Reward weight for each drive finger in distance contact and moving
    tangentially (genuine turning work)."""

    w_contact_authority: float = 8.0
    """Reward for maintaining the active task's sustained contact contract.

    This is a shaping term only: shaft progress and turn reward remain guarded by
    the hard sustained gate. It makes the validated contact grasp preferable to
    the otherwise attractive upright-but-no-contact local optimum.
    """

    reward_fall_weight: float = 5_000.0
    """One-shot fall penalty for this curriculum phase.

    Early training must be allowed to explore contact transitions; the final
    phase still uses the full deployment-grade penalty.
    """

    w_excess: float = 0.5
    """Penalty weight on commanded-target penetration beyond measured motion.

    This force-free squeeze-intent proxy replaces the former force ceiling.
    """

    w_wrong: float = 1.0
    """Penalty weight on the binary non-fingertip contact safety predicate."""

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

    Observation space (52-D, latent-conditioned/deployable):
      [finger_q(16), cur_targets(16), privileged(20)].  A custom rl_games network
      encodes the privileged tail into a low-D latent; the raw euler is no longer
      a standalone actor input (it lives at privileged[0:3]).
    Action space (16-D): HORA-style delta targets for the 16 independent finger
      DOFs (index/middle/ring/pinky x 3 + thumb x 4); 5 mimic distal joints follow
      via COUPLED_JOINTS.
    Privileged obs (20-D): euler(3)+angvel(3)+rel_pos(3)+quat(4)+load_proxy(1)+
      contact_friction(1)+per_finger_distance_contact_score(5). Geometry DR
      appends diameter/length scales for the final 22-D top-down contract.
    """

    observation_semantics_version: str = "linker-l20-force-free-reset-dr-v1"

    # ---- Gym spaces ----
    # HORA-faithful deployable mode: the actor obs is [finger_q(16),
    # cur_targets(16), privileged(20)] = 52-D (the raw euler is no longer a
    # standalone obs; it lives inside the privileged tail the network encodes).
    # The shape is finalised in __post_init__ once privileged_obs_dim is known
    # (it bumps +2 under geometry DR).  Box(52) is the default (no geometry DR).
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
    state_space = 0

    # HORA-faithful latent-conditioned actor (deployable). See base cfg.
    latent_conditioned: bool = True

    # ---- Active fingers ----
    fingers: tuple[str, ...] = ("index", "middle", "ring", "pinky", "thumb")
    role_neutral_fingertip_contact: bool = False
    """Count any fingertip contact with any screwdriver part for turn authority.

    The lateral task keeps its historical index-cap/drive-body roles. Palm-down
    top-down variants enable this flag because those side-grasp roles do not
    describe their contact geometry.
    """
    role_neutral_min_contact_fingers: int = 3
    """Minimum simultaneous fingertip contacts in role-neutral tasks."""

    # ---- Turn direction ----
    # The Linker is a LEFT hand (mirror of the right-handed Allegro), so the
    # natural grip drives the screwdriver the opposite way: +1 (vs Allegro -1).
    turn_direction: float = 1.0

    # ---- Force-free contact model ----
    # Contact authority and shaping use fingertip-to-handle-axis distance only.
    # Contact sensors remain enabled for diagnostics and the binary wrong-surface
    # safety predicate, never as reward magnitudes or privileged observations.
    require_pad_facing: bool = False

    contact_d_margin: float = 0.008
    """Surface-clearance threshold (m) for binary kinematic contact."""
    contact_d_far_margin: float = 0.020
    """Surface-clearance outer edge (m) of the linear distance-score ramp."""
    contact_d_margin_by_finger: dict[str, float] = field(default_factory=dict)
    """Optional calibrated per-finger contact thresholds.

    An override shifts both distance-window endpoints by the same amount relative
    to contact_d_margin. This compensates fixed distal-body frame/pad offsets
    while preserving the shared ramp width.
    """
    pen_deadband: float = 0.06
    """Free target-tracking error (rad) before the squeeze-intent proxy activates.

    Calibrated for a zero-tension episode start (``reset_zero_tension_targets``):
    free-motion tracking lag stays well inside the band, so only deliberate
    squeezing past the surface is charged.  Without the zero-tension snap the
    validated hold itself sits ~0.19 rad beyond this band and is permanently
    taxed — do not lower this band without re-checking that interaction.
    """
    target_penetration_scale: float = 10.0
    """Scale from summed rad² target penetration to reward units.

    At 10, an exploratory squeeze (~0.1 rad past the deadband on three joints)
    costs ~0.05-0.5/step across the curriculum — cheap enough for PPO to
    discover active turning — while a crushing 0.3 rad five-finger squeeze
    costs ~5-13/step.  The original 100 priced exploratory squeezes at ~5/step,
    which blocked the gradient path to active rotation entirely (the 20260728
    frozen-finger/creep-exploit run).
    """
    wrong_surface_force_threshold: float = 1.0e-3
    """Diagnostic force floor (N) used only as a binary wrong-surface predicate."""

    # Legacy force-window values remain available to calibration/audit tooling.
    # They are not read by the reward or privileged-observation paths.
    contact_f_min: float = 0.1
    contact_f_lo: float = 0.5
    contact_f_hi: float = 4.0
    contact_f_max: float = 8.0

    # Turn progress is authorized only while the prescribed contact roles are
    # physically present for several consecutive policy steps.  At 10 Hz, three
    # steps reject single-frame contact flicker while adding only 0.2 s before
    # the first authorized transition (steps 1 and 2 prime the streak).
    turn_contact_hold_steps: int = 3
    """Consecutive policy steps required before turn reward/progress is credited."""
    turn_require_index_cap: bool = True
    """Require the index stabilizer to press the cap as well as drive fingers."""

    # ---- Co-motion authorization (anti-coasting / anti-creep) ----
    # Handle rotation only pays while enough contacting fingertips move *with*
    # the handle surface.  Distance contact alone cannot distinguish an active
    # drive from solver creep under motionless fingers (the 20260728 run froze
    # the hand and collected turn reward from creep); the surface-co-motion
    # projection can.  See ``rewards.surface_co_motion``.
    turn_motion_authorized: bool = False
    """Multiply turn reward, milestone and qualified-progress metrics by the
    co-motion authorization gate.  Off by default; the top-down task enables it."""
    turn_motion_min_fingers: float = 2.0
    """Contacting, co-moving fingertips needed for full motion authorization
    (soft count: partial credit below, saturates at this many)."""
    turn_motion_surface_speed_floor: float = 0.002
    """Local surface speed (m/s) at or below which the handle counts as static
    and the gate stays open — starting a stationary handle must not be vetoed."""
    reset_action_hold_steps: int = 5
    """Policy steps after reset during which the validated grasp is held."""
    reset_action_ramp_steps: int = 10
    """Policy steps used to ramp actions from zero to their commanded value."""
    absolute_action_targets: bool = False
    """Map actions directly to home-relative joint targets instead of integrating
    delta targets. Disabled by default for checkpoint compatibility."""

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

    w_target_bound: float = 0.0
    """Penalty on accumulated joint targets entering the outer 20% of their
    available action band. Disabled by default; used for auditable A/B runs
    that test whether target saturation prevents cyclic finger gaiting."""

    reverse_to_turn_ratio: float = 1.0
    """Multiplier on the active turn weight when reverse matching is enabled."""
    turn_reward_power: float = 1.0
    """Exponent applied symmetrically to forward and reverse angular speed.
    One preserves the legacy linear, target-free progress objective."""

    # A tilt termination must be costly in the same transition that receives
    # the final turn reward. The final phase is 600 policy steps and a viable
    # policy earns roughly 100 shaped reward per full episode. With the PPO
    # reward-shaper scale of 0.005, 30,000 raw units price one drop at -150:
    # enough to make a mid/late-episode fall unprofitable, unlike the rejected
    # 5,000 (-25) corrective branch.
    reward_fall_weight: float = 30000.0
    """One-shot penalty when tilt crosses the active phase termination limit."""

    # ---- Curriculum (3 phases; load scale PINNED to 1.0 throughout) ----
    curriculum_phases: list[LinkerCurriculumPhaseCfg] = field(
        default_factory=lambda: [
            LinkerCurriculumPhaseCfg(
                # --- P0: establish a stable grip; gentle turning ---
                step_start=0,
                dynamics_randomization_scale=0.25,
                reward_turn_weight=120.0,
                screwdriver_load_scale=1.0,
                min_drive_fingers=2.0,
                w_grip=1.5,
                w_index_cap=0.5,
                w_drive=0.5,
                w_contact_authority=8.0,
                reward_fall_weight=5_000.0,
                w_excess=0.5,
                w_wrong=1.0,
                w_idle=0.3,
                episode_length_s=25.0,
                upright_termination_threshold=1.5,
            ),
            LinkerCurriculumPhaseCfg(
                # --- P1: steady multi-finger rotation; tighten contact quality ---
                step_start=40_000_000,
                dynamics_randomization_scale=0.60,
                reward_turn_weight=170.0,
                screwdriver_load_scale=1.0,
                min_drive_fingers=3.0,
                w_grip=0.6,
                w_index_cap=1.0,
                w_drive=1.0,
                w_contact_authority=5.0,
                reward_fall_weight=15_000.0,
                w_excess=1.0,
                w_wrong=2.0,
                w_idle=0.5,
                episode_length_s=45.0,
                upright_termination_threshold=1.2,
            ),
            LinkerCurriculumPhaseCfg(
                # --- P2: refined steady rotation; strict upright + anti-crush ---
                step_start=90_000_000,
                dynamics_randomization_scale=1.0,
                reward_turn_weight=200.0,
                screwdriver_load_scale=1.0,
                min_drive_fingers=3.0,
                w_grip=0.3,
                w_index_cap=1.2,
                w_drive=1.2,
                w_contact_authority=3.0,
                reward_fall_weight=30_000.0,
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
    privileged_obs_dim: int = 20
    """3 euler + 3 angvel + 3 rel-pos + 4 quat + 1 friction + 5 distance scores.
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
            # Hardware-safe legacy-task seed. The original lateral posture used
            # PIP values above the real 1.08-rad limit. These conservative values
            # retain >=0.10 rad margin (including the thin-handle flex offset);
            # baseline/DR policies still require their own posture search and
            # retraining before release.
            pos=(0.15107654, -0.06682857, 1.32253946),
            rot=(0.44578500, -0.47244983, -0.22820989, 0.72524971),
            joint_pos={
                # index / middle / ring / pinky: roll, pitch, pip, dip(=0.8917*pip)
                "index_mcp_roll": 0.070000, "index_mcp_pitch": 0.140535,
                "index_pip": 0.970000, "index_dip": 0.864949,
                "middle_mcp_roll": -0.070000, "middle_mcp_pitch": 0.184823,
                "middle_pip": 0.930000, "middle_dip": 0.829281,
                "ring_mcp_roll": -0.070000, "ring_mcp_pitch": 0.449762,
                "ring_pip": 0.930000, "ring_dip": 0.829281,
                "pinky_mcp_roll": 0.045669, "pinky_mcp_pitch": 0.654302,
                "pinky_pip": 0.930000, "pinky_dip": 0.829281,
                "thumb_cmc_yaw": 0.673745, "thumb_cmc_roll": 1.120000,
                "thumb_cmc_pitch": 0.100000, "thumb_mcp": 0.876434,
                "thumb_ip": 1.018329,
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
            "index":  (0.070000, 0.140535, 0.970000),
            "middle": (-0.070000, 0.184823, 0.930000),
            "ring":   (-0.070000, 0.449762, 0.930000),
            "pinky":  (0.045669, 0.654302, 0.930000),
            "thumb":  (0.673745, 1.120000, 0.100000, 0.876434),
        }
    )

    def __post_init__(self) -> None:
        # Base wires the geometry MultiAssetSpawner + replicate_physics when
        # randomize_geometry is on; then we add the LinkerL20-specific pieces:
        # the +2 privileged-obs channels and the per-(d,L)-bucket pregrasp table.
        if self.contact_d_far_margin <= self.contact_d_margin:
            raise ValueError(
                "contact_d_far_margin must be greater than contact_d_margin"
            )
        if self.pen_deadband < 0.0 or self.target_penetration_scale < 0.0:
            raise ValueError(
                "force-free penetration proxy parameters must be non-negative"
            )
        if any(value < 0.0 for value in self.contact_d_margin_by_finger.values()):
            raise ValueError(
                "per-finger contact distance margins must be non-negative"
            )
        super().__post_init__()
        if self.domain_rand.randomize_geometry:
            self.privileged_obs_dim += 2  # +diameter scale, +length scale → 22
            manifest_path = Path(self.screwdriver_variants_dir) / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.pregrasp_positions_buckets = seed_pregrasp_buckets(
                self.pregrasp_positions, manifest, _FLEX_WEIGHTS
            )
            base_length = float(manifest["base"]["length"])
            root_offsets = [(0.0, 0.0, 0.0)] * int(manifest["num_buckets"])
            for variant in manifest["variants"]:
                root_offsets[int(variant["bucket"])] = (
                    0.0,
                    0.0,
                    float(variant["length"]) - base_length,
                )
            self.pregrasp_root_pos_offsets_buckets = root_offsets

        # Finalise the actor obs space for the latent-conditioned (deployable)
        # mode: [proprio(history_obs_dim) , privileged(privileged_obs_dim)].
        # history_obs_dim = 2 * n_finger_dofs = 32; privileged_obs_dim is 20
        # (or 22 under geometry DR, bumped just above).
        if self.latent_conditioned:
            obs_dim = self.history_obs_dim + self.privileged_obs_dim
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
