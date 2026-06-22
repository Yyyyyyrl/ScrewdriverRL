"""Hand-agnostic configuration for the continuous screwdriver rotation task.

Design philosophy
-----------------
The screwdriver is mounted on a 3-DOF universal joint (tilt-x, tilt-y,
rotation-z) and must be continuously rotated in one direction while staying
upright.  The policy must use fingertip contacts, not slaps or knocks, to
produce sustained rotation.

This module holds everything that is identical across hands: the curriculum,
domain randomisation, reward weights, screwdriver asset, and simulation
config.  The hand-specific fields (``robot_cfg``, ``fingers``,
``pregrasp_positions``, the gym spaces, the privileged/history dims, and the
fingertip pad axis) are declared ``MISSING`` here and filled in by each hand's
subclass (e.g. ``AllegroScrewdriverRotationEnvCfg``,
``LinkerL20ScrewdriverRotationEnvCfg``).

All reward weights carry inline justifications so the numbers are traceable.

Curriculum
----------
Training is split into three phases controlled by the global step count:

  Phase 0 — "Reach & grasp"
    The near-reward dominates so the policy first learns to surround the
    handle with fingertips.  The contact gate is ON but generous (0.10 m),
    the pad-facing factor is a lenient soft ramp, and the handle is
    free-spinning (no screw load).

  Phase 1 — "Contact rotation"
    The turn reward increases, the screw load ramps to half, the contact gate
    tightens (0.07 m), the pad-facing threshold tightens, and the
    proximal-link penalty activates to shape toward fingertip-only contact.

  Phase 2 — "Sustained fingertip rotation"
    Full reward weights, contact gate at 0.05 m, strict pad-facing, full screw
    load, longer episodes, strict upright termination.
"""

from __future__ import annotations

import math
import os as _os
from dataclasses import MISSING, field
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
# Assets are bundled inside this repository (see ``<root>/assets``) so the task
# has no dependency on any files outside ScrewdriverRL.
# Override by setting the environment variable SCREWDRIVER_RL_ASSET_ROOT.
#
# This file is at <root>/screwdriver_rl/tasks/base/ , so parents[3] is the
# repo root that holds the bundled ``assets`` directory.
_DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
ASSET_ROOT = Path(_os.environ.get("SCREWDRIVER_RL_ASSET_ROOT", str(_DEFAULT_ASSET_ROOT)))


# ---------------------------------------------------------------------------
# Curriculum phase config
# ---------------------------------------------------------------------------

@configclass
class CurriculumPhaseCfg:
    """Reward weights for one curriculum training phase.

    The env selects the phase whose ``step_start`` is the largest value that
    does not exceed the global step counter.
    """

    step_start: int = 0
    """Global step at which this phase activates."""

    # ---- Turn reward ----
    reward_turn_weight: float = 30.0
    """Forward-rotation reward weight (rad/s × weight per policy step)."""

    turn_reward_contact_distance: float = 0.0
    """Axis-distance threshold for the fingertip-contact gate on turn reward.
    0 disables the gate.  Set to ~5× handle radius in Phase 1, tighten to
    ~3.75× in Phase 2."""

    turn_reward_min_contact_fingers: int = 2
    """Minimum number of fingertips inside ``turn_reward_contact_distance``
    for the contact gate to open.  2 prevents a single-finger poke from
    counting as manipulation."""

    turn_reward_min_fingertip_speed: float = 0.0
    """Fingertip speed (m/s) below which the motion gate is fully closed.
    Prevents earning reward from static finger pressure with no push motion."""

    turn_reward_full_fingertip_speed: float = 0.015
    """Fingertip speed at which the motion gate is fully open."""

    # ---- Screwdriver rotational load ----
    screwdriver_load_scale: float = 1.0
    """Multiplier on ``cfg.screwdriver_load_torque`` for this phase.  Ramp from
    0 (free-spinning handle — first learn to rotate at all) to 1.0 (full screw
    resistance) so the policy is not crushed by load before it can turn."""

    # ---- Proximal-link penalty ----
    reward_proximal_penalty_weight: float = 0.0
    """Penalty weight for proximal/medial link proximity to the handle.
    Encourages fingertip-only contact.  Off in Phase 0 to not confuse
    the policy before it has learned to approach at all."""

    # ---- Near reward ----
    near_reward_weight: float = 0.8
    """Fingertip proximity reward weight.  High in Phase 0 (dominant signal
    to encourage approaching), tapers in later phases."""

    # ---- Episode length ----
    episode_length_s: float = 20.0
    """Episode length for this phase.  Short in Phase 0; longer later so the
    policy has time to accumulate many turns."""

    # ---- Termination leniency ----
    upright_termination_threshold: float = 2.0
    """Tilt norm (rad) above which the episode terminates.  Lenient in Phase 0
    (explore freely), strict in Phase 2 (upright must be maintained)."""


# ---------------------------------------------------------------------------
# Domain randomisation config
# ---------------------------------------------------------------------------

@configclass
class DomainRandCfg:
    """Per-episode physics randomisation that forces the Stage 2 adaptation
    network to actually learn something.

    Without this, all envs have identical dynamics every episode and the
    proprioceptive history looks the same regardless of which env the robot is
    in, so the adaptation network has zero gradient signal.  With DR, each env
    gets different damping/mass/gains at every reset, producing diverse
    proprioceptive signatures that the network must distinguish.

    All ranges are multiplicative scales on the base values from the
    articulation configs.
    """

    enabled: bool = True

    # ------------------------------------------------------------------
    # Screwdriver dynamics
    # ------------------------------------------------------------------
    rotation_damping_range: tuple[float, float] = (0.5, 2.0)
    """Multiplicative scale on the base rotation joint damping.  This is the
    primary friction proxy: low values mean the screwdriver spins more freely;
    high values require more sustained force."""

    screwdriver_mass_range: tuple[float, float] = (0.5, 2.0)
    """Multiplicative scale on the screwdriver body mass.  Mass variation is the
    second-largest source of sim-to-real gap in in-hand tasks after friction."""

    # ------------------------------------------------------------------
    # Screwdriver rotational load
    # ------------------------------------------------------------------
    screwdriver_load_torque_range: tuple[float, float] = (0.5, 1.5)
    """Multiplicative scale on the base ``screwdriver_load_torque`` per reset.
    Once a load is present this is the dominant sim-to-real factor (it replaces
    rotation damping as the primary 'friction' the Stage-2 network must infer)."""

    # ------------------------------------------------------------------
    # Hand actuator gains
    # ------------------------------------------------------------------
    finger_stiffness_range: tuple[float, float] = (0.8, 1.2)
    """Multiplicative scale on finger joint stiffness (±20%).  Gain variation
    mostly affects compliance, not contact quality; too wide a range would
    destabilise the grasp rather than improve generalisation."""

    finger_damping_range: tuple[float, float] = (0.8, 1.2)
    """Multiplicative scale on finger joint damping (±20%)."""

    # ------------------------------------------------------------------
    # Observation noise
    # ------------------------------------------------------------------
    obs_noise_std: float = 0.01
    """Gaussian noise std added to every policy observation at each step.
    Simulates encoder quantisation, IMU noise, and marker-tracking jitter."""


# ---------------------------------------------------------------------------
# Main environment config (hand-agnostic base)
# ---------------------------------------------------------------------------

@configclass
class ScrewdriverRotationEnvCfg(DirectRLEnvCfg):
    """Hand-agnostic base configuration for the continuous screwdriver rotation
    task.

    Subclasses MUST set the hand-specific fields (declared ``MISSING`` below):
    ``observation_space``, ``action_space``, ``fingers``, ``pregrasp_positions``,
    ``privileged_obs_dim``, ``history_obs_dim``, and ``robot_cfg``.

    Observation layout (per hand, D = num finger DOFs):
      policy:  [finger_q(D), cur_targets(D), screwdriver_euler(3)]
      critic:  [euler(3), angvel(3), rel_pos(3), quat(4), friction(1),
                fingertip_axis_dist(num_fingers)]
    """

    # ------------------------------------------------------------------
    # Gym spaces (hand-specific — set by subclass)
    # ------------------------------------------------------------------
    observation_space = MISSING
    action_space = MISSING
    state_space = 0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    decimation: int = 6
    """Physics sub-steps per policy step.  6 × (1/60 s) = 0.1 s policy dt,
    i.e. 10 Hz.  Lower frequencies hurt contact fidelity, higher frequencies
    slow wall-clock training."""

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.5,
            # Higher friction than MFR's 1.0: the finger must *roll* the
            # handle to get spin, not just tap it.
        ),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            gpu_max_rigid_patch_count=2**22,
        ),
    )

    # episode_length_s must match curriculum_phases[0].  The env overrides
    # max_episode_length dynamically at each curriculum transition, but
    # super().__init__() uses this value for the initial setup and the
    # episode-start stagger.
    episode_length_s: float = 30.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=1.5, replicate_physics=True
    )

    # ------------------------------------------------------------------
    # Task — active fingers (hand-specific, set by subclass)
    # ------------------------------------------------------------------
    fingers: tuple[str, ...] = MISSING
    """Active fingers driven by the policy."""

    # HORA-style incremental action (delta targets).  action=0 holds the
    # current target; the policy naturally learns to hold its grip.
    action_delta: bool = True
    action_delta_scale: float = 0.05
    """0.05 rad/step × 10 Hz = 0.5 rad/s max joint velocity."""
    action_clip: float = 1.0
    clamp_joint_targets: bool = True
    joint_target_margin: float = 0.02

    randomize_obj_start: bool = True
    """Randomise screwdriver initial Z angle over [−π, π] so the policy
    generalises to all orientations, not just the reset pose."""

    reset_contact_steps: int = 32
    """Physics settling steps after reset to stabilise initial contacts."""

    turn_direction: float = -1.0
    """Sign of the desired rotation: −1 = negative-z (right-hand rule: CCW
    when viewed from above).  A mirror-image (left) hand may need +1."""

    turn_velocity_clip: float = 1.0
    """Cap on instantaneous turn velocity used in the reward (rad/s).
    Prevents a single forceful flick from producing an outsized reward spike."""

    # ------------------------------------------------------------------
    # Physics realism
    # ------------------------------------------------------------------
    friction_coefficient: float = 1.5
    """Contact friction on the ground plane.  Matches sim.physics_material."""

    # ------------------------------------------------------------------
    # Screwdriver rotational load (models the resistance of driving a screw)
    # ------------------------------------------------------------------
    screwdriver_load_torque: float = 0.045
    """Constant (Coulomb) resistive torque on the screwdriver rotation joint
    (N·m), opposing the direction of motion.  This is the breakaway "stiction" a
    real screw presents: below it the handle does not rotate at all, so a
    sub-threshold finger nudge cannot make it creep forward and farm turn reward.
    Raised from 0.02 (where feeble low-torque nudging still spun a near-free
    handle) to 0.045.  Set 0.0 to recover a free-spinning handle."""

    screwdriver_load_viscous: float = 0.0
    """Extra speed-proportional resistance (N·m·s/rad) on top of the actuator's
    bearing damping.  Usually 0 — the Coulomb term above is dominant."""

    screwdriver_load_omega_eps: float = 0.05
    """Velocity (rad/s) over which the Coulomb torque is smoothly ramped via
    tanh(omega/eps), so it passes through zero without solver chatter."""

    # ------------------------------------------------------------------
    # Reward — stable weights (phase-independent)
    # ------------------------------------------------------------------
    reward_reverse_weight: float = 220.0
    """Penalises backward rotation.  Slightly above reward_turn_weight so the
    optimal behaviour strictly favours forward turns.  Both the turn reward and
    reverse penalty share the SAME contact gate so the expected value of contact
    is always positive."""

    reward_upright_weight: float = 200.0
    """Additive cost for tilt: upright_weight * sum(tilt_xy²)."""

    reward_tilt_velocity_weight: float = 7
    """Penalises rate of change of tilt (L1 of d(tilt_xy)/dt) — the direct lever
    against side-to-side rocking during turning."""

    turn_upright_gate_std: float = 0.2
    """std of the multiplicative Gaussian gate on the turn reward:
      gate = exp(-(tilt_norm / std)²).
    Makes the upright penalty multiplicative rather than additive, which an
    additive penalty cannot achieve against a large turn reward."""

    use_shaft_spin_measure: bool = True
    """Measure spin as the quaternion delta projected onto the shaft axis rather
    than the Euler-z joint coordinate (which advances under precession)."""

    # ------------------------------------------------------------------
    # Contact distance proxy
    # ------------------------------------------------------------------
    use_axis_contact_proxy: bool = True
    """Compute fingertip distances to the handle *axis segment* (handle
    origin → cap origin) instead of the handle body origin."""

    # Action / finger regularisation (phase-independent)
    reward_action_weight: float = 0.25
    reward_action_rate_weight: float = 0.1
    reward_finger_velocity_weight: float = 0.001

    # Near-reward shape
    near_reward_std: float = 0.03
    """Exponential decay scale for fingertip proximity: near = exp(-d/std)."""
    near_reward_top_k: int = 2
    """Average only the top-k closest non-thumb fingertips.  Prevents the
    near-reward from forcing all fingers onto the same side of the handle."""

    # Milestone (sparse progress bonus)
    milestone_angle: float = 0.5 * math.pi
    """Sparse bonus fires every half-turn (π/2 rad) of net forward progress."""
    milestone_bonus: float = 0.25

    # ------------------------------------------------------------------
    # RMA / asymmetric observations
    # ------------------------------------------------------------------
    asymmetric_obs: bool = False
    privileged_obs_dim: int = MISSING
    """3 euler + 3 angvel + 3 rel-pos + 4 quat + 1 friction +
    num_fingers fingertip-dist (= 14 + len(fingers)).  Hand-specific."""
    prop_hist_len: int = 30
    history_obs_dim: int = MISSING
    """[finger_q(D), cur_targets(D)] per frame (= 2 × num finger DOFs).
    Hand-specific."""

    # ------------------------------------------------------------------
    # Curriculum phases (GENERIC FALLBACK — each hand owns its own)
    # ------------------------------------------------------------------
    # This default exists so the base config is usable on its own, but every
    # hand subclass OVERRIDES ``curriculum_phases`` with values tuned to its own
    # finger count and geometry (e.g. ``turn_reward_min_contact_fingers`` and the
    # per-phase contact distances differ between the 3-active-finger Allegro and
    # the 5-finger LinkerL20).  Edit a hand's curriculum in its own cfg module,
    # not here.
    curriculum_phases: list[CurriculumPhaseCfg] = field(
        default_factory=lambda: [
            CurriculumPhaseCfg(
                # --- P0: learn PAD contact from the start ---
                # No ungated warm-up phase: with the turn gate off the policy
                # just wobbles the handle to farm reward (high OscRatio).  The
                # ungated ``near_reward`` (0.8) already provides the "approach
                # the handle" gradient, so we start with the contact gate ON
                # (generous 0.10 m), a lenient SOFT pad factor (gradient, not a
                # cliff), and a free-spinning handle (no load).
                step_start=0,
                reward_turn_weight=120.0,
                turn_reward_contact_distance=0.10,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.0,
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
                screwdriver_load_scale=1.0,
                reward_proximal_penalty_weight=5.0,
                near_reward_weight=0.15,
                episode_length_s=60.0,
                upright_termination_threshold=1.0,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Robot (hand-specific — set by subclass)
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = MISSING

    # ------------------------------------------------------------------
    # Screwdriver (shared across hands)
    # ------------------------------------------------------------------
    screwdriver_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Screwdriver",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "screwdriver/screwdriver_isaaclab.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
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
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Mount shifted -0.009 m in x (was 0.0) to reduce the LinkerL20
            # screwdriver's settled lean from ~10deg to ~6deg WHILE KEEPING the
            # lean toward the four fingers (-x).  Past ~-0.011 the lean flips to
            # the thumb/palm side (+x), which is not wanted, so -0.009 sits just
            # short of that crossover.  This base is SHARED, so each hand's
            # init_state.pos is compensated by the SAME -0.009 m x shift to
            # preserve its grasp geometry (Allegro: 0.0 -> -0.009; LinkerL20 hand x
            # left at 0.13 so it gains the relative shift).  Re-derive with
            # render_posture.py if the screwdriver asset or any hand pregrasp changes.
            pos=(-0.009, 0.0, 1.205),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "tilt": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_1", "table_screwdriver_joint_2"],
                stiffness=0.0,
                # Weak restoring resistance that helps the screwdriver return to
                # upright after small perturbations without making it rigid.
                damping=0.003
            ),
            "rotation": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_3"],
                stiffness=0.0,
                # Damping 0.5: with the handle z-inertia (izz~6e-5) the velocity
                # time constant tau = I/c ~ 1.2e-4 s is far shorter than one
                # physics step (1/60 s), so the handle stops the instant the
                # finger leaves — it cannot coast forward for reward.  Raised from
                # 0.15 (which still let a steady sub-threshold push spin it up).
                damping=0.5,
            ),
            "cap": ImplicitActuatorCfg(
                joint_names_expr=["screwdriver_body_cap_joint"],
                stiffness=0.0,
                # Light damping (was 0.0): couples the free-spinning cap to the
                # body so it cannot keep rotating on its own bearing and look like
                # the screwdriver is "spinning by itself".
                damping=0.05,
            ),
        },
    )

    # ------------------------------------------------------------------
    # Pregrasp joint positions (hand-specific — set by subclass)
    # ------------------------------------------------------------------
    pregrasp_positions: dict[str, tuple[float, ...]] = MISSING
    """Independent finger joint positions at episode reset, keyed by finger,
    in the same semantic order as the hand's ``FINGER_JOINT_NAMES`` tuples."""

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------
    domain_rand: DomainRandCfg = field(default_factory=DomainRandCfg)
