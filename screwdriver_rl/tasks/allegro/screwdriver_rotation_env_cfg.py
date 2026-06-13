"""Configuration for Allegro hand continuous screwdriver rotation.

Design philosophy
-----------------
The screwdriver is mounted on a 3-DOF universal joint (tilt-x, tilt-y,
rotation-z) and must be continuously rotated in the negative-z direction
while staying upright.  The policy must use fingertip contacts, not slaps
or knocks, to produce sustained rotation.

All reward weights carry inline justifications so the numbers are traceable.

Curriculum
----------
Training is split into three phases controlled by ``common_step_counter``:

  Phase 1 — "Reach & grasp" (0 → phase2_start steps)
    The near-reward dominates so the policy first learns to surround the
    handle with fingertips.  Turn reward is very weak and the contact gate
    is OFF so even accidental rotations provide a tiny signal.  This
    prevents a common failure mode where the policy never approaches the
    object because the reward landscape is completely flat.

  Phase 2 — "Contact rotation" (phase2_start → phase3_start steps)
    The turn reward increases substantially.  The contact gate is turned ON
    with a generous threshold (0.10 m ≈ 5× handle radius) so the policy
    must have a finger near the handle to earn turn reward, preventing
    "flick-and-coast" behaviour where the policy knocks the screwdriver and
    then retreats.  Proximal-link penalty is activated to begin shaping
    toward fingertip-only contact.

  Phase 3 — "Sustained fingertip rotation" (phase3_start → ∞)
    Full reward weights.  Contact gate tightened to 0.075 m (≈ handle
    surface + fingertip pad thickness).  Episode length extended.  The
    policy now must maintain deliberate fingertip manipulation for the
    entire episode.
"""

from __future__ import annotations

import math
from dataclasses import field
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
# Default: reuse assets from the sibling MFR_benchmark repository.
# Override by setting the environment variable SCREWDRIVER_RL_ASSET_ROOT.
import os as _os

# This file is at <root>/screwdriver_rl/tasks/allegro/ , so parents[3] is the
# repo root and parents[4] is the directory that holds the sibling
# MFR_benchmark checkout.
_DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[4] / "MFR_benchmark" / "MFR_benchmark" / "assets"
ASSET_ROOT = Path(_os.environ.get("SCREWDRIVER_RL_ASSET_ROOT", str(_DEFAULT_ASSET_ROOT)))


# ---------------------------------------------------------------------------
# Curriculum phase config
# ---------------------------------------------------------------------------

@configclass
class CurriculumPhaseCfg:
    """Reward weights for one curriculum training phase.

    The env reads ``common_step_counter`` and selects the phase whose
    ``step_start`` is the largest value that does not exceed the counter.
    """

    step_start: int = 0
    """Global step at which this phase activates."""

    # ---- Turn reward ----
    reward_turn_weight: float = 30.0
    """Forward-rotation reward weight (rad/s × weight per policy step)."""

    turn_reward_contact_distance: float = 0.0
    """Axis-distance threshold for the fingertip-contact gate on turn reward.
    0 disables the gate.  Set to ~5× handle radius in Phase 2, tighten to
    ~3.75× in Phase 3."""

    turn_reward_min_contact_fingers: int = 2
    """Minimum number of fingertips inside ``turn_reward_contact_distance``
    for the contact gate to open.  2 prevents a single-finger poke from
    counting as manipulation."""

    turn_reward_min_fingertip_speed: float = 0.0
    """Fingertip speed (m/s) below which the motion gate is fully closed.
    Prevents earning reward from static finger pressure with no push motion."""

    turn_reward_full_fingertip_speed: float = 0.015
    """Fingertip speed at which the motion gate is fully open."""

    # ---- Proximal-link penalty ----
    reward_proximal_penalty_weight: float = 0.0
    """Penalty weight for proximal/medial Allegro link proximity to the handle.
    Encourages fingertip-only contact.  Off in Phase 1 to not confuse
    the policy before it has learned to approach at all."""

    # ---- Near reward ----
    near_reward_weight: float = 0.8
    """Fingertip proximity reward weight.  High in Phase 1 (dominant signal
    to encourage approaching), tapers in later phases."""

    # ---- Episode length ----
    episode_length_s: float = 20.0
    """Episode length for this phase.  Short in Phase 1; longer later so the
    policy has time to accumulate many turns."""

    # ---- Termination leniency ----
    upright_termination_threshold: float = 2.0
    """Tilt norm (rad) above which the episode terminates.  Lenient in Phase 1
    (explore freely), strict in Phase 3 (upright must be maintained)."""


# ---------------------------------------------------------------------------
# Domain randomisation config
# ---------------------------------------------------------------------------

@configclass
class DomainRandCfg:
    """Per-episode physics randomisation that forces the Stage 2 adaptation
    network to actually learn something.

    Without this, all 2048 envs have identical dynamics every episode.
    The proprioceptive history looks the same regardless of which env the
    robot is in, so the adaptation network has zero gradient signal — MSE
    loss is equally minimised by predicting the dataset mean.  With DR,
    each env gets different damping/mass/gains at every reset, producing
    diverse proprioceptive signatures that the network must distinguish.

    All ranges are multiplicative scales on the base values from the
    articulation configs.  Keeping them multiplicative (not additive)
    means the ratio of variation is constant regardless of the base value,
    which is what physically motivated randomisation should look like.
    """

    enabled: bool = True

    # ------------------------------------------------------------------
    # Screwdriver dynamics
    # ------------------------------------------------------------------
    rotation_damping_range: tuple[float, float] = (0.5, 2.0)
    """Multiplicative scale on the base rotation joint damping (0.15 N·m·s/rad).
    Range → [0.075, 0.30].  This is the primary friction proxy: low values
    mean the screwdriver spins more freely (easy to turn, hard to stop);
    high values require more sustained force (harder to turn, stops quickly).
    The 4× range (0.5–2.0×) matches HORA's friction randomisation range
    (0.3–3.0 absolute, ~10× range; we use a tighter range because the
    rotation damping and contact friction are not the same parameter)."""

    screwdriver_mass_range: tuple[float, float] = (0.5, 2.0)
    """Multiplicative scale on the screwdriver body mass (0.3 kg base).
    Range → [0.15, 0.60] kg.  A heavier object requires more impulse to
    accelerate — the policy must press harder and longer.  A lighter object
    accelerates easily but the fingers must also decelerate it (or it
    oscillates).  Mass variation is the second-largest source of sim-to-real
    gap in in-hand tasks after friction (RMA, Kumar et al. 2021)."""

    # ------------------------------------------------------------------
    # Hand actuator gains
    # ------------------------------------------------------------------
    finger_stiffness_range: tuple[float, float] = (0.8, 1.2)
    """Multiplicative scale on finger joint stiffness (6.0 base).
    Range → [4.8, 7.2].  Tighter range (±20%) than screwdriver dynamics
    because gain variation mostly affects compliance, not contact quality;
    too wide a range would destabilise the grasp rather than improve
    generalisation.  Matches HORA's PD gain randomisation policy."""

    finger_damping_range: tuple[float, float] = (0.8, 1.2)
    """Multiplicative scale on finger joint damping (1.0 base) → [0.8, 1.2]."""

    # ------------------------------------------------------------------
    # Observation noise
    # ------------------------------------------------------------------
    obs_noise_std: float = 0.01
    """Gaussian noise std added to every policy observation at each step.
    Simulates encoder quantisation, IMU noise, and marker-tracking jitter.
    At std=0.01 rad the noise is ~0.6° — below human perception threshold
    for joint angle, but meaningful for a neural network trained to detect
    slip patterns in the proprioceptive history."""


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------

@configclass
class AllegroScrewdriverRotationEnvCfg(DirectRLEnvCfg):
    """Full configuration for the Allegro continuous screwdriver rotation task.

    Observation space (27-D):
      [finger_q(12), cur_targets(12), screwdriver_euler(3)]

    Action space (12-D):
      HORA-style delta targets, clipped to [-1, 1].
      target[t] = target[t-1] + action_delta_scale * action
      Scale 0.05 rad/step at 10 Hz → max 0.5 rad/s per joint.

    Privileged observations (for teacher-student / RMA):
      [screwdriver_euler(3), screwdriver_angvel(3), screwdriver_pos(3),
       screwdriver_quat(4), friction(1), fingertip_axis_dist(3)] = 17-D
    """

    # ------------------------------------------------------------------
    # Gym spaces
    # ------------------------------------------------------------------
    # 12 finger positions + 12 targets + 3 screwdriver Euler angles
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
    # 12 DOF: index(4) + middle(4) + thumb(4)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    state_space = 0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    decimation: int = 6
    """Physics sub-steps per policy step.  6 × (1/60 s) = 0.1 s policy dt,
    i.e. 10 Hz.  Same as MFR benchmark; lower frequencies hurt contact
    fidelity, higher frequencies slow wall-clock training."""

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.5,
            # Higher friction than MFR's 1.0: the finger must *roll* the
            # handle to get spin, not just tap it.  A tapped object on a
            # low-friction surface can coast; on a high-friction surface it
            # stops quickly and requires sustained contact force.
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

    # episode_length_s must match Phase-1 value in curriculum_phases[0].
    # The env overrides max_episode_length dynamically at each curriculum
    # transition, but super().__init__() uses this value for the initial setup
    # and for the episode-start stagger.
    episode_length_s: float = 20.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=1.5, replicate_physics=True
    )

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------
    fingers: tuple[str, ...] = ("index", "middle", "thumb")
    """Active fingers.  3-finger configuration matches the most-desired
    outcome: 1–2 fingers stabilise, 1–2 push/reposition."""

    # HORA-style incremental action (delta targets).  action=0 holds the
    # current target; the policy naturally learns to hold its grip.
    action_delta: bool = True
    action_delta_scale: float = 0.05
    """0.05 rad/step × 10 Hz = 0.5 rad/s max joint velocity.  Matches HORA's
    1/24 ≈ 0.042 rad/step at 20 Hz; scaled for our 10 Hz policy."""
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
    when viewed from above)."""

    turn_velocity_clip: float = 1.0
    """Cap on instantaneous turn velocity used in the reward (rad/s).
    Prevents a single forceful flick from producing an outsized reward spike
    that masks the true learning signal."""

    # ------------------------------------------------------------------
    # Physics realism
    # ------------------------------------------------------------------
    friction_coefficient: float = 1.5
    """Contact friction on the ground plane.  Matches sim.physics_material."""

    # Screwdriver actuator damping is set in screwdriver_cfg below.
    # rotation_joint_damping = 0.15  (documented there)

    # ------------------------------------------------------------------
    # Reward — stable weights (phase-independent)
    # ------------------------------------------------------------------
    reward_reverse_weight: float = 220.0
    """Penalises backward rotation.  Slightly above reward_turn_weight so
    the optimal behaviour strictly favours forward turns over backward ones.
    Both the turn reward and reverse penalty share the SAME contact gate so
    that the expected value of contact is always positive (forward > reverse
    when the gate opens).  If only the forward reward were gated, the
    policy would learn to keep fingers off the handle to avoid the reverse
    penalty."""

    reward_upright_weight: float = 200.0
    """Additive cost for tilt: upright_weight * sum(tilt_xy²).  Acts as a
    baseline penalty so the policy pays for every step of tilt regardless
    of whether it is turning."""

    reward_tilt_velocity_weight: float = 5.0
    """Penalises rate of change of tilt.  Discourages oscillatory wobble
    that would otherwise be invisible to the static tilt cost."""

    turn_upright_gate_std: float = 0.25
    """std of the multiplicative Gaussian gate on the turn reward:
      gate = exp(-(tilt_norm / std)²).
    At tilt_norm = std (0.25 rad ≈ 14°): gate = e⁻¹ ≈ 0.37.
    At tilt_norm = 2×std (0.5 rad ≈ 29°): gate ≈ 0.02.
    This means the dominant positive reward term is almost entirely
    suppressed at moderate tilt, making the penalty multiplicative rather
    than additive.  An additive penalty cannot race a large turn reward
    (tilt-and-scrape exploit); a multiplicative gate can."""

    use_shaft_spin_measure: bool = True
    """Measure spin as the quaternion delta projected onto the shaft axis
    rather than the Euler-z joint coordinate.  Euler-z also advances under
    precession of a tilted shaft, rewarding wobble-scraping as if it were
    genuine rotation."""

    use_axis_contact_proxy: bool = True
    """Compute fingertip distances to the handle *axis segment* (handle
    origin → cap origin) instead of the handle body origin.  The axis
    distance is ~0.03 m for a fingertip pad touching the 0.02 m-radius
    handle, giving a physically interpretable threshold."""

    # Action / finger regularisation (phase-independent)
    reward_action_weight: float = 0.25
    reward_action_rate_weight: float = 0.1
    reward_finger_velocity_weight: float = 0.001

    # Near-reward shape
    near_reward_std: float = 0.03
    """Exponential decay scale for fingertip proximity: near = exp(-d/std).
    At d = 0.03 m (handle contact distance), near ≈ e⁻¹ ≈ 0.37.
    At d = 0.075 m (contact-gate threshold), near ≈ e⁻²·⁵ ≈ 0.08."""
    near_reward_top_k: int = 2
    """Average only the top-k closest non-thumb fingertips.  Prevents the
    near-reward from forcing all fingers onto the same side of the handle."""

    # Milestone (sparse progress bonus)
    milestone_angle: float = 0.5 * math.pi
    """Sparse bonus fires every half-turn (π/2 rad) of net forward progress.
    Provides a long-horizon signal that persists in the value function
    when the dense turn reward is near zero (e.g., contact gate just closed)."""
    milestone_bonus: float = 0.25

    # ------------------------------------------------------------------
    # RMA / asymmetric observations
    # ------------------------------------------------------------------
    asymmetric_obs: bool = False
    privileged_obs_dim: int = 17
    """3 euler + 3 angvel + 3 rel-pos + 4 quat + 1 friction + 3 fingertip-dist."""
    prop_hist_len: int = 30
    history_obs_dim: int = 24
    """[finger_q(12), cur_targets(12)] per frame."""

    # ------------------------------------------------------------------
    # Curriculum phases
    # ------------------------------------------------------------------
    curriculum_phases: list[CurriculumPhaseCfg] = field(
        default_factory=lambda: [
            CurriculumPhaseCfg(
                step_start=0,
                reward_turn_weight=30.0,
                # Contact gate OFF: even accidental rotations provide signal.
                # The policy cannot learn to turn before it knows how to
                # approach and hold the handle; starting with a strict gate
                # yields zero gradient for turn reward the entire first phase.
                turn_reward_contact_distance=0.0,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.0,
                # No proximal penalty yet: first learn to approach.
                reward_proximal_penalty_weight=0.0,
                # Near-reward is the dominant training signal in Phase 1.
                near_reward_weight=0.8,
                episode_length_s=20.0,
                # Lenient termination: the policy will naturally tilt while
                # it is still exploring; terminating too early removes the
                # recovery data from the replay buffer.
                upright_termination_threshold=2.0,
            ),
            CurriculumPhaseCfg(
                step_start=15_000_000,
                reward_turn_weight=150.0,
                # Contact gate ON, generous (0.10 m ≈ 5× handle radius).
                # The policy must have a fingertip near the handle to receive
                # turn reward; this eliminates "flick-and-coast" where the
                # policy knocks and retreats to let the screwdriver coast.
                turn_reward_contact_distance=0.10,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.003,
                # Mild proximal penalty: begin discouraging knuckle/palm use.
                reward_proximal_penalty_weight=2.0,
                near_reward_weight=0.3,
                episode_length_s=40.0,
                upright_termination_threshold=1.5,
            ),
            CurriculumPhaseCfg(
                step_start=60_000_000,
                reward_turn_weight=200.0,
                # Tighten gate to 0.05 m = 50 mm (≈ 2.5× handle radius).
                # Handle radius = 20 mm, fingertip pad adds ~10 mm, so
                # touching the surface sits at ~30 mm axis distance.
                # 50 mm means the tip must be within ~20 mm of the surface —
                # genuine near-contact, not just hovering in the vicinity.
                turn_reward_contact_distance=0.05,
                turn_reward_min_contact_fingers=2,
                turn_reward_min_fingertip_speed=0.003,
                # Stronger proximal penalty: enforce fingertip-only style.
                reward_proximal_penalty_weight=5.0,
                near_reward_weight=0.15,
                episode_length_s=60.0,
                # Strict upright maintenance required in final phase.
                upright_termination_threshold=1.0,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Robot (Allegro hand)
    # ------------------------------------------------------------------
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
            pos=(0.0, -0.095, 1.33),
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

    # ------------------------------------------------------------------
    # Screwdriver
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
            pos=(0.0, 0.0, 1.205),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "tilt": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_1", "table_screwdriver_joint_2"],
                stiffness=0.0,
                # Slightly higher tilt damping (0.001) than MFR (0.0001).
                # Provides a weak restoring resistance that helps the screwdriver
                # return to upright after small perturbations, without making it
                # completely rigid.  Full uprightness is enforced by the reward,
                # not by physics damping.
                damping=0.001,
            ),
            "rotation": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_3"],
                stiffness=0.0,
                # Damping 0.15 (3× MFR's 0.05).
                # Physical reasoning: screwdriver body inertia
                #   I ≈ 0.5 × 0.3 kg × (0.02 m)² ≈ 6×10⁻⁵ kg⋅m²
                # Time constant τ = I/b = 6×10⁻⁵ / 0.15 ≈ 0.0004 s.
                # This means the screwdriver loses 63% of its angular velocity
                # every 0.4 ms — far faster than one physics step (1/60 s ≈ 16 ms).
                # Effectively: the screwdriver stops the moment the finger leaves.
                # A "small touch" (1 physics step of 0.001 N⋅m torque) imparts
                # ω = T×dt/I ≈ 0.001×0.017/6×10⁻⁵ ≈ 0.28 rad/s, which decays
                # completely within one policy step.  The policy cannot earn
                # sustained turn reward from a single tap.
                damping=0.15,
            ),
            "cap": ImplicitActuatorCfg(
                joint_names_expr=["screwdriver_body_cap_joint"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # ------------------------------------------------------------------
    # Pregrasp joint positions (per finger, 4 joints each)
    # ------------------------------------------------------------------
    pregrasp_positions: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "index":  (0.1, 0.6, 0.6, 0.6),
            "middle": (-0.1, 0.5, 0.9, 0.9),
            "ring":   (0.0, 0.5, 0.65, 0.65),
            "thumb":  (1.2, 0.3, 0.3, 1.2),
        }
    )
    """Finger joint positions at episode reset.  Identical to MFR benchmark:
    index/middle/thumb wrap around the handle body from above;  the thumb
    opposes from the side.  This creates a 3-point pinch from which the policy
    should discover the repositioning/stabilising strategy."""

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------
    domain_rand: DomainRandCfg = field(default_factory=DomainRandCfg)
