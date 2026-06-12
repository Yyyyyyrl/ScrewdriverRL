"""Curriculum phase definitions (pure python, no isaaclab).

A curriculum is an ordered list of :class:`CurriculumPhase`. The stage-1
trainer applies each phase's ``env_overrides`` (dotted paths into the live env
config) and advances when the windowed env metrics satisfy the phase's
:class:`AdvanceCriteria`. Full rationale: docs/curriculum.md.

Design constraints baked into ``allegro_default``:
  * Reverse weight strictly above turn weight in EVERY phase (project
    requirement): backward rotation is always net-negative, so oscillation
    can never farm reward. This is safe against the "never touch" collapse
    because the reverse cost carries the same contact/motion/upright gates as
    the turn reward — an idle hand collects neither — and the early phases
    keep a strong near-contact reward so touching the handle has positive
    expected value even before clean turning emerges.
  * The upright weight rises only after spinning is established, otherwise
    "hold perfectly still" dominates the discovery phase.
  * Contact-gate distances are fingertip-to-handle-axis values
    (handle radius 0.02 m => pad contact ~0.03 m) and each step-down stays
    inside the previous phase's converged distances to avoid a gate-collapse
    reward cliff at the transition.
  * Upright termination is active from phase 1 (HORA's drop-termination
    analogue for a mounted screwdriver); thresholds tighten per phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdvanceCriteria:
    """Gate for moving to the next phase. All present checks must hold on the
    windowed means of env metrics (keys are env ``extras`` names)."""

    min_phase_steps: float = 0.0
    min_episode_length: float | None = None
    # metric key -> (op, threshold) where op is ">=" or "<="
    metric_bounds: dict[str, tuple[str, float]] = field(default_factory=dict)
    # forward minus reverse turn velocity margin (rad/s)
    min_fwd_minus_rev: float | None = None


@dataclass
class CurriculumPhase:
    name: str
    env_overrides: dict[str, Any] = field(default_factory=dict)
    advance: AdvanceCriteria | None = None  # None = final phase


def _phase(name, overrides, advance=None):
    return CurriculumPhase(name=name, env_overrides=overrides, advance=advance)


ALLEGRO_DEFAULT: list[CurriculumPhase] = [
    _phase(
        "phase1_upright_spin_discovery",
        {
            "reward_turn_weight": 1000.0,
            "reward_reverse_weight": 1100.0,  # > turn weight (see module docstring)
            "turn_velocity_clip": 1.0,
            "turn_upright_gate_std": 0.45,
            "reward_upright_weight": 50.0,
            "reward_tilt_velocity_weight": 0.5,
            "upright_termination_threshold": 0.6,
            "reward_action_weight": 0.03,
            "reward_action_rate_weight": 0.06,
            "reward_finger_pose_weight": 0.002,
            "reward_finger_velocity_weight": 0.0002,
            "reward_joint_limit_weight": 0.0,
            "reward_work_weight": 0.0,
            "milestone_bonus": 0.0,
            # Strong, wide approach gradient: touching must have positive EV
            # while the (gated) reverse penalty exceeds the turn reward.
            "near_reward_weight": 1.2,
            "near_reward_std": 0.12,
            "near_reward_top_k": 3,
            "turn_reward_contact_distance": 0.06,
            "turn_reward_min_contact_fingers": 2,
            "turn_reward_min_fingertip_speed": 0.0,
            "turn_reward_full_fingertip_speed": 0.003,
            "lost_contact_termination_distance": 0.0,
            "stagnation_variance_eps": 0.0,
            # DR: friction only, narrow; no per-step noise yet.
            "dr.randomize_friction": True,
            "dr.friction_range": (0.8, 1.2),
            "dr.randomize_mass": False,
            "dr.randomize_gains": False,
            "dr.randomize_com": False,
            "dr.randomize_z_damping": False,
            "dr.init_tilt_max": 0.0,
            "dr.pregrasp_noise": 0.01,
            "dr.obs_noise_std": 0.0,
            "dr.action_noise_std": 0.0,
        },
        AdvanceCriteria(
            min_phase_steps=120e6,
            min_episode_length=540,  # 90% of the 600-step episode
            metric_bounds={
                "eval_net_turns": (">=", 0.10),
                "eval_screwdriver_upright_norm": ("<=", 0.35),
                "eval_mean_fingertip_dist": ("<=", 0.08),
            },
            min_fwd_minus_rev=0.04,
        ),
    ),
    _phase(
        "phase2_contacted_direction",
        {
            "reward_turn_weight": 1000.0,
            "reward_reverse_weight": 1150.0,
            "turn_velocity_clip": 1.0,
            "turn_upright_gate_std": 0.30,
            "reward_upright_weight": 100.0,
            "reward_tilt_velocity_weight": 1.0,
            "upright_termination_threshold": 0.5,
            "reward_action_weight": 0.10,
            "reward_action_rate_weight": 0.06,
            "reward_finger_pose_weight": 0.005,
            "reward_finger_velocity_weight": 0.0005,
            "reward_joint_limit_weight": 150.0,
            "reward_work_weight": 0.0,
            "milestone_bonus": 0.05,
            "near_reward_weight": 0.15,
            "near_reward_std": 0.12,
            "near_reward_top_k": 3,
            "turn_reward_contact_distance": 0.045,
            "turn_reward_full_fingertip_speed": 0.006,
            "lost_contact_termination_distance": 0.0,
            "stagnation_variance_eps": 0.0,
            "dr.friction_range": (0.7, 1.3),
            "dr.randomize_mass": True,
            "dr.randomize_gains": True,
            "dr.pregrasp_noise": 0.02,
        },
        AdvanceCriteria(
            min_phase_steps=120e6,
            min_episode_length=540,
            metric_bounds={
                "eval_net_turns": (">=", 0.15),
                "eval_screwdriver_upright_norm": ("<=", 0.30),
                "eval_mean_fingertip_dist": ("<=", 0.06),
            },
            min_fwd_minus_rev=0.06,
        ),
    ),
    _phase(
        "phase3_stable_contact_turning",
        {
            "reward_turn_weight": 500.0,
            "reward_reverse_weight": 575.0,
            "turn_velocity_clip": 0.75,
            "turn_upright_gate_std": 0.20,
            "reward_upright_weight": 200.0,
            "reward_tilt_velocity_weight": 1.5,
            "upright_termination_threshold": 0.45,
            "reward_action_weight": 0.10,
            "reward_action_rate_weight": 0.06,
            "reward_finger_pose_weight": 0.01,
            "reward_finger_velocity_weight": 0.001,
            "reward_joint_limit_weight": 150.0,
            "reward_work_weight": 0.01,
            "milestone_bonus": 0.10,
            "near_reward_weight": 0.12,
            "turn_reward_contact_distance": 0.04,
            "turn_reward_min_fingertip_speed": 0.003,
            "turn_reward_full_fingertip_speed": 0.015,
            "lost_contact_termination_distance": 0.0,
            "stagnation_variance_eps": 0.0,
            "dr.friction_range": (0.6, 1.4),
            "dr.randomize_com": True,
            "dr.randomize_z_damping": True,
            "dr.init_tilt_max": 0.03,
            "dr.pregrasp_noise": 0.03,
        },
        AdvanceCriteria(
            min_phase_steps=120e6,
            min_episode_length=540,
            metric_bounds={
                "eval_net_turns": (">=", 0.10),
                "eval_screwdriver_upright_norm": ("<=", 0.25),
                "eval_mean_fingertip_dist": ("<=", 0.05),
            },
            min_fwd_minus_rev=0.04,
        ),
    ),
    _phase(
        "phase4_strict_continuous_turning",
        {
            "reward_turn_weight": 200.0,
            "reward_reverse_weight": 220.0,
            "turn_velocity_clip": 0.5,
            "turn_upright_gate_std": 0.15,
            "reward_upright_weight": 400.0,
            "reward_tilt_velocity_weight": 5.0,
            "upright_termination_threshold": 0.4,
            "reward_action_weight": 0.25,
            "reward_action_rate_weight": 0.1,
            "reward_finger_pose_weight": 0.02,
            "reward_finger_velocity_weight": 0.001,
            "reward_joint_limit_weight": 200.0,
            "reward_work_weight": 0.01,
            "milestone_bonus": 0.25,
            "near_reward_weight": 0.0,
            "turn_reward_contact_distance": 0.035,
            "turn_reward_min_fingertip_speed": 0.003,
            "turn_reward_full_fingertip_speed": 0.015,
            "lost_contact_termination_distance": 0.06,
            "lost_contact_min_fingers": 1,
            "lost_contact_grace_steps": 5,
            "stagnation_variance_eps": 0.003,
            # Full DR + per-step noise for the sim2real-ready teacher.
            "dr.friction_range": (0.5, 1.5),
            "dr.init_tilt_max": 0.05,
            "dr.obs_noise_std": 0.01,
            "dr.action_noise_std": 0.01,
        },
        advance=None,  # final phase
    ),
]


CURRICULA: dict[str, list[CurriculumPhase]] = {
    "allegro_default": ALLEGRO_DEFAULT,
    "none": [],
}


def get_curriculum(name: str) -> list[CurriculumPhase]:
    if name not in CURRICULA:
        raise KeyError(f"Unknown curriculum {name!r}. Available: {sorted(CURRICULA)}")
    return CURRICULA[name]
