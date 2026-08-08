"""Tests for the task-adaptive terminal logger (``RotationTrainingLogger``).

These run on CPU without Isaac — ``screwdriver_rl.utils.logging`` only needs torch.
They lock in the per-task dispatch: the distance/motion-gated Allegro layout shows
its real terms with no ``nan`` (and no pad/force fields), while the LinkerL20
force-window layout is unchanged.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import torch

from screwdriver_rl.utils.logging import RotationTrainingLogger


def _tensorise(values: dict[str, float]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor([v], dtype=torch.float32) for k, v in values.items()}


# Shared frame keys emitted by every task.
_FRAME = {
    "eval_curriculum_phase": 1.0,
    "eval_num_phases": 3.0,
    "eval_total_reward": 0.5,
}

# Keys the restored 044e558 Allegro reward emits (distance+motion contact gate,
# near, milestone, proximal — no pad-facing, no contact force, no rotate-reward).
_ALLEGRO = {
    **_FRAME,
    "eval_total_turns": 0.5, "eval_net_turns": 0.4, "eval_osc_ratio": 0.1,
    "eval_tilt_norm": 0.15, "eval_upright_gate": 0.9,
    "eval_contact_gate": 0.7, "eval_binary_gate": 0.8, "eval_motion_gate": 0.6,
    "eval_min_tip_dist": 0.04, "eval_fwd_vel": 0.3, "eval_rev_vel": 0.02,
    "eval_turn_reward": 1.2, "eval_reverse_cost": 0.1, "eval_near_reward": 0.3,
    "eval_proximal_cost": 0.0, "eval_upright_cost": 0.2, "eval_action_cost": 0.05,
}

# Keys the force-based LinkerL20 reward emits.
_LINKER = {
    **_FRAME,
    "eval_total_turns": 0.5, "eval_net_turns": 0.4, "eval_osc_ratio": 0.1,
    "eval_tilt_norm": 0.15, "eval_upright_gate": 0.9, "eval_contact_gate": 0.7,
    "eval_binary_gate": 0.8, "eval_contact_force": 2.0, "eval_fwd_vel": 0.3,
    "eval_rev_vel": 0.02, "eval_turn_reward": 1.0, "eval_reverse_cost": 0.1,
    "eval_upright_cost": 0.2, "eval_action_cost": 0.05,
    # force-layout signature + its rows
    "eval_in_window": 0.6, "eval_drive_count": 2.0, "eval_idle_count": 0.0,
    "eval_index_cap_force": 1.5, "eval_wrong_surface_force": 0.0,
    "eval_max_joint_dev": 0.1, "eval_index_cap_reward": 0.3,
    "eval_contact_authority_reward": 1.1,
    "eval_drive_reward": 0.4, "eval_grip_reward": 0.2, "eval_excess_cost": 0.0,
    "eval_wrong_surface_cost": 0.0, "eval_home_dev_cost": 0.1, "eval_idle_cost": 0.0,
    "eval_fall_cost": 0.0, "eval_tilt_vel_cost": 0.3,
}


def _capture(extras: dict[str, float]) -> str:
    logger = RotationTrainingLogger(log_interval_steps=1)  # header prints in __init__
    buf = io.StringIO()
    with redirect_stdout(buf):
        logger.log(global_steps=10, extras=_tensorise(extras), epoch=3)
    return buf.getvalue()


def test_allegro_distance_layout_has_no_nan() -> None:
    out = _capture(_ALLEGRO)
    assert "nan" not in out, out
    # distance/motion-gated terms shown
    for label in ("Contact quality", "ContactGate", "BinaryGate", "MotionGate",
                  "MinTipDist", "TurnRew", "NearRew", "ProxCost", "UprightGate"):
        assert label in out, (label, out)
    # the pad/force fields and the HORA layout must be gone
    for absent in ("PadFactor", "PadCos", "ContactForce", "RotateRew",
                   "WorkCost", "InWindow", "DriveCnt"):
        assert absent not in out, (absent, out)


def test_linker_layout_unchanged() -> None:
    out = _capture(_LINKER)
    assert "nan" not in out, out
    # force-layout signature rows still render
    for label in ("Contact quality", "InWindow", "DriveCnt", "IndexCapF",
                  "Authority", "Grip", "WrongSurf"):
        assert label in out, (label, out)
    # Allegro-only / HORA terms must NOT leak into the Linker layout
    for absent in ("RotateRew", "WorkCost", "NearRew", "MinTipDist"):
        assert absent not in out, (absent, out)


def test_episode_outcomes_render_only_once_populated() -> None:
    # fall_rate / authorized net turns are the metrics eval decides on; they must
    # be visible in the training log so checkpoint choice is not driven by the
    # mid-episode FwdVel/NetTurns readouts alone.
    warm_up = {**_LINKER, "eval_ep_outcome_n": 0.0,
               "eval_ep_fall_rate": 0.0, "eval_ep_net_turns": 0.0}
    assert "EpFallRate" not in _capture(warm_up)

    populated = {**_LINKER, "eval_ep_outcome_n": 512.0,
                 "eval_ep_fall_rate": 0.08, "eval_ep_net_turns": 1.2}
    out = _capture(populated)
    assert "nan" not in out, out
    for label in ("EpFallRate", "EpNetTurns", "n= 512"):
        assert label in out, (label, out)
    assert "⚠ FALLING" not in out

    falling = {**populated, "eval_ep_fall_rate": 0.42}
    assert "⚠ FALLING" in _capture(falling)


def test_dispatch_is_by_in_window_key() -> None:
    # eval_in_window selects the force layout; without it, the distance layout.
    assert "InWindow" in _capture(_LINKER)
    assert "InWindow" not in _capture(_ALLEGRO) and "MinTipDist" in _capture(_ALLEGRO)
