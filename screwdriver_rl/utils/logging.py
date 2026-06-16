"""Terminal logging for the screwdriver rotation task.

Prints a formatted summary every ``log_interval_steps`` global steps.
The output is designed to give at-a-glance diagnostics for the most
common failure modes:

  - Oscillation ratio > 0.3  → net turns ≪ forward turns; back-and-forth.
  - Upright gate mean < 0.3  → screwdriver is consistently tilting.
  - Contact gate mean < 0.1  → fingers are not staying near the handle.
  - Rev-vel > Fwd-vel        → policy is pushing backward more than forward.
  - Proximal cost > 0        → palm/knuckle contacts occurring (Phase 2+).
"""

from __future__ import annotations

import time
from typing import Any

import torch


# ANSI colours (disabled on non-TTY)
import sys

_USE_COLOUR = sys.stdout.isatty()
_R = "\033[91m" if _USE_COLOUR else ""   # red
_Y = "\033[93m" if _USE_COLOUR else ""   # yellow
_G = "\033[92m" if _USE_COLOUR else ""   # green
_B = "\033[96m" if _USE_COLOUR else ""   # cyan
_W = "\033[97m" if _USE_COLOUR else ""   # white bold
_N = "\033[0m"  if _USE_COLOUR else ""   # reset


def _mean(t: torch.Tensor | None) -> float:
    if t is None:
        return float("nan")
    return t.mean().item()


def _colour(value: float, lo: float, hi: float, invert: bool = False) -> str:
    """Return value string wrapped in a colour based on thresholds."""
    good = value >= hi if not invert else value <= lo
    bad = value <= lo if not invert else value >= hi
    if good:
        return f"{_G}{value:.3f}{_N}"
    if bad:
        return f"{_R}{value:.3f}{_N}"
    return f"{_Y}{value:.3f}{_N}"


class RotationTrainingLogger:
    """Periodic terminal logger for screwdriver rotation training.

    Parameters
    ----------
    log_interval_steps:
        Approximate number of global env steps between log lines.
        Actual interval may differ slightly due to batched step counting.
    """

    def __init__(self, log_interval_steps: int = 500) -> None:
        self._interval = log_interval_steps
        self._last_log_step: int = 0
        self._start_time: float = time.monotonic()
        self._last_time: float = self._start_time

        # Print header once.
        print(self._header(), flush=True)

    # ------------------------------------------------------------------

    def log(self, global_steps: int, extras: dict[str, Any]) -> None:
        if global_steps - self._last_log_step < self._interval:
            return
        delta_steps = global_steps - self._last_log_step
        now = time.monotonic()
        elapsed = now - self._start_time
        interval_s = now - self._last_time
        self._last_log_step = global_steps
        self._last_time = now

        sps = delta_steps / max(interval_s, 1e-6)

        def m(key: str) -> float:
            v = extras.get(key)
            return _mean(v) if isinstance(v, torch.Tensor) else float("nan")

        fwd_turns = m("eval_total_turns")
        net_turns  = m("eval_net_turns")
        osc_ratio  = m("eval_osc_ratio")
        tilt_norm  = m("eval_tilt_norm")
        u_gate     = m("eval_upright_gate")
        c_gate     = m("eval_contact_gate")
        b_gate     = m("eval_binary_gate")
        mot_gate   = m("eval_motion_gate")
        pad_fac    = m("eval_pad_gate")
        pad_cos    = m("eval_pad_cos")
        cforce     = m("eval_contact_force")
        turn_vel   = m("eval_fwd_vel")
        rev_vel    = m("eval_rev_vel")
        tip_dist   = m("eval_min_tip_dist")
        phase      = m("eval_curriculum_phase")

        turn_rew   = m("eval_turn_reward")
        rev_cost   = m("eval_reverse_cost")
        near_rew   = m("eval_near_reward")
        prox_cost  = m("eval_proximal_cost")
        up_cost    = m("eval_upright_cost")
        act_cost   = m("eval_action_cost")
        total_rew  = m("eval_total_reward")

        # Failure-mode flags
        osc_warn  = " ⚠ OSCILLATION"   if osc_ratio > 0.35  else ""
        tilt_warn = " ⚠ TILT"          if tilt_norm > 0.4   else ""
        cont_warn = " ⚠ NO-CONTACT"    if b_gate < 0.15     else ""
        rev_warn  = " ⚠ BACKWARD"      if rev_vel > turn_vel else ""

        phase_str = f"Ph@{int(phase):,}" if not isinstance(phase, float) or not phase != phase else "Ph?"

        lines = [
            f"{_W}{'─'*72}{_N}",
            (
                f"  {_B}Step{_N} {global_steps:>12,}  "
                f"{_B}Elapsed{_N} {int(elapsed//3600):02d}h{int((elapsed%3600)//60):02d}m  "
                f"{_B}SPS{_N} {sps:>8,.0f}  "
                f"{_B}Curriculum{_N} {phase_str}"
            ),
            f"  {_W}Progress{_N}",
            (
                f"    FwdTurns {_colour(fwd_turns, 0.0, 2.0):>14}  "
                f"NetTurns {_colour(net_turns, 0.0, 1.5):>14}  "
                f"OscRatio {_colour(osc_ratio, 0.4, 0.15, invert=True):>14}{osc_warn}"
            ),
            f"  {_W}Object state{_N}",
            (
                f"    TiltNorm {_colour(tilt_norm, 0.5, 0.15, invert=True):>14}  "
                f"UprightGate {_colour(u_gate, 0.3, 0.8):>14}{tilt_warn}"
            ),
            f"  {_W}Contact quality{_N}",
            (
                f"    ContactGate {_colour(c_gate, 0.2, 0.6):>14}  "
                f"BinaryGate {_colour(b_gate, 0.2, 0.7):>14}  "
                f"MotionGate {_colour(mot_gate, 0.2, 0.7):>14}{cont_warn}"
            ),
            (
                f"    PadFactor {_colour(pad_fac, 0.2, 0.7):>14}  "
                f"PadCos {_colour(pad_cos, 0.0, 0.5):>16}  "
                f"ContactForce {cforce:>7.3f}N"
            ),
            (
                f"    MinTipDist {_colour(tip_dist, 0.08, 0.035, invert=True):>13}  "
                f"FwdVel {_colour(turn_vel, 0.1, 0.5):>14}  "
                f"RevVel {_colour(rev_vel, 0.4, 0.1, invert=True):>14}{rev_warn}"
            ),
            f"  {_W}Reward breakdown{_N}",
            (
                f"    TurnRew {turn_rew:>10.3f}  RevCost {rev_cost:>9.3f}  "
                f"NearRew {near_rew:>9.3f}  ProxCost {prox_cost:>8.3f}"
            ),
            (
                f"    UprightCost {up_cost:>8.3f}  ActionCost {act_cost:>7.3f}  "
                f"TotalRew {total_rew:>9.3f}"
            ),
        ]
        print("\n".join(lines), flush=True)

    @staticmethod
    def _header() -> str:
        return (
            f"\n{'='*72}\n"
            f"  ScrewdriverRL — Allegro Continuous Rotation Training\n"
            f"  Colour guide: {_G}good{_N}  {_Y}ok{_N}  {_R}bad{_N}\n"
            f"  OscRatio < 0.15 = good  |  UprightGate > 0.8 = good\n"
            f"  ContactGate > 0.6 = good  |  RevVel < FwdVel = good\n"
            f"{'='*72}"
        )
