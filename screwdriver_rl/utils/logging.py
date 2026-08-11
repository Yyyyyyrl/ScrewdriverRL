"""Terminal logging for the screwdriver rotation task.

Prints a formatted summary every ``log_interval_steps`` global steps.
The output is designed to give at-a-glance diagnostics for the most
common failure modes:

  - Oscillation ratio > 0.3  → net turns ≪ forward turns; back-and-forth.
  - Upright gate mean < 0.3  → screwdriver is consistently tilting.
  - Contact gate mean < 0.1  → not enough drive fingers engaged.
  - Rev-vel > Fwd-vel        → policy is pushing backward more than forward.
  - Idle count > 0           → some finger is hanging unused.
  - WrongSurf > 0            → back/knuckle/palm contact with the screwdriver.

The body adapts per task via the ``eval_in_window`` signature key in ``extras``
(see ``_render_body``): the force-window LinkerL20 layout when present, otherwise
the distance/motion-gated Allegro layout.  The shared frame (rule + Epoch line +
TOTAL REWARD footer) is task-independent.  Missing keys render as ``nan``
(``_mean`` returns NaN), so any task is safe.
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

    def __init__(self, log_interval_steps: int = 2000) -> None:
        self._interval = log_interval_steps
        self._last_log_step: int = 0
        self._start_time: float = time.monotonic()
        self._last_time: float = self._start_time

        # Print header once.
        print(self._header(), flush=True)

    # ------------------------------------------------------------------

    def log(
        self,
        global_steps: int,
        extras: dict[str, Any],
        epoch: int = 0,
        stage: int = 1,
        iter_num: int | None = None,
        total_iters: int | None = None,
        loss: float | None = None,
    ) -> None:
        # Stage 1 throttles to ~one line per ``_interval`` global steps; Stage 2
        # is driven once per adaptation iter by the trainer, so it controls its
        # own cadence and bypasses the throttle.
        if stage == 1 and global_steps - self._last_log_step < self._interval:
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

        # ---- Shared frame scalars ----
        phase      = m("eval_curriculum_phase")
        num_phases = m("eval_num_phases")
        total_rew  = m("eval_total_reward")

        if phase == phase and num_phases == num_phases:  # not NaN
            phase_lbl = f"Phase {int(phase)}/{int(num_phases)}"
        elif phase == phase:
            phase_lbl = f"Phase {int(phase)}"
        else:
            phase_lbl = "Phase ?"

        # ---- Stage 2: compact adaptation layout (no reward breakdown) ----
        if stage == 2:
            print(
                "\n".join(
                    self._render_stage2(
                        m, global_steps, elapsed, sps, phase_lbl,
                        iter_num, total_iters, loss,
                    )
                ),
                flush=True,
            )
            return

        # ``phase``/``num_phases`` arrive as floats (env emits a 1-indexed phase
        # number and the phase count); ``phase_lbl`` (computed above) renders
        # "Phase n/total" or "Phase ?" when the key is missing.
        # ---- Shared frame: top rule + Epoch line, TOTAL REWARD footer ----
        top = [
            f"{_W}{'─'*72}{_N}",
            (
                f"  {_B}Epoch{_N} {epoch:>8,}  "
                f"{_B}Step{_N} {global_steps:>12,}  "
                f"{_B}Elapsed{_N} {int(elapsed//3600):02d}h{int((elapsed%3600)//60):02d}m  "
                f"{_B}SPS{_N} {sps:>7,.0f}  "
                f"{_B}Curriculum{_N} {_W}{phase_lbl}{_N}"
            ),
        ]
        footer = [
            f"{_W}{'─'*72}{_N}",
            (
                f"  {_W}TOTAL REWARD{_N} {_colour(total_rew, 0.0, 1.0):>12}"
                f"   {_B}(mean per-step, Phase {int(phase) if phase == phase else '?'}){_N}"
            ),
        ]

        # ---- Task-adaptive body: LinkerL20 force-window vs Allegro distance gate ----
        body = self._render_body(m, extras)

        print("\n".join(top + body + footer), flush=True)

    # ------------------------------------------------------------------
    # Stage-2 (adaptation) compact renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render_stage2(
        m,
        global_steps: int,
        elapsed: float,
        sps: float,
        phase_lbl: str,
        iter_num: int | None,
        total_iters: int | None,
        loss: float | None,
    ) -> list[str]:
        """Compact Stage-2 (proprioceptive adaptation) block.

        Shows the adaptation loss + frozen-teacher rollout health.  The Stage-1
        reward breakdown / TOTAL REWARD footer are dropped because Stage 2
        discards reward (supervised MSE only).  Reuses the shared ``eval_*``
        keys, so it renders identically for both hands with no task branch.
        """
        if iter_num is not None and total_iters is not None:
            iter_str = f"{iter_num:>4,}/{total_iters:,}"
        elif iter_num is not None:
            iter_str = f"{iter_num:>4,}"
        else:
            iter_str = "?"
        loss_str = f"{loss:.6f}" if loss is not None and loss == loss else "nan"

        fwd_vel   = m("eval_fwd_vel")
        net_turns = m("eval_net_turns")
        osc_ratio = m("eval_osc_ratio_aggregate")
        if osc_ratio != osc_ratio:
            osc_ratio = m("eval_osc_ratio")
        u_gate    = m("eval_upright_gate")
        c_gate    = m("eval_contact_gate")
        osc_warn  = " ⚠ OSCILLATION" if osc_ratio > 0.35 else ""

        return [
            f"{_W}{'─'*72}{_N}",
            (
                f"  {_W}Stage 2{_N} · {_B}Iter{_N} {_W}{iter_str}{_N}   "
                f"{_B}Step{_N} {global_steps:>12,}   "
                f"{_B}Elapsed{_N} {int(elapsed//3600):02d}h{int((elapsed%3600)//60):02d}m   "
                f"{_B}SPS{_N} {sps:>7,.0f}"
            ),
            (
                f"  {_B}AdaptLoss{_N} {_W}{loss_str}{_N}   "
                f"{_B}Curriculum{_N} {_W}{phase_lbl}{_N}"
            ),
            (
                f"  {_W}Teacher{_N}  "
                f"FwdVel {_colour(fwd_vel, 0.1, 0.5)}  "
                f"NetTurns {_colour(net_turns, 0.0, 1.5)}  "
                f"OscRatio {_colour(osc_ratio, 0.4, 0.15, invert=True)}{osc_warn}"
            ),
            (
                f"           "
                f"UprightGate {_colour(u_gate, 0.3, 0.8)}  "
                f"ContactGate {_colour(c_gate, 0.2, 0.6)}"
            ),
            f"{_W}{'─'*72}{_N}",
        ]

    # ------------------------------------------------------------------
    # Task-adaptive body renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render_inhand_body(m) -> list[str]:
        """Compact body for the HORA free-object in-hand rotation task.

        The in-hand env has its own reward and fall semantics, but still exposes
        physical forward/reverse motion, net turns and oscillation.  Those are
        the long-run policy-quality signals; showing only clipped RotateReward
        previously hid a spin-and-drop or back-and-forth policy.
        """
        rot_rew   = m("eval_rotate_reward")
        fwd_vel   = m("eval_fwd_vel")
        rev_vel   = m("eval_rev_vel")
        net_turns = m("eval_net_turns")
        osc_ratio = m("eval_osc_ratio_aggregate")
        if osc_ratio != osc_ratio:
            osc_ratio = m("eval_osc_ratio")
        obj_z     = m("eval_obj_z")
        fall      = m("eval_fall_frac")
        c_gate    = m("eval_contact_gate")
        ep_len    = m("eval_ep_len")
        hold      = m("eval_hold_frac")
        linvel_c  = m("eval_linvel_cost")
        pose_c    = m("eval_pose_cost")
        torque_c  = m("eval_torque_cost")
        work_c    = m("eval_work_cost")
        fall_warn = " ⚠ DROPPING" if fall > 0.3 else ""
        return [
            f"  {_W}Rotation{_N}",
            (
                f"    RotateReward {_colour(rot_rew, 0.05, 0.3):>14}  "
                f"FwdVel {_colour(fwd_vel, 0.1, 0.5):>14}  "
                f"RevVel {_colour(rev_vel, 0.4, 0.1, invert=True):>14}"
            ),
            (
                f"    NetTurns {_colour(net_turns, 0.0, 1.5):>14}  "
                f"OscRatio {_colour(osc_ratio, 0.4, 0.15, invert=True):>14}  "
                f"ObjZ {obj_z:>7.3f}  "
                f"FallFrac {_colour(fall, 0.3, 0.02, invert=True):>14}{fall_warn}"
            ),
            (
                f"    ContactGate {_colour(c_gate, 0.2, 0.6):>14}"
            ),
            f"  {_W}Episodes (completed){_N}",
            (
                # EpLen = mean steps at episode end (timeout = max, 400 by default);
                # HoldFrac = fraction of episodes that timed out still holding.
                f"    EpLen {_colour(ep_len, 50.0, 300.0):>14}  "
                f"HoldFrac {_colour(hold, 0.1, 0.6):>14}"
            ),
            f"  {_W}Penalties (pre-weight){_N}",
            (
                f"    LinVel {linvel_c:>8.3f}  PoseDev {pose_c:>8.3f}  "
                f"Torque {torque_c:>8.3f}  Work {work_c:>8.3f}"
            ),
        ]

    @staticmethod
    def _render_body(m, extras: dict[str, Any]) -> list[str]:
        """Per-task body: the HORA in-hand layout (``eval_rotate_reward``), the
        force-window LinkerL20 layout (``eval_in_window``), and the
        distance/motion-gated Allegro layout (otherwise)."""
        if "eval_rotate_reward" in extras:
            return RotationTrainingLogger._render_inhand_body(m)

        fwd_turns = m("eval_total_turns")
        net_turns  = m("eval_net_turns")
        osc_ratio  = m("eval_osc_ratio")
        tilt_norm  = m("eval_tilt_norm")
        u_gate     = m("eval_upright_gate")
        c_gate     = m("eval_contact_gate")
        b_gate     = m("eval_binary_gate")
        cforce     = m("eval_contact_force")
        turn_vel   = m("eval_fwd_vel")
        rev_vel    = m("eval_rev_vel")

        turn_rew   = m("eval_turn_reward")
        rev_cost   = m("eval_reverse_cost")
        up_cost    = m("eval_upright_cost")
        act_cost   = m("eval_action_cost")

        # Failure-mode flags
        osc_warn  = " ⚠ OSCILLATION"   if osc_ratio > 0.35  else ""
        tilt_warn = " ⚠ TILT"          if tilt_norm > 0.4   else ""
        cont_warn = " ⚠ NO-CONTACT"    if b_gate < 0.15     else ""
        rev_warn  = " ⚠ BACKWARD"      if rev_vel > turn_vel else ""

        lines = [
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
        ]

        # Episode-level outcomes.  FwdTurns/NetTurns above are instantaneous
        # per-env accumulators mid-episode; these two are averages over completed
        # episodes and are the quantities eval reports.  Selecting checkpoints on
        # the mid-episode numbers alone has already gone wrong once.
        if "eval_ep_outcome_n" in extras and m("eval_ep_outcome_n") > 0:
            fall_rate = m("eval_ep_fall_rate")
            lines[1:1] = [
                (
                    f"    EpFallRate {_colour(fall_rate, 0.30, 0.05, invert=True):>13}  "
                    f"EpNetTurns {_colour(m('eval_ep_net_turns'), 0.0, 1.5):>13}  "
                    f"n={int(m('eval_ep_outcome_n')):>4d}"
                    f"{' ⚠ FALLING' if fall_rate > 0.25 else ''}"
                ),
            ]

        # Force-based (LinkerL20) vs distance/pad-based (legacy) contact layout.
        if "eval_in_window" in extras:
            in_win    = m("eval_in_window")
            drive_cnt = m("eval_drive_count")
            idle_cnt  = m("eval_idle_count")
            cap_force = m("eval_index_cap_force")
            wrong_f   = m("eval_wrong_surface_force")
            max_dev   = m("eval_max_joint_dev")
            idle_warn  = " ⚠ IDLE-FINGER"  if idle_cnt > 0.5     else ""
            wrong_warn = " ⚠ WRONG-SURF"   if wrong_f  > 0.5     else ""
            lines += [
                (
                    f"    ContactGate {_colour(c_gate, 0.2, 0.6):>14}  "
                    f"InWindow {_colour(in_win, 0.2, 0.6):>14}  "
                    f"DriveCnt {_colour(drive_cnt, 1.5, 3.0):>14}{cont_warn}"
                ),
                (
                    f"    ContactForce {cforce:>7.3f}N  "
                    f"IndexCapF {cap_force:>8.3f}N  "
                    f"IdleCnt {_colour(idle_cnt, 0.5, 0.0, invert=True):>14}{idle_warn}"
                ),
                (
                    f"    WrongSurf {_colour(wrong_f, 0.5, 0.0, invert=True):>13}N  "
                    f"MaxJointDev {max_dev:>7.3f}rad{wrong_warn}"
                ),
                (
                    f"    FwdVel {_colour(turn_vel, 0.1, 0.5):>14}  "
                    f"RevVel {_colour(rev_vel, 0.4, 0.1, invert=True):>14}{rev_warn}"
                ),
                f"  {_W}Reward breakdown{_N}",
                (
                    f"    TurnRew {turn_rew:>9.3f}  Authority "
                    f"{m('eval_contact_authority_reward'):>8.3f}  "
                    f"IdxCap {m('eval_index_cap_reward'):>8.3f}"
                ),
                (
                    f"    Drive {m('eval_drive_reward'):>8.3f}  "
                    f"Grip {m('eval_grip_reward'):>8.3f}"
                ),
                (
                    f"    RevCost {rev_cost:>9.3f}  Excess {m('eval_excess_cost'):>8.3f}  "
                    f"WrongC {m('eval_wrong_surface_cost'):>8.3f}  HomeDev {m('eval_home_dev_cost'):>7.3f}"
                ),
                (
                    f"    UprightCost {up_cost:>8.3f}  IdleCost {m('eval_idle_cost'):>8.3f}  "
                    f"ActionCost {act_cost:>7.3f}"
                ),
                # FallCost and TiltVel were the only reward terms with no log
                # readout, which made the one term big enough to dominate the
                # objective (a one-shot 30,000 at the terminating step) invisible
                # from the training log alone.  Both are now printed so the active
                # fall-penalty contract can be verified from the log rather than
                # inferred, per the "read the contract back" rule.
                (
                    f"    FallCost {m('eval_fall_cost'):>8.3f}  "
                    f"TiltVel {m('eval_tilt_vel_cost'):>8.3f}"
                ),
            ]
        else:
            mot_gate = m("eval_motion_gate")
            tip_dist = m("eval_min_tip_dist")
            near_rew = m("eval_near_reward")
            prox_cost = m("eval_proximal_cost")
            lines += [
                (
                    f"    ContactGate {_colour(c_gate, 0.2, 0.6):>14}  "
                    f"BinaryGate {_colour(b_gate, 0.2, 0.7):>14}  "
                    f"MotionGate {_colour(mot_gate, 0.2, 0.7):>14}{cont_warn}"
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
                    f"    UprightCost {up_cost:>8.3f}  ActionCost {act_cost:>7.3f}"
                ),
            ]
        return lines

    @staticmethod
    def _header() -> str:
        return (
            f"\n{'='*72}\n"
            f"  ScrewdriverRL — Continuous Rotation Training\n"
            f"  Colour guide: {_G}good{_N}  {_Y}ok{_N}  {_R}bad{_N}\n"
            f"  OscRatio < 0.15 = good  |  UprightGate > 0.8 = good\n"
            f"  ContactGate > 0.6 = good  |  RevVel < FwdVel = good\n"
            f"{'='*72}"
        )
