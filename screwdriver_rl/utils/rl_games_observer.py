"""rl_games algo observer that adds curriculum-aware behaviour to Stage 1 PPO.

Two jobs on top of the stock :class:`IsaacAlgoObserver`:

1. **Epoch sync** — the true training epoch lives at the rl_games Runner/algo
   level, not in the env.  Each epoch we push it into the env so the terminal
   :class:`~screwdriver_rl.utils.logging.RotationTrainingLogger` can display it.

2. **End-of-phase checkpoints** — the curriculum advances Phase 1 → 2 → 3 as the
   global step counter crosses the per-phase ``step_start`` thresholds.  When a
   phase completes (the env reports a higher phase index) we save a tagged
   checkpoint ``<name>_phase{N}.pth`` capturing the policy at the end of that
   phase.  The final phase has no transition to trigger on, so it is saved by
   :meth:`save_final_phase`, which the caller invokes once training ends.

These checkpoints are *additional* to rl_games' own ``_best``/``_last`` saves.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from rl_games.common.algo_observer import IsaacAlgoObserver
except ImportError:  # very old Isaac Lab layout
    from isaaclab_tasks.utils.wrappers.rl_games import RlGamesAlgoObserver as IsaacAlgoObserver


class PhaseCheckpointObserver(IsaacAlgoObserver):
    """Sync the epoch into the env and save a checkpoint at each phase boundary.

    Parameters
    ----------
    env:
        The *unwrapped* screwdriver rotation env.  Must expose
        ``cfg.curriculum_phases`` (the phase list), ``_curriculum_phase`` (the
        currently active phase object), and ``_current_epoch`` (written here).
    """

    def __init__(self, env: Any) -> None:
        super().__init__()
        self._env = env
        self._algo = None
        self._saved_phases: set[int] = set()

    # ------------------------------------------------------------------

    def after_init(self, algo) -> None:
        super().after_init(algo)
        self._algo = algo

    def after_print_stats(self, frame, epoch_num, total_time) -> None:
        super().after_print_stats(frame, epoch_num, total_time)
        # Feed the true epoch to the env's terminal logger.
        self._env._current_epoch = int(epoch_num)
        # Save a tagged checkpoint for every phase that has fully completed.
        cur_idx = self._current_phase_index()
        for done_idx in range(cur_idx):
            self._save_phase(done_idx + 1, epoch_num)

    def save_final_phase(self) -> None:
        """Save the highest-reached phase as ``<name>_phase{N}.pth``.

        Called once after training ends to capture the final phase, which never
        completes via a curriculum transition.  No-op if the algo never ran
        (e.g. training failed before ``after_init``).
        """
        if self._algo is None:
            return
        cur_idx = self._current_phase_index()
        self._save_phase(cur_idx + 1, getattr(self._algo, "epoch_num", 0))

    # ------------------------------------------------------------------

    def _current_phase_index(self) -> int:
        phases = self._env.cfg.curriculum_phases
        return phases.index(self._env._curriculum_phase)

    def _save_phase(self, phase_num: int, epoch_num) -> None:
        if phase_num in self._saved_phases or self._algo is None:
            return
        self._saved_phases.add(phase_num)
        name = f"{self._algo.config['name']}_phase{phase_num}"
        path = os.path.join(self._algo.nn_dir, name)
        self._algo.save(path)
        print(
            f"[phase-checkpoint] saved {path}.pth (end of Phase {phase_num}, "
            f"epoch {int(epoch_num)})",
            flush=True,
        )
