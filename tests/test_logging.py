"""Unit tests for the terminal logger and the phase-checkpoint observer.

No Isaac Sim required — both are exercised with synthetic data / fakes.

Run:  python -m pytest tests/test_logging.py -q   (or python tests/test_logging.py)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.utils.logging import RotationTrainingLogger  # noqa: E402
from screwdriver_rl.utils.rl_games_observer import PhaseCheckpointObserver  # noqa: E402


def _extras(phase: int, num_phases: int = 3) -> dict:
    """A minimal extras dict with a couple of representative metrics."""
    one = torch.ones(4)
    keys = [
        "eval_total_turns", "eval_net_turns", "eval_osc_ratio", "eval_tilt_norm",
        "eval_upright_gate", "eval_contact_gate", "eval_binary_gate",
        "eval_motion_gate", "eval_pad_gate", "eval_pad_cos", "eval_contact_force",
        "eval_fwd_vel", "eval_rev_vel", "eval_min_tip_dist", "eval_turn_reward",
        "eval_reverse_cost", "eval_near_reward", "eval_proximal_cost",
        "eval_upright_cost", "eval_action_cost",
    ]
    extras = {k: one * 0.5 for k in keys}
    extras["eval_total_reward"] = one * 1.25
    extras["eval_curriculum_phase"] = one * float(phase)
    extras["eval_num_phases"] = one * float(num_phases)
    return extras


def test_logger_shows_epoch_phase_and_total(capsys):
    logger = RotationTrainingLogger(log_interval_steps=2000)
    # First call below the interval should not print a block.
    logger.log(1000, _extras(phase=2), epoch=5)
    capsys.readouterr()  # drop the header + nothing
    # Crossing the interval prints a block.
    logger.log(2000, _extras(phase=2), epoch=5)
    out = capsys.readouterr().out
    assert "Epoch" in out and "5" in out
    assert "Phase 2/3" in out
    assert "TOTAL REWARD" in out
    assert "1.250" in out


def test_logger_handles_missing_phase(capsys):
    logger = RotationTrainingLogger(log_interval_steps=1)
    extras = _extras(phase=1)
    del extras["eval_curriculum_phase"]
    del extras["eval_num_phases"]
    logger.log(10, extras, epoch=0)
    out = capsys.readouterr().out
    assert "Phase ?" in out


# --------------------------------------------------------------------------
# Phase-checkpoint observer
# --------------------------------------------------------------------------

class _FakePhase:
    pass


class _FakeCfg:
    def __init__(self, phases):
        self.curriculum_phases = phases


class _FakeEnv:
    def __init__(self, phases):
        self.cfg = _FakeCfg(phases)
        self._curriculum_phase = phases[0]
        self._current_epoch = 0


class _FakeAlgo:
    def __init__(self, nn_dir):
        self.nn_dir = str(nn_dir)
        self.config = {"name": "model"}
        self.epoch_num = 0
        self.saved = []
        # Attributes the stock IsaacAlgoObserver.after_init touches.
        self.games_to_track = 100
        self.ppo_device = "cpu"
        self.device = "cpu"
        self.writer = None

    def save(self, fn):
        self.saved.append(fn)
        Path(fn + ".pth").write_text("ckpt")


def test_observer_saves_checkpoint_at_each_phase(tmp_path):
    phases = [_FakePhase(), _FakePhase(), _FakePhase()]
    env = _FakeEnv(phases)
    algo = _FakeAlgo(tmp_path)

    obs = PhaseCheckpointObserver(env)
    obs.after_init(algo)

    # Phase 1 (index 0): no completed phase yet.
    obs.after_print_stats(frame=1, epoch_num=1, total_time=1.0)
    assert env._current_epoch == 1
    assert not list(tmp_path.glob("*.pth"))

    # Advance to Phase 2 (index 1): Phase 1 is now complete -> phase1 ckpt.
    env._curriculum_phase = phases[1]
    obs.after_print_stats(frame=2, epoch_num=2, total_time=2.0)
    assert (tmp_path / "model_phase1.pth").exists()

    # Advance to Phase 3 (index 2): Phase 2 complete -> phase2 ckpt.
    env._curriculum_phase = phases[2]
    obs.after_print_stats(frame=3, epoch_num=3, total_time=3.0)
    assert (tmp_path / "model_phase2.pth").exists()

    # Idempotent: re-running the same phase does not re-save.
    n_saves = len(algo.saved)
    obs.after_print_stats(frame=4, epoch_num=4, total_time=4.0)
    assert len(algo.saved) == n_saves

    # Final phase saved explicitly at training end.
    obs.save_final_phase()
    assert (tmp_path / "model_phase3.pth").exists()


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["python", "-m", "pytest", __file__, "-q"]))
