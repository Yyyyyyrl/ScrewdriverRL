"""End-to-end trainer tests on a fake env. No Isaac Sim required.

The FakeEnv mimics the DirectRLEnv interface the trainers consume:
``reset() -> (obs_dict, extras)`` and
``step(a) -> (obs_dict, rew, terminated, timed_out, extras)``.

Run:  python tests/test_algo.py
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.algo import PPO, ProprioAdapt  # noqa: E402
from screwdriver_rl.configs import (  # noqa: E402
    AdvanceCriteria,
    CurriculumPhase,
    NetworkCfg,
    PPOTrainCfg,
    StudentTrainCfg,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class FakeEnv:
    """Random-dynamics stand-in with the right shapes and extras keys."""

    def __init__(self, num_envs=16, obs_dim=24, act_dim=12, priv_dim=22, hist_len=30):
        self.num_envs = num_envs
        self.device = DEVICE
        self.single_action_space = gym.spaces.Box(-1, 1, (act_dim,), dtype=np.float32)
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)}
        )
        self.cfg = SimpleNamespace(
            privileged_obs_dim=priv_dim,
            prop_hist_len=hist_len,
            history_obs_dim=obs_dim,
            reward_turn_weight=0.0,  # curriculum override target
            dr=SimpleNamespace(obs_noise_std=0.0),
        )
        self._obs_dim, self._act_dim, self._priv_dim, self._hist_len = obs_dim, act_dim, priv_dim, hist_len
        self._t = torch.zeros(num_envs, device=DEVICE)

    def _obs(self):
        return {
            "policy": torch.randn(self.num_envs, self._obs_dim, device=DEVICE),
            "critic": torch.randn(self.num_envs, self._priv_dim, device=DEVICE),
            "proprio_hist": torch.randn(self.num_envs, self._hist_len, self._obs_dim, device=DEVICE),
        }

    def reset(self):
        self._t.zero_()
        return self._obs(), {}

    def step(self, actions):
        self._t += 1
        rew = -actions.pow(2).sum(dim=-1)
        timed_out = self._t >= 20
        terminated = torch.zeros_like(timed_out)
        self._t[timed_out] = 0
        extras = {
            "eval_net_turns": torch.rand(self.num_envs, device=DEVICE),
            "eval_forward_turn_velocity": torch.full((self.num_envs,), 0.5, device=DEVICE),
            "eval_reverse_turn_velocity": torch.full((self.num_envs,), 0.1, device=DEVICE),
            "eval_screwdriver_upright_norm": torch.full((self.num_envs,), 0.05, device=DEVICE),
            "eval_mean_fingertip_dist": torch.full((self.num_envs,), 0.03, device=DEVICE),
        }
        return self._obs(), rew, terminated, timed_out, extras


def test_ppo_trains_and_curriculum_advances():
    env = FakeEnv()
    cfg = PPOTrainCfg(horizon_length=8, minibatch_size=64, mini_epochs=2, max_agent_steps=8 * 16 * 6)
    curriculum = [
        CurriculumPhase(
            "p1",
            {"reward_turn_weight": 123.0},
            AdvanceCriteria(
                min_phase_steps=0,
                min_episode_length=1,
                metric_bounds={"eval_net_turns": (">=", 0.0)},
                min_fwd_minus_rev=0.1,
            ),
        ),
        CurriculumPhase("p2", {"reward_turn_weight": 456.0}, None),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        trainer = PPO(env, tmp, cfg=cfg, net_cfg=NetworkCfg(), curriculum=curriculum, device=DEVICE)
        assert env.cfg.reward_turn_weight == 123.0, "phase-1 override not applied"
        trainer.train()
        assert env.cfg.reward_turn_weight == 456.0, "curriculum did not advance to phase 2"
        assert trainer.agent_steps >= cfg.max_agent_steps

        # checkpoint round-trip with training state
        ckpt = str(Path(tmp) / "stage1_nn" / "last")
        trainer.save(ckpt)
        trainer2 = PPO(env, tmp, cfg=cfg, net_cfg=NetworkCfg(), curriculum=curriculum, device=DEVICE)
        trainer2.restore(ckpt + ".pth")
        assert trainer2.agent_steps == trainer.agent_steps
        assert trainer2.phase_idx == trainer.phase_idx
        return ckpt + ".pth"


def test_padapt_trains_from_teacher(teacher_ckpt):
    env = FakeEnv()
    cfg = StudentTrainCfg(max_agent_steps=16 * 60, bc_weight=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        student = ProprioAdapt(env, tmp, cfg=cfg, net_cfg=NetworkCfg(), device=DEVICE)
        student.restore_from_teacher(teacher_ckpt)
        # Only adapt_tconv should be trainable.
        trainable = {n for n, p in student.model.named_parameters() if p.requires_grad}
        assert trainable and all("adapt_tconv" in n for n in trainable)
        frozen_before = student.model.actor_mlp.mlp[0].weight.clone()
        student.train()
        assert torch.equal(frozen_before, student.model.actor_mlp.mlp[0].weight), "actor weights moved"


if __name__ == "__main__":
    import shutil

    keep_dir = Path(tempfile.mkdtemp())
    try:
        ckpt = test_ppo_trains_and_curriculum_advances()
        # the tempdir from the ppo test is gone; save a fresh teacher ckpt for padapt
        env = FakeEnv()
        trainer = PPO(env, str(keep_dir), cfg=PPOTrainCfg(horizon_length=8, minibatch_size=64,
                      mini_epochs=1, max_agent_steps=8 * 16), net_cfg=NetworkCfg(), device=DEVICE)
        trainer.train()
        teacher = str(keep_dir / "stage1_nn" / "final.pth")
        print("[PASS] test_ppo_trains_and_curriculum_advances")
        test_padapt_trains_from_teacher(teacher)
        print("[PASS] test_padapt_trains_from_teacher")
    finally:
        shutil.rmtree(keep_dir, ignore_errors=True)
    print("all algo tests passed")
