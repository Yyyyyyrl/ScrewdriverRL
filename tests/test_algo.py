"""Tests for the Stage-2 proprioceptive-adaptation algorithm. No Isaac Sim.

Covers the adaptation network shapes, its ability to fit a supervised target,
and the end-to-end ``ProprioAdaptTrainer`` loop driven by a fake env that mimics
the ``DirectRLEnv`` interface the trainer consumes:
``reset() -> (obs_dict, info)`` and
``step(a) -> (obs_dict, rew, terminated, truncated, info)``.

Run:  python tests/test_algo.py   (or  python -m pytest tests/ -q)
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.algos.proprio_adapt import (  # noqa: E402
    AdaptTrainCfg,
    ProprioAdaptNet,
    ProprioAdaptTrainer,
)

DEVICE = "cpu"


class FakeEnv:
    """Minimal stand-in for the screwdriver env's gym interface.

    Produces the three observation groups the trainer reads (``policy``,
    ``critic``, ``proprio_hist``).  The privileged ``critic`` vector is a fixed
    linear function of the latest proprio frame (plus small noise) so the
    adaptation network has a learnable signal rather than pure noise.
    """

    def __init__(self, num_envs=16, policy_dim=27, priv_dim=17, hist_len=30,
                 frame_dim=24, act_dim=12, device=DEVICE):
        self.num_envs = num_envs
        self.device = device
        self._policy_dim = policy_dim
        self._priv_dim = priv_dim
        self._hist_len = hist_len
        self._frame_dim = frame_dim
        self._act_dim = act_dim
        # Fixed (unknown to the net) mapping latest-frame -> privileged obs.
        self._proj = torch.randn(frame_dim, priv_dim, device=device)

    def _obs(self):
        hist = torch.randn(self.num_envs, self._hist_len, self._frame_dim, device=self.device)
        critic = hist[:, -1] @ self._proj + 0.01 * torch.randn(self.num_envs, self._priv_dim, device=self.device)
        return {
            "policy": torch.randn(self.num_envs, self._policy_dim, device=self.device),
            "critic": critic,
            "proprio_hist": hist,
        }

    def reset(self):
        return self._obs(), {}

    def step(self, actions):
        rew = -actions.pow(2).sum(dim=-1)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(terminated)
        return self._obs(), rew, terminated, truncated, {}


def test_net_output_shapes():
    net = ProprioAdaptNet(frame_dim=24, hist_len=30, out_dim=17)
    out = net(torch.randn(8, 30, 24))
    assert out.shape == (8, 17), f"got {tuple(out.shape)}"
    # Custom dims also resolve the conv-flatten size dynamically.
    net2 = ProprioAdaptNet(frame_dim=12, hist_len=40, out_dim=9)
    assert net2(torch.randn(4, 40, 12)).shape == (4, 9)


def test_net_overfits_supervised_batch():
    """The network must be able to reduce MSE on a fixed (hist -> priv) batch."""
    torch.manual_seed(0)
    net = ProprioAdaptNet(frame_dim=24, hist_len=30, out_dim=17)
    proj = torch.randn(24, 17)
    hist = torch.randn(64, 30, 24)
    target = hist[:, -1] @ proj
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.mse_loss

    init_loss = loss_fn(net(hist), target).item()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(net(hist), target)
        loss.backward()
        opt.step()
    final_loss = loss_fn(net(hist), target).item()
    assert final_loss < 0.5 * init_loss, f"loss did not drop: {init_loss:.4f} -> {final_loss:.4f}"


def test_trainer_runs_and_saves():
    env = FakeEnv()
    actor = lambda obs: torch.zeros(obs.shape[0], 12, device=obs.device)  # noqa: E731
    cfg = AdaptTrainCfg(
        rollout_steps=4, num_iters=3, batch_size=32,
        num_epochs_per_iter=2, log_interval=100,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = ProprioAdaptTrainer(
            env=env, stage1_actor_fn=actor, cfg=cfg, out_dir=tmp, device=DEVICE,
            priv_obs_dim=17, frame_dim=24, hist_len=30,
        )
        ckpt = trainer.train()
        assert Path(ckpt).exists(), "trainer did not save a checkpoint"
        # The saved checkpoint must reload into a matching network.
        state = torch.load(ckpt, map_location=DEVICE)
        net = ProprioAdaptNet(frame_dim=24, hist_len=30, out_dim=17)
        net.load_state_dict(state["net"])


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    raise SystemExit(1 if failures else 0)
