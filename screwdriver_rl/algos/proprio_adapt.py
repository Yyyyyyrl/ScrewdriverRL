"""Proprioceptive adaptation network and Stage 2 training loop.

Stage 2 of the RMA pipeline:
  1. Load the frozen Stage 1 actor.
  2. Roll out the environment collecting (proprio_history, privileged_obs) pairs.
  3. Train ProprioAdaptNet to predict privileged_obs from proprio_history alone,
     so at deployment the policy can infer environment properties without
     access to ground-truth state (friction, exact pose, etc.).

The adaptation network is a lightweight temporal conv (following HORA's
ProprioAdaptTConv) that maps the last 30 policy steps of [finger_q, targets]
to the 17-D privileged observation vector used by the Stage 1 critic.

Usage (called from train.py --stage 2):
  ProprioAdaptTrainer(env, stage1_checkpoint, cfg).train()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adaptation network
# ---------------------------------------------------------------------------

class ProprioAdaptNet(nn.Module):
    """Maps proprioceptive history (T × frame_dim) → privileged_obs_dim.

    Architecture (HORA-style temporal conv):
      frame_dim  →  frame encoder  →  32-D per step
      30 steps   →  1D conv stack  →  pooled 96-D
                 →  linear         →  privileged_obs_dim
    """

    def __init__(self, frame_dim: int = 24, hist_len: int = 30, out_dim: int = 17) -> None:
        super().__init__()
        self.frame_enc = nn.Sequential(
            nn.Linear(frame_dim, 32), nn.ELU(),
            nn.Linear(32, 32), nn.ELU(),
        )
        # Input to conv: (batch, channels=32, seq=hist_len)
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=9, stride=2),  # → seq ≈ 11
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),  # → seq ≈ 7
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),  # → seq ≈ 3
            nn.ELU(),
        )
        # Determine flattened size from a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 32, hist_len)
            flat = self.temporal(dummy).flatten(1).shape[1]
        self.head = nn.Linear(flat, out_dim)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hist: (batch, hist_len, frame_dim)
        Returns:
            pred: (batch, out_dim)
        """
        b, t, d = hist.shape
        frames = self.frame_enc(hist.view(b * t, d)).view(b, t, 32)
        # (batch, channels, seq) for Conv1d
        x = self.temporal(frames.permute(0, 2, 1))
        return self.head(x.flatten(1))


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptTrainCfg:
    rollout_steps: int = 512
    """Policy steps to collect per data-gathering iteration."""
    num_iters: int = 500
    """Training iterations (each collects rollout_steps × num_envs transitions)."""
    batch_size: int = 4096
    learning_rate: float = 1e-3
    num_epochs_per_iter: int = 5
    """Gradient epochs over each collected batch."""
    log_interval: int = 20
    """Print a summary every N training iterations."""


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ProprioAdaptTrainer:
    """Stage 2 teacher-student adaptation trainer."""

    def __init__(
        self,
        env,
        stage1_actor_fn,  # callable: obs_tensor → action_tensor (frozen policy)
        cfg: AdaptTrainCfg,
        out_dir: str,
        device: str = "cuda:0",
        priv_obs_dim: int = 17,
        frame_dim: int = 24,
        hist_len: int = 30,
    ) -> None:
        self.env = env
        self.actor = stage1_actor_fn
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = device

        self.net = ProprioAdaptNet(frame_dim=frame_dim, hist_len=hist_len, out_dim=priv_obs_dim).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        os.makedirs(out_dir, exist_ok=True)

    def _collect(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out the frozen policy and collect (proprio_hist, priv_obs) pairs."""
        hists, privs = [], []
        obs_dict, _ = self.env.reset()
        for _ in range(self.cfg.rollout_steps):
            with torch.no_grad():
                action = self.actor(obs_dict["policy"])
            obs_dict, _, terminated, truncated, _ = self.env.step(action)
            hists.append(obs_dict["proprio_hist"].detach())  # (N, T, D)
            privs.append(obs_dict["critic"].detach())        # (N, priv_dim)
        # (rollout_steps × N, ...)
        return torch.cat(hists, dim=0), torch.cat(privs, dim=0)

    def _train_on_batch(self, hists: torch.Tensor, privs: torch.Tensor) -> float:
        """One pass of supervised regression on the collected batch."""
        n = hists.shape[0]
        indices = torch.randperm(n, device=self.device)
        total_loss = 0.0
        steps = 0
        bs = self.cfg.batch_size
        for epoch in range(self.cfg.num_epochs_per_iter):
            for start in range(0, n, bs):
                idx = indices[start:start + bs]
                pred = self.net(hists[idx])
                loss = F.mse_loss(pred, privs[idx])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                steps += 1
        return total_loss / max(steps, 1)

    def train(self) -> str:
        """Run full Stage 2 training.  Returns path to the saved checkpoint."""
        print(
            f"\n{'='*60}\n"
            f"  Stage 2 — Proprioceptive Adaptation Training\n"
            f"  Iters: {self.cfg.num_iters}  |  "
            f"Rollout steps/iter: {self.cfg.rollout_steps}\n"
            f"  Network params: {sum(p.numel() for p in self.net.parameters()):,}\n"
            f"{'='*60}\n",
            flush=True,
        )
        t0 = time.monotonic()
        for it in range(1, self.cfg.num_iters + 1):
            hists, privs = self._collect()
            loss = self._train_on_batch(hists, privs)

            if it % self.cfg.log_interval == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"  iter {it:>4}/{self.cfg.num_iters}  "
                    f"loss {loss:.5f}  "
                    f"elapsed {int(elapsed//60):02d}m{int(elapsed%60):02d}s",
                    flush=True,
                )

        ckpt_path = os.path.join(self.out_dir, "proprio_adapt.pth")
        torch.save({"net": self.net.state_dict()}, ckpt_path)
        print(f"\n[stage2] Saved adaptation network → {ckpt_path}\n", flush=True)
        return ckpt_path
