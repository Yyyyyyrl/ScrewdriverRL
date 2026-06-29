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
    """Unused since Stage-2 progress moved to the env's terminal logger (one
    compact block per iter).  Retained for backward compatibility with callers
    that still pass it."""
    save_interval: int = 50
    """Write an intermediate checkpoint every N iterations (0 disables).  Each
    cadence point writes ``proprio_adapt_iter_<it>.pth`` and overwrites a
    rolling ``proprio_adapt_last.pth``, so an interrupted run loses at most
    ``save_interval`` iterations of work instead of everything."""


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
        self._net_dims = {"frame_dim": frame_dim, "hist_len": hist_len, "out_dim": priv_obs_dim}
        # Base (unwrapped) env, used to drive the Stage-2 terminal log.  Resolved
        # via getattr so the no-Isaac FakeEnv in tests works unchanged.
        self._env_unwrapped = getattr(env, "unwrapped", env)
        self._last_extras: dict = {}
        os.makedirs(out_dir, exist_ok=True)

    def _collect(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out the frozen policy and collect (proprio_hist, priv_obs) pairs."""
        hists, privs = [], []
        obs_dict, _ = self.env.reset()
        info: dict = {}
        for _ in range(self.cfg.rollout_steps):
            with torch.no_grad():
                action = self.actor(obs_dict["policy"])
            obs_dict, _, terminated, truncated, info = self.env.step(action)
            hists.append(obs_dict["proprio_hist"].detach())  # (N, T, D)
            privs.append(obs_dict["critic"].detach())        # (N, priv_dim)
        # Keep the final step's info (env extras) for the per-iter Stage-2 log.
        self._last_extras = info if isinstance(info, dict) else {}
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

    def _save_ckpt(self, filename: str, it: int, loss: float) -> str:
        """Atomically write a checkpoint to ``out_dir/filename``.

        Keeps the ``"net"`` key that loaders (play/eval, test_algo) expect and
        adds the iteration, loss, and network dims so a checkpoint is
        self-describing.  Writes to a temp file then ``os.replace``s it into
        place, so a crash mid-save cannot leave a truncated/corrupt checkpoint.
        """
        path = os.path.join(self.out_dir, filename)
        tmp = path + ".tmp"
        torch.save(
            {
                "net": self.net.state_dict(),
                "iter": it,
                "loss": loss,
                "net_dims": self._net_dims,
            },
            tmp,
        )
        os.replace(tmp, path)
        return path

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
        # Put the env into Stage-2 logging mode: its per-step log is suppressed
        # and we drive the logger once per iter below (see env _get_rewards).
        setattr(self._env_unwrapped, "_log_stage", 2)

        loss = float("nan")
        for it in range(1, self.cfg.num_iters + 1):
            hists, privs = self._collect()
            loss = self._train_on_batch(hists, privs)

            # Drive the env's terminal logger once per iter with the compact
            # Stage-2 layout (adapt loss + frozen-teacher rollout health).  All
            # access is getattr-guarded so the no-Isaac FakeEnv in tests no-ops.
            setattr(self._env_unwrapped, "_current_epoch", it)
            setattr(self._env_unwrapped, "_stage2_loss", loss)
            logger = getattr(self._env_unwrapped, "_logger", None)
            if logger is not None:
                logger.log(
                    getattr(self._env_unwrapped, "_global_steps", 0),
                    self._last_extras,
                    stage=2,
                    iter_num=it,
                    total_iters=self.cfg.num_iters,
                    loss=loss,
                )

            # Intermediate checkpoints so an interrupted run loses at most
            # save_interval iters.  The final iter is skipped here — it is saved
            # canonically as proprio_adapt.pth below.
            is_last = it == self.cfg.num_iters
            if self.cfg.save_interval > 0 and it % self.cfg.save_interval == 0 and not is_last:
                snap = self._save_ckpt(f"proprio_adapt_iter_{it:05d}.pth", it, loss)
                self._save_ckpt("proprio_adapt_last.pth", it, loss)
                print(
                    f"  [stage2] checkpoint → {os.path.basename(snap)} "
                    f"(+ proprio_adapt_last.pth)  loss {loss:.5f}",
                    flush=True,
                )

        ckpt_path = self._save_ckpt("proprio_adapt.pth", self.cfg.num_iters, loss)
        print(f"\n[stage2] Saved adaptation network → {ckpt_path}\n", flush=True)
        return ckpt_path
