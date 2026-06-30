"""Proprioceptive adaptation network and Stage 2 training loop.

Stage 2 of the RMA pipeline (HORA-faithful latent mode):
  1. Load the frozen Stage 1 actor (incl. its privileged encoder ``env_mlp``).
  2. Roll out the environment collecting (proprio_history, teacher_latent) pairs,
     where ``teacher_latent = tanh(env_mlp(normalize(privileged)))`` is exactly
     the latent the Stage-1 actor consumed.
  3. Train ProprioAdaptNet to reproduce that latent from proprio_history alone,
     so at deployment the frozen actor runs on a proprioceptively-inferred latent
     (pure RMA) — no privileged/simulation-only state, no external tracker.
  4. Write a self-contained, directly-deployable ``deploy.pth`` (the analogue of
     HORA's ``stage2_nn/best.pth``) merging the Stage-1 actor + obs normaliser +
     the trained adapter.  Consumed by ``screwdriver_rl/deploy/policy.py``.

The adaptation network is a lightweight temporal conv (following HORA's
ProprioAdaptTConv) over the last 30 policy steps of [finger_q, targets].

Legacy fallback: when no ``teacher_latent_fn`` is supplied the adapter instead
regresses the raw privileged-obs vector (used by the no-Isaac unit tests).

Usage (called from train.py --stage 2):
  ProprioAdaptTrainer(env, frozen_actor, cfg, ..., deploy_meta=..., latent_dim=...,
                      teacher_latent_fn=..., actor_with_latent_fn=...).train()
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

    # ---- On-policy latent refinement (HORA-faithful latent mode only) ----
    onpolicy_latent: bool = False
    """After a warmup, drive the frozen actor with the adapter's *predicted*
    latent (ramped true→predicted) instead of the teacher latent, so the
    collected proprio-history distribution matches deployment.

    **Default OFF.**  On the screwdriver task this destabilises training: feeding
    a not-yet-converged predicted latent to the frozen actor tips the tool over,
    so the rollout collapses (oscillation / falls), the teacher-latent targets go
    out-of-distribution, and ``AdaptLoss`` climbs as the mix coefficient ramps up
    (observed collapse from ~iter 110 as alpha→1).  The upright constraint makes
    this far more fragile than HORA's pure-rotation task.  Pure supervised
    regression from healthy teacher-driven rollouts (OFF) is the safe default; the
    residual sim-to-deploy distribution gap is then measured by the eval gate.
    Opt in with ``train.py --adapt_onpolicy`` only with a gentle schedule."""
    onpolicy_warmup_iters: int = 50
    """Iterations of pure teacher-latent driving before refinement kicks in (let
    the adapter learn a sane latent first)."""
    onpolicy_ramp_iters: int = 100
    """Iterations to ramp the mix coefficient 0→1 (teacher→predicted) after warmup."""


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ProprioAdaptTrainer:
    """Stage 2 teacher-student adaptation trainer."""

    def __init__(
        self,
        env,
        stage1_actor_fn,  # callable: obs_tensor → action_tensor (frozen policy, true priv)
        cfg: AdaptTrainCfg,
        out_dir: str,
        device: str = "cuda:0",
        priv_obs_dim: int = 17,
        frame_dim: int = 24,
        hist_len: int = 30,
        deploy_meta: dict | None = None,
        latent_dim: int | None = None,
        teacher_latent_fn=None,        # callable: policy_obs → teacher latent (N, latent_dim)
        actor_with_latent_fn=None,     # callable: (policy_obs, latent) → action (for on-policy)
    ) -> None:
        self.env = env
        self.actor = stage1_actor_fn
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = device
        self.deploy_meta = deploy_meta

        # HORA-faithful latent mode: the adapter regresses the teacher *latent*
        # ``tanh(env_mlp(priv))`` (out_dim = latent_dim) rather than the raw
        # privileged vector.  Falls back to the legacy priv-vector target when no
        # ``teacher_latent_fn`` is given (keeps the FakeEnv tests working).
        self._latent_mode = teacher_latent_fn is not None and latent_dim is not None
        self.teacher_latent_fn = teacher_latent_fn
        self.actor_with_latent_fn = actor_with_latent_fn
        out_dim = int(latent_dim) if self._latent_mode else int(priv_obs_dim)

        self.net = ProprioAdaptNet(frame_dim=frame_dim, hist_len=hist_len, out_dim=out_dim).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self._net_dims = {"frame_dim": frame_dim, "hist_len": hist_len, "out_dim": out_dim}
        # Base (unwrapped) env, used to drive the Stage-2 terminal log.  Resolved
        # via getattr so the no-Isaac FakeEnv in tests works unchanged.
        self._env_unwrapped = getattr(env, "unwrapped", env)
        self._last_extras: dict = {}
        os.makedirs(out_dir, exist_ok=True)

    def _onpolicy_alpha(self, it: int) -> float:
        """Mix coefficient for on-policy latent refinement (0=teacher, 1=predicted)."""
        if not (self._latent_mode and self.cfg.onpolicy_latent
                and self.actor_with_latent_fn is not None):
            return 0.0
        if it <= self.cfg.onpolicy_warmup_iters:
            return 0.0
        ramp = max(1, self.cfg.onpolicy_ramp_iters)
        return float(min(1.0, (it - self.cfg.onpolicy_warmup_iters) / ramp))

    def _collect(self, it: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out the frozen policy and collect (proprio_hist, target) pairs.

        ``target`` is the teacher latent in latent mode, else the privileged obs.
        With on-policy refinement (after warmup) the env is stepped using the
        adapter's predicted latent (ramped in), so the collected history matches
        the deployment distribution; the regression target stays the teacher latent.
        """
        hists, targets = [], []
        obs_dict, _ = self.env.reset()
        info: dict = {}
        alpha = self._onpolicy_alpha(it)
        for _ in range(self.cfg.rollout_steps):
            policy_obs = obs_dict["policy"]
            with torch.no_grad():
                if self._latent_mode and alpha > 0.0:
                    pred_lat = self.net(obs_dict["proprio_hist"])
                    teacher_lat = self.teacher_latent_fn(policy_obs)
                    used_lat = (1.0 - alpha) * teacher_lat + alpha * pred_lat
                    action = self.actor_with_latent_fn(policy_obs, used_lat)
                else:
                    action = self.actor(policy_obs)
            obs_dict, _, terminated, truncated, info = self.env.step(action)
            hists.append(obs_dict["proprio_hist"].detach())  # (N, T, D)
            if self._latent_mode:
                with torch.no_grad():
                    targets.append(self.teacher_latent_fn(obs_dict["policy"]).detach())
            else:
                targets.append(obs_dict["critic"].detach())  # (N, priv_dim)
        # Keep the final step's info (env extras) for the per-iter Stage-2 log.
        self._last_extras = info if isinstance(info, dict) else {}
        # (rollout_steps × N, ...)
        return torch.cat(hists, dim=0), torch.cat(targets, dim=0)

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

    def _save_deploy(self, filename: str, it: int, loss: float) -> str | None:
        """Write the self-contained, directly-deployable ``deploy.pth`` bundle.

        Merges the Stage-1 actor + obs-normaliser + deployment config captured in
        ``deploy_meta`` (see ``train.py:_build_deploy_meta``) with the trained
        adaptation network.  This is the ScrewdriverRL analogue of HORA's
        ``stage2_nn/best.pth`` — ``screwdriver_rl/deploy/policy.py:DeployPolicy``
        consumes it with neither Isaac nor rl_games.  No-op (returns ``None``)
        when no ``deploy_meta`` was provided.
        """
        if self.deploy_meta is None:
            return None
        bundle = dict(self.deploy_meta)
        bundle["adapter"] = self.net.state_dict()
        bundle["net_dims"] = self._net_dims
        bundle["iter"] = it
        bundle["loss"] = loss
        path = os.path.join(self.out_dir, filename)
        tmp = path + ".tmp"
        torch.save(bundle, tmp)
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

        if self._latent_mode:
            print(
                f"  Mode: HORA-faithful latent (adapter → {self._net_dims['out_dim']}-D teacher latent)"
                f"  |  on-policy refinement: "
                f"{'on' if (self.cfg.onpolicy_latent and self.actor_with_latent_fn is not None) else 'off'}"
                f"\n  Deploy bundle: {'deploy.pth will be written' if self.deploy_meta else 'DISABLED (no deploy_meta)'}\n",
                flush=True,
            )

        loss = float("nan")
        for it in range(1, self.cfg.num_iters + 1):
            hists, privs = self._collect(it)
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
                deploy_last = self._save_deploy("deploy_last.pth", it, loss)
                print(
                    f"  [stage2] checkpoint → {os.path.basename(snap)} "
                    f"(+ proprio_adapt_last.pth"
                    f"{' + deploy_last.pth' if deploy_last else ''})  loss {loss:.5f}",
                    flush=True,
                )

        ckpt_path = self._save_ckpt("proprio_adapt.pth", self.cfg.num_iters, loss)
        deploy_path = self._save_deploy("deploy.pth", self.cfg.num_iters, loss)
        print(f"\n[stage2] Saved adaptation network → {ckpt_path}", flush=True)
        if deploy_path:
            print(f"[stage2] Saved deployable bundle → {deploy_path}\n", flush=True)
        else:
            print("[stage2] WARNING: no deploy_meta → adapter-only checkpoint (not "
                  "directly deployable).\n", flush=True)
        return ckpt_path
