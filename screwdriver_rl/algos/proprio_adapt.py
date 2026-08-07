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

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from screwdriver_rl.deploy.codecs import ProprioCodecSpec


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
    continuous_rollouts: bool = False
    """Keep environment state across iterations. With a frozen teacher this
    preserves long-horizon episode coverage when the same samples/iteration are
    collected as more parallel environments and shorter rollout chunks."""
    num_iters: int = 500
    """Training iterations (each collects rollout_steps × num_envs transitions)."""
    batch_size: int = 4096
    action_loss_weight: float = 0.0
    """Weight on an action-space term: ``|actor(proprio, adapter(hist)) -
    actor(proprio, teacher_latent)|²`` through the frozen actor.  Plain latent MSE
    treats all 8 latent dims as equally important, but they are not — deployment
    failures are driven by the dims the actor's action is *sensitive* to, and
    capacity spent on dims the policy ignores (e.g. the spin-phase channel) buys
    nothing.  This term puts the error where it actually changes the action.
    0 disables (pure latent MSE)."""
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
    resume_checkpoint: str | None = None
    """Optional adapter checkpoint to resume from.  Stage 2 is supervised
    online regression, so the network weights and saved global iteration are
    restored while the Adam optimizer intentionally starts fresh."""

    # ---- On-policy latent refinement (HORA-faithful latent mode only) ----
    onpolicy_latent: bool = True
    """After a warmup, drive the frozen actor with the adapter's *predicted*
    latent (ramped true→predicted) instead of the teacher latent, so the
    collected proprio-history distribution matches deployment.

    ``train.py`` keeps this OFF unless ``--adapt_onpolicy`` is passed: a purely
    off-policy adapter collapses at deployment on the in-hand rotation task —
    2026-07-16 probe: oracle fall% 40 / rotate 0.18 vs deploy fall% 71 /
    rotate 0.095 — the classic imitation-drift failure this refinement exists
    to fix.  On the screwdriver task this destabilises training: feeding
    a not-yet-converged predicted latent to the frozen actor tips the tool over,
    so the rollout collapses (oscillation / falls), the teacher-latent targets go
    out-of-distribution, and ``AdaptLoss`` climbs as the mix coefficient ramps up
    (observed collapse from ~iter 110 as alpha→1).  The upright constraint makes
    this far more fragile than HORA's pure-rotation task — use
    leave ``--adapt_onpolicy`` unset there; the residual sim-to-deploy
    distribution gap is then measured by the eval gate."""
    onpolicy_warmup_iters: int = 50
    """Iterations of pure teacher-latent driving before refinement kicks in (let
    the adapter learn a sane latent first)."""
    onpolicy_ramp_iters: int = 100
    """Iterations to ramp the mix coefficient 0→1 (teacher→predicted) after warmup."""


def _held_out_split_indices(
    num_samples: int,
    device: str | torch.device,
    sample_group_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
    """Return train/validation indices without leaking grouped histories.

    Production Stage 2 emits one group id per simulator environment.  Holding
    out complete environments prevents heavily overlapping 30-step histories
    from the same trajectory appearing on both sides of the validation split.
    The sample-random fallback keeps legacy single-environment tests usable.
    """
    if num_samples < 2:
        raise ValueError("Stage-2 held-out validation requires at least two samples")

    if sample_group_ids is not None:
        groups = sample_group_ids.reshape(-1).to(device=device)
        if groups.numel() != num_samples:
            raise ValueError("Stage-2 sample group ids must align with collected samples")
        unique_groups = torch.unique(groups)
        if unique_groups.numel() >= 2:
            order = torch.randperm(unique_groups.numel(), device=device)
            shuffled_groups = unique_groups[order]
            val_group_count = max(1, unique_groups.numel() // 10)
            val_groups = shuffled_groups[:val_group_count]
            val_mask = torch.isin(groups, val_groups)
            val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1)
            train_idx = torch.nonzero(~val_mask, as_tuple=False).squeeze(-1)
            if val_idx.numel() > 0 and train_idx.numel() > 0:
                return train_idx, val_idx, {
                    "validation_split": "environment_group",
                    "validation_group_count": int(val_group_count),
                    "train_group_count": int(
                        unique_groups.numel() - val_group_count
                    ),
                }

    indices = torch.randperm(num_samples, device=device)
    val_count = max(1, num_samples // 10)
    return indices[val_count:], indices[:val_count], {
        "validation_split": "sample_random_fallback",
        "validation_group_count": 0,
        "train_group_count": 0,
    }


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
        actor_mu_grad_fn=None,         # callable: (policy_obs, latent) → mu, DIFFERENTIABLE wrt latent
    ) -> None:
        self.env = env
        self.actor = stage1_actor_fn
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = device
        self.deploy_meta = deploy_meta
        if deploy_meta is not None:
            config = deploy_meta.get("config")
            actor_arch = deploy_meta.get("actor_arch")
            if not isinstance(config, Mapping) or not isinstance(actor_arch, Mapping):
                raise ValueError("deploy metadata requires config and actor_arch objects")
            codec_value = config.get("proprio_codec")
            if not isinstance(codec_value, Mapping):
                raise ValueError("deploy metadata is missing its explicit ProprioCodec")
            codec = ProprioCodecSpec.from_dict(codec_value)
            if codec.frame_dim != frame_dim or codec.history_length != hist_len:
                raise ValueError("deploy ProprioCodec does not match adaptation history")
            latent_width = int(actor_arch.get("latent_dim", 0))
            expected_actor_width = codec.actor_frame_count * codec.frame_dim
            if latent_width == 0:
                expected_actor_width += int(config.get("euler_dim", 3))
            if int(actor_arch.get("proprio_dim", -1)) != expected_actor_width:
                raise ValueError("deploy actor proprio width does not match ProprioCodec")
            if int(actor_arch.get("action_dim", -1)) != codec.joint_count:
                raise ValueError("deploy actor action width does not match ProprioCodec")

        # HORA-faithful latent mode: the adapter regresses the teacher *latent*
        # ``tanh(env_mlp(priv))`` (out_dim = latent_dim) rather than the raw
        # privileged vector.  Falls back to the legacy priv-vector target when no
        # ``teacher_latent_fn`` is given (keeps the FakeEnv tests working).
        self._latent_mode = teacher_latent_fn is not None and latent_dim is not None
        self.teacher_latent_fn = teacher_latent_fn
        self.actor_with_latent_fn = actor_with_latent_fn
        self.actor_mu_grad_fn = actor_mu_grad_fn
        self._action_loss_on = bool(
            cfg.action_loss_weight > 0.0 and actor_mu_grad_fn is not None
        )
        self._policy_obs_buf: torch.Tensor | None = None
        self._priv_obs_dim = int(priv_obs_dim)
        out_dim = int(latent_dim) if self._latent_mode else int(priv_obs_dim)

        self.net = ProprioAdaptNet(frame_dim=frame_dim, hist_len=hist_len, out_dim=out_dim).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        # Diagnostic-only auxiliary decoder. The deploy path still consumes only
        # the latent adapter above; this probe makes the raw privileged channels'
        # held-out predictability measurable as required by the force-free plan.
        self.priv_probe = (
            ProprioAdaptNet(
                frame_dim=frame_dim,
                hist_len=hist_len,
                out_dim=self._priv_obs_dim,
            ).to(device)
            if self._latent_mode
            else None
        )
        self.priv_probe_optim = (
            torch.optim.Adam(
                self.priv_probe.parameters(),
                lr=cfg.learning_rate,
            )
            if self.priv_probe is not None
            else None
        )
        self._last_validation: dict = {}
        self._net_dims = {"frame_dim": frame_dim, "hist_len": hist_len, "out_dim": out_dim}
        # Base (unwrapped) env, used to drive the Stage-2 terminal log.  Resolved
        # via getattr so the no-Isaac FakeEnv in tests works unchanged.
        self._env_unwrapped = getattr(env, "unwrapped", env)
        self._last_extras: dict = {}
        self._rollout_obs: dict | None = None
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

    def _collect(
        self, it: int = 1
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """Roll out the frozen policy and collect (proprio_hist, target) pairs.

        ``target`` is the teacher latent in latent mode, else the privileged obs.
        With on-policy refinement (after warmup) the env is stepped using the
        adapter's predicted latent (ramped in), so the collected history matches
        the deployment distribution; the regression target stays the teacher latent.
        """
        hists, targets, raw_privs = [], [], []
        policy_obses: list[torch.Tensor] = []
        if self.cfg.continuous_rollouts and self._rollout_obs is not None:
            obs_dict = self._rollout_obs
        else:
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
                if self._action_loss_on:
                    # Same post-step obs the target latent is taken from, so the
                    # actor sees a consistent (proprio, latent) pair.
                    policy_obses.append(obs_dict["policy"].detach().clone())
                critic = obs_dict.get("critic")
                if critic is None:
                    raise RuntimeError(
                        "latent Stage 2 requires critic observations for "
                        "held-out privileged-channel diagnostics"
                    )
                raw_privs.append(critic.detach())
            else:
                targets.append(obs_dict["critic"].detach())  # (N, priv_dim)
        # Keep the final step's info (env extras) for the per-iter Stage-2 log.
        self._last_extras = info if isinstance(info, dict) else {}
        if self.cfg.continuous_rollouts:
            self._rollout_obs = obs_dict
        # (rollout_steps × N, ...)
        collected_hists = torch.cat(hists, dim=0)
        collected_targets = torch.cat(targets, dim=0)
        self._policy_obs_buf = torch.cat(policy_obses, dim=0) if policy_obses else None
        if self._latent_mode:
            # torch.cat above is step-major: each contiguous block contains one
            # row per simulator environment.  Repeating env ids in the same
            # order supplies leakage-free validation groups.
            sample_group_ids = torch.arange(
                hists[0].shape[0], device=self.device
            ).repeat(len(hists))
            return (
                collected_hists,
                collected_targets,
                torch.cat(raw_privs, dim=0),
                sample_group_ids,
            )
        return collected_hists, collected_targets

    def _train_on_batch(
        self,
        hists: torch.Tensor,
        targets: torch.Tensor,
        raw_privs: torch.Tensor | None = None,
        sample_group_ids: torch.Tensor | None = None,
    ) -> float:
        """Fit on a train split and retain an untouched split for diagnostics.

        In latent mode ``targets`` contains the teacher latent consumed by the
        frozen actor.  ``raw_privs`` is used only by ``priv_probe`` to measure
        how much of each force-free privileged channel can be inferred from
        proprioceptive history; neither the probe nor raw privileged data is
        included in the deploy bundle.
        """
        n = hists.shape[0]
        if targets.shape[0] != n or (raw_privs is not None and raw_privs.shape[0] != n):
            raise ValueError("Stage-2 histories and regression targets must align")
        if n < 2:
            raise ValueError("Stage-2 held-out validation requires at least two samples")

        train_idx, val_idx, split_diagnostics = _held_out_split_indices(
            n,
            self.device,
            sample_group_ids,
        )
        total_loss = 0.0
        steps = 0
        bs = self.cfg.batch_size
        for _ in range(self.cfg.num_epochs_per_iter):
            train_idx = train_idx[torch.randperm(train_idx.numel(), device=self.device)]
            for start in range(0, train_idx.numel(), bs):
                idx = train_idx[start:start + bs]
                pred = self.net(hists[idx])
                loss = F.mse_loss(pred, targets[idx])
                if self._action_loss_on and self._policy_obs_buf is not None:
                    obs_b = self._policy_obs_buf[idx]
                    mu_pred = self.actor_mu_grad_fn(obs_b, pred)
                    with torch.no_grad():
                        mu_teacher = self.actor_mu_grad_fn(obs_b, targets[idx])
                    loss = loss + self.cfg.action_loss_weight * F.mse_loss(
                        mu_pred, mu_teacher
                    )
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.priv_probe is not None:
                    if raw_privs is None or self.priv_probe_optim is None:
                        raise RuntimeError(
                            "latent Stage 2 requires raw privileged targets for diagnostics"
                        )
                    raw_pred = self.priv_probe(hists[idx])
                    raw_loss = F.mse_loss(raw_pred, raw_privs[idx])
                    self.priv_probe_optim.zero_grad()
                    raw_loss.backward()
                    self.priv_probe_optim.step()

                total_loss += loss.item()
                steps += 1

        with torch.no_grad():
            target_err = self.net(hists[val_idx]) - targets[val_idx]
            target_mse_by_channel = target_err.square().mean(dim=0)
            validation = {
                "mode": "teacher_latent" if self._latent_mode else "raw_privileged",
                "sample_count": int(val_idx.numel()),
                "train_sample_count": int(train_idx.numel()),
                **split_diagnostics,
                "adapter_target_mse": float(target_err.square().mean().item()),
                "adapter_target_mae": float(target_err.abs().mean().item()),
                "adapter_target_mse_by_channel": target_mse_by_channel.cpu().tolist(),
            }
            if self._latent_mode:
                validation["adapter_latent_mse"] = validation["adapter_target_mse"]
                validation["adapter_latent_mae"] = validation["adapter_target_mae"]
                validation["adapter_latent_mse_by_channel"] = validation[
                    "adapter_target_mse_by_channel"
                ]

            if self.priv_probe is not None and raw_privs is not None:
                raw_err = self.priv_probe(hists[val_idx]) - raw_privs[val_idx]
                raw_mse_by_channel = raw_err.square().mean(dim=0)
                validation.update(
                    {
                        "raw_privileged_mse": float(raw_err.square().mean().item()),
                        "raw_privileged_mae": float(raw_err.abs().mean().item()),
                        "raw_privileged_mse_by_channel": raw_mse_by_channel.cpu().tolist(),
                    }
                )
                # The final force-free Linker L20 layout is 22-D.  Keep these
                # semantic summaries gated on that exact width so historical
                # 19/21-D checkpoints are not mislabeled.
                if raw_err.shape[1] == 22:
                    validation.update(
                        {
                            "screw_relative_position_mse": float(
                                raw_err[:, 6:9].square().mean().item()
                            ),
                            "distance_contact_score_mse": float(
                                raw_err[:, 15:20].square().mean().item()
                            ),
                            "raw_privileged_channel_layout": {
                                "screw_euler": [0, 3],
                                "screw_angular_velocity": [3, 6],
                                "screw_relative_position": [6, 9],
                                "screw_quaternion": [9, 13],
                                "load_proxy": [13, 14],
                                "contact_friction": [14, 15],
                                "distance_contact_scores": [15, 20],
                                "screw_geometry": [20, 22],
                            },
                        }
                    )
            self._last_validation = validation
        return total_loss / max(steps, 1)

    def _save_validation(self, it: int, loss: float) -> str:
        """Atomically persist the latest true held-out diagnostics as JSON."""
        report = dict(self._last_validation)
        report["iter"] = int(it)
        report["training_loss"] = float(loss)
        self._last_validation = report
        path = os.path.join(self.out_dir, "adaptation_validation.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
        return path

    def _save_ckpt(self, filename: str, it: int, loss: float) -> str:
        """Atomically write a checkpoint to ``out_dir/filename``.

        Keeps the ``"net"`` key that loaders (play/eval, test_algo) expect and
        adds the iteration, loss, and network dims so a checkpoint is
        self-describing.  Writes to a temp file then ``os.replace``s it into
        place, so a crash mid-save cannot leave a truncated/corrupt checkpoint.
        """
        path = os.path.join(self.out_dir, filename)
        tmp = path + ".tmp"
        state = {
            "net": self.net.state_dict(),
            "iter": it,
            "loss": loss,
            "net_dims": self._net_dims,
            "validation": self._last_validation,
        }
        if self.priv_probe is not None:
            state["priv_probe"] = self.priv_probe.state_dict()
            state["priv_probe_dims"] = {
                "frame_dim": self._net_dims["frame_dim"],
                "hist_len": self._net_dims["hist_len"],
                "out_dim": self._priv_obs_dim,
            }
        torch.save(state, tmp)
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
        bundle["adaptation_validation"] = self._last_validation
        path = os.path.join(self.out_dir, filename)
        tmp = path + ".tmp"
        torch.save(bundle, tmp)
        os.replace(tmp, path)
        return path

    def train(self) -> str:
        """Run full Stage 2 training.  Returns path to the saved checkpoint."""
        start_iter = 0
        if self.cfg.resume_checkpoint:
            resume_path = os.path.abspath(os.path.expanduser(self.cfg.resume_checkpoint))
            state = torch.load(resume_path, map_location=self.device)
            if not isinstance(state, Mapping) or "net" not in state:
                raise ValueError(f"invalid Stage-2 resume checkpoint: {resume_path}")
            saved_dims = state.get("net_dims")
            if saved_dims != self._net_dims:
                raise ValueError(
                    "Stage-2 resume network dimensions do not match: "
                    f"checkpoint={saved_dims}, expected={self._net_dims}"
                )
            start_iter = int(state.get("iter", 0))
            if not 0 < start_iter < self.cfg.num_iters:
                raise ValueError(
                    "Stage-2 resume iteration must be between 1 and "
                    f"{self.cfg.num_iters - 1}, got {start_iter}"
                )
            self.net.load_state_dict(state["net"], strict=True)
            if self.priv_probe is not None and "priv_probe" in state:
                self.priv_probe.load_state_dict(state["priv_probe"], strict=True)
            saved_validation = state.get("validation")
            if isinstance(saved_validation, Mapping):
                self._last_validation = dict(saved_validation)

        print(
            f"\n{'='*60}\n"
            f"  Stage 2 — Proprioceptive Adaptation Training\n"
            f"  Iters: {self.cfg.num_iters}  |  "
            f"Rollout steps/iter: {self.cfg.rollout_steps}\n"
            f"  Continuous rollout chunks: {self.cfg.continuous_rollouts}\n"
            f"  Resume: {start_iter} completed iteration(s)"
            f"{' from ' + self.cfg.resume_checkpoint if start_iter else ''}\n"
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
        for it in range(start_iter + 1, self.cfg.num_iters + 1):
            collected = self._collect(it)
            sample_group_ids = None
            if len(collected) == 4:
                hists, targets, raw_privs, sample_group_ids = collected
            elif len(collected) == 3:
                hists, targets, raw_privs = collected
            else:
                hists, targets = collected
                raw_privs = None
            loss = self._train_on_batch(
                hists,
                targets,
                raw_privs,
                sample_group_ids,
            )
            validation_path = self._save_validation(it, loss)

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
        print(f"[stage2] Saved held-out diagnostics → {validation_path}", flush=True)
        if deploy_path:
            print(f"[stage2] Saved deployable bundle → {deploy_path}\n", flush=True)
        else:
            print("[stage2] WARNING: no deploy_meta → adapter-only checkpoint (not "
                  "directly deployable).\n", flush=True)
        return ckpt_path
