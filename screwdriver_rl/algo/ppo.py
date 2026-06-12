"""Stage-1 teacher: PPO with privileged-information encoder + curriculum.

A bespoke PPO (HORA-lineage) rather than rl_games/rsl-rl because the
teacher-student pipeline needs (a) dict observations {obs, priv_info,
proprio_hist} threaded through rollout storage, (b) the privileged encoder
inside the model so stage 2 can reuse the identical ActorCritic with only the
adaptation module swapped in, and (c) curriculum hooks that read env metrics
every epoch.

Expects an Isaac Lab DirectRLEnv whose ``step`` returns
``(obs_dict, rewards, terminated, timed_out, extras)`` with obs_dict keys
``policy`` / ``critic`` / ``proprio_hist``.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from ..configs.curricula import CurriculumPhase
from ..configs.train_cfg import NetworkCfg, PPOTrainCfg
from ..utils.cli import set_dotted
from .experience import ExperienceBuffer
from .metrics import AverageScalarMeter, EnvMetricsTracker, format_console_metrics
from .models import ActorCritic
from .running_mean_std import RunningMeanStd


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma) -> torch.Tensor:
    c1 = torch.log(p1_sigma / p0_sigma + 1.0e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1.0e-5))
    return (c1 + c2 - 0.5).sum(dim=-1).mean()


class AdaptiveKLScheduler:
    """Raise LR when the policy moves too little, lower it when too much."""

    def __init__(self, kl_threshold: float, min_lr: float = 1.0e-6, max_lr: float = 1.0e-2):
        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr

    def update(self, lr: float, kl: float) -> float:
        if kl > 2.0 * self.kl_threshold:
            return max(lr / 1.5, self.min_lr)
        if kl < 0.5 * self.kl_threshold:
            return min(lr * 1.5, self.max_lr)
        return lr


class PPO:
    def __init__(
        self,
        env,
        output_dir: str,
        cfg: PPOTrainCfg | None = None,
        net_cfg: NetworkCfg | None = None,
        curriculum: list[CurriculumPhase] | None = None,
        device: str = "cuda:0",
    ):
        self.env = env
        self.device = device
        self.cfg = cfg or PPOTrainCfg()
        self.net_cfg = net_cfg or NetworkCfg()
        self.curriculum = curriculum or []

        self.num_envs = env.num_envs
        self.actions_num = env.single_action_space.shape[0]
        obs_space = env.single_observation_space
        self.obs_dim = (
            obs_space["policy"].shape[0] if hasattr(obs_space, "spaces") else obs_space.shape[0]
        )
        self.priv_dim = int(env.cfg.privileged_obs_dim)
        self.hist_len = int(env.cfg.prop_hist_len)
        self.hist_dim = int(env.cfg.history_obs_dim)

        self.model = ActorCritic(
            actions_num=self.actions_num,
            obs_dim=self.obs_dim,
            actor_units=tuple(self.net_cfg.actor_units),
            priv_mlp_units=tuple(self.net_cfg.priv_mlp_units),
            priv_info=True,
            proprio_adapt=False,
            priv_info_dim=self.priv_dim,
            adapt_obs_dim=self.hist_dim,
            adapt_history_len=self.hist_len,
            adapt_hidden_dim=self.net_cfg.adapt_hidden_dim,
            adapt_conv_kernels=tuple(self.net_cfg.adapt_conv_kernels),
        ).to(device)

        self.obs_rms = RunningMeanStd((self.obs_dim,)).to(device)
        self.value_rms = RunningMeanStd((1,)).to(device)

        self.last_lr = float(self.cfg.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr)
        self.scheduler = AdaptiveKLScheduler(self.cfg.kl_threshold)

        self.horizon = int(self.cfg.horizon_length)
        self.batch_size = self.horizon * self.num_envs
        self.storage = ExperienceBuffer(
            self.num_envs, self.horizon, self.cfg.minibatch_size,
            self.obs_dim, self.actions_num, self.priv_dim, device,
        )

        # ---- output / logging ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(output_dir, "stage1_nn")
        os.makedirs(self.nn_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(os.path.join(output_dir, "stage1_tb"))

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.env_metrics = EnvMetricsTracker()

        # ---- counters / episode tracking ----
        self.epoch_num = 0
        self.agent_steps = 0
        self.best_rewards = -1.0e9
        self.obs: dict | None = None
        self.current_rewards = torch.zeros((self.num_envs, 1), device=device)
        self.current_lengths = torch.zeros(self.num_envs, device=device)
        self.collect_time = 0.0
        self.train_time = 0.0

        # ---- curriculum ----
        self.phase_idx = -1
        self.phase_start_steps = 0
        self.phase_wait_reason = ""
        if self.curriculum:
            self._enter_phase(0, initial=True)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _enter_phase(self, idx: int, initial: bool = False) -> None:
        phase = self.curriculum[idx]
        applied = {}
        for path, value in phase.env_overrides.items():
            if set_dotted(self.env.cfg, path, value):
                applied[path] = value
            else:
                print(f"[curriculum] WARNING: env cfg has no field {path!r}")
        self.phase_idx = idx
        self.phase_start_steps = self.agent_steps
        self.phase_wait_reason = ""
        prefix = "Initial curriculum phase" if initial else "Curriculum phase"
        print(f"{prefix} {idx + 1}/{len(self.curriculum)}: {phase.name} ({len(applied)} overrides)")
        self.writer.add_scalar("curriculum/phase_index", idx, self.agent_steps)

    def _phase_can_advance(self, mean_lengths: float) -> tuple[bool, str]:
        if not self.curriculum or self.phase_idx >= len(self.curriculum) - 1:
            return False, "final phase"
        advance = self.curriculum[self.phase_idx].advance
        if advance is None:
            return False, "no advance rule"
        phase_steps = self.agent_steps - self.phase_start_steps
        if phase_steps < advance.min_phase_steps:
            return False, f"steps {phase_steps / 1e6:.0f}M < {advance.min_phase_steps / 1e6:.0f}M"
        if advance.min_episode_length is not None and mean_lengths < advance.min_episode_length:
            return False, f"Len {mean_lengths:.0f} < {advance.min_episode_length:.0f}"
        for key, (op, threshold) in advance.metric_bounds.items():
            value = self.env_metrics.get(key)
            if value is None:
                return False, f"missing {key}"
            if op == ">=" and value < threshold:
                return False, f"{key} {value:.3f} < {threshold:.3f}"
            if op == "<=" and value > threshold:
                return False, f"{key} {value:.3f} > {threshold:.3f}"
        if advance.min_fwd_minus_rev is not None:
            fwd = self.env_metrics.get("eval_forward_turn_velocity")
            rev = self.env_metrics.get("eval_reverse_turn_velocity")
            if fwd is None or rev is None:
                return False, "missing fwd/rev velocity"
            if fwd - rev < advance.min_fwd_minus_rev:
                return False, f"fwd-rev {fwd - rev:.3f} < {advance.min_fwd_minus_rev:.3f}"
        return True, "ready"

    def _maybe_advance_curriculum(self, mean_lengths: float) -> None:
        can, reason = self._phase_can_advance(mean_lengths)
        self.phase_wait_reason = reason
        if not can:
            return
        old = self.curriculum[self.phase_idx]
        self.save(os.path.join(self.nn_dir, f"curriculum_exit_{self.phase_idx + 1}_{old.name}"))
        self._enter_phase(self.phase_idx + 1)
        # New phase => new reward scale regime; old "best" is incomparable.
        self.best_rewards = -1.0e9
        print(
            f"Curriculum advanced: {old.name} -> {self.curriculum[self.phase_idx].name} "
            f"at {self.agent_steps / 1e6:.0f}M steps"
        )

    # ------------------------------------------------------------------
    # Train / eval mode helpers
    # ------------------------------------------------------------------

    def set_eval(self):
        self.model.eval()
        self.obs_rms.eval()
        self.value_rms.eval()

    def set_train(self):
        self.model.train()
        if self.cfg.normalize_input:
            self.obs_rms.train()
        if self.cfg.normalize_value:
            self.value_rms.train()

    @staticmethod
    def _prepare_obs(obs_dict: dict) -> dict:
        return {
            "obs": obs_dict["policy"],
            "priv_info": obs_dict.get("critic"),
            "proprio_hist": obs_dict.get("proprio_hist"),
        }

    def _model_act(self, obs: dict) -> dict:
        input_dict = {
            "obs": self.obs_rms(obs["obs"]) if self.cfg.normalize_input else obs["obs"],
            "priv_info": obs["priv_info"],
        }
        result = self.model.act(input_dict)
        if self.cfg.normalize_value:
            result["values"] = self.value_rms(result["values"], unnorm=True)
        return result

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _play_steps(self) -> None:
        for t in range(self.horizon):
            res = self._model_act(self.obs)
            self.storage.update("obses", t, self.obs["obs"])
            self.storage.update("priv_info", t, self.obs["priv_info"])
            for key in ("actions", "neglogpacs", "values", "mus", "sigmas"):
                self.storage.update(key, t, res[key])

            actions = torch.clamp(res["actions"], -1.0, 1.0)
            obs_dict, rewards, terminated, timed_out, extras = self.env.step(actions)
            self.env_metrics.accumulate(extras)

            dones = terminated | timed_out
            rewards = rewards.unsqueeze(1)
            shaped = self.cfg.reward_scale * rewards.clone()
            if self.cfg.value_bootstrap:
                # Timeout is not failure: bootstrap the cut-off return.
                shaped += self.cfg.gamma * res["values"] * timed_out.unsqueeze(1).float()
            self.storage.update("dones", t, dones.to(torch.uint8))
            self.storage.update("rewards", t, shaped)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_idx = dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_idx])
            self.episode_lengths.update(self.current_lengths[done_idx])
            not_done = (1.0 - dones.float())
            self.current_rewards *= not_done.unsqueeze(1)
            self.current_lengths *= not_done

            self.obs = self._prepare_obs(obs_dict)

        self.env_metrics.finalize()
        last_values = self._model_act(self.obs)["values"]
        self.agent_steps += self.batch_size

        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.tau)
        self.storage.prepare_training(self.cfg.normalize_advantage)
        if self.cfg.normalize_value:
            self.value_rms.train()
            self.storage.data["values"] = self.value_rms(self.storage.data["values"])
            self.storage.data["returns"] = self.value_rms(self.storage.data["returns"])
            self.value_rms.eval()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _train_epoch(self):
        t0 = time.time()
        self.set_eval()
        self._play_steps()
        self.collect_time += time.time() - t0

        t0 = time.time()
        self.set_train()
        a_losses, c_losses, b_losses, entropies, kls = [], [], [], [], []
        for _ in range(self.cfg.mini_epochs):
            epoch_kls = []
            for i in range(len(self.storage)):
                batch = self.storage[i]
                obs = self.obs_rms(batch["obses"]) if self.cfg.normalize_input else batch["obses"]
                res = self.model(
                    {"obs": obs, "priv_info": batch["priv_info"], "prev_actions": batch["actions"]}
                )

                ratio = torch.exp(batch["neglogpacs"] - res["prev_neglogp"])
                surr1 = batch["advantages"] * ratio
                surr2 = batch["advantages"] * torch.clamp(
                    ratio, 1.0 - self.cfg.e_clip, 1.0 + self.cfg.e_clip
                )
                a_loss = torch.max(-surr1, -surr2).mean()

                values = res["values"]
                if self.cfg.clip_value:
                    value_clipped = batch["values"] + (values - batch["values"]).clamp(
                        -self.cfg.e_clip, self.cfg.e_clip
                    )
                    c_loss = torch.max(
                        (values - batch["returns"]) ** 2, (value_clipped - batch["returns"]) ** 2
                    ).mean()
                else:
                    c_loss = ((values - batch["returns"]) ** 2).mean()

                if self.cfg.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    b_loss = (
                        (torch.clamp_max(res["mus"] - soft_bound, 0.0) ** 2
                         + torch.clamp_max(-res["mus"] + soft_bound, 0.0) ** 2)
                        .sum(dim=-1)
                        .mean()
                    )
                else:
                    b_loss = torch.zeros((), device=self.device)
                entropy = res["entropy"].mean()

                loss = (
                    a_loss
                    + 0.5 * c_loss * self.cfg.critic_coef
                    - entropy * self.cfg.entropy_coef
                    + b_loss * self.cfg.bounds_loss_coef
                )
                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.truncate_grads:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl = policy_kl(
                        res["mus"].detach(), res["sigmas"].detach(), batch["mus"], batch["sigmas"]
                    )
                self.storage.update_mu_sigma(res["mus"].detach(), res["sigmas"].detach())
                epoch_kls.append(kl)
                a_losses.append(a_loss.detach())
                c_losses.append(c_loss.detach())
                b_losses.append(b_loss.detach())
                entropies.append(entropy.detach())

            mean_kl = torch.stack(epoch_kls).mean().item()
            self.last_lr = self.scheduler.update(self.last_lr, mean_kl)
            for group in self.optimizer.param_groups:
                group["lr"] = self.last_lr
            kls.append(mean_kl)

        self.train_time += time.time() - t0
        return a_losses, c_losses, b_losses, entropies, kls

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        start = time.time()
        last = start
        obs_dict = self.env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        self.obs = self._prepare_obs(obs_dict)
        self.agent_steps = max(self.agent_steps, self.batch_size)

        while self.agent_steps < self.cfg.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self._train_epoch()
            self.storage.data = None

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            if self.curriculum:
                self._maybe_advance_curriculum(mean_lengths)

            all_fps = self.agent_steps / (time.time() - start)
            last_fps = self.batch_size / (time.time() - last)
            last = time.time()
            phase = (
                f"Phase: {self.curriculum[self.phase_idx].name} | " if self.curriculum else ""
            )
            gate = (
                f" | Gate: {self.phase_wait_reason}"
                if self.curriculum and self.phase_idx < len(self.curriculum) - 1
                else ""
            )
            print(
                f"Steps: {self.agent_steps / 1e6:07.1f}M | FPS: {all_fps:.0f} (last {last_fps:.0f}) | "
                f"{phase}Reward: {mean_rewards:.2f} | Len: {mean_lengths:.0f} | "
                f"Best: {self.best_rewards:.2f}"
                f"{format_console_metrics(self.env_metrics.last_means)}{gate}"
            )

            # ---- tensorboard ----
            w = self.writer
            w.add_scalar("episode/rewards", mean_rewards, self.agent_steps)
            w.add_scalar("episode/lengths", mean_lengths, self.agent_steps)
            for key, value in self.env_metrics.last_means.items():
                w.add_scalar(f"env/{key}", value, self.agent_steps)
            if a_losses:
                w.add_scalar("losses/actor", torch.stack(a_losses).mean().item(), self.agent_steps)
                w.add_scalar("losses/critic", torch.stack(c_losses).mean().item(), self.agent_steps)
                w.add_scalar("losses/bounds", torch.stack(b_losses).mean().item(), self.agent_steps)
                w.add_scalar("losses/entropy", torch.stack(entropies).mean().item(), self.agent_steps)
            if kls:
                w.add_scalar("info/kl", sum(kls) / len(kls), self.agent_steps)
            w.add_scalar("info/lr", self.last_lr, self.agent_steps)
            w.add_scalar("performance/collect_fps", self.agent_steps / max(self.collect_time, 1e-6), self.agent_steps)
            if self.curriculum:
                w.add_scalar("curriculum/phase_index", self.phase_idx, self.agent_steps)

            # ---- checkpoints ----
            if self.cfg.save_frequency_epochs > 0 and self.epoch_num % self.cfg.save_frequency_epochs == 0:
                self.save(os.path.join(self.nn_dir, f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):05d}M"))
                self.save(os.path.join(self.nn_dir, "last"))
            if mean_rewards > self.best_rewards and self.epoch_num >= self.cfg.save_best_after_epochs:
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, "best"))

        print("max_agent_steps reached")
        self.save(os.path.join(self.nn_dir, "final"))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "obs_rms": self.obs_rms.state_dict(),
                "value_rms": self.value_rms.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "agent_steps": self.agent_steps,
                "epoch_num": self.epoch_num,
                "best_rewards": self.best_rewards,
                "phase_idx": self.phase_idx,
            },
            f"{path}.pth",
        )

    def restore(self, path: str, resume_training_state: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=False)
        if "obs_rms" in ckpt:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        if "value_rms" in ckpt:
            self.value_rms.load_state_dict(ckpt["value_rms"])
        if resume_training_state:
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.agent_steps = ckpt.get("agent_steps", 0)
            self.epoch_num = ckpt.get("epoch_num", 0)
            self.best_rewards = ckpt.get("best_rewards", -1.0e9)
            phase_idx = ckpt.get("phase_idx", -1)
            if self.curriculum and 0 <= phase_idx < len(self.curriculum):
                self._enter_phase(phase_idx, initial=True)
