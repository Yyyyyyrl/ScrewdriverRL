"""Stage-2 student: proprioceptive adaptation via teacher-student distillation.

Loads a frozen stage-1 (teacher) checkpoint and trains ONLY the adaptation
module (ProprioAdaptTConv) so the policy can run without privileged
information. On-policy rollouts are collected with the *student's* own
actions (DAgger-style: the adaptation module sees the state distribution it
will face at deployment).

Loss = MSE(z_student, z_teacher) + bc_weight * MSE(mu_student, mu_teacher)

The optional behavior-cloning term (DexScrew lesson) directly anchors the
student's actions to the frozen teacher's, which helps when small latent
errors map to large action differences; ``bc_weight=0`` reproduces pure HORA.
"""

from __future__ import annotations

import os
import time

import torch

from ..configs.train_cfg import NetworkCfg, StudentTrainCfg
from .metrics import AverageScalarMeter, EnvMetricsTracker, format_console_metrics
from .models import ActorCritic
from .running_mean_std import RunningMeanStd


class ProprioAdapt:
    def __init__(
        self,
        env,
        output_dir: str,
        cfg: StudentTrainCfg | None = None,
        net_cfg: NetworkCfg | None = None,
        device: str = "cuda:0",
    ):
        self.env = env
        self.device = device
        self.cfg = cfg or StudentTrainCfg()
        self.net_cfg = net_cfg or NetworkCfg()

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
            proprio_adapt=True,
            priv_info_dim=self.priv_dim,
            adapt_obs_dim=self.hist_dim,
            adapt_history_len=self.hist_len,
            adapt_hidden_dim=self.net_cfg.adapt_hidden_dim,
            adapt_conv_kernels=tuple(self.net_cfg.adapt_conv_kernels),
        ).to(device)
        self.model.eval()

        self.obs_rms = RunningMeanStd((self.obs_dim,)).to(device)
        self.obs_rms.eval()  # frozen: stage-1 statistics
        self.hist_rms = RunningMeanStd((self.hist_len, self.hist_dim)).to(device)

        # Train only the adaptation module.
        adapt_params = []
        for name, param in self.model.named_parameters():
            if "adapt_tconv" in name:
                adapt_params.append(param)
            else:
                param.requires_grad = False
        self.optimizer = torch.optim.Adam(adapt_params, lr=self.cfg.learning_rate)

        self.output_dir = output_dir
        self.nn_dir = os.path.join(output_dir, "stage2_nn")
        os.makedirs(self.nn_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(os.path.join(output_dir, "stage2_tb"))

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.env_metrics = EnvMetricsTracker()
        self.current_rewards = torch.zeros(self.num_envs, device=device)
        self.current_lengths = torch.zeros(self.num_envs, device=device)
        self.agent_steps = 0
        self.epoch_num = 0
        self.best_rewards = -1.0e9

    @staticmethod
    def _prepare_obs(obs_dict: dict) -> dict:
        return {
            "obs": obs_dict["policy"],
            "priv_info": obs_dict.get("critic"),
            "proprio_hist": obs_dict.get("proprio_hist"),
        }

    def train(self):
        start = time.time()
        obs_dict = self.env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]

        log_every = 50  # env steps per console/tensorboard update

        while self.agent_steps < self.cfg.max_agent_steps:
            self.epoch_num += 1
            obs = self._prepare_obs(obs_dict)

            self.hist_rms.train()
            input_dict = {
                "obs": self.obs_rms(obs["obs"]).detach(),
                "priv_info": obs["priv_info"],
                "proprio_hist": self.hist_rms(obs["proprio_hist"].detach()),
            }
            mu_student, z, z_gt, mu_teacher = self.model.forward_stage2(input_dict)

            latent_loss = ((z - z_gt.detach()) ** 2).mean()
            if self.cfg.bc_weight > 0.0:
                bc_loss = ((mu_student - mu_teacher.detach()) ** 2).mean()
            else:
                bc_loss = torch.zeros((), device=self.device)
            loss = latent_loss + self.cfg.bc_weight * bc_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Step the env with the student's own (deterministic) actions.
            actions = torch.clamp(mu_student.detach(), -1.0, 1.0)
            obs_dict, rewards, terminated, timed_out, extras = self.env.step(actions)
            self.env_metrics.accumulate(extras)
            dones = terminated | timed_out
            self.agent_steps += self.num_envs

            self.current_rewards += rewards
            self.current_lengths += 1
            done_idx = dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_idx])
            self.episode_lengths.update(self.current_lengths[done_idx])
            not_done = 1.0 - dones.float()
            self.current_rewards *= not_done
            self.current_lengths *= not_done

            if self.epoch_num % log_every == 0:
                means = self.env_metrics.finalize()
                mean_rewards = self.episode_rewards.get_mean()
                fps = self.agent_steps / (time.time() - start)
                print(
                    f"Steps: {self.agent_steps / 1e6:07.1f}M | FPS: {fps:.0f} | "
                    f"LatentMSE: {latent_loss.item():.5f} | BC: {bc_loss.item():.5f} | "
                    f"Reward: {mean_rewards:.2f} | Len: {self.episode_lengths.get_mean():.0f} | "
                    f"Best: {self.best_rewards:.2f}{format_console_metrics(means)}"
                )
                w = self.writer
                w.add_scalar("losses/latent_mse", latent_loss.item(), self.agent_steps)
                w.add_scalar("losses/bc", bc_loss.item(), self.agent_steps)
                w.add_scalar("episode/rewards", mean_rewards, self.agent_steps)
                w.add_scalar("episode/lengths", self.episode_lengths.get_mean(), self.agent_steps)
                for key, value in means.items():
                    w.add_scalar(f"env/{key}", value, self.agent_steps)

                if self.cfg.save_frequency_epochs > 0 and (
                    self.epoch_num % (log_every * self.cfg.save_frequency_epochs) == 0
                ):
                    self.save(os.path.join(self.nn_dir, "last"))
                if mean_rewards > self.best_rewards and self.epoch_num >= self.cfg.save_best_after_epochs:
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, "best"))

        self.save(os.path.join(self.nn_dir, "final"))
        print("max_agent_steps reached")

    # ------------------------------------------------------------------

    def restore_from_teacher(self, path: str) -> None:
        """Load a stage-1 checkpoint; adapt_tconv stays freshly initialized."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        adapt_missing = [k for k in missing if "adapt_tconv" in k]
        other_missing = [k for k in missing if "adapt_tconv" not in k]
        if other_missing or unexpected:
            raise RuntimeError(
                f"Teacher checkpoint mismatch: missing={other_missing} unexpected={unexpected}"
            )
        print(f"Loaded teacher weights; adaptation module initialized fresh ({len(adapt_missing)} tensors).")
        if "obs_rms" in ckpt:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])

    def restore(self, path: str) -> None:
        """Load a full stage-2 checkpoint (for eval/deployment)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.obs_rms.load_state_dict(ckpt["obs_rms"])
        if "hist_rms" in ckpt:
            self.hist_rms.load_state_dict(ckpt["hist_rms"])

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "obs_rms": self.obs_rms.state_dict(),
                "hist_rms": self.hist_rms.state_dict(),
                "agent_steps": self.agent_steps,
            },
            f"{path}.pth",
        )
