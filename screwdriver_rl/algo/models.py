"""Actor-critic and adaptation-module networks (pure PyTorch).

Architecture follows HORA (Qi et al. 2022) / DexScrew: the privileged
environment state is compressed by ``env_mlp`` into a small latent that is
concatenated to the proprioceptive observation; the stage-2 student replaces
that encoder with ``adapt_tconv``, a temporal conv over proprioceptive
history, trained to reproduce the same latent.

Modes:
    teacher (priv_info=True, proprio_adapt=False):
        z = tanh(env_mlp(priv))            -> actor([obs, z])
    student (priv_info=True, proprio_adapt=True):
        z = tanh(adapt_tconv(hist))        -> actor([obs, z])
        z_gt = tanh(env_mlp(priv))         (distillation target, train only)
    plain (priv_info=False): actor(obs)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Stack of Linear + ELU layers."""

    def __init__(self, units: tuple[int, ...], input_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        for out_size in units:
            layers.append(nn.Linear(input_size, out_size))
            layers.append(nn.ELU())
            input_size = out_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    """Temporal-conv encoder: proprio history -> environment latent.

    (N, history_len, obs_dim) -> per-frame MLP -> Conv1d stack over time ->
    linear projection to latent_dim.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        history_len: int = 30,
        hidden_dim: int = 32,
        conv_kernels: tuple[tuple[int, int], ...] = ((9, 2), (5, 1), (5, 1)),
    ):
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        conv_layers: list[nn.Module] = []
        for kernel, stride in conv_kernels:
            conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel, stride=stride))
            conv_layers.append(nn.ReLU(inplace=True))
        self.temporal_aggregation = nn.Sequential(*conv_layers)

        out_len = history_len
        for kernel, stride in conv_kernels:
            out_len = (out_len - kernel) // stride + 1
        if out_len <= 0:
            raise ValueError(
                f"conv kernels {conv_kernels} reduce history_len {history_len} to {out_len} <= 0"
            )
        self.low_dim_proj = nn.Linear(hidden_dim * out_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_transform(x)  # (N, T, H)
        x = x.permute(0, 2, 1)  # (N, H, T)
        x = self.temporal_aggregation(x)  # (N, H, T')
        return self.low_dim_proj(x.flatten(1))


class ActorCritic(nn.Module):
    """Gaussian policy + value head with optional privileged/adaptation latent."""

    def __init__(
        self,
        actions_num: int,
        obs_dim: int,
        actor_units: tuple[int, ...] = (512, 256, 128),
        priv_mlp_units: tuple[int, ...] = (256, 128, 8),
        priv_info: bool = True,
        proprio_adapt: bool = False,
        priv_info_dim: int = 0,
        adapt_obs_dim: int = 24,
        adapt_history_len: int = 30,
        adapt_hidden_dim: int = 32,
        adapt_conv_kernels: tuple[tuple[int, int], ...] = ((9, 2), (5, 1), (5, 1)),
    ):
        super().__init__()
        self.priv_info = priv_info
        self.proprio_adapt = proprio_adapt
        self.latent_dim = priv_mlp_units[-1] if priv_info else 0

        actor_input = obs_dim + self.latent_dim
        if priv_info:
            self.env_mlp = MLP(priv_mlp_units, input_size=priv_info_dim)
            if proprio_adapt:
                self.adapt_tconv = ProprioAdaptTConv(
                    obs_dim=adapt_obs_dim,
                    latent_dim=self.latent_dim,
                    history_len=adapt_history_len,
                    hidden_dim=adapt_hidden_dim,
                    conv_kernels=adapt_conv_kernels,
                )

        self.actor_mlp = MLP(actor_units, input_size=actor_input)
        self.value = nn.Linear(actor_units[-1], 1)
        self.mu = nn.Linear(actor_units[-1], actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, dtype=torch.float32))

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                fan_out = module.kernel_size[0] * module.out_channels
                module.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.sigma, 0.0)

    # ------------------------------------------------------------------

    def _latents(self, obs_dict: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (policy latent, ground-truth latent or None)."""
        if not self.priv_info:
            return None, None
        if self.proprio_adapt:
            z = torch.tanh(self.adapt_tconv(obs_dict["proprio_hist"]))
            z_gt = (
                torch.tanh(self.env_mlp(obs_dict["priv_info"]))
                if obs_dict.get("priv_info") is not None
                else None
            )
            return z, z_gt
        z = torch.tanh(self.env_mlp(obs_dict["priv_info"]))
        return z, z

    def _heads(self, obs: torch.Tensor, latent: torch.Tensor | None):
        x = obs if latent is None else torch.cat([obs, latent], dim=-1)
        x = self.actor_mlp(x)
        return self.mu(x), self.value(x)

    def _actor_critic(self, obs_dict: dict):
        latent, latent_gt = self._latents(obs_dict)
        mu, value = self._heads(obs_dict["obs"], latent)
        sigma = self.sigma.expand_as(mu)
        return mu, sigma, value, latent, latent_gt

    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs_dict: dict) -> dict:
        """Sample stochastic actions for rollout collection."""
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        actions = distr.sample()
        return {
            "actions": actions,
            "neglogpacs": -distr.log_prob(actions).sum(dim=-1),
            "values": value,
            "mus": mu,
            "sigmas": sigma,
        }

    @torch.no_grad()
    def act_inference(self, obs_dict: dict) -> torch.Tensor:
        """Deterministic actions for evaluation/deployment."""
        mu, _, _, _, _ = self._actor_critic(obs_dict)
        return mu

    def forward(self, input_dict: dict) -> dict:
        """Training forward pass (PPO update)."""
        prev_actions = input_dict.get("prev_actions")
        mu, logstd, value, latent, latent_gt = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        return {
            "prev_neglogp": torch.squeeze(-distr.log_prob(prev_actions).sum(dim=-1)),
            "values": value,
            "entropy": distr.entropy().sum(dim=-1),
            "mus": mu,
            "sigmas": sigma,
            "latent": latent,
            "latent_gt": latent_gt,
        }

    def forward_stage2(self, obs_dict: dict):
        """Stage-2 forward: student action + latents + frozen-teacher action.

        Returns (mu_student, z_student, z_gt, mu_teacher). ``mu_teacher`` is
        the action the frozen policy takes given the ground-truth latent —
        used by the optional behavior-cloning loss.
        """
        if not (self.priv_info and self.proprio_adapt):
            raise RuntimeError("forward_stage2 requires priv_info and proprio_adapt")
        z = torch.tanh(self.adapt_tconv(obs_dict["proprio_hist"]))
        mu_student, _ = self._heads(obs_dict["obs"], z)
        z_gt, mu_teacher = None, None
        if obs_dict.get("priv_info") is not None:
            with torch.no_grad():
                z_gt = torch.tanh(self.env_mlp(obs_dict["priv_info"]))
                mu_teacher, _ = self._heads(obs_dict["obs"], z_gt)
        return mu_student, z, z_gt, mu_teacher
