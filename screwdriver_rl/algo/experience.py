"""Rollout storage with GAE for PPO (pure PyTorch)."""

from __future__ import annotations

import math

import torch


def _flatten_time(arr: torch.Tensor) -> torch.Tensor:
    """(T, N, ...) -> (N*T, ...)."""
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer:
    """Fixed-horizon rollout storage over parallel envs.

    Stores observations, privileged info, actions and PPO bookkeeping;
    computes advantages/returns with GAE and serves shuffled-free contiguous
    minibatches (the data is already decorrelated across thousands of envs).
    """

    def __init__(
        self,
        num_envs: int,
        horizon_length: int,
        minibatch_size: int,
        obs_dim: int,
        act_dim: int,
        priv_dim: int,
        device: str,
    ):
        self.device = device
        self.num_envs = num_envs
        self.horizon = horizon_length
        self.batch_size = num_envs * horizon_length
        self.minibatch_size = max(1, min(int(minibatch_size), self.batch_size))
        self.num_minibatches = max(1, math.ceil(self.batch_size / self.minibatch_size))

        T, N = horizon_length, num_envs
        self.storage: dict[str, torch.Tensor] = {
            "obses": torch.zeros((T, N, obs_dim), device=device),
            "priv_info": torch.zeros((T, N, priv_dim), device=device),
            "actions": torch.zeros((T, N, act_dim), device=device),
            "mus": torch.zeros((T, N, act_dim), device=device),
            "sigmas": torch.zeros((T, N, act_dim), device=device),
            "rewards": torch.zeros((T, N, 1), device=device),
            "values": torch.zeros((T, N, 1), device=device),
            "returns": torch.zeros((T, N, 1), device=device),
            "neglogpacs": torch.zeros((T, N), device=device),
            "dones": torch.zeros((T, N), dtype=torch.uint8, device=device),
        }
        self.data: dict[str, torch.Tensor] | None = None
        self._last_range = (0, 0)

    def update(self, name: str, t: int, value: torch.Tensor) -> None:
        self.storage[name][t] = value

    def compute_returns(self, last_values: torch.Tensor, gamma: float, tau: float) -> None:
        """GAE: A_t = delta_t + gamma*tau*(1-done) * A_{t+1}."""
        advantage = 0.0
        for t in reversed(range(self.horizon)):
            next_values = last_values if t == self.horizon - 1 else self.storage["values"][t + 1]
            not_done = (1.0 - self.storage["dones"][t].float()).unsqueeze(1)
            delta = (
                self.storage["rewards"][t]
                + gamma * next_values * not_done
                - self.storage["values"][t]
            )
            advantage = delta + gamma * tau * not_done * advantage
            self.storage["returns"][t] = advantage + self.storage["values"][t]

    def prepare_training(self, normalize_advantage: bool = True) -> None:
        self.data = {k: _flatten_time(v) for k, v in self.storage.items()}
        advantages = self.data["returns"] - self.data["values"]
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        self.data["advantages"] = advantages.squeeze(1)

    def update_mu_sigma(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        start, end = self._last_range
        self.data["mus"][start:end] = mu
        self.data["sigmas"][start:end] = sigma

    def __len__(self) -> int:
        return self.num_minibatches

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.minibatch_size
        end = min((idx + 1) * self.minibatch_size, self.batch_size)
        self._last_range = (start, end)
        return {k: v[start:end] for k, v in self.data.items()}
