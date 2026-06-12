"""Running mean/std input normalization (Welford-style parallel update).

Statistics are stored as float64 buffers inside an nn.Module so they ride
along in checkpoints. In train mode each forward() updates the statistics; in
eval mode they are frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1.0e-5, clamp: float = 5.0):
        super().__init__()
        self.epsilon = epsilon
        self.clamp = clamp
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    @torch.no_grad()
    def _update(self, batch: torch.Tensor) -> None:
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]

        delta = batch_mean.double() - self.running_mean
        total = self.count + batch_count
        self.running_mean += delta * batch_count / total
        m_a = self.running_var * self.count
        m_b = batch_var.double() * batch_count
        self.running_var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def forward(self, x: torch.Tensor, unnorm: bool = False) -> torch.Tensor:
        if self.training and not unnorm:
            self._update(x)
        mean = self.running_mean.float()
        std = torch.sqrt(self.running_var.float() + self.epsilon)
        if unnorm:
            return torch.clamp(x, -self.clamp, self.clamp) * std + mean
        return torch.clamp((x - mean) / std, -self.clamp, self.clamp)
