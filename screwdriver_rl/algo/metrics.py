"""Windowed scalar meters + env-extras aggregation for training logs."""

from __future__ import annotations

import torch


class AverageScalarMeter:
    """Running average over (approximately) the last ``window_size`` values."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0.0

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        new_mean = torch.mean(values.float()).cpu().item()
        size = min(values.numel(), self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        total = old_size + size
        self.current_size = total
        self.mean = (self.mean * old_size + new_mean * size) / total

    def clear(self) -> None:
        self.current_size = 0
        self.mean = 0.0

    def get_mean(self) -> float:
        return self.mean


class EnvMetricsTracker:
    """Aggregates per-step env ``extras`` tensors into per-rollout means.

    Tracks every ``eval_*`` key the env emits, so new reward terms show up in
    tensorboard without touching trainer code.
    """

    def __init__(self, prefix: str = "eval_"):
        self.prefix = prefix
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self.last_means: dict[str, float] = {}

    def accumulate(self, extras: dict) -> None:
        if not isinstance(extras, dict):
            return
        for key, value in extras.items():
            if not key.startswith(self.prefix):
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                scalar = torch.nan_to_num(value.float()).mean().item()
            else:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
            self._sums[key] = self._sums.get(key, 0.0) + scalar
            self._counts[key] = self._counts.get(key, 0) + 1

    def finalize(self) -> dict[str, float]:
        if self._sums:
            self.last_means = {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}
        self._sums, self._counts = {}, {}
        return self.last_means

    def get(self, key: str) -> float | None:
        value = self.last_means.get(key)
        return None if value is None else float(value)


# Short console aliases for the most interesting metrics, printed in order.
CONSOLE_METRICS: tuple[tuple[str, str], ...] = (
    ("eval_net_turns", "NetTurns"),
    ("eval_total_turns", "FwdTurns"),
    ("eval_forward_turn_velocity", "FwdVel"),
    ("eval_reverse_turn_velocity", "RevVel"),
    ("eval_screwdriver_upright_norm", "Tilt"),
    ("eval_turn_contact_gate", "ContactGate"),
    ("eval_turn_motion_gate", "MotionGate"),
    ("eval_turn_upright_gate", "UprightGate"),
    ("eval_mean_fingertip_dist", "TipDist"),
    ("eval_turn_reward", "TurnRew"),
    ("eval_near_reward", "NearRew"),
)


def format_console_metrics(means: dict[str, float]) -> str:
    parts = [f"{alias}: {means[key]:.3f}" for key, alias in CONSOLE_METRICS if key in means]
    return (" | " + " | ".join(parts)) if parts else ""
