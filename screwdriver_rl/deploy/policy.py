"""Env-free deployable controller for a Stage-2 ``deploy.pth`` bundle.

This is the ScrewdriverRL analogue of HORA's ``deploy_ros2.py`` inference path.
A Stage-2 ``deploy.pth`` bundle is self-contained::

    {
      "actor":      <canonical actor state: actor_mlp.* / mu.* / running_mean_std.*
                     (the normaliser sliced to the proprio block)>,
      "actor_arch": {mlp_units, activation, proprio_dim, latent_dim, action_dim,
                     normalize_input, clip_obs},
      "adapter":    <ProprioAdaptNet state dict (history → latent)>,
      "net_dims":   {frame_dim, hist_len, out_dim(=latent_dim)},
      "config":     {n_finger, action_delta_scale, finger_lower, finger_upper,
                     home_targets, prop_hist_len, history_obs_dim, task},
    }

``DeployPolicy.act(finger_q)`` consumes *only* proprioception — joint encoders
plus the ``cur_targets`` it integrates itself plus its rolling history — and emits
the 16-D action.  In the HORA-faithful latent design the actor consumes
``[proprio(P), latent(K)]``; the latent is **inferred proprioceptively** by the
adaptation network (``latent = adapter(history)``), so no privileged/simulation-
only state and no external object tracker are required (pure RMA).

(Legacy euler-bridge bundles — ``actor_arch`` with no ``latent_dim`` — are still
supported: the actor consumes ``[finger_q, cur_targets, euler]`` and the adapter
supplies the euler.  New LinkerL20 bundles use the latent path.)

The actor is rebuilt as a plain MLP rather than via rl_games, so this module
imports cleanly with neither Isaac Sim nor rl_games installed.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from screwdriver_rl.algos.proprio_adapt import ProprioAdaptNet

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "gelu": nn.GELU,
}


class RunningMeanStd(nn.Module):
    """Inference-only replica of rl_games' input normaliser.

    Matches rl_games' ``RunningMeanStd.forward`` in eval mode:
    ``y = clamp((x - mean) / sqrt(var + eps), -5, 5)``.  Buffer names
    (``running_mean``/``running_var``/``count``) match rl_games so the trained
    statistics load directly.
    """

    def __init__(self, dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.ones((), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
        return torch.clamp(y, -5.0, 5.0)


class DeployActor(nn.Module):
    """Deterministic actor: ``mu(actor_mlp([normalize(clip(proprio)) , latent]))``.

    Reproduces the forward pass of the HORA-faithful rl_games actor
    (``screwdriver_rl/algos/latent_network.py``): the proprio block is clipped +
    normalised exactly as in training, the latent (already a bounded
    ``tanh(env_mlp(priv))`` analogue produced by the adaptation net) is
    concatenated **without** normalisation — matching Stage 1, where the latent is
    formed inside the network after the obs normaliser — then the shared
    ``actor_mlp`` + ``mu`` head produce the deterministic action.

    Legacy mode (``latent_dim == 0``): the whole observation is passed as
    ``proprio`` and normalised; ``forward(obs)`` reproduces the old shared-trunk
    actor 1:1.
    """

    def __init__(self, arch: Mapping) -> None:
        super().__init__()
        units = list(arch["mlp_units"])
        act_name = str(arch.get("activation", "elu")).lower()
        act = _ACTIVATIONS.get(act_name, nn.ELU)
        action_dim = int(arch["action_dim"])
        self.latent_dim = int(arch.get("latent_dim", 0))
        # The normalised input block.  Latent mode: proprio_dim (32); legacy: the
        # full obs_dim.
        self.proprio_dim = int(arch.get("proprio_dim", arch.get("obs_dim")))
        self.clip_obs = float(arch.get("clip_obs", 5.0))
        self.normalize_input = bool(arch.get("normalize_input", True))

        layers: list[nn.Module] = []
        d = self.proprio_dim + self.latent_dim
        for u in units:
            layers += [nn.Linear(d, u), act()]
            d = u
        self.actor_mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(d, action_dim)
        self.running_mean_std = RunningMeanStd(self.proprio_dim) if self.normalize_input else None

    def forward(self, proprio: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.clamp(proprio, -self.clip_obs, self.clip_obs)
        if self.running_mean_std is not None:
            x = self.running_mean_std(x)
        if self.latent_dim > 0 and latent is not None:
            x = torch.cat([x, latent], dim=-1)
        return self.mu(self.actor_mlp(x))


def canonicalize_actor_state(
    rlgames_state: Mapping[str, torch.Tensor], proprio_dim: int | None = None
) -> dict:
    """Extract a :class:`DeployActor` state dict from a full rl_games model state.

    Keeps only the actor trunk, the ``mu`` head and the input normaliser, re-keyed
    to :class:`DeployActor`'s submodule names.  rl_games stores these under the
    ``a2c_network.actor_mlp.*`` / ``a2c_network.mu.*`` / ``running_mean_std.*``
    prefixes (``value``/``sigma``/``value_mean_std``, the privileged ``env_mlp``
    encoder, and the central-value critic are dropped — deployment never needs
    them; the latent comes from the adaptation net, not ``env_mlp``).

    ``proprio_dim``: in the latent design the rl_games obs normaliser covers the
    full ``[proprio, privileged]`` obs, but the deploy actor only normalises the
    proprio block (the latent is concatenated post-normalisation).  When given,
    the ``running_mean`` / ``running_var`` buffers are sliced to the first
    ``proprio_dim`` entries so they load into ``DeployActor.running_mean_std``.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in rlgames_state.items():
        if k.startswith("a2c_network.actor_mlp.") or k.startswith("a2c_network.mu."):
            out[k[len("a2c_network."):]] = v
        elif k.startswith("running_mean_std.") and not k.startswith("running_mean_std.running_mean_std"):
            out[k] = v
    if proprio_dim is not None:
        for key in ("running_mean_std.running_mean", "running_mean_std.running_var"):
            t = out.get(key)
            if t is not None and t.dim() == 1 and t.shape[0] > proprio_dim:
                out[key] = t[:proprio_dim].contiguous().clone()
    return out


class DeployPolicy:
    """Proprioception-only controller built from a Stage-2 ``deploy.pth`` bundle."""

    def __init__(self, bundle: str | Mapping, device: str = "cpu") -> None:
        if isinstance(bundle, str):
            bundle = torch.load(bundle, map_location=device)
        self.device = torch.device(device)
        cfg = dict(bundle.get("config", {}))
        self.cfg = cfg

        # Actor.
        self.actor = DeployActor(bundle["actor_arch"]).to(self.device).eval()
        missing, unexpected = self.actor.load_state_dict(bundle["actor"], strict=False)
        if missing or unexpected:
            # running_mean_std may be absent when normalize_input is False; the
            # actor MLP + mu must always be present.
            core_missing = [m for m in missing if not m.startswith("running_mean_std")]
            if core_missing:
                raise RuntimeError(f"actor state dict missing core keys: {core_missing}")

        # Adaptation network (euler estimator).
        nd = bundle["net_dims"]
        self.adapter = ProprioAdaptNet(
            frame_dim=int(nd["frame_dim"]), hist_len=int(nd["hist_len"]), out_dim=int(nd["out_dim"])
        ).to(self.device).eval()
        self.adapter.load_state_dict(bundle["adapter"])

        # Geometry / integration constants.
        self.n_finger = int(cfg.get("n_finger", bundle["actor_arch"]["action_dim"]))
        # Latent mode (latent_dim>0): actor consumes [proprio, latent]; the
        # adapter supplies the latent.  Legacy mode (0): actor consumes
        # [finger_q, cur_targets, euler]; the adapter supplies the euler.
        self.latent_dim = int(bundle["actor_arch"].get("latent_dim", 0))
        self.euler_dim = int(cfg.get("euler_dim", 3))
        self.hist_len = int(nd["hist_len"])
        self.frame_dim = int(nd["frame_dim"])
        self.action_delta_scale = float(cfg.get("action_delta_scale", 0.05))

        def _vec(key, fill):
            v = cfg.get(key)
            t = torch.tensor(v, dtype=torch.float32) if v is not None else torch.full((self.n_finger,), fill)
            return t.to(self.device).view(1, -1)

        self.finger_lower = _vec("finger_lower", -3.14)
        self.finger_upper = _vec("finger_upper", 3.14)
        self.home_targets = _vec("home_targets", 0.0)

        # Integration state.
        self.cur_targets = self.home_targets.clone()
        self.hist = torch.zeros(1, self.hist_len, self.frame_dim, device=self.device)
        self.reset()

    # -- helpers ----------------------------------------------------------- #
    def _to_row(self, x) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, -1)
        if t.shape[1] != self.n_finger:
            raise ValueError(f"expected {self.n_finger} finger positions, got {t.shape[1]}")
        return t

    def _frame(self, finger_q: torch.Tensor) -> torch.Tensor:
        raw = torch.cat([finger_q, self.cur_targets], dim=-1)  # (1, 2*n_finger)
        frame = torch.zeros(1, self.frame_dim, device=self.device)
        n = min(self.frame_dim, raw.shape[1])
        frame[:, :n] = raw[:, :n]
        return frame

    # -- public API -------------------------------------------------------- #
    def reset(self, finger_q=None) -> None:
        """Reset integration state.  Seeds ``cur_targets`` to home and fills the
        whole history with the first frame (mirrors the env's reset seeding)."""
        self.cur_targets = self.home_targets.clone()
        fq = self.home_targets.clone() if finger_q is None else self._to_row(finger_q)
        self.hist = self._frame(fq).unsqueeze(1).repeat(1, self.hist_len, 1)

    @torch.no_grad()
    def act(self, finger_q, return_action: bool = False):
        """One control step (pure proprioceptive RMA).

        Args:
            finger_q: measured positions of the 16 independent finger joints (rad).
            return_action: also return the raw 16-D action (delta), not just the
                integrated absolute targets.

        Returns:
            ``cur_targets`` (1, n_finger) absolute joint targets, or
            ``(cur_targets, action)`` if ``return_action``.
        """
        fq = self._to_row(finger_q)

        # Roll history and append the latest [finger_q, cur_targets] frame.
        self.hist = torch.roll(self.hist, shifts=-1, dims=1)
        self.hist[:, -1] = self._frame(fq)

        pred = self.adapter(self.hist)
        if self.latent_dim > 0:
            # HORA-faithful latent path: actor = mu(actor_mlp([norm(proprio), latent])).
            proprio = torch.cat([fq, self.cur_targets], dim=-1)
            action = torch.clamp(self.actor(proprio, pred[:, : self.latent_dim]), -1.0, 1.0)
        else:
            # Legacy euler-bridge path: actor = mu(actor_mlp(norm([fq, targets, euler]))).
            obs = torch.cat([fq, self.cur_targets, pred[:, : self.euler_dim]], dim=-1)
            action = torch.clamp(self.actor(obs), -1.0, 1.0)

        self.cur_targets = torch.clamp(
            self.cur_targets + self.action_delta_scale * action, self.finger_lower, self.finger_upper
        )
        if return_action:
            return self.cur_targets, action
        return self.cur_targets
