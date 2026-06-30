"""Custom rl_games network: a privileged-latent-conditioned actor-critic.

This is the ScrewdriverRL port of HORA's ``ActorCritic`` (``hora/algo/models/
models.py:138-177``), wired into rl_games so Stage 1 trains a HORA-faithful,
*deployable* policy.

The actor observation is ``[proprio(P), privileged(V)]`` (the LinkerL20 env's
``latent_conditioned`` mode).  Instead of feeding the raw privileged tail to the
policy MLP, this network compresses it through a small ``env_mlp`` bottleneck::

    proprio = obs[:, :P]
    priv    = obs[:, P:]
    latent  = tanh(env_mlp(priv))          # K-D (default K=8)
    out     = actor_mlp(cat([proprio, latent]))   # then mu / value heads

At Stage 2 / deploy the privileged tail is unavailable; the Stage-2 adaptation
network reproduces ``latent`` from proprioceptive history alone (pure RMA).  The
actor MLP + mu head are identical in both regimes — only the source of ``latent``
changes — so the trained policy deploys directly (see ``screwdriver_rl/deploy/``).

The obs normaliser lives in the rl_games *model* wrapper (it normalises the full
``[proprio, priv]`` obs before ``forward``), so ``env_mlp`` sees the normalised
privileged slice — exactly what the Stage-2 teacher latent must match.

Register with ``register_latent_network()`` before ``Runner.load`` and set
``network.name: priv_latent_actor_critic`` in the rl_games YAML, alongside
``proprio_dim``, ``latent_dim`` and ``priv_mlp_units``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import A2CBuilder

# Name used both in the YAML (``network.name``) and the registry.
LATENT_NETWORK_NAME = "priv_latent_actor_critic"


class PrivLatentA2CBuilder(A2CBuilder):
    """A2C builder whose actor compresses the privileged obs tail into a latent."""

    # ``A2CBuilder.build`` hardcodes ``A2CBuilder.Network``; override so the
    # registry instantiates *our* Network subclass.
    def build(self, name, **kwargs):
        return PrivLatentA2CBuilder.Network(self.params, **kwargs)

    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # Custom dims (read before the base __init__ resizes the MLP input).
            self.proprio_dim = int(params["proprio_dim"])
            self.latent_dim = int(params["latent_dim"])
            self.priv_mlp_units = [int(u) for u in params.get("priv_mlp_units", [256, 128])]

            input_shape = kwargs["input_shape"]
            self.full_obs_dim = int(input_shape[0])
            self.priv_dim = self.full_obs_dim - self.proprio_dim
            if self.priv_dim <= 0:
                raise ValueError(
                    f"priv_latent network: full obs {self.full_obs_dim} <= proprio_dim "
                    f"{self.proprio_dim}; expected obs = [proprio, privileged]."
                )

            # Build the actor/critic MLPs sized for [proprio, latent] by handing
            # the base __init__ a reduced input_shape.  The model-level obs
            # normaliser and forward() still see the full obs; we slice inside
            # forward().
            kwargs = dict(kwargs)
            kwargs["input_shape"] = (self.proprio_dim + self.latent_dim,)
            super().__init__(params, **kwargs)

            # Privileged encoder: priv_dim -> priv_mlp_units -> latent_dim.
            # tanh is applied in forward() (bounded latent, matching HORA).
            layers: list[nn.Module] = []
            d = self.priv_dim
            for u in self.priv_mlp_units:
                layers += [nn.Linear(d, u), self.activations_factory.create(self.activation)]
                d = u
            layers += [nn.Linear(d, self.latent_dim)]
            self.env_mlp = nn.Sequential(*layers)

            # env_mlp is created after the base init loop, so initialise it here
            # with the same initializer the rest of the MLP uses.
            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.env_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)

        def _encode(self, obs: torch.Tensor) -> torch.Tensor:
            """obs=[proprio, priv] (normalised) -> [proprio, tanh(env_mlp(priv))]."""
            proprio = obs[:, : self.proprio_dim]
            priv = obs[:, self.proprio_dim :]
            latent = torch.tanh(self.env_mlp(priv))
            return torch.cat([proprio, latent], dim=-1)

        def forward(self, obs_dict):
            merged = dict(obs_dict)
            merged["obs"] = self._encode(obs_dict["obs"])
            return super().forward(merged)


def register_latent_network() -> str:
    """Register the latent network with rl_games (idempotent).  Call before
    ``Runner.load``.  Returns the network name to put in ``network.name``."""
    from rl_games.algos_torch import model_builder

    model_builder.register_network(LATENT_NETWORK_NAME, PrivLatentA2CBuilder)
    return LATENT_NETWORK_NAME
