"""Training hyperparameter dataclasses (pure python, no isaaclab).

Defaults reproduce the HORA-style setup tuned for this task family:
600-step episodes at 10 Hz control, thousands of parallel envs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetworkCfg:
    """ActorCritic + adaptation module dimensions."""

    actor_units: tuple[int, ...] = (512, 256, 128)
    # Privileged encoder; the last entry is the environment latent dim that
    # the stage-2 adaptation module must reproduce.
    priv_mlp_units: tuple[int, ...] = (256, 128, 8)
    adapt_hidden_dim: int = 32
    # (kernel, stride) per Conv1d layer; designed for history_len=30 -> 3.
    adapt_conv_kernels: tuple[tuple[int, int], ...] = ((9, 2), (5, 1), (5, 1))


@dataclass
class PPOTrainCfg:
    """Stage-1 teacher PPO hyperparameters."""

    learning_rate: float = 5.0e-3  # KL-adaptive from here
    kl_threshold: float = 0.02
    # gamma=0.999 at 10 Hz control: effective horizon ~1000 steps = 100 s,
    # matched to the 60 s episode. Move together with env decimation.
    gamma: float = 0.999
    tau: float = 0.95  # GAE lambda
    horizon_length: int = 32  # ~5% of an episode per rollout
    minibatch_size: int = 32768
    mini_epochs: int = 5
    e_clip: float = 0.2
    clip_value: bool = True
    critic_coef: float = 4.0
    entropy_coef: float = 0.0
    bounds_loss_coef: float = 1.0e-4
    grad_norm: float = 1.0
    truncate_grads: bool = True
    value_bootstrap: bool = True
    normalize_input: bool = True
    normalize_value: bool = True
    normalize_advantage: bool = True
    # Reward scale applied before GAE (keeps values in a friendly range).
    reward_scale: float = 0.01
    max_agent_steps: int = 1_500_000_000
    save_frequency_epochs: int = 200
    save_best_after_epochs: int = 100


@dataclass
class StudentTrainCfg:
    """Stage-2 proprioceptive-adaptation training."""

    learning_rate: float = 3.0e-4
    # Latent imitation is the core HORA/RMA loss; the optional behavior
    # cloning term (dexscrew lesson) additionally matches the student's
    # actions to the frozen teacher's. 0 reproduces pure HORA.
    bc_weight: float = 0.1
    max_agent_steps: int = 300_000_000
    save_frequency_epochs: int = 500
    save_best_after_epochs: int = 0


@dataclass
class TrainRunCfg:
    """Top-level bundle dumped to config.json for reproducibility."""

    network: NetworkCfg = field(default_factory=NetworkCfg)
    ppo: PPOTrainCfg = field(default_factory=PPOTrainCfg)
    student: StudentTrainCfg = field(default_factory=StudentTrainCfg)
