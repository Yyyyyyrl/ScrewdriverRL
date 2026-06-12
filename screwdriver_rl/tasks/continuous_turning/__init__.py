import gymnasium as gym

from .env_cfg import ContinuousTurningEnvCfg, DomainRandCfg

gym.register(
    id="Screwdriver-Continuous-Turning-v0",
    entry_point="screwdriver_rl.tasks.continuous_turning.env:ContinuousTurningEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": ContinuousTurningEnvCfg},
)

__all__ = ["ContinuousTurningEnvCfg", "DomainRandCfg"]
