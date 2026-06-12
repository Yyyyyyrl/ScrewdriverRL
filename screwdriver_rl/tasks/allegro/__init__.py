"""Allegro hand screwdriver rotation task for Isaac Lab.

Registers:
  Isaac-Allegro-Screwdriver-Rotation-Direct-v0
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Allegro-Screwdriver-Rotation-Direct-v0",
    entry_point=(
        "screwdriver_rl.tasks.allegro."
        "screwdriver_rotation_env:AllegroScrewdriverRotationEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "screwdriver_rl.tasks.allegro."
            "screwdriver_rotation_env_cfg:AllegroScrewdriverRotationEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
