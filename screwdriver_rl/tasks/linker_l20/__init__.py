"""Linker Hand L20 (Left) screwdriver rotation task for Isaac Lab.

Registers:
  Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0
  Isaac-LinkerL20-Screwdriver-Rotation-Top-Grasp-Direct-v0
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0",
    entry_point=(
        "screwdriver_rl.tasks.linker_l20."
        "screwdriver_rotation_env:LinkerL20ScrewdriverRotationEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20."
            "screwdriver_rotation_env_cfg:LinkerL20ScrewdriverRotationEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-LinkerL20-Screwdriver-Rotation-Top-Grasp-Direct-v0",
    entry_point=(
        "screwdriver_rl.tasks.linker_l20."
        "screwdriver_rotation_env:LinkerL20ScrewdriverRotationEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20."
            "screwdriver_rotation_top_grasp_env_cfg:"
            "LinkerL20ScrewdriverRotationTopGraspEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-LinkerL20-Screwdriver-Rotation-DR-Direct-v0",
    entry_point=(
        "screwdriver_rl.tasks.linker_l20."
        "screwdriver_rotation_env:LinkerL20ScrewdriverRotationEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20."
            "screwdriver_rotation_dr_env_cfg:"
            "LinkerL20ScrewdriverRotationDREnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
