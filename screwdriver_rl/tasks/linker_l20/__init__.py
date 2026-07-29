"""Linker Hand L20 (Left) screwdriver rotation task for Isaac Lab.

Registers:
  Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0
  Isaac-LinkerL20-Screwdriver-Rotation-Top-Grasp-Direct-v0
  Isaac-LinkerL20-Screwdriver-Rotation-Topdown
  Isaac-LinkerL20-Screwdriver-Rotation-DR-Direct-v0
  Isaac-LinkerL20-Inhand-Rotation        (HORA free-cylinder in-hand rotation)
  Isaac-LinkerL20-Inhand-GraspGen        (grasp-cache collection for the above)
"""

import gymnasium as gym

from . import agents
from . import screwdriver_rotation_topdown_registration  # noqa: F401


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

# ---------------------------------------------------------------------------
# HORA in-hand cylinder rotation (free object) + its grasp-cache generator.
# ---------------------------------------------------------------------------
gym.register(
    id="Isaac-LinkerL20-Inhand-Rotation",
    entry_point=(
        "screwdriver_rl.tasks.linker_l20."
        "inhand_rotation_env:LinkerL20InhandRotationEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20."
            "inhand_rotation_env_cfg:LinkerL20InhandRotationEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_inhand_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-LinkerL20-Inhand-GraspGen",
    entry_point=(
        "screwdriver_rl.tasks.linker_l20."
        "inhand_grasp_gen_env:LinkerL20InhandGraspGenEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20."
            "inhand_rotation_env_cfg:LinkerL20InhandGraspGenEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_inhand_ppo_cfg.yaml",
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
