"""Gym registration for the exact, intentionally unsuffixed top-down task ID."""

import gymnasium as gym


gym.register(
    id="Isaac-LinkerL20-Screwdriver-Rotation-Topdown",
    entry_point=(
        "screwdriver_rl.tasks.linker_l20."
        "screwdriver_rotation_env:LinkerL20ScrewdriverRotationEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20."
            "screwdriver_rotation_topdown_diameter_env_cfg:"
            "LinkerL20ScrewdriverRotationTopdownDiameterRandEnvCfg"
        ),
        "rl_games_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20.agents:rl_games_ppo_cfg.yaml"
        ),
    },
)


# HORA-faithful extrinsics-split validation variant: fixed 64 mm handle, full
# deployability DR except geometry, but the actor latent encodes only slow
# extrinsics (load proxy + contact friction) — see the cfg module docstring.
gym.register(
    id="Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora",
    entry_point=(
        "screwdriver_rl.tasks.linker_l20."
        "screwdriver_rotation_env:LinkerL20ScrewdriverRotationEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20."
            "screwdriver_rotation_topdown_hora_env_cfg:"
            "LinkerL20ScrewdriverRotationTopdownHoraEnvCfg"
        ),
        "rl_games_cfg_entry_point": (
            "screwdriver_rl.tasks.linker_l20.agents:rl_games_ppo_cfg.yaml"
        ),
    },
)

