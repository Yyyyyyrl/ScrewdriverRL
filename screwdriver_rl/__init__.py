"""ScrewdriverRL: continuous screwdriver turning with dexterous hands in Isaac Lab.

Package layout:
    assets/   -- URDF + mesh assets (hands, objects)
    robots/   -- hand-agnostic HandSpec contract + per-hand specs (imports isaaclab)
    tasks/    -- DirectRLEnv task implementations (imports isaaclab)
    algo/     -- PPO / ProprioAdapt teacher-student trainers (pure PyTorch)
    configs/  -- training + curriculum dataclasses (pure python)
    utils/    -- CLI override helpers (pure python)

Import rules: ``algo``, ``configs`` and ``utils`` never import isaaclab so they
can be unit-tested without Isaac Sim. ``robots`` and ``tasks`` must only be
imported after the Omniverse app has been launched (see scripts/).
"""

from pathlib import Path

__version__ = "0.1.0"

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
