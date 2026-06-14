"""Hand-agnostic shared base for the screwdriver continuous-rotation task.

This subpackage holds the reward/curriculum/observation/reset logic that is
common to every hand.  It does NOT register a gym environment — each hand
(``tasks/allegro``, ``tasks/linker_l20``, ...) subclasses ``ScrewdriverRotationEnv``
and ``ScrewdriverRotationEnvCfg`` here, fills in the hand-specific joint/body
maps and articulation config, and registers its own gym id.
"""

from .screwdriver_rotation_env import ScrewdriverRotationEnv
from .screwdriver_rotation_env_cfg import (
    ASSET_ROOT,
    CurriculumPhaseCfg,
    DomainRandCfg,
    ScrewdriverRotationEnvCfg,
)

__all__ = [
    "ScrewdriverRotationEnv",
    "ScrewdriverRotationEnvCfg",
    "CurriculumPhaseCfg",
    "DomainRandCfg",
    "ASSET_ROOT",
]
