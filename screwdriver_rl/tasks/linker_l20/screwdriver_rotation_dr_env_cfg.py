"""LinkerL20 screwdriver rotation with geometry + physics domain randomization.

A thin subclass of :class:`LinkerL20ScrewdriverRotationEnvCfg` that turns on the
new DR axes *at construction time* (so the cfg ``__post_init__`` can bump the
privileged-obs dim 19→21 and build the per-(diameter,length)-bucket pregrasp
table).  Enabling the flags via a post-construction CLI override would miss
``__post_init__`` and is therefore not supported — use this task id instead.

Prerequisite: run ``python tools/generate_screwdriver_variants.py`` once so the
variant URDFs + ``manifest.json`` exist.
"""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass

from screwdriver_rl.tasks.base.screwdriver_rotation_env_cfg import DomainRandCfg
from screwdriver_rl.tasks.linker_l20.screwdriver_rotation_env_cfg import (
    LinkerL20ScrewdriverRotationEnvCfg,
)


@configclass
class LinkerL20ScrewdriverRotationDREnvCfg(LinkerL20ScrewdriverRotationEnvCfg):
    """LinkerL20 + per-env handle geometry, contact friction, and tilt damping DR."""

    domain_rand: DomainRandCfg = field(
        default_factory=lambda: DomainRandCfg(
            randomize_geometry=True,
            randomize_contact_friction=True,
            randomize_tilt_damping=True,
        )
    )
