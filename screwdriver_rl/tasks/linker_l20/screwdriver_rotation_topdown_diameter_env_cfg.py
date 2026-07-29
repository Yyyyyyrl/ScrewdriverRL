"""Diameter-domain-randomized configuration for the Linker L20 top-down task.

This thin subclass keeps the already validated 64 mm task as its nominal
configuration, then swaps only the screwdriver spawn for an independent,
fixed-length 60/64/68 mm asset bank.  Observation/action/reward/curriculum,
dynamics randomization, controller and PPO settings continue to come from the
baseline Linker screwdriver task.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.utils import configclass

from screwdriver_rl.utils.linker_topdown_diameter_postures import (
    TOPDOWN_PREGRASP_POSITIONS_BUCKETS,
    TOPDOWN_RESET_POSITIONS_BUCKETS,
    TOPDOWN_ROOT_POS_OFFSETS_BUCKETS,
    TOPDOWN_ROOT_QUATS_WXYZ_BUCKETS,
    TOPDOWN_SCREWDRIVER_TILT_XY_BUCKETS,
)
from .screwdriver_rotation_topdown_env_cfg import (
    ASSET_ROOT,
    LinkerL20ScrewdriverRotationTopdownEnvCfg,
)


@configclass
class LinkerL20ScrewdriverRotationTopdownDiameterRandEnvCfg(
    LinkerL20ScrewdriverRotationTopdownEnvCfg
):
    """Top-down task with equal-weight 60/64/68 mm handle diameters."""

    # The tangential-speed reward reference is radius-derived for this task.
    # The baseline Linker task retains its original scalar behavior.
    scale_drive_speed_with_geometry: bool = True

    def __post_init__(self) -> None:
        # Build the validated nominal configuration first.  Its explicit guard
        # protects direct users of the fixed-64-mm class from accidentally
        # selecting the unrelated baseline 34/40/46-mm variant bank.
        self.domain_rand.randomize_geometry = False
        super().__post_init__()

        self.screwdriver_variants_dir = str(
            ASSET_ROOT / "screwdriver" / "topdown_variants"
        )
        self.geometry_variant_assignment = "cyclic"
        self.domain_rand.randomize_geometry = True
        self._enable_geometry_randomisation()

        # Geometry is privileged environment state.  Length remains fixed but
        # the second channel stays at 1.0 to preserve the shared DR observation
        # contract [diameter_scale, length_scale].
        self.privileged_obs_dim += 2
        if self.latent_conditioned:
            obs_dim = self.history_obs_dim + self.privileged_obs_dim
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.pregrasp_positions_buckets = [
            {finger: tuple(values) for finger, values in bucket.items()}
            for bucket in TOPDOWN_PREGRASP_POSITIONS_BUCKETS
        ]
        self.reset_joint_positions_buckets = [
            {finger: tuple(values) for finger, values in bucket.items()}
            for bucket in TOPDOWN_RESET_POSITIONS_BUCKETS
        ]
        self.pregrasp_root_pos_offsets_buckets = list(
            TOPDOWN_ROOT_POS_OFFSETS_BUCKETS
        )
        self.pregrasp_root_quats_buckets = list(TOPDOWN_ROOT_QUATS_WXYZ_BUCKETS)
        self.reset_screwdriver_tilt_xy_buckets = list(
            TOPDOWN_SCREWDRIVER_TILT_XY_BUCKETS
        )
