"""Top-down pre-grasp configuration for Linker L20 in-hand rotation.

The task dynamics, object distribution, observations, rewards, curriculum, and
policy configuration are inherited unchanged from the standard Linker L20
in-hand task.  This module only supplies:

* a fixed hand root above the object with a 45-degree down-facing palm;
* fingertip-only canonical seeds fitted separately to cylinders, cuboids, and
  spheres; and
* a dedicated grasp-cache namespace and grasp-generation configuration.

The per-shape seeds are not training reset caches.  They are collision-checked
starting points from which Isaac Lab harvests settled, per-prototype caches.
The training config deliberately refuses to start without the complete
top-down cache set instead of silently training from an unvalidated fallback.
"""

from __future__ import annotations

import copy

from isaaclab.utils import configclass

from .inhand_rotation_env_cfg import (
    LinkerL20InhandGraspGenEnvCfg,
    LinkerL20InhandRotationEnvCfg,
    _joint_pos_with_mimics,
)


# Strict world -Z was tested first, but every 64 mm cube candidate dropped
# after release: 0 timeouts across the broad and mesh-guided searches.  Tilting
# the down-facing palm 45 degrees lets the index distal curl under the cube
# while the other fingertips oppose it.  The production seed was selected by
# 20 s searches, then its cache rows were replay-certified with strict non-tip
# rejection across independent full-domain-randomisation episodes.
# Base-local palm +X maps to (0,+sqrt(1/2),-sqrt(1/2)); finger direction +Z
# maps to (0,-sqrt(1/2),-sqrt(1/2)).
TOPDOWN_HAND_POS: tuple[float, float, float] = (0.0, 0.0, 0.725)
TOPDOWN_HAND_ROT: tuple[float, float, float, float] = (
    0.27059805007309845,
    0.6532814824381883,
    0.6532814824381883,
    -0.27059805007309845,
)

# Physics-selected nominal operator-load centre.  Production training resets
# still come from the versioned, per-scale cache rather than this single seed.
TOPDOWN_OBJECT_INIT_POS: tuple[float, float, float] = (
    -0.009065027347372086,
    -0.08247900578976629,
    0.5695643860739765,
)
TOPDOWN_OBJECT_INIT_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Reward rotation is aligned with the gravity vector exactly, per task design.
TOPDOWN_ROT_AXIS: tuple[float, float, float] = (0.0, 0.0, -1.0)

# The cuboid seed below is the physics-selected production pose for the 64 mm
# cube.  The legacy non-production shape seeds remain for tooling compatibility;
# the cache CLI only permits cuboid generation for this deployment pipeline.
TOPDOWN_PREGRASP_BY_SHAPE: dict[str, dict[str, tuple[float, ...]]] = {
    "cylinder": {
        "index": (0.130000, 0.464707, 1.065931),
        "middle": (0.029007, 0.540377, 0.663232),
        "ring": (0.013229, 0.370236, 0.849679),
        "pinky": (0.097696, 0.326228, 0.970907),
        "thumb": (0.915593, 0.586972, 0.146910, 0.446909),
    },
    "cuboid": {
        "index": (0.12000000000000001, 0.27542790417018326, 1.044469372720009),
        "middle": (-0.12000000000000001, 0.6532812108471646, 0.7279558832753705),
        "ring": (0.09427624393544437, 0.526782625834149, 0.6824411534684619),
        "pinky": (0.11521066185128623, 0.39424901833581494, 0.6763672454250939),
        "thumb": (0.7750714874040479, 1.0784517282082406, 0.42034865845335045, 0.4406184473026461),
    },
    "sphere": {
        "index": (0.129891, 0.465224, 1.066452),
        "middle": (0.028877, 0.593924, 0.711791),
        "ring": (0.013671, 0.409232, 0.887721),
        "pinky": (0.098823, 0.361088, 1.000636),
        "thumb": (0.915231, 0.586550, 0.146607, 0.446640),
    },
}


def _apply_topdown_seed(cfg, shape: str) -> None:
    """Apply one shape-family seed after the base config builds its assets."""
    if shape not in TOPDOWN_PREGRASP_BY_SHAPE:
        raise ValueError(
            f"Unsupported top-down grasp shape {shape!r}; expected one of "
            f"{tuple(TOPDOWN_PREGRASP_BY_SHAPE)}"
        )

    pregrasp = copy.deepcopy(TOPDOWN_PREGRASP_BY_SHAPE[shape])
    cfg.rot_axis = TOPDOWN_ROT_AXIS
    cfg.pregrasp_positions = pregrasp
    cfg.robot_cfg.init_state.pos = TOPDOWN_HAND_POS
    cfg.robot_cfg.init_state.rot = TOPDOWN_HAND_ROT
    cfg.robot_cfg.init_state.joint_pos = _joint_pos_with_mimics(pregrasp)
    cfg.grasp_gen_obj_init_pos = TOPDOWN_OBJECT_INIT_POS
    cfg.grasp_gen_obj_init_rot = TOPDOWN_OBJECT_INIT_ROT
    cfg.object_cfg.init_state.pos = TOPDOWN_OBJECT_INIT_POS
    cfg.object_cfg.init_state.rot = TOPDOWN_OBJECT_INIT_ROT


@configclass
class LinkerL20InhandRotationTopdownEnvCfg(LinkerL20InhandRotationEnvCfg):
    """Training config for the gravity-axis, tilted top-down fingertip grasp."""

    rot_axis: tuple[float, float, float] = TOPDOWN_ROT_AXIS
    # Preserve a ~27.6 mm centre-drop allowance below the selected cage centre.
    reset_height_threshold: float = 0.542
    grasp_cache_name: str = "linker_l20_topdown_cube64_v2"
    grasp_orientation: str = "top-down-tilted-45deg-v2-replay-certified"
    require_complete_grasp_cache: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        # The bottom-up D=0.1 ablation passed its palm gate, but the same
        # damping made this gravity-axis grasp rest on hand_base_link for
        # 1.451% of step-env samples (>1% hard limit).  Keep top-down on the
        # replay-certified G20 damping; reward/actor changes remain shared.
        self.robot_cfg.actuators["fingers"].damping = 1.0
        # The production object is a 64 mm cube.  All actual training resets
        # come from the required per-scale cuboid caches.
        _apply_topdown_seed(self, "cuboid")


@configclass
class LinkerL20InhandRotationTopdownGraspGenEnvCfg(
    LinkerL20InhandGraspGenEnvCfg
):
    """Cache harvester using the top-down seed for the requested shape."""

    grasp_cache_name: str = "linker_l20_topdown_cube64_v2"
    grasp_orientation: str = "top-down-tilted-45deg-v2-replay-certified"
    # This class inherits the generic bottom-up generator directly, so repeat
    # the orientation-specific height frame explicitly.
    reset_height_threshold: float = 0.542
    # The replay-certified candidate has a broad stable basin.  Independent
    # 0.02 rad q16 noise retained 89.45% of rows across four full-DR replays
    # while preserving useful reset diversity.
    grasp_gen_pose_noise: float = 0.02
    # Require harvested rows to sit at least 10 mm above the actual fall gate.
    grasp_gen_accept_z_margin: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_topdown_seed(self, str(self.cache_shape))
        self.grasp_gen_pose_noise = 0.02

    def configure_cache_shape(self, shape: str) -> None:
        """Refresh the seed after the cache tool pins a shape and rebuilds it."""
        self.cache_shape = str(shape)
        _apply_topdown_seed(self, self.cache_shape)
