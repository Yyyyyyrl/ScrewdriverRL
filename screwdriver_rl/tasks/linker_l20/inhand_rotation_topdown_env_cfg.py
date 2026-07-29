"""Top-down pre-grasp configuration for Linker L20 in-hand rotation.

The task dynamics, object distribution, observations, rewards, curriculum, and
policy configuration are inherited unchanged from the standard Linker L20
in-hand task.  This module only supplies:

* a fixed hand root above the object with the palm normal exactly world ``-Z``;
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


# The L20 palm outward normal is base-local +X.  This unit quaternion maps +X
# to world -Z and the finger direction (+Z) to world -Y: an exact, level
# top-down palm rather than an oblique side grasp.
TOPDOWN_HAND_POS: tuple[float, float, float] = (0.0, 0.0, 0.615)
TOPDOWN_HAND_ROT: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)

# Representative scale-0.8 object centre used by the static mesh fit.  The hand
# root is 6.5 cm above the centre and the fingertip cage reaches down around it.
TOPDOWN_OBJECT_INIT_POS: tuple[float, float, float] = (0.0023, -0.1783, 0.55)
TOPDOWN_OBJECT_INIT_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Reward rotation is aligned with the gravity vector exactly, per task design.
TOPDOWN_ROT_AXIS: tuple[float, float, float] = (0.0, 0.0, -1.0)

# Full-URDF-mesh optimized seeds.  At scale 0.8 each seed has contact on all
# five distal meshes, no object contact on the palm/metacarpal/proximal/middle
# links, no blocking self collision, and >= 0.04 rad joint-limit margin.
TOPDOWN_PREGRASP_BY_SHAPE: dict[str, dict[str, tuple[float, ...]]] = {
    "cylinder": {
        "index": (0.130000, 0.464707, 1.065931),
        "middle": (0.029007, 0.540377, 0.663232),
        "ring": (0.013229, 0.370236, 0.849679),
        "pinky": (0.097696, 0.326228, 0.970907),
        "thumb": (0.915593, 0.586972, 0.146910, 0.446909),
    },
    "cuboid": {
        "index": (0.123436, 0.496310, 1.105027),
        "middle": (0.029018, 0.601187, 0.719678),
        "ring": (0.012912, 0.393539, 0.881700),
        "pinky": (0.094346, 0.314292, 0.975750),
        "thumb": (0.940885, 0.592575, 0.151998, 0.453834),
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
    """Training config for the gravity-axis, top-down fingertip grasp."""

    rot_axis: tuple[float, float, float] = TOPDOWN_ROT_AXIS
    grasp_cache_name: str = "linker_l20_topdown"
    require_complete_grasp_cache: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        # Mixed training env 0 is a cylinder; all actual training resets come
        # from the required per-shape/per-prototype caches.
        _apply_topdown_seed(self, "cylinder")


@configclass
class LinkerL20InhandRotationTopdownGraspGenEnvCfg(
    LinkerL20InhandGraspGenEnvCfg
):
    """Cache harvester using the top-down seed for the requested shape."""

    grasp_cache_name: str = "linker_l20_topdown"

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_topdown_seed(self, str(self.cache_shape))

    def configure_cache_shape(self, shape: str) -> None:
        """Refresh the seed after the cache tool pins a shape and rebuilds it."""
        self.cache_shape = str(shape)
        _apply_topdown_seed(self, self.cache_shape)
