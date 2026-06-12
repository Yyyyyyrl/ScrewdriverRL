"""Allegro Hand V4 (right, xela tactile model) HandSpec.

Asset: ``assets/hands/allegro/allegro_hand_right.urdf`` — the Isaac Lab-ready
URDF from the MFR benchmark (xela variant). Joint and fingertip names below
are the names that exist in that URDF, not upstream Wonik Allegro names.

The base pose, pregrasp and actuator gains reproduce the MFR screwdriver task
initial posture: hand above a mounted screwdriver at (0, 0, 1.205), palm
facing down-forward, three fingertips (index, middle, thumb) on the handle and
the ring finger parked. See docs/porting_notes.md for provenance.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .. import ASSETS_DIR
from .hand_spec import HandSpec

_URDF_PATH = ASSETS_DIR / "hands" / "allegro" / "allegro_hand_right.urdf"

_FINGER_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "index": (
        "allegro_hand_hitosashi_finger_finger_joint_0",
        "allegro_hand_hitosashi_finger_finger_joint_1",
        "allegro_hand_hitosashi_finger_finger_joint_2",
        "allegro_hand_hitosashi_finger_finger_joint_3",
    ),
    "middle": (
        "allegro_hand_naka_finger_finger_joint_4",
        "allegro_hand_naka_finger_finger_joint_5",
        "allegro_hand_naka_finger_finger_joint_6",
        "allegro_hand_naka_finger_finger_joint_7",
    ),
    "ring": (
        "allegro_hand_kusuri_finger_finger_joint_8",
        "allegro_hand_kusuri_finger_finger_joint_9",
        "allegro_hand_kusuri_finger_finger_joint_10",
        "allegro_hand_kusuri_finger_finger_joint_11",
    ),
    "thumb": (
        "allegro_hand_oya_finger_joint_12",
        "allegro_hand_oya_finger_joint_13",
        "allegro_hand_oya_finger_joint_14",
        "allegro_hand_oya_finger_joint_15",
    ),
}

_FINGERTIP_BODY_NAMES: dict[str, str] = {
    "index": "hitosashi_ee",
    "middle": "naka_ee",
    "ring": "kusuri_ee",
    "thumb": "oya_ee",
}

_PREGRASP: dict[str, tuple[float, ...]] = {
    "index": (0.1, 0.6, 0.6, 0.6),
    "middle": (-0.1, 0.5, 0.9, 0.9),
    "ring": (0.0, 0.5, 0.65, 0.65),
    "thumb": (1.2, 0.3, 0.3, 1.2),
}


def allegro_hand_spec(
    controlled_fingers: tuple[str, ...] = ("index", "middle", "thumb"),
) -> HandSpec:
    """Build the Allegro HandSpec.

    Args:
        controlled_fingers: Fingers driven by the policy (action-vector order).
            Defaults to the MFR 3-finger setup; pass all four for 16-DOF control.
    """
    parked = tuple(f for f in _FINGER_JOINT_NAMES if f not in controlled_fingers)

    joint_pos = {
        name: value
        for finger, names in _FINGER_JOINT_NAMES.items()
        for name, value in zip(names, _PREGRASP[finger])
    }

    articulation_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(_URDF_PATH),
            fix_base=True,
            # Keep fixed joints: merging them removes the *_ee fingertip bodies.
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=True,
            make_instanceable=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.095, 1.33),
            rot=(0.664463, 0.2418448, 0.2418448, 0.664463),  # wxyz
            joint_pos=joint_pos,
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=6.0,
                damping=1.0,
                armature=0.001,
            )
        },
    )

    return HandSpec(
        name="allegro",
        articulation_cfg=articulation_cfg,
        finger_joint_names=_FINGER_JOINT_NAMES,
        fingertip_body_names=_FINGERTIP_BODY_NAMES,
        controlled_fingers=controlled_fingers,
        parked_fingers=parked,
        pregrasp=_PREGRASP,
        thumb_name="thumb",
    )
