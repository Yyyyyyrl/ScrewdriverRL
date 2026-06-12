"""Hand-agnostic contract between the task environment and a specific hand.

A task env never references hand-specific names (joints, fingertip bodies,
pregrasp angles) directly — it only consumes a :class:`HandSpec`. Adding a new
hand to ScrewdriverRL means writing one new module under ``robots/`` that
returns a ``HandSpec`` and registering it in ``robots/__init__.py``; no task
code changes. See docs/adding_new_hands.md.

NOTE: this module imports isaaclab and must only be imported after the
Omniverse app has been launched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab.assets import ArticulationCfg


@dataclass
class HandSpec:
    """Everything the screwdriver tasks need to know about a hand.

    Attributes:
        name: Registry key, e.g. ``"allegro"``.
        articulation_cfg: Full Isaac Lab articulation config (spawner, initial
            base pose, default joint state, actuators). The ``prim_path`` is
            overridden by the task.
        finger_joint_names: Ordered (proximal -> distal) joint names per finger.
        fingertip_body_names: Fingertip end-effector body name per finger. Used
            for the contact proxy / near-contact shaping; must survive URDF
            import (keep ``merge_fixed_joints=False`` in the spawner).
        controlled_fingers: Fingers driven by the policy, in action-vector
            order. Action dim = sum of their joint counts.
        parked_fingers: Fingers held at their pregrasp pose every step.
        pregrasp: Initial/reference joint angles per finger (same order as
            ``finger_joint_names``).
        thumb_name: Finger treated as the thumb by thumb-weighted shaping
            terms; None if the hand has no opposing thumb.
    """

    name: str
    articulation_cfg: ArticulationCfg
    finger_joint_names: dict[str, tuple[str, ...]]
    fingertip_body_names: dict[str, str]
    controlled_fingers: tuple[str, ...]
    parked_fingers: tuple[str, ...] = ()
    pregrasp: dict[str, tuple[float, ...]] = field(default_factory=dict)
    thumb_name: str | None = "thumb"

    def __post_init__(self) -> None:
        all_fingers = set(self.controlled_fingers) | set(self.parked_fingers)
        missing_joints = all_fingers - set(self.finger_joint_names)
        if missing_joints:
            raise ValueError(f"HandSpec {self.name!r}: no joint names for fingers {sorted(missing_joints)}")
        missing_pregrasp = all_fingers - set(self.pregrasp)
        if missing_pregrasp:
            raise ValueError(f"HandSpec {self.name!r}: no pregrasp for fingers {sorted(missing_pregrasp)}")
        missing_tips = set(self.controlled_fingers) - set(self.fingertip_body_names)
        if missing_tips:
            raise ValueError(f"HandSpec {self.name!r}: no fingertip body for fingers {sorted(missing_tips)}")
        for finger in all_fingers:
            n_joints = len(self.finger_joint_names[finger])
            n_pregrasp = len(self.pregrasp[finger])
            if n_joints != n_pregrasp:
                raise ValueError(
                    f"HandSpec {self.name!r}: finger {finger!r} has {n_joints} joints "
                    f"but {n_pregrasp} pregrasp values"
                )
        if self.thumb_name is not None and self.thumb_name not in self.controlled_fingers:
            self.thumb_name = None

    @property
    def num_action_dofs(self) -> int:
        return sum(len(self.finger_joint_names[f]) for f in self.controlled_fingers)
