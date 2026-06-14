"""Allegro hand continuous screwdriver rotation environment.

All task logic lives in the hand-agnostic
:class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnv`.  This module only
supplies the Allegro-specific joint/body name maps; the matching articulation,
pregrasp, and gym spaces live in
:class:`AllegroScrewdriverRotationEnvCfg`.
"""

from __future__ import annotations

from screwdriver_rl.tasks.base.screwdriver_rotation_env import ScrewdriverRotationEnv

from .screwdriver_rotation_env_cfg import AllegroScrewdriverRotationEnvCfg


class AllegroScrewdriverRotationEnv(ScrewdriverRotationEnv):
    """Continuous screwdriver rotation with the Allegro (right) hand."""

    cfg: AllegroScrewdriverRotationEnvCfg

    # Fingertip (distal pad) bodies — only these should touch the handle.
    FINGERTIP_BODY_NAMES = {
        "index":  "hitosashi_ee",
        "middle": "naka_ee",
        "ring":   "kusuri_ee",
        "thumb":  "oya_ee",
    }

    # Proximal and medial phalange links to penalise when close to the handle.
    # These are the links BEHIND the fingertip: if they touch the handle the
    # policy is using the finger body rather than the fingertip pad.
    PROXIMAL_BODY_PATTERNS = [
        r"^allegro_hand_base_link$",                         # palm
        r"^allegro_hand_hitosashi_finger_finger_link_0$",    # index proximal
        r"^allegro_hand_hitosashi_finger_finger_link_1$",    # index medial
        r"^allegro_hand_naka_finger_finger_link_4$",         # middle proximal
        r"^allegro_hand_naka_finger_finger_link_5$",         # middle medial
        r"^allegro_hand_kusuri_finger_finger_link_8$",       # ring proximal
        r"^allegro_hand_kusuri_finger_finger_link_9$",       # ring medial
        r"^allegro_hand_oya_finger_link_12$",                # thumb proximal
        r"^allegro_hand_oya_finger_link_13$",                # thumb medial
    ]

    # Per-finger joint name tuples (4 DOF each, semantic order).
    FINGER_JOINT_NAMES = {
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
    # Allegro has no mimic/coupled joints — COUPLED_JOINTS inherits {} from base.
