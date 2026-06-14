"""Linker Hand L20 (Left) continuous screwdriver rotation environment.

All task logic lives in the hand-agnostic
:class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnv`.  This module only
supplies the Linker-specific joint/body name maps and the mimic-joint coupling;
the matching articulation, pregrasp, and gym spaces live in
:class:`LinkerL20ScrewdriverRotationEnvCfg`.

The Linker L20 has 16 independently-driven finger DOFs (index/middle/ring/pinky:
mcp_roll, mcp_pitch, pip; thumb: cmc_yaw, cmc_roll, cmc_pitch, mcp) plus 5 mimic
distal joints (``*_dip`` follow ``*_pip``; ``thumb_ip`` follows ``thumb_mcp``).
The mimic joints are handled by ``COUPLED_JOINTS`` — see the base class for how
this is robust to either URDF-import behaviour (independent vs. PhysX-coupled).
"""

from __future__ import annotations

from screwdriver_rl.tasks.base.screwdriver_rotation_env import ScrewdriverRotationEnv

from .screwdriver_rotation_env_cfg import LinkerL20ScrewdriverRotationEnvCfg


class LinkerL20ScrewdriverRotationEnv(ScrewdriverRotationEnv):
    """Continuous screwdriver rotation with the Linker Hand L20 (left)."""

    cfg: LinkerL20ScrewdriverRotationEnvCfg

    # Fingertip (distal pad) bodies — only these should touch the handle.
    FINGERTIP_BODY_NAMES = {
        "index":  "index_distal",
        "middle": "middle_distal",
        "ring":   "ring_distal",
        "pinky":  "pinky_distal",
        "thumb":  "thumb_distal",
    }

    # Non-fingertip links to penalise when close to the handle (palm, metacarpals,
    # proximal and medial phalanges).  Everything BEHIND the distal pads.
    PROXIMAL_BODY_PATTERNS = [
        r"^hand_base_link$",                          # palm
        r"^(index|middle|ring|pinky)_metacarpals$",   # knuckle bases
        r"^(index|middle|ring|pinky)_proximal$",      # proximal phalanges
        r"^(index|middle|ring|pinky)_middle$",        # medial phalanges
        r"^thumb_metacarpals_base[12]$",              # thumb CMC staging
        r"^thumb_metacarpals$",
        r"^thumb_proximal$",
    ]

    # Per-finger INDEPENDENT joint names (semantic order).  Mimic distal joints
    # (*_dip, thumb_ip) are NOT listed here — they are driven via COUPLED_JOINTS.
    FINGER_JOINT_NAMES = {
        "index":  ("index_mcp_roll", "index_mcp_pitch", "index_pip"),
        "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
        "ring":   ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
        "pinky":  ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
        "thumb":  ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
    }

    # Mimic followers: follower -> (master, multiplier, offset).  Multipliers
    # taken verbatim from the URDF <mimic> tags.
    COUPLED_JOINTS = {
        "index_dip":  ("index_pip", 0.8917, 0.0),
        "middle_dip": ("middle_pip", 0.8917, 0.0),
        "ring_dip":   ("ring_pip", 0.8917, 0.0),
        "pinky_dip":  ("pinky_pip", 0.8917, 0.0),
        "thumb_ip":   ("thumb_mcp", 1.1619, 0.0),
    }

    # Self-collision pair filters: physically-impossible overlaps created by the
    # inflated collision hulls near the palm.  These links are rigidly clustered
    # at the palm and cannot touch on the real hand, so filtering them is
    # sim-to-real-safe.  The deployment-critical collisions (fingertip<->fingertip,
    # a finger crossing into a neighbour's middle/distal) are NOT filtered.
    SELF_COLLISION_FILTER_PAIRS = [
        # palm <-> each finger's proximal phalanx (can't fold back into the palm)
        ("hand_base_link", "index_proximal"),
        ("hand_base_link", "middle_proximal"),
        ("hand_base_link", "ring_proximal"),
        ("hand_base_link", "pinky_proximal"),
        # palm <-> thumb's non-adjacent CMC chain + proximal
        ("hand_base_link", "thumb_metacarpals_base1"),
        ("hand_base_link", "thumb_metacarpals"),
        ("hand_base_link", "thumb_proximal"),
        # adjacent knuckle bases (rigidly packed at the palm)
        ("index_metacarpals", "middle_metacarpals"),
        ("middle_metacarpals", "ring_metacarpals"),
        ("ring_metacarpals", "pinky_metacarpals"),
        ("thumb_metacarpals_base2", "index_metacarpals"),
        # thumb's nested 3-stage CMC chain: non-adjacent internal segments whose
        # convex hulls overlap (base2->base1->metacarpals->proximal->distal).
        # These are a rigid chain that can't self-collide on the real thumb.
        # NOTE: thumb_distal <-> FINGERS / palm stay ACTIVE (real opposition).
        ("thumb_metacarpals_base2", "thumb_metacarpals"),
        ("thumb_metacarpals_base2", "thumb_proximal"),
        ("thumb_metacarpals_base2", "thumb_distal"),
        ("thumb_metacarpals_base1", "thumb_proximal"),
        ("thumb_metacarpals_base1", "thumb_distal"),
        ("thumb_metacarpals", "thumb_distal"),
    ]
