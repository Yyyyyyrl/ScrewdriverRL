"""Map the 16-D ScrewdriverRL policy joints onto LinkerHand L20/G20 SDK commands.

The trained policy outputs 16 *independent* finger-joint targets (radians, in the
training-URDF convention).  The LinkerHand SDK accepts a length-20 vector of
``0..255`` "range" values per hand, published as ``sensor_msgs/JointState.position``
to ``/cb_<side>_hand_control_cmd`` (or passed to ``LinkerHandApi.finger_move``).

SDK 20-slot layout (confirmed by the vendor's own Isaac Gym example,
``examples/L20/l20_isaacgym/l20_example.py:reorderList``):

    0-4   root-flex   {thumb, index, middle, ring, pinky}
    5-9   side-bend   {thumb, index, middle, ring, pinky}   (abduction)
    10    thumb rotation
    11-14 reserved (left untouched / 0)
    15-19 bend / tip  {thumb, index, middle, ring, pinky}

The tip slot drives PIP with DIP following mechanically (one bend slot per
finger), matching our ``COUPLED_JOINTS`` — so the 5 mimic joints are not sent.

Radian ↔ 0..255 calibration
---------------------------
``L20_L_MIN/MAX/DIRECT`` below are vendored **verbatim** from the live SDK's own
conversion tables — ``linker_hand_sdk_ros/scripts/LinkerHand/utils/mapping.py``
(``l20_l_min`` / ``l20_l_max`` / ``l20_l_derict``, the block marked "L20 L OK").
That module is what ``LinkerHandApi`` itself uses, so it is the authoritative
description of the hand: with ``DIRECT == -1`` on every active left-hand slot,
range 255 = arc min = open/extended and range 0 = arc max = flexed.

⚠️ The SDK repo also ships a *stale* standalone copy of these tables in
``range_to_arc/scripts/utils/linker_range_arc.py`` (different endpoints, and
direction 0 instead of -1 on slots 0 and 10 — i.e. inverted thumb root-flex and
thumb rotation).  Never use that module.  We deliberately vendor the good tables
instead of importing the SDK at runtime; ``tests/test_linker_sdk_map.py``
re-reads the SDK checkout and fails if these constants ever drift.

Joint mapping strategy
----------------------
Our training-URDF ranges differ from the SDK's calibrated ranges (e.g. finger
abduction ±0.17 vs ±0.26, pip 0..1.57 vs tip arc 0..1.08), so each joint is
mapped by *normalized fraction of range*: the same grip fraction in our
convention maps to the same fraction of the SDK range.  The map is affine and
bijective per joint, and the state read-back uses the exact inverse, so the
policy always sees a self-consistent world; the residual absolute-angle
distortion is verified on hardware via ``hand_check pose`` (visual parity with
the sim render).

Per-joint sign conventions (especially abduction and the thumb) cannot be
derived from code alone.  ``hand_check wiggle`` runs a guided per-joint motion
test on the real hand and writes a small **calibration overlay** (JSON) that
:func:`apply_calibration` loads at runtime — no code edits needed:

    {"version": 1, "joints": {"index_mcp_roll": {"flip": true},
                              "thumb_cmc_pitch": {"slot": 0, "flip": false}}}

The overlay may also override a joint's **mapping range** (``lo``/``hi``, in
our training-URDF radians).  Setting a joint's ``lo``/``hi`` to its SDK slot's
calibrated arc range (``L20_L_MIN/MAX[slot]``) turns the fraction map into an
**absolute-angle map** for that joint (``arc == clamp(value, lo, hi)``): the
physical angle then matches the policy's angle instead of its range fraction.
Measured on hardware 2026-07-06, the default fraction map leaves the four
fingertips 0.20–0.36 rad straighter than sim at the pregrasp (URDF pip range
0..1.57 vs SDK tip arc 0..1.08) — see ``linker_calib_absolute.json``.
"""

from __future__ import annotations

import json
from typing import Mapping, NamedTuple, Sequence

# --------------------------------------------------------------------------- #
# L20/G20 *left-hand* calibration — vendored verbatim from the live SDK
# (linker_hand_sdk_ros/scripts/LinkerHand/utils/mapping.py, "# L20 L OK").
# Guarded against drift by tests/test_linker_sdk_map.py::test_tables_match_sdk.
# --------------------------------------------------------------------------- #
L20_L_MIN = [0, 0, 0, 0, 0, -0.297, -0.26, -0.26, -0.26, -0.26, 0.122, 0, 0, 0, 0, 0, 0, 0, 0, 0]
L20_L_MAX = [0.87, 1.4, 1.4, 1.4, 1.4, 0.683, 0.26, 0.26, 0.26, 0.26, 1.78, 0, 0, 0, 0, 1.29, 1.08, 1.08, 1.08, 1.08]
L20_L_DIRECT = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1]

# Slots the SDK leaves untouched (no actuator); never feed these to scale_value
# (their min == max would divide by zero).
_RESERVED_SLOTS = (11, 12, 13, 14)


def _scale(v: float, a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return (v - a_min) * (b_max - b_min) / (a_max - a_min) + b_min


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def arc_to_range_left(arc20: Sequence[float]) -> list[float]:
    """Radian (SDK convention) → 0..255 per slot, for the L20/G20 left hand.

    Reproduces the live SDK's ``arc_to_range_left(..., hand_joint="L20")``
    exactly (reserved slots 11-14 left at 0).
    """
    out = [0.0] * 20
    for i in range(20):
        if i in _RESERVED_SLOTS:
            continue
        v = _clamp(arc20[i], min(L20_L_MIN[i], L20_L_MAX[i]), max(L20_L_MIN[i], L20_L_MAX[i]))
        if L20_L_DIRECT[i] == -1:
            out[i] = _scale(v, L20_L_MIN[i], L20_L_MAX[i], 255, 0)
        else:
            out[i] = _scale(v, L20_L_MIN[i], L20_L_MAX[i], 0, 255)
    return out


def range_to_arc_left(range20: Sequence[float]) -> list[float]:
    """0..255 per slot → radian (SDK convention), inverse of :func:`arc_to_range_left`."""
    out = [0.0] * 20
    for i in range(20):
        if i in _RESERVED_SLOTS:
            continue
        v = _clamp(range20[i], 0, 255)
        if L20_L_DIRECT[i] == -1:
            out[i] = _scale(v, 0, 255, L20_L_MAX[i], L20_L_MIN[i])
        else:
            out[i] = _scale(v, 0, 255, L20_L_MIN[i], L20_L_MAX[i])
    return out


# --------------------------------------------------------------------------- #
# Semantic 16 → 20 mapping (our training-URDF joint order → SDK slot).
#
#   lo/hi : our training-URDF soft limits (radians) for range normalisation
#   flip  : reverse the grip fraction before mapping into the SDK range
#           (set per hand via the calibration overlay after `hand_check wiggle`)
#
# Our 16-joint order is fingers (index, middle, ring, pinky, thumb) × joints,
# matching screwdriver_rl/tasks/linker_l20 FINGER_JOINT_NAMES.  Thumb slot
# assignment {pitch→0, roll→5, yaw→10, mcp→15} follows the vendor Isaac Gym
# example (root-flex/side-bend/rotation/bend); signs remain hardware-verified.
# --------------------------------------------------------------------------- #

class JointSpec(NamedTuple):
    name: str
    slot: int
    lo: float
    hi: float
    flip: bool


DEFAULT_JOINTS: tuple[JointSpec, ...] = (
    JointSpec("index_mcp_roll", 6, -0.17, 0.17, False),   # abduction
    JointSpec("index_mcp_pitch", 1, 0.00, 1.40, False),   # root-flex
    JointSpec("index_pip", 16, 0.00, 1.57, False),        # bend (dip mimics)
    JointSpec("middle_mcp_roll", 7, -0.17, 0.17, False),
    JointSpec("middle_mcp_pitch", 2, 0.00, 1.40, False),
    JointSpec("middle_pip", 17, 0.00, 1.57, False),
    JointSpec("ring_mcp_roll", 8, -0.17, 0.17, False),
    JointSpec("ring_mcp_pitch", 3, 0.00, 1.40, False),
    JointSpec("ring_pip", 18, 0.00, 1.57, False),
    JointSpec("pinky_mcp_roll", 9, -0.17, 0.17, False),
    JointSpec("pinky_mcp_pitch", 4, 0.00, 1.40, False),
    JointSpec("pinky_pip", 19, 0.00, 1.57, False),
    JointSpec("thumb_cmc_yaw", 10, 0.00, 1.40, False),    # → thumb rotation
    JointSpec("thumb_cmc_roll", 5, 0.00, 1.22, False),    # → thumb side-bend
    JointSpec("thumb_cmc_pitch", 0, 0.00, 0.79, False),   # → thumb root-flex
    JointSpec("thumb_mcp", 15, 0.00, 1.05, False),        # → thumb bend (ip mimics)
)

N_FINGER_JOINTS = len(DEFAULT_JOINTS)  # 16

_JOINT_KEYS = frozenset({"flip", "slot", "lo", "hi"})
_TOP_KEYS = frozenset({"version", "note", "joints"})

_active_joints: tuple[JointSpec, ...] = DEFAULT_JOINTS


def load_calibration_file(path: str) -> dict:
    """Read a calibration overlay JSON file (schema in the module docstring)."""
    with open(path, "r", encoding="utf-8") as f:
        overlay = json.load(f)
    if not isinstance(overlay, dict):
        raise ValueError(f"calibration overlay must be a JSON object, got {type(overlay).__name__}")
    return overlay


def build_joint_table(overlay: Mapping | None = None) -> list[JointSpec]:
    """Return the 16-joint table with a calibration overlay applied.

    Validates strictly (unknown joint/key, reserved or out-of-range slot,
    duplicate slots, non-bool flip → ``ValueError``) so a typo in a hand-written
    overlay cannot silently mis-route a joint.
    """
    table = {js.name: js for js in DEFAULT_JOINTS}
    if overlay:
        for key in overlay:
            if key not in _TOP_KEYS:
                raise ValueError(f"calibration overlay: unknown top-level key {key!r}")
        joints = overlay.get("joints", {})
        if not isinstance(joints, Mapping):
            raise ValueError("calibration overlay: 'joints' must be an object")
        for name, spec in joints.items():
            if name not in table:
                raise ValueError(f"calibration overlay: unknown joint {name!r}")
            if not isinstance(spec, Mapping):
                raise ValueError(f"calibration overlay: joint {name!r} entry must be an object")
            for k in spec:
                if k not in _JOINT_KEYS:
                    raise ValueError(f"calibration overlay: joint {name!r} has unknown key {k!r}")
            js = table[name]
            if "flip" in spec:
                if not isinstance(spec["flip"], bool):
                    raise ValueError(f"calibration overlay: joint {name!r} flip must be a bool")
                js = js._replace(flip=spec["flip"])
            if "slot" in spec:
                slot = spec["slot"]
                if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot <= 19:
                    raise ValueError(f"calibration overlay: joint {name!r} slot must be an int in 0..19")
                if slot in _RESERVED_SLOTS:
                    raise ValueError(f"calibration overlay: joint {name!r} slot {slot} is reserved")
                js = js._replace(slot=slot)
            for key in ("lo", "hi"):
                if key in spec:
                    v = spec[key]
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise ValueError(f"calibration overlay: joint {name!r} {key} must be a number")
                    js = js._replace(**{key: float(v)})
            if js.hi <= js.lo:
                raise ValueError(
                    f"calibration overlay: joint {name!r} needs hi > lo, got [{js.lo}, {js.hi}]")
            table[name] = js
    result = [table[js.name] for js in DEFAULT_JOINTS]  # preserve semantic order
    slots = [js.slot for js in result]
    if len(set(slots)) != len(slots):
        dupes = sorted({s for s in slots if slots.count(s) > 1})
        raise ValueError(f"calibration overlay: duplicate SDK slot(s) {dupes}")
    return result


def apply_calibration(overlay: Mapping | str | None) -> tuple[JointSpec, ...]:
    """Set the module-active joint table from an overlay dict or JSON path.

    ``None`` resets to :data:`DEFAULT_JOINTS`.  Returns the active table.
    """
    global _active_joints
    if isinstance(overlay, str):
        overlay = load_calibration_file(overlay)
    _active_joints = tuple(build_joint_table(overlay))
    return _active_joints


def reset_calibration() -> None:
    """Restore :data:`DEFAULT_JOINTS` as the active table."""
    global _active_joints
    _active_joints = DEFAULT_JOINTS


def active_joints() -> tuple[JointSpec, ...]:
    """The joint table currently used by the conversion functions."""
    return _active_joints


# --------------------------------------------------------------------------- #
# Reference poses (semantic 16-joint order).
# --------------------------------------------------------------------------- #

# Vendored from screwdriver_rl/tasks/linker_l20/screwdriver_rotation_env_cfg.py
# ``pregrasp_positions`` (keep in sync — tests/test_linker_sdk_map.py checks).
# At runtime prefer the deploy bundle's ``home_targets`` (per-run truth, may
# differ under geometry-bucket randomization); this is the bundle-less fallback
# used by hand_check.
PREGRASP_16: list[float] = [
    0.070000, 0.140535, 0.970000,    # index  (mcp_roll, mcp_pitch, pip)
    -0.070000, 0.184823, 0.930000,   # middle
    -0.070000, 0.449762, 0.930000,   # ring
    0.045669, 0.654302, 0.930000,    # pinky
    0.673745, 1.120000, 0.100000, 0.876434,  # thumb (cmc_yaw, cmc_roll, cmc_pitch, mcp)
]


def open_pose_16() -> list[float]:
    """A relaxed open-hand pose: flexion joints extended, abduction neutral."""
    return [js.lo if js.lo >= 0.0 else 0.5 * (js.lo + js.hi) for js in DEFAULT_JOINTS]


# --------------------------------------------------------------------------- #
# Conversions.
# --------------------------------------------------------------------------- #

def joints16_to_sdk_arc(t16: Sequence[float]) -> list[float]:
    """Place our 16 policy targets (radians) into the SDK's 20-slot radian vector.

    Reserved slots (11-14) are left at 0.  Each joint is mapped by normalized
    fraction of its training range onto the SDK slot's calibrated range.
    """
    if len(t16) != N_FINGER_JOINTS:
        raise ValueError(f"expected {N_FINGER_JOINTS} joint targets, got {len(t16)}")
    arc = [0.0] * 20
    for val, js in zip(t16, _active_joints):
        frac = _clamp((float(val) - js.lo) / (js.hi - js.lo), 0.0, 1.0) if js.hi > js.lo else 0.0
        if js.flip:
            frac = 1.0 - frac
        arc[js.slot] = L20_L_MIN[js.slot] + frac * (L20_L_MAX[js.slot] - L20_L_MIN[js.slot])
    return arc


def joints16_to_sdk_range(t16: Sequence[float]) -> list[int]:
    """Full forward map: 16 radian targets → length-20 ``uint8`` (0..255) command."""
    arc = joints16_to_sdk_arc(t16)
    rng = arc_to_range_left(arc)
    return [int(round(_clamp(v, 0, 255))) for v in rng]


def sdk_range_to_joints16(range20: Sequence[float]) -> list[float]:
    """Inverse map (for reading hand state): length-20 0..255 → our 16 radians.

    Only the slots our policy drives are recovered; the mechanical mimic joints
    are ignored.  Used by the deploy node to build ``finger_q`` from joint state.
    """
    if len(range20) != 20:
        raise ValueError(f"expected 20 SDK slots, got {len(range20)}")
    arc = range_to_arc_left(range20)
    out: list[float] = []
    for js in _active_joints:
        span = L20_L_MAX[js.slot] - L20_L_MIN[js.slot]
        frac = (arc[js.slot] - L20_L_MIN[js.slot]) / span if span != 0 else 0.0
        if js.flip:
            frac = 1.0 - frac
        out.append(js.lo + _clamp(frac, 0.0, 1.0) * (js.hi - js.lo))
    return out
