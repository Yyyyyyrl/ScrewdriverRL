"""Map the 16-D ScrewdriverRL policy joints onto LinkerHand L20/G20 SDK commands.

The trained policy outputs 16 *independent* finger-joint targets (radians, in the
training-URDF convention).  The LinkerHand SDK accepts a length-20 vector of
``0..255`` "range" values per hand, published as ``sensor_msgs/JointState.position``
to ``/cb_<side>_hand_control_cmd`` (or passed to ``LinkerHandApi.finger_move``).

Two conversions are needed:

1. **radian → 0..255** — the SDK already provides this as
   ``range_to_arc/scripts/utils/linker_range_arc.py:arc_to_range_left`` (calibrated
   per slot with ``l20_l_min/max`` and direction flips).  We *reuse* that module
   when the SDK is importable, and otherwise fall back to a vendored copy of the
   exact same constants + math (so this module — and its tests — run standalone).

2. **our 16 joints → the SDK's 20 slots (radian convention)** — authored here.
   The SDK 20-slot layout (from ``examples/L20/l20_isaacgym/l20_example.py``):

       0-4   root-flex   {thumb, index, middle, ring, pinky}
       5-9   abduction   {thumb, index, middle, ring, pinky}
       10    thumb rotation
       11-14 reserved (left untouched / 0)
       15-19 finger-bend {thumb, index, middle, ring, pinky}

   The dip/tip joints couple mechanically on the real hand (one bend slot per
   finger), matching our ``COUPLED_JOINTS`` — so the 5 mimic joints are not sent.

Our training-URDF ranges differ from the SDK's calibrated ranges (e.g. abduction
±0.17 vs ±0.26), so each joint is mapped by *normalized fraction of range*: the
same grip fraction in our convention maps to the same fraction of the SDK range.

⚠️ THUMB MAPPING IS PROVISIONAL.  The 4 thumb DOFs → SDK thumb slots {0,5,10,15}
and their sign/flip need verification against the SDK's own URDF on real hardware.
Everything thumb-related is isolated in ``_OUR_JOINTS`` below (slot + flip per
joint) so it can be corrected in one place without touching the conversion logic.
The four finger chains (index/middle/ring/pinky) are well-determined.
"""

from __future__ import annotations

from typing import Sequence

# --------------------------------------------------------------------------- #
# SDK radian↔0..255 conversion: reuse the real SDK module if importable, else
# fall back to a vendored copy of its exact constants + math.
# --------------------------------------------------------------------------- #

_DEFAULT_SDK_UTILS = "/home/user/linkerhand-ros-sdk/range_to_arc/scripts/utils"


def _load_sdk_range_arc():
    """Return the SDK's ``linker_range_arc`` module, or ``None`` if unavailable."""
    try:  # already on the path (e.g. inside the ROS workspace)
        import linker_range_arc as _m  # type: ignore

        return _m
    except Exception:
        pass
    import importlib.util
    import os

    for d in (_DEFAULT_SDK_UTILS, os.environ.get("LINKERHAND_SDK_UTILS", "")):
        path = os.path.join(d, "linker_range_arc.py") if d else ""
        if path and os.path.exists(path):
            spec = importlib.util.spec_from_file_location("linker_range_arc", path)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                return mod
            except Exception:
                return None
    return None


_SDK = _load_sdk_range_arc()

# Vendored L20 *left-hand* calibration — copied verbatim from the SDK's
# ``linker_range_arc.py`` (l20_l_min / l20_l_max / l20_l_derict).  Per slot these
# are the arc (radian) endpoints and the direction flag used by the 0..255 map.
L20_L_MIN = [-1.57, 0, 0, 0, 0, 0, -0.26, -0.26, -0.26, -0.26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
L20_L_MAX = [0, 1.57, 1.57, 1.57, 1.57, 1.57, 0.26, 0.26, 0.26, 0.26, 0, 0, 0, 0, 0, 1.57, 1.57, 1.57, 1.57, 1.57]
L20_L_DIRECT = [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1]

# Slots the SDK leaves untouched (no actuator); never feed these to scale_value
# (their min == max would divide by zero).
_RESERVED_SLOTS = (11, 12, 13, 14)


def _scale(v: float, a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return (v - a_min) * (b_max - b_min) / (a_max - a_min) + b_min


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def arc_to_range_left(arc20: Sequence[float]) -> list[float]:
    """Radian (SDK convention) → 0..255 per slot, for the L20/G20 left hand.

    Delegates to the real SDK module when present; the vendored path reproduces
    its exact behaviour (reserved slots 11-14 left at 0).
    """
    if _SDK is not None and hasattr(_SDK, "arc_to_range_left"):
        return list(_SDK.arc_to_range_left(list(arc20)))
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
    if _SDK is not None and hasattr(_SDK, "range_to_arc_left"):
        return list(_SDK.range_to_arc_left(list(range20)))
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
# Each entry: (name, sdk_slot, our_lo, our_hi, flip)
#   our_lo/our_hi : our training-URDF soft limits (radians) for normalisation
#   flip          : reverse the grip fraction before mapping into the SDK range
# Our 16-joint order is fingers (index, middle, ring, pinky, thumb) × joints,
# matching screwdriver_rl/tasks/linker_l20 FINGER_JOINT_NAMES.
# --------------------------------------------------------------------------- #
_OUR_JOINTS = [
    # name,               slot, our_lo, our_hi, flip
    ("index_mcp_roll",     6, -0.17, 0.17, False),   # abduction
    ("index_mcp_pitch",    1,  0.00, 1.40, False),   # root-flex
    ("index_pip",         16,  0.00, 1.57, False),   # bend
    ("middle_mcp_roll",    7, -0.17, 0.17, False),
    ("middle_mcp_pitch",   2,  0.00, 1.40, False),
    ("middle_pip",        17,  0.00, 1.57, False),
    ("ring_mcp_roll",      8, -0.17, 0.17, False),
    ("ring_mcp_pitch",     3,  0.00, 1.40, False),
    ("ring_pip",          18,  0.00, 1.57, False),
    ("pinky_mcp_roll",     9, -0.17, 0.17, False),
    ("pinky_mcp_pitch",    4,  0.00, 1.40, False),
    ("pinky_pip",         19,  0.00, 1.57, False),
    # ── thumb: PROVISIONAL slot/flip — verify against the SDK URDF on hardware ──
    ("thumb_cmc_yaw",     10,  0.00, 1.40, False),   # → thumb rotation  (slot 10)
    ("thumb_cmc_roll",     5,  0.00, 1.22, False),   # → thumb abduction (slot 5)
    ("thumb_cmc_pitch",    0,  0.00, 0.79, False),   # → thumb root-flex (slot 0)
    ("thumb_mcp",         15,  0.00, 1.05, False),   # → thumb bend      (slot 15)
]

N_FINGER_JOINTS = len(_OUR_JOINTS)  # 16


def joints16_to_sdk_arc(t16: Sequence[float]) -> list[float]:
    """Place our 16 policy targets (radians) into the SDK's 20-slot radian vector.

    Reserved slots (11-14) are left at 0.  Each joint is mapped by normalized
    fraction of its training range onto the SDK slot's calibrated range.
    """
    if len(t16) != N_FINGER_JOINTS:
        raise ValueError(f"expected {N_FINGER_JOINTS} joint targets, got {len(t16)}")
    arc = [0.0] * 20
    for val, (_name, slot, lo, hi, flip) in zip(t16, _OUR_JOINTS):
        frac = _clamp((float(val) - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0
        if flip:
            frac = 1.0 - frac
        arc[slot] = L20_L_MIN[slot] + frac * (L20_L_MAX[slot] - L20_L_MIN[slot])
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
    for _name, slot, lo, hi, flip in _OUR_JOINTS:
        span = L20_L_MAX[slot] - L20_L_MIN[slot]
        frac = (arc[slot] - L20_L_MIN[slot]) / span if span != 0 else 0.0
        if flip:
            frac = 1.0 - frac
        out.append(lo + _clamp(frac, 0.0, 1.0) * (hi - lo))
    return out


def using_real_sdk() -> bool:
    """True if the real SDK ``linker_range_arc`` module is being used (vs vendored)."""
    return _SDK is not None
