"""Hardware-side helpers shared by ``deploy_linker`` and ``hand_check``.

Stdlib-only: importing this module must never require torch, ROS, or the
LinkerHand SDK, so the deploy tooling stays testable on any machine.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Sequence


def smoothstep(u: float) -> float:
    """Cubic ease 3u² − 2u³ on [0, 1] (zero velocity at both ends)."""
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def ramp_frames(
    start16: Sequence[float],
    end16: Sequence[float],
    duration_s: float,
    rate_hz: float = 20.0,
    ease=smoothstep,
) -> list[list[float]]:
    """Interpolated joint-space frames from ``start16`` to ``end16``.

    The final frame is ``end16`` verbatim (no float residue), so a follow-up
    command computed from ``end16`` is continuous with the ramp.  Frames are
    per-joint monotone for a monotone ``ease``.  ``duration_s <= 0`` degenerates
    to a single instant frame ``[end16]``.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    if len(start16) != len(end16):
        raise ValueError(f"start/end length mismatch: {len(start16)} vs {len(end16)}")
    end = [float(v) for v in end16]
    if duration_s <= 0:
        return [end]
    start = [float(v) for v in start16]
    n = max(1, round(duration_s * rate_hz))
    frames: list[list[float]] = []
    for k in range(1, n):
        w = ease(k / n)
        frames.append([s + w * (e - s) for s, e in zip(start, end)])
    frames.append(end)
    return frames


def validate_state20(x) -> list[float] | None:
    """Return the state as 20 floats, or ``None`` if it is unusable.

    Rejects everything the LinkerHand CAN layer is known to produce on failure:
    wrong length, the ``[-1]*20`` parse-failure vector, ``""`` cache
    placeholders (any non-numeric entry, including bools), non-finite values,
    and anything outside the 0..255 command range.
    """
    try:
        seq = list(x)
    except TypeError:
        return None
    if len(seq) != 20:
        return None
    out: list[float] = []
    for v in seq:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        f = float(v)
        if not math.isfinite(f) or f < 0.0 or f > 255.0:
            return None
        out.append(f)
    return out


def bootstrap_sdk(sdk_root: str | None = None) -> str:
    """Make the LinkerHand SDK importable; returns the scripts dir added to path.

    Resolution order: explicit ``sdk_root`` arg > ``$LINKERHAND_SDK_ROOT`` >
    ``/home/user/linkerhand-ros-sdk``.  The SDK's Python packages
    live under ``<root>/linker_hand_sdk_ros/scripts`` (top-level ``LinkerHand``
    package, which itself imports its own top-level ``utils``/``core`` — avoid
    running from a cwd that shadows those names).
    """
    root = sdk_root or os.environ.get("LINKERHAND_SDK_ROOT") or "/home/user/linkerhand-ros-sdk"
    scripts = os.path.join(root, "linker_hand_sdk_ros", "scripts")
    if not os.path.isdir(os.path.join(scripts, "LinkerHand")):
        raise FileNotFoundError(
            f"LinkerHand SDK not found under {root!r} (expected {scripts}/LinkerHand). "
            "Pass --sdk-root or set $LINKERHAND_SDK_ROOT to the linkerhand-ros-sdk checkout."
        )
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return scripts


def parse_five(spec: str | Sequence[int], name: str = "value") -> list[int]:
    """Parse a per-finger 5-vector: ``"120"`` → ``[120]*5``, or ``"a,b,c,d,e"``.

    Finger order for L20/G20 ``set_speed``/``set_torque`` is
    (thumb, index, middle, ring, little); a single number sidesteps ordering.
    """
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        try:
            vals = [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"{name}: expected an int or 5 comma-separated ints, got {spec!r}") from exc
    else:
        vals = [int(v) for v in spec]
    if len(vals) == 1:
        vals = vals * 5
    if len(vals) != 5:
        raise ValueError(f"{name}: expected 1 or 5 values, got {len(vals)} from {spec!r}")
    for v in vals:
        if not 0 <= v <= 255:
            raise ValueError(f"{name}: values must be in 0..255, got {v}")
    return vals
