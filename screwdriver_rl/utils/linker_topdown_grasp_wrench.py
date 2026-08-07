"""Whether a top-down grasp can actually turn the handle, not merely touch it.

The static mesh validator checks contact, clearance, self-collision and joint
margin.  A posture can pass all four and still be useless: on 2026-08-06 a fully
validated posture put index/middle/ring/pinky at 99.8-102.6% of handle height --
perched on the top rim -- with the thumb alone at 54.6% on the opposite side.
Every gate was green.  The grasp could not rotate the handle, because the four
rim contacts press axially rather than tangentially, and the thumb pushing
inward at mid-height with no partner at its own height tips the object instead
of gripping it.

So the missing predicate is about the *distribution* of contacts:

``height_band``
    Every contact must sit inside a usable band of the handle body.  A contact
    at the rim has almost no wrap and slides off under tangential load.

``radial_closure``
    The contact radial directions must not all lie in one hemisphere, or the
    normal forces sum to a net lateral push instead of a squeeze.  This is the
    load-bearing criterion.  The handle hangs on a universal joint, so there is
    no bearing to react a net lateral force and any imbalance becomes tilt.

``opposition_height``
    Reported as a diagnostic only.  It was briefly enforced and was wrong: see
    ``DEFAULT_MAX_OPPOSITION_HEIGHT_MISMATCH_M``.

These are necessary conditions, deliberately cheap and kinematic.  They do not
prove force closure and do not replace the PhysX functional contact gate; they
reject the specific failure that static validation cannot see.

The thresholds are calibrated against the reference configuration -- the last
posture with measured fall-rate and net-turn evidence -- not derived from first
principles.  That ordering matters: the first revision of this module enforced a
height band and an opposition-height limit that the reference violates, which
would have rejected a working grasp in favour of one that tipped the handle
within 30 simulation steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


#: Usable fraction of the handle body, measured from its base.  Deliberately
#: permissive at the top: the reference configuration contacts at 87-100% and
#: still drove the handle, so a rim contact is not by itself disqualifying.
DEFAULT_HEIGHT_BAND = (0.20, 1.05)
#: Reported, NOT enforced.  An earlier revision required the thumb to act within
#: 20 mm of its opposing fingers, reasoning that a height difference turns
#: opposed radial forces into a tipping couple.  The reference configuration --
#: the only one with measured fall-rate and net-turn evidence behind it --
#: violates that by 55-61 mm and works, so the criterion is kept as a diagnostic
#: rather than a gate.  Do not promote it to a gate without evidence that beats
#: the reference.
DEFAULT_MAX_OPPOSITION_HEIGHT_MISMATCH_M = float("inf")
#: Maximum ||mean radial unit vector||.  This is the criterion that survived
#: contact with the evidence.  The handle hangs on a universal joint, not a
#: rigid bearing, so it cannot react a net lateral force: imbalance becomes
#: tilt.  Measured on the reference configuration this is 0.218 (reset) and
#: 0.364 (controller target); a candidate at 0.532 tipped the handle 1.458 rad
#: and terminated within 30 steps.  0.40 sits just above the reference target
#: and well below the value that demonstrably fails.
DEFAULT_MAX_RADIAL_CLOSURE = 0.40


def contact_cylindrical_coordinates(
    contact_points_w: Mapping[str, Sequence[float]],
    axis_xy: Sequence[float],
    body_base_z: float,
    body_top_z: float,
) -> dict[str, dict[str, float]]:
    """Per-finger contact height fraction, radius and azimuth about the axis."""

    axis = np.asarray(axis_xy, dtype=np.float64)
    length = float(body_top_z) - float(body_base_z)
    if length <= 0.0:
        raise ValueError("handle body must have positive length")
    out: dict[str, dict[str, float]] = {}
    for finger, point in contact_points_w.items():
        p = np.asarray(point, dtype=np.float64)
        delta = p[:2] - axis
        out[finger] = {
            "z_m": float(p[2]),
            "height_fraction": float((p[2] - float(body_base_z)) / length),
            "radius_m": float(np.linalg.norm(delta)),
            "azimuth_rad": float(np.arctan2(delta[1], delta[0])),
        }
    return out


def grasp_wrench_gate(
    contact_points_w: Mapping[str, Sequence[float]],
    axis_xy: Sequence[float],
    body_base_z: float,
    body_top_z: float,
    *,
    thumb: str = "thumb",
    active_fingers: Sequence[str] | None = None,
    height_band: tuple[float, float] = DEFAULT_HEIGHT_BAND,
    max_opposition_height_mismatch_m: float = DEFAULT_MAX_OPPOSITION_HEIGHT_MISMATCH_M,
    max_radial_closure: float = DEFAULT_MAX_RADIAL_CLOSURE,
) -> dict[str, Any]:
    """Evaluate whether the contact distribution can apply torque about the axis.

    ``active_fingers`` restricts the check to fingers that actually carry load;
    a finger measured at zero force should not be allowed to satisfy the
    distribution requirements on paper.
    """

    coords = contact_cylindrical_coordinates(
        contact_points_w, axis_xy, body_base_z, body_top_z
    )
    names = list(active_fingers) if active_fingers is not None else list(coords)
    missing = [name for name in names if name not in coords]
    if missing:
        raise KeyError(f"no contact point for {missing}")

    band_failures = {
        name: coords[name]["height_fraction"]
        for name in names
        if not height_band[0] <= coords[name]["height_fraction"] <= height_band[1]
    }

    opposing = [name for name in names if name != thumb]
    if thumb in names and opposing:
        opposing_mean_z = float(np.mean([coords[name]["z_m"] for name in opposing]))
        mismatch = abs(coords[thumb]["z_m"] - opposing_mean_z)
    else:
        opposing_mean_z = float("nan")
        mismatch = 0.0

    units = np.asarray(
        [
            (np.cos(coords[name]["azimuth_rad"]), np.sin(coords[name]["azimuth_rad"]))
            for name in names
        ],
        dtype=np.float64,
    )
    closure = float(np.linalg.norm(units.mean(axis=0))) if len(units) else 1.0

    band_pass = not band_failures
    opposition_pass = mismatch <= max_opposition_height_mismatch_m
    closure_pass = closure <= max_radial_closure
    return {
        "pass": bool(band_pass and opposition_pass and closure_pass),
        "height_band_pass": bool(band_pass),
        "height_band_failures": band_failures,
        "opposition_height_pass": bool(opposition_pass),
        "opposition_height_mismatch_m": mismatch,
        "opposing_mean_z_m": opposing_mean_z,
        "radial_closure_pass": bool(closure_pass),
        "radial_closure": closure,
        "evaluated_fingers": names,
        "contacts": coords,
        "thresholds": {
            "height_band": list(height_band),
            "max_opposition_height_mismatch_m": max_opposition_height_mismatch_m,
            "max_radial_closure": max_radial_closure,
        },
    }


#: Minimum dot(pad axis, contact direction) for a contact to be acceptable.
#:
#: This encodes an explicit operator requirement, not a derived number: a
#: contact somewhat around the side of the fingertip is acceptable, a contact on
#: the BACK of the finger never is.  Side-of-finger contact reads near zero, so
#: the faithful encoding of "not the back" is exactly 0.0.  Resist widening this
#: to a comfortable-looking positive margin -- that would reject the side
#: contacts the operator explicitly accepted.  Numerical noise on the cosine is
#: order 1e-3, far below any contact that is genuinely ambiguous.
DEFAULT_MIN_PAD_FACING = 0.0


def pad_axis(
    distal_origin_w: Sequence[float],
    tip_marker_w: Sequence[float],
    tip_marker_flexed_w: Sequence[float],
) -> np.ndarray:
    """Unit vector from the distal link toward its pad, perpendicular to the bone.

    Derived kinematically: flexing a finger swings its tip toward the palm, so
    the tip's motion under a small positive flexion points to the pad side.
    Verified by a fist test -- with the fingers strongly flexed the pads face the
    palm, and this axis agrees with the direction to the palm centroid at
    dot 0.91-1.00 for all five digits.  The distal collision mesh is NOT a
    reliable cue: it is thicker on the dorsal side, so inferring the pad from
    mesh asymmetry gives the opposite answer.
    """
    origin = np.asarray(distal_origin_w, dtype=np.float64)
    span = np.asarray(tip_marker_w, dtype=np.float64) - origin
    span /= max(float(np.linalg.norm(span)), 1.0e-9)
    motion = np.asarray(tip_marker_flexed_w, dtype=np.float64) - np.asarray(
        tip_marker_w, dtype=np.float64
    )
    pad = motion - float(np.dot(motion, span)) * span
    return pad / max(float(np.linalg.norm(pad)), 1.0e-9)


def pad_facing(
    distal_origin_w: Sequence[float],
    tip_marker_w: Sequence[float],
    pad_axis_w: Sequence[float],
    contact_point_w: Sequence[float],
) -> float:
    """Cosine between the pad axis and the contact direction; >0 means pad-borne.

    A back-of-finger contact cannot drive the handle: there is no pad friction
    patch and the flexion actuators pull the contact *away* from the object.
    Nothing in the static validator measured this, and it turned out to be the
    dominant defect -- the reference configuration, which held the best measured
    fall rate and net turns on this task, contacts the handle with the backs of
    all five digits (pad facing -1.00 to +0.06).
    """
    origin = np.asarray(distal_origin_w, dtype=np.float64)
    span = np.asarray(tip_marker_w, dtype=np.float64) - origin
    span /= max(float(np.linalg.norm(span)), 1.0e-9)
    rel = np.asarray(contact_point_w, dtype=np.float64) - origin
    rel = rel - float(np.dot(rel, span)) * span
    norm = float(np.linalg.norm(rel))
    if norm < 1.0e-9:
        return 0.0
    return float(np.dot(rel / norm, np.asarray(pad_axis_w, dtype=np.float64)))


#: Pad direction in each DISTAL LINK's own frame, as a unit vector.
#:
#: Constant by construction: re-derived at four flexion levels the local axis is
#: invariant to 0.0 deg for the four fingers and 0.3 deg for the thumb.  That is
#: what makes an in-simulation measurement possible -- the pad direction in world
#: frame is just this vector rotated by the distal body's orientation, with no
#: geometric reconstruction of contact points, object pose or wrist placement.
#:
#: That matters because reconstruction is exactly what went wrong: four separate
#: offline attempts to decide pad-versus-back contact each failed differently
#: (evaluating the reset instead of the settled state, taking the nearest point
#: on the lateral wall instead of on the object, ignoring the domain-randomised
#: wrist offset, ignoring the object's own settled tilt), and the last one still
#: reported 37.6 N on a fingertip measured 7.4 mm clear of the object.
#:
#: Sign convention verified by a fist test: with the fingers strongly flexed the
#: pads face the palm, and this axis agrees with the direction to the palm
#: centroid at dot 0.907-1.000 for all five digits.  The distal collision mesh is
#: NOT a usable cue -- it is thicker on the dorsal side.
PAD_AXIS_LOCAL: dict[str, tuple[float, float, float]] = {
    "index": (0.7974, 0.0, 0.6035),
    "middle": (0.7974, 0.0, 0.6035),
    "ring": (0.7974, 0.0, 0.6035),
    "pinky": (0.7974, 0.0, 0.6035),
    "thumb": (0.9049, 0.0005, 0.4257),
}


def pad_facing_from_force(
    contact_force_w: Sequence[float],
    pad_axis_w: Sequence[float],
) -> float:
    """Cosine of the contact force against the inward pad normal; >0 is pad-borne.

    ``contact_force_w`` is the force the object exerts ON the fingertip, so a
    pad contact pushes the finger back along ``-pad_axis``.  Returns 0.0 for a
    force below numerical noise rather than a direction from nothing.

    This measures where the load actually acts, which is the question that
    matters: a fingertip can be geometrically nearest the object on its dorsal
    side while carrying no load there at all.
    """
    force = np.asarray(contact_force_w, dtype=np.float64)
    magnitude = float(np.linalg.norm(force))
    if magnitude < 1.0e-6:
        return 0.0
    axis = np.asarray(pad_axis_w, dtype=np.float64)
    return float(np.dot(force / magnitude, -axis / max(np.linalg.norm(axis), 1e-12)))
