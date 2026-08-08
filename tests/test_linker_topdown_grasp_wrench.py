"""The grasp-wrench gate: can the posture turn the handle, not just touch it.

Regression cover for the 2026-08-06 failure in which a posture passed every
static gate (contact, clearance, self-collision, joint margin) while
index/middle/ring/pinky sat at 99.8-102.6% of handle height and the thumb alone
at 54.6%.  Nothing in the validator priced that, so a grasp that tips the handle
rather than turning it was reported fully valid.
"""

from __future__ import annotations

import math

import pytest

from screwdriver_rl.utils.linker_topdown_grasp_wrench import (
    contact_cylindrical_coordinates,
    grasp_wrench_gate,
)


AXIS = (0.0, 0.0)
BASE_Z = 1.305
TOP_Z = 1.405
RADIUS = 0.032
DRIVE = ("middle", "ring", "pinky", "thumb")


def _point(azimuth_deg: float, height_fraction: float) -> tuple[float, float, float]:
    angle = math.radians(azimuth_deg)
    return (
        RADIUS * math.cos(angle),
        RADIUS * math.sin(angle),
        BASE_Z + height_fraction * (TOP_Z - BASE_Z),
    )


def _gate(points, **kwargs):
    return grasp_wrench_gate(points, AXIS, BASE_Z, TOP_Z, active_fingers=DRIVE, **kwargs)


def test_cylindrical_coordinates_round_trip() -> None:
    coords = contact_cylindrical_coordinates(
        {"middle": _point(45.0, 0.60)}, AXIS, BASE_Z, TOP_Z
    )["middle"]
    assert coords["height_fraction"] == pytest.approx(0.60)
    assert coords["radius_m"] == pytest.approx(RADIUS)
    assert coords["azimuth_rad"] == pytest.approx(math.radians(45.0))


def test_balanced_opposed_grasp_passes() -> None:
    result = _gate({
        "middle": _point(-87.0, 0.70),
        "ring": _point(-33.0, 0.60),
        "pinky": _point(13.0, 0.50),
        "thumb": _point(147.0, 0.60),
    })
    assert result["pass"], result
    assert result["opposition_height_mismatch_m"] == pytest.approx(0.0, abs=1.0e-9)


def test_reference_configuration_is_accepted() -> None:
    # The measured OLD TARGET (physics candidate 117): fingers high, thumb low,
    # 55 mm of opposition offset.  It is the only configuration with measured
    # fall-rate and net-turn evidence, so the gate must not reject it.  An
    # earlier revision did, via a height band and an opposition-height limit.
    #
    # All five fingers are evaluated because all five contact the body in that
    # posture -- the index sits at 87% on the lateral surface, not on the cap.
    # radial_closure depends on which set is summed, so the set must match what
    # the validator passes in, which is "whichever fingers touch the body".
    points = {
        "index": _point(-127.5, 0.870),
        "middle": _point(-91.9, 1.000),
        "ring": _point(-58.5, 0.948),
        "pinky": _point(2.3, 1.000),
        "thumb": _point(91.5, 0.400),
    }
    result = grasp_wrench_gate(points, AXIS, BASE_Z, TOP_Z)
    assert result["pass"], result
    assert result["radial_closure"] == pytest.approx(0.364, abs=0.005)
    # The offset is measured and reported, just not enforced.
    assert result["opposition_height_mismatch_m"] > 0.050
    assert result["opposition_height_pass"]


def test_laterally_unbalanced_five_finger_grasp_is_rejected() -> None:
    # Same five-finger accounting as the reference, for the posture that tipped
    # the handle 1.458 rad within 30 steps.
    points = {
        "index": _point(-132.9, 0.782),
        "middle": _point(-95.4, 0.674),
        "ring": _point(-64.9, 0.599),
        "pinky": _point(-8.4, 0.618),
        "thumb": _point(171.7, 0.618),
    }
    result = grasp_wrench_gate(points, AXIS, BASE_Z, TOP_Z)
    assert result["radial_closure"] == pytest.approx(0.532, abs=0.01)
    assert not result["pass"]


def test_laterally_unbalanced_grasp_is_rejected() -> None:
    # radial_closure 0.53 tipped the handle 1.458 rad within 30 steps: the
    # universal joint cannot react a net lateral push.
    result = _gate({
        "middle": _point(-95.4, 0.674),
        "ring": _point(-64.9, 0.599),
        "pinky": _point(-8.4, 0.618),
        "thumb": _point(171.7, 0.618),
    })
    assert result["radial_closure"] > 0.40
    assert not result["radial_closure_pass"]
    assert not result["pass"]


def test_same_hemisphere_contacts_are_rejected() -> None:
    # All four crowded onto one side: normal forces sum to a lateral push.
    result = _gate({
        "middle": _point(0.0, 0.60),
        "ring": _point(12.0, 0.60),
        "pinky": _point(24.0, 0.60),
        "thumb": _point(36.0, 0.60),
    })
    assert result["height_band_pass"]
    assert result["radial_closure"] == pytest.approx(1.0, abs=0.05)
    assert not result["radial_closure_pass"]
    assert not result["pass"]


def test_index_is_excluded_by_role_not_by_accident() -> None:
    # The index presses the cap axially; including it would fail the band check
    # for a grasp that is otherwise correct.
    points = {
        "index": _point(-174.0, 1.20),
        "middle": _point(-87.0, 0.70),
        "ring": _point(-33.0, 0.60),
        "pinky": _point(13.0, 0.50),
        "thumb": _point(147.0, 0.60),
    }
    assert _gate(points)["pass"]
    assert not grasp_wrench_gate(points, AXIS, BASE_Z, TOP_Z)["pass"]


def test_unknown_active_finger_is_an_error() -> None:
    with pytest.raises(KeyError):
        grasp_wrench_gate(
            {"middle": _point(0.0, 0.6)}, AXIS, BASE_Z, TOP_Z,
            active_fingers=["middle", "thumb"],
        )


def test_pad_axis_points_toward_the_palm_when_the_hand_makes_a_fist() -> None:
    """Anchor the pad axis sign against an unambiguous physical fact.

    The sign matters and is easy to get backwards: the distal collision mesh is
    thicker on the DORSAL side, so inferring the pad from mesh asymmetry gives
    the opposite answer to the kinematic derivation.  With the fingers strongly
    flexed the pads face the palm, which settles it.
    """
    import numpy as np

    from screwdriver_rl.utils.linker_topdown_geometry import (
        FINGERTIP_LINKS,
        TIP_MARKER_LINKS,
        UrdfGeometry,
    )
    from screwdriver_rl.utils.linker_topdown_grasp_wrench import pad_axis

    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    model = UrdfGeometry(repo / "assets/linker_hand_l20/linkerhand_l20_left.urdf")
    limits = model.joint_limits(0.05)
    palm = np.asarray(
        model.collision_mesh_local("hand_base_link").vertices
    ).mean(axis=0)
    flex_joint = {
        "index": "index_pip", "middle": "middle_pip", "ring": "ring_pip",
        "pinky": "pinky_pip", "thumb": "thumb_mcp",
    }
    root, quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    for finger, distal, tip in zip(flex_joint, FINGERTIP_LINKS, TIP_MARKER_LINKS):
        q = {name: 0.0 for name in model.independent_joint_names}
        for joint in (f"{finger}_mcp_pitch", f"{finger}_pip"):
            if joint in q:
                q[joint] = limits[joint][1] * 0.85
        if finger == "thumb":
            q["thumb_mcp"] = limits["thumb_mcp"][1] * 0.85
            q["thumb_cmc_roll"] = limits["thumb_cmc_roll"][1] * 0.6
        fk = model.forward_kinematics(q, root, quat)
        flexed = dict(q)
        joint = flex_joint[finger]
        flexed[joint] = min(flexed[joint] + 0.02, limits[joint][1])
        fk_flexed = model.forward_kinematics(flexed, root, quat)

        axis = pad_axis(fk[distal][:3, 3], fk[tip][:3, 3], fk_flexed[tip][:3, 3])
        span = fk[tip][:3, 3] - fk[distal][:3, 3]
        span = span / np.linalg.norm(span)
        to_palm = palm - fk[distal][:3, 3]
        to_palm = to_palm - float(np.dot(to_palm, span)) * span
        to_palm = to_palm / np.linalg.norm(to_palm)
        assert float(np.dot(axis, to_palm)) > 0.85, finger


def test_side_of_finger_contact_is_accepted_and_back_is_not() -> None:
    """The operator's acceptance rule: around the side is fine, the back never is.

    Encoded as a threshold of exactly 0.0.  Widening it to a positive margin
    would reject side contacts that were explicitly accepted; the measured
    values that must be rejected are strongly negative (-0.22 to -1.00 on the
    reference configuration, which contacts with the backs of all five digits).
    """
    import numpy as np

    from screwdriver_rl.utils.linker_topdown_grasp_wrench import (
        DEFAULT_MIN_PAD_FACING,
        pad_facing,
    )

    origin = np.zeros(3)
    tip = np.array([0.0, 0.0, 0.03])          # bone axis along +Z
    pad = np.array([0.0, 1.0, 0.0])           # pad faces +Y

    def facing(direction):
        return pad_facing(origin, tip, pad, np.asarray(direction) * 0.01)

    pad_on = facing([0.0, 1.0, 0.0])
    side = facing([1.0, 0.02, 0.0])           # essentially the side of the tip
    back = facing([0.0, -1.0, 0.0])

    assert pad_on == pytest.approx(1.0, abs=1e-6)
    assert back == pytest.approx(-1.0, abs=1e-6)
    assert side == pytest.approx(0.02, abs=0.01)

    assert pad_on >= DEFAULT_MIN_PAD_FACING
    assert side >= DEFAULT_MIN_PAD_FACING, "side contact must remain acceptable"
    assert back < DEFAULT_MIN_PAD_FACING, "back-of-finger contact must be rejected"
    # The reference configuration's measured values must all be rejected.
    for measured in (-0.484, -1.000, -0.218, -0.592):
        assert measured < DEFAULT_MIN_PAD_FACING
