"""Offline tests for the Phase B MCP-roll detector.

No camera and no hand motion. The synthetic frames mirror the real roll-view
layout: one near-vertical palm tape and four near-horizontal proximal tapes.
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import pytest

from tools.measure_g20_mcp_roll import (
    FINGERS_BOTTOM_TO_TOP,
    abduction_angle,
    measure_roll_frame,
    palm_reference_heading,
)

BLUE_BGR = (255, 0, 0)
DEPTH_SCALE = 0.001
INTRINSICS = {"width": 1280, "height": 720, "fx": 640.0, "fy": 640.0,
              "ppx": 640.0, "ppy": 360.0, "model": "brown_conrady",
              "coeffs": [0.0] * 5}


def _args(**overrides) -> argparse.Namespace:
    values = {
        "roi": (600, 150, 1000, 560),
        "palm_roi": (500, 150, 610, 560),
        "hsv_lower": (75, 40, 20),
        "hsv_upper": (165, 255, 255),
        "component_min_area": 15,
        "palm_min_area": 1000,
        "min_marker_row_gap_px": 20.0,
        "min_elongation2": 4.0,
        "segmentation_depth_min_m": 0.245,
        "segmentation_depth_max_m": 0.280,
        "target_finger": None,
        "target_row_band": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _bar(color: np.ndarray, center: tuple[float, float], length: float,
         thickness: int, heading_deg: float) -> None:
    """Draw a blue bar whose major axis has the given image heading."""
    radians = math.radians(heading_deg)
    dx, dy = math.cos(radians) * length / 2.0, math.sin(radians) * length / 2.0
    cv2.line(color,
             (int(round(center[0] - dx)), int(round(center[1] - dy))),
             (int(round(center[0] + dx)), int(round(center[1] + dy))),
             BLUE_BGR, thickness)


def _scene(finger_headings: dict[str, float], palm_heading: float = 90.0,
           rows: dict[str, int] | None = None,
           depth_raw: int = 260) -> tuple[np.ndarray, np.ndarray]:
    color = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.full((720, 1280), depth_raw, dtype=np.uint16)
    _bar(color, (556.0, 348.0), 320.0, 50, palm_heading)
    default_rows = {"index": 469, "middle": 384, "ring": 299, "pinky": 215}
    rows = rows or default_rows
    for finger, heading in finger_headings.items():
        _bar(color, (676.0, float(rows[finger])), 84.0, 12, heading)
    return color, depth


def _measure(color, depth, args):
    return measure_roll_frame(color, depth, INTRINSICS, DEPTH_SCALE, args)


def test_palm_reference_is_stable_across_the_wrap_boundary():
    # A near-vertical tape reports +90 or -90 depending on numerical noise in
    # the PCA sign convention. Both must map to the same reference.
    assert palm_reference_heading(90.0) == pytest.approx(0.0)
    assert palm_reference_heading(-90.0) == pytest.approx(0.0)
    assert palm_reference_heading(89.5) == pytest.approx(-0.5)
    assert palm_reference_heading(-89.5) == pytest.approx(0.5)
    # The pair straddling the boundary must not differ by ~180 degrees.
    assert abs(palm_reference_heading(89.9) - palm_reference_heading(-89.9)) < 0.3


def test_abduction_is_zero_when_finger_is_perpendicular_to_palm():
    assert abduction_angle(90.0, 0.0) == pytest.approx(0.0)
    assert abduction_angle(-90.0, 0.0) == pytest.approx(0.0)


def test_abduction_sign_is_signed_not_absolute():
    positive = abduction_angle(90.0, -5.0)
    negative = abduction_angle(90.0, 5.0)
    assert positive == pytest.approx(5.0)
    assert negative == pytest.approx(-5.0)
    assert positive == -negative


def test_identity_is_positional_bottom_to_top():
    # Give each finger a distinct heading, then check the label-to-row mapping.
    headings = {"index": -6.0, "middle": -2.0, "ring": 2.0, "pinky": 6.0}
    color, depth = _scene(headings)
    markers, angles, diagnostics = _measure(color, depth, _args())

    assert set(markers) == {"palm", *FINGERS_BOTTOM_TO_TOP}
    rows = diagnostics["marker_rows_bottom_to_top"]
    assert rows == sorted(rows, reverse=True), "index must be the bottom marker"
    assert markers["index"].centroid_uv[1] > markers["pinky"].centroid_uv[1]
    for finger, heading in headings.items():
        assert angles[f"{finger}_abduction_deg_2d"] == pytest.approx(
            -heading, abs=0.6
        )


def test_frame_fails_when_a_finger_marker_is_missing():
    color, depth = _scene({"index": 0.0, "middle": 0.0, "ring": 0.0})
    with pytest.raises(ValueError, match="expected 4 finger markers"):
        _measure(color, depth, _args())


def test_frame_fails_when_two_markers_could_be_confused_by_row():
    # 16 px apart: far enough that the 12 px bars stay separate components,
    # close enough that the row-gap guard must reject the identity.
    rows = {"index": 469, "middle": 384, "ring": 315, "pinky": 299}
    color, depth = _scene(
        {finger: 0.0 for finger in FINGERS_BOTTOM_TO_TOP}, rows=rows
    )
    with pytest.raises(ValueError, match="too close to separate by row"):
        _measure(color, depth, _args())


def test_frame_fails_when_depth_puts_fingers_outside_the_gate():
    # Background at 0.67 m must never be accepted as a finger marker.
    color, depth = _scene(
        {finger: 0.0 for finger in FINGERS_BOTTOM_TO_TOP}, depth_raw=670
    )
    with pytest.raises(ValueError, match="expected 4 finger markers"):
        _measure(color, depth, _args())


def test_palm_is_not_depth_gated():
    # The real palm tape has ~5% valid depth; a fully invalid palm must still
    # measure, because only the finger markers are depth gated.
    color, depth = _scene({finger: 0.0 for finger in FINGERS_BOTTOM_TO_TOP})
    depth[150:560, 500:610] = 0
    markers, angles, _ = _measure(color, depth, _args())
    assert markers["palm"].depth_valid_fraction == pytest.approx(0.0)
    assert angles["palm_reference_heading_deg_2d"] == pytest.approx(0.0, abs=0.6)


def test_palm_tape_outside_finger_roi_is_not_counted_as_a_finger():
    color, depth = _scene({finger: 0.0 for finger in FINGERS_BOTTOM_TO_TOP})
    _, _, diagnostics = _measure(color, depth, _args())
    assert len(diagnostics["marker_rows_bottom_to_top"]) == 4


def test_rotating_the_palm_reference_moves_all_abductions_together():
    headings = {finger: 0.0 for finger in FINGERS_BOTTOM_TO_TOP}
    base = _measure(*_scene(headings, palm_heading=90.0), _args())[1]
    tilted = _measure(*_scene(headings, palm_heading=87.0), _args())[1]
    for finger in FINGERS_BOTTOM_TO_TOP:
        shift = (tilted[f"{finger}_abduction_deg_2d"]
                 - base[f"{finger}_abduction_deg_2d"])
        assert shift == pytest.approx(-3.0, abs=0.6)


# --- sweep mode: only the target finger's tape has to be visible -------------
# The non-target fingers are pitched out of the way, so their proximal tapes
# leave the depth gate. Identity must still never be inferred by elimination.


def _sweep_args(finger: str, band: tuple[float, float], **overrides):
    return _args(target_finger=finger, target_row_band=band, **overrides)


def test_sweep_mode_measures_the_lone_target_marker():
    color, depth = _scene({"index": -4.0})
    markers, angles, diagnostics = _measure(
        color, depth, _sweep_args("index", (440.0, 500.0))
    )
    assert set(markers) == {"palm", "index"}
    assert diagnostics["measured_fingers"] == ["index"]
    assert angles["index_abduction_deg_2d"] == pytest.approx(4.0, abs=0.6)
    assert "middle_abduction_deg_2d" not in angles


def test_sweep_mode_rejects_a_marker_outside_the_target_row_band():
    # A lone marker at the ring row must not be accepted as index just because
    # it is the only one left.
    color, depth = _scene({"ring": 0.0})
    with pytest.raises(ValueError, match="outside its expected band"):
        _measure(color, depth, _sweep_args("index", (440.0, 500.0)))


def test_sweep_mode_rejects_more_than_one_visible_finger_marker():
    color, depth = _scene({"index": 0.0, "middle": 0.0})
    with pytest.raises(ValueError, match="expected exactly 1 finger marker"):
        _measure(color, depth, _sweep_args("index", (440.0, 500.0)))


def test_sweep_mode_rejects_a_frame_with_no_finger_marker():
    color, depth = _scene({})
    with pytest.raises(ValueError, match="expected exactly 1 finger marker"):
        _measure(color, depth, _sweep_args("index", (440.0, 500.0)))


@pytest.mark.parametrize("finger,row,band", [
    ("index", 469, (440.0, 500.0)),
    ("middle", 384, (355.0, 415.0)),
    ("ring", 299, (270.0, 330.0)),
    ("pinky", 215, (185.0, 245.0)),
])
def test_sweep_mode_works_for_every_finger(finger, row, band):
    color, depth = _scene({finger: -3.0}, rows={finger: row})
    _, angles, diagnostics = _measure(color, depth, _sweep_args(finger, band))
    assert diagnostics["measured_fingers"] == [finger]
    assert angles[f"{finger}_abduction_deg_2d"] == pytest.approx(3.0, abs=0.6)


def test_cli_requires_the_row_band_whenever_a_target_finger_is_given():
    from tools.measure_g20_mcp_roll import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--out-prefix", "/tmp/x", "--target-finger", "index"])
    args = parse_args(["--out-prefix", "/tmp/x", "--target-finger", "index",
                       "--target-row-band", "440,500"])
    assert args.target_row_band == (440.0, 500.0)
