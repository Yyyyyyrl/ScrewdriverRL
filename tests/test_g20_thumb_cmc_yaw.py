"""Offline tests for the thumb CMC-yaw detector and single-step runner.

No camera, no CAN, no motion.
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import pytest

from tools.measure_g20_thumb_yaw import measure_yaw_frame, yaw_angle
from tools.summarize_g20_thumb_yaw import (
    axis_delta_deg as summary_axis_delta_deg,
    palm_reference_delta,
    robust_yaw,
)
from tools.run_g20_thumb_cmc_yaw_candidate_step import (
    THUMB_FRAME_SLOTS,
    THUMB_MCP_SLOT,
    THUMB_PITCH_SLOT,
    THUMB_ROLL_SLOT,
    THUMB_YAW_SLOT,
    build_thumb_yaw_command,
)

BLUE = (255, 0, 0)
DEPTH_SCALE = 0.001

# Measured read-only snapshot 2026-08-03 with the thumb-yaw isolation posture.
BASE_STATE20 = [247, 247, 252, 246, 248, 254, 80, 130, 166, 187, 253,
                0, 0, 0, 0, 255, 255, 255, 255, 255]


def _frame(state20=None):
    s = state20 or BASE_STATE20
    return [s[5], s[10], s[0], 0, 0, s[15]]


def _call(target_raw, state20=None, frame6=None, **over):
    s = list(state20 or BASE_STATE20)
    kwargs = dict(expected_start_raw=s[THUMB_YAW_SLOT], target_raw=target_raw,
                  roll_hold_raw=s[THUMB_ROLL_SLOT],
                  pitch_hold_raw=s[THUMB_PITCH_SLOT],
                  mcp_hold_raw=s[THUMB_MCP_SLOT])
    kwargs.update(over)
    return build_thumb_yaw_command(
        s, frame6 if frame6 is not None else _frame(s), **kwargs)


# --- angle convention -------------------------------------------------------

def test_reference_offset_recentres_the_angle_away_from_the_wrap():
    # Measured pose: thumb +76.2, palm +0.15, so the raw difference sits 14 deg
    # from +90 and a Phase-B-sized sweep would wrap. With the offset removed
    # first, the same pose reads near zero.
    assert yaw_angle(0.15, 76.22, 75.9) == pytest.approx(0.17, abs=0.02)
    assert abs(yaw_angle(0.15, 76.22, 0.0)) > 70.0


def test_offset_gives_ninety_degrees_of_headroom_each_way():
    for travel in (-85.0, -45.0, 0.0, 45.0, 85.0):
        got = yaw_angle(0.0, 75.9 + travel, 75.9)
        assert got == pytest.approx(travel, abs=1e-6), travel


def test_without_the_offset_the_same_travel_would_flip_sign():
    # +20 deg of travel from the measured pose crosses +90 and wraps negative.
    naive = yaw_angle(0.15, 76.22 + 20.0, 0.0)
    assert naive < 0.0
    corrected = yaw_angle(0.15, 76.22 + 20.0, 75.9)
    assert corrected == pytest.approx(20.17, abs=0.02)


def test_yaw_sign_is_signed_not_absolute():
    assert yaw_angle(0.0, 75.9 + 5.0, 75.9) == pytest.approx(5.0)
    assert yaw_angle(0.0, 75.9 - 5.0, 75.9) == pytest.approx(-5.0)


# --- detector ---------------------------------------------------------------

def _det_args(**over):
    values = {"thumb_roi": (330, 170, 900, 470), "palm_roi": (440, 430, 900, 660),
              "hsv_lower": (75, 40, 20), "hsv_upper": (165, 255, 255),
              "thumb_min_area": 500, "palm_min_area": 500,
              "thumb_depth_band": (0.265, 0.305),
              "palm_depth_band": (0.305, 0.350),
              "min_elongation2": 4.0, "reference_offset_deg": 118.0}
    values.update(over)
    return argparse.Namespace(**values)


def _bar(color, depth, cx, cy, length, thick, deg, depth_raw):
    r = math.radians(deg)
    dx, dy = math.cos(r) * length / 2, math.sin(r) * length / 2
    p0 = (int(cx - dx), int(cy - dy))
    p1 = (int(cx + dx), int(cy + dy))
    cv2.line(color, p0, p1, BLUE, thick)
    cv2.line(depth, p0, p1, int(depth_raw), thick + 6)


def _scene(thumb_deg=-36.0, thumb_cross_deg=50.0, palm_deg=0.0,
           palm_cross_deg=81.0, thumb_depth=283, palm_depth=332,
           touching=False):
    """Four markers: thumb L nearer the camera, palm L further away."""
    color = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.full((720, 1280), 900, dtype=np.uint16)
    _bar(color, depth, 700, 300, 130, 26, thumb_deg, thumb_depth)
    _bar(color, depth, 560, 410, 108, 26, thumb_cross_deg, thumb_depth + 9)
    palm_cx = 600 if touching else 700
    _bar(color, depth, palm_cx, 455 if touching else 500, 150, 26, palm_deg, palm_depth)
    _bar(color, depth, 529, 576, 90, 38, palm_cross_deg, palm_depth - 4)
    # The real scene has a big blue blob far behind; depth must exclude it.
    cv2.rectangle(color, (0, 153), (116, 332), BLUE, -1)
    cv2.rectangle(depth, (0, 153), (116, 332), 540, -1)
    return color, depth


def test_detector_labels_all_four_markers():
    angles, detail = measure_yaw_frame(*_scene(), DEPTH_SCALE, _det_args())
    assert set(detail) == {"thumb", "thumb_cross", "palm", "palm_cross"}
    # thumb is the stripe farther from the palm, not merely the longer one
    assert detail["thumb"]["depth_median_m"] < detail["palm"]["depth_median_m"]
    assert angles["thumb_heading_deg_2d"] == pytest.approx(-36.0, abs=1.0)
    assert angles["palm_heading_deg_2d"] == pytest.approx(0.0, abs=1.0)


def test_l_angles_are_reported_for_both_segments():
    angles, _ = measure_yaw_frame(*_scene(), DEPTH_SCALE, _det_args())
    assert angles["thumb_l_angle_deg"] == pytest.approx(86.0, abs=1.5)
    assert angles["palm_l_angle_deg"] == pytest.approx(81.0, abs=1.5)


def test_l_angle_tracks_out_of_plane_tilt_independently_of_area():
    # The witness for out-of-plane motion must be the angle inside the L, not
    # marker area, because ROI clipping also shrinks area.
    base, _ = measure_yaw_frame(*_scene(), DEPTH_SCALE, _det_args())
    tilted, _ = measure_yaw_frame(*_scene(thumb_cross_deg=20.0), DEPTH_SCALE,
                                  _det_args())
    assert tilted["thumb_l_angle_deg"] < base["thumb_l_angle_deg"] - 20.0


def test_depth_separates_thumb_and_palm_even_when_they_touch():
    # The thumb sweeps across the palm; at one end the tapes overlap in image
    # space and would fuse into one component without depth-first segmentation.
    angles, detail = measure_yaw_frame(*_scene(touching=True), DEPTH_SCALE,
                                       _det_args())
    assert detail["thumb"]["depth_median_m"] < 0.305
    assert detail["palm"]["depth_median_m"] > 0.305
    assert angles["thumb_heading_deg_2d"] == pytest.approx(-36.0, abs=1.0)


def test_far_background_blob_is_excluded_by_depth():
    _, detail = measure_yaw_frame(*_scene(), DEPTH_SCALE, _det_args())
    for marker in detail.values():
        assert marker["depth_median_m"] < 0.35


def test_detector_fails_when_a_thumb_stripe_is_missing():
    color, depth = _scene()
    color[250:350, 640:770] = 0
    with pytest.raises(ValueError, match="expected exactly 2 thumb markers"):
        measure_yaw_frame(color, depth, DEPTH_SCALE, _det_args())


def test_detector_fails_when_a_palm_stripe_is_missing():
    color, depth = _scene()
    color[470:530, 615:780] = 0
    with pytest.raises(ValueError, match="expected exactly 2 palm markers"):
        measure_yaw_frame(color, depth, DEPTH_SCALE, _det_args())


def test_detector_fails_on_a_round_marker_with_no_usable_heading():
    color, depth = _scene()
    color[240:360, 630:780] = 0
    cv2.circle(color, (700, 300), 45, BLUE, -1)
    cv2.circle(depth, (700, 300), 48, 283, -1)
    with pytest.raises(ValueError, match="elongation"):
        measure_yaw_frame(color, depth, DEPTH_SCALE, _det_args())


# --- step runner ------------------------------------------------------------

def test_only_the_yaw_slot_changes():
    after, frame6, diff = _call(BASE_STATE20[THUMB_YAW_SLOT] - 12)
    assert [d["slot"] for d in diff] == [THUMB_YAW_SLOT]
    assert frame6[1] == BASE_STATE20[THUMB_YAW_SLOT] - 12
    assert frame6[0] == BASE_STATE20[THUMB_ROLL_SLOT]
    assert frame6[2] == BASE_STATE20[THUMB_PITCH_SLOT]
    assert frame6[5] == BASE_STATE20[THUMB_MCP_SLOT]
    assert [frame6[i] for i in (3, 4)] == [0, 0]


def test_frame_slot_table_matches_the_phase_a_thumb_runner():
    assert THUMB_FRAME_SLOTS == (5, 10, 0, 11, 12, 15)


def test_no_long_finger_slot_is_ever_touched():
    after, _, _ = _call(BASE_STATE20[THUMB_YAW_SLOT] - 12)
    for slot in (1, 2, 3, 4, 6, 7, 8, 9, 16, 17, 18, 19):
        assert after[slot] == BASE_STATE20[slot]


@pytest.mark.parametrize("target", [235, 253])
def test_step_outside_one_to_seventeen_raw_is_refused(target):
    # yaw rests at 253, so an 18 raw step down and a zero-length step are the
    # two in-range ways to violate the adjacency rule.
    with pytest.raises(ValueError, match="must be 1..17 raw"):
        _call(target)


def test_upward_step_over_seventeen_raw_is_refused():
    state = list(BASE_STATE20)
    state[THUMB_YAW_SLOT] = 200
    with pytest.raises(ValueError, match="must be 1..17 raw"):
        _call(218, state20=state)


def test_expected_start_must_match_the_measured_slot():
    with pytest.raises(ValueError, match="must start within"):
        _call(240, expected_start_raw=200)


def test_hold_off_its_reference_blocks_the_step():
    state = list(BASE_STATE20)
    state[THUMB_ROLL_SLOT] = 200
    with pytest.raises(ValueError, match="hold slot 5 must be within"):
        _call(241, state20=state, roll_hold_raw=254)


def test_frame_disagreeing_with_raw20_blocks_the_step():
    frame6 = _frame()
    frame6[1] -= 6
    with pytest.raises(ValueError, match="element 1/slot 10"):
        _call(241, frame6=frame6)


def test_nonzero_reserved_frame_element_blocks_the_step():
    frame6 = _frame()
    frame6[3] = 5
    with pytest.raises(ValueError, match="element 3"):
        _call(241, frame6=frame6)


def test_nonzero_reserved_raw20_slot_blocks_the_step():
    state = list(BASE_STATE20)
    state[12] = 3
    with pytest.raises(ValueError, match="reserved slots 11..14"):
        _call(241, state20=state)


def test_target_outside_raw_range_is_refused():
    with pytest.raises(ValueError, match="target raw must be in 0..255"):
        _call(300)


# --- robust palm-L summary reference ----------------------------------------

def _summary_angles(thumb: float, palm_long: float, palm_cross: float):
    return {
        "thumb_heading_deg_2d": {"median": thumb},
        "palm_heading_deg_2d": {"median": palm_long},
        "palm_cross_heading_deg_2d": {"median": palm_cross},
    }


def test_summary_axis_delta_handles_undirected_line_wrap():
    assert summary_axis_delta_deg(-89.0, 89.0) == pytest.approx(2.0)
    assert summary_axis_delta_deg(89.0, -89.0) == pytest.approx(-2.0)


def test_robust_palm_reference_fuses_agreeing_l_markers():
    zero = {
        "thumb_heading_deg_2d": 0.0,
        "palm_heading_deg_2d": 0.0,
        "palm_cross_heading_deg_2d": 0.0,
    }
    angles = _summary_angles(5.0, 0.2, 0.1)
    palm, mode = palm_reference_delta(angles, zero)
    assert mode == "fused_long_and_cross"
    assert palm == pytest.approx(0.15)
    assert robust_yaw(angles, zero) == pytest.approx(4.85)


def test_robust_palm_reference_falls_back_when_long_is_occluded():
    zero = {
        "thumb_heading_deg_2d": 0.0,
        "palm_heading_deg_2d": 0.0,
        "palm_cross_heading_deg_2d": 0.0,
    }
    angles = _summary_angles(5.0, 2.0, 0.1)
    palm, mode = palm_reference_delta(angles, zero)
    assert mode == "cross_fallback_long_occluded"
    assert palm == pytest.approx(0.1)
    assert robust_yaw(angles, zero) == pytest.approx(4.9)
