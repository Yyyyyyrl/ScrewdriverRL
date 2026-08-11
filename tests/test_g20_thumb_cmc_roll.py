"""Offline tests for thumb CMC-roll detection and single-frame safety logic."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pytest

from tools.measure_g20_thumb_roll import measure_roll_frame
from tools.run_g20_thumb_cmc_roll_candidate_step import (
    MAX_ADJACENT_STEP_RAW,
    THUMB_FRAME_SLOTS,
    THUMB_MCP_SLOT,
    THUMB_PITCH_SLOT,
    THUMB_ROLL_SLOT,
    THUMB_YAW_SLOT,
    build_thumb_roll_command,
    parse_args,
)


BASE = [247, 250, 251, 249, 249, 58, 107, 125, 120, 122, 125,
        0, 0, 0, 0, 254, 255, 255, 255, 247]


def _frame(state=BASE):
    return [state[5], state[10], state[0], 0, 0, state[15]]


def _command(target, state=None, frame=None, **overrides):
    state = list(state or BASE)
    kwargs = {
        "expected_start_raw": state[THUMB_ROLL_SLOT],
        "target_raw": target,
        "yaw_hold_raw": 125,
        "pitch_hold_raw": 247,
        "mcp_hold_raw": 254,
    }
    kwargs.update(overrides)
    return build_thumb_roll_command(
        state, frame if frame is not None else _frame(state), **kwargs
    )


def test_thumb_frame_slot_contract_and_only_roll_changes():
    after, frame, diff = _command(72)
    assert THUMB_FRAME_SLOTS == (5, 10, 0, 11, 12, 15)
    assert [item["slot"] for item in diff] == [THUMB_ROLL_SLOT]
    assert frame == [72, 125, 247, 0, 0, 254]
    for slot in range(20):
        if slot != THUMB_ROLL_SLOT:
            assert after[slot] == BASE[slot]


@pytest.mark.parametrize("target", [58, 73])
def test_zero_or_over_limit_step_is_refused(target):
    with pytest.raises(ValueError, match=f"must be 1..{MAX_ADJACENT_STEP_RAW} raw"):
        _command(target)


def test_drifted_hold_axis_is_refused():
    state = list(BASE)
    state[THUMB_YAW_SLOT] = 120
    with pytest.raises(ValueError, match="hold slot 10"):
        _command(68, state=state)


def test_reserved_or_mismatched_thumb_frame_is_refused():
    frame = _frame()
    frame[3] = 1
    with pytest.raises(ValueError, match="element 3.*reserved slot 11"):
        _command(68, frame=frame)
    frame = _frame()
    frame[2] -= 5
    with pytest.raises(ValueError, match="element 2/slot 0"):
        _command(68, frame=frame)


def test_cli_is_no_send_without_execute():
    args = parse_args([
        "--sdk-root", "/tmp/sdk", "--calib", "/tmp/calib.json",
        "--expected-start-raw", "58", "--target-raw", "72",
        "--yaw-hold-raw", "125", "--pitch-hold-raw", "247",
        "--mcp-hold-raw", "254", "--out", "/tmp/out.json",
    ])
    assert args.execute is False


def _detector_args():
    return argparse.Namespace(
        hsv_lower=(75, 40, 20),
        hsv_upper=(165, 255, 255),
        palm_depth_band=(0.275, 0.310),
        thumb_depth_band=(0.43, 0.50),
        palm_roi=(670, 230, 850, 315),
        thumb_roi=(450, 380, 750, 630),
        palm_min_area=5000,
        thumb_min_area=600,
        min_thumb_separation_m=0.025,
        max_thumb_separation_m=0.080,
        max_palm_plane_rms_m=0.008,
    )


def _scene(second_center=(625, 530), include_white=True):
    color = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.full((720, 1280), 900, dtype=np.uint16)
    if include_white:
        cv2.rectangle(color, (680, 240), (840, 300), (255, 255, 255), -1)
        cv2.rectangle(depth, (680, 240), (840, 300), 290, -1)
    for center in ((600, 455), second_center):
        cv2.circle(color, center, 20, (255, 0, 0), -1)
        cv2.circle(depth, center, 20, 460, -1)
    intr = {"fx": 600.0, "fy": 600.0, "ppx": 640.0, "ppy": 360.0}
    return color, depth, intr


def test_detector_recovers_white_palm_and_two_thumb_markers():
    color, depth, intr = _scene()
    values, details = measure_roll_frame(
        color, depth, 0.001, intr, _detector_args()
    )
    assert set(details) >= {"palm", "thumb_upper", "thumb_lower"}
    assert details["palm"]["area_px"] > 5000
    assert values["thumb_marker_separation_m"] == pytest.approx(0.059, abs=0.004)
    assert 60.0 < values["thumb_cmc_roll_deg_3d"] < 90.0
    assert values["palm_plane_rms_m"] < 0.001


def test_detector_requires_the_unique_white_reference():
    color, depth, intr = _scene(include_white=False)
    with pytest.raises(ValueError, match="exactly 1 palm"):
        measure_roll_frame(color, depth, 0.001, intr, _detector_args())


def test_detector_rejects_implausible_thumb_marker_separation():
    color, depth, intr = _scene(second_center=(730, 600))
    with pytest.raises(ValueError, match="separation"):
        measure_roll_frame(color, depth, 0.001, intr, _detector_args())


def test_hold_slot_constants_match_g20_thumb_contract():
    assert (THUMB_ROLL_SLOT, THUMB_YAW_SLOT, THUMB_PITCH_SLOT, THUMB_MCP_SLOT) == (
        5, 10, 0, 15
    )
