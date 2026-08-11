from __future__ import annotations

import pytest

from tools.run_g20_pinky_candidate_step import build_pinky_command


def _state() -> list[int]:
    state = [255] * 20
    state[4] = 253
    state[9] = 125
    state[11:15] = [0, 0, 0, 0]
    return state


def _frame(state: list[int]) -> list[int]:
    return [state[9], state[14], state[4], 0, 0, state[19]]


def test_pip_step_changes_only_slot19_and_holds_pitch_and_side():
    state = _state()
    after, frame, diff = build_pinky_command(
        state,
        _frame(state),
        joint="pinky_pip",
        expected_start_raw=255,
        target_raw=240,
        side_hold_raw=125,
        other_flex_hold_raw=255,
    )
    assert diff == [
        {"slot": 4, "before": 253, "after": 255},
        {"slot": 19, "before": 255, "after": 240},
    ]
    assert frame == [125, 0, 255, 0, 0, 240]
    assert after[19] == 240


def test_pitch_step_changes_only_slot4_and_holds_pip_and_side():
    state = _state()
    state[4] = 240
    after, frame, diff = build_pinky_command(
        state,
        _frame(state),
        joint="pinky_mcp_pitch",
        expected_start_raw=240,
        target_raw=224,
        side_hold_raw=125,
        other_flex_hold_raw=255,
    )
    assert diff == [{"slot": 4, "before": 240, "after": 224}]
    assert frame == [125, 0, 224, 0, 0, 255]
    assert after[19] == 255


def test_reverse_bridge_uses_actual_readback_for_step_limit():
    state = _state()
    state[19] = 78
    _, _, diff = build_pinky_command(
        state,
        _frame(state),
        joint="pinky_pip",
        expected_start_raw=80,
        target_raw=94,
        side_hold_raw=125,
        start_tolerance_raw=2,
    )
    assert diff[-1] == {"slot": 19, "before": 78, "after": 94}


def test_rejects_step_larger_than_17_from_actual_readback():
    state = _state()
    state[19] = 77
    with pytest.raises(ValueError, match="step must be 1..17"):
        build_pinky_command(
            state,
            _frame(state),
            joint="pinky_pip",
            expected_start_raw=79,
            target_raw=95,
            side_hold_raw=125,
            start_tolerance_raw=2,
        )


def test_rejects_fresh_frame_mapping_mismatch():
    state = _state()
    frame = _frame(state)
    frame[2] = 200
    with pytest.raises(ValueError, match="fresh pinky frame"):
        build_pinky_command(
            state,
            frame,
            joint="pinky_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )


def test_rejects_nonzero_reserved_slot():
    state = _state()
    state[14] = 1
    with pytest.raises(ValueError, match="fresh pinky frame|reserved"):
        build_pinky_command(
            state,
            _frame(state),
            joint="pinky_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )


def test_rejects_side_hold_drift():
    state = _state()
    with pytest.raises(ValueError, match="hold slot 9"):
        build_pinky_command(
            state,
            _frame(state),
            joint="pinky_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=130,
            hold_tolerance_raw=2,
        )


def test_low_pitch_endpoint_recovery_allows_bounded_side_correction():
    state = _state()
    state[4] = 3
    state[9] = 140
    after, frame, diff = build_pinky_command(
        state,
        _frame(state),
        joint="pinky_mcp_pitch",
        expected_start_raw=3,
        target_raw=6,
        side_hold_raw=153,
        other_flex_hold_raw=255,
        hold_tolerance_raw=3,
        side_start_tolerance_raw=15,
    )
    assert diff == [
        {"slot": 4, "before": 3, "after": 6},
        {"slot": 9, "before": 140, "after": 153},
    ]
    assert frame == [153, 0, 6, 0, 0, 255]


def test_low_pitch_endpoint_recovery_rejects_excessive_side_error():
    state = _state()
    state[4] = 3
    state[9] = 137
    with pytest.raises(ValueError, match="hold slot 9"):
        build_pinky_command(
            state,
            _frame(state),
            joint="pinky_mcp_pitch",
            expected_start_raw=3,
            target_raw=6,
            side_hold_raw=153,
            other_flex_hold_raw=255,
            hold_tolerance_raw=3,
            side_start_tolerance_raw=15,
        )


def test_reserved_bytes_are_not_aliased_from_thumb_or_index_slots():
    state = _state()
    state[15] = 66
    state[16] = 77
    after, frame, _ = build_pinky_command(
        state,
        _frame(state),
        joint="pinky_pip",
        expected_start_raw=255,
        target_raw=240,
        side_hold_raw=125,
    )
    assert after[15] == 66
    assert after[16] == 77
    assert frame[4] == 0

    bad3 = _frame(state)
    bad3[3] = 66
    with pytest.raises(ValueError, match="reserved byte 3"):
        build_pinky_command(
            state,
            bad3,
            joint="pinky_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )

    bad = _frame(state)
    bad[4] = 77
    with pytest.raises(ValueError, match="reserved byte 4"):
        build_pinky_command(
            state,
            bad,
            joint="pinky_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )
