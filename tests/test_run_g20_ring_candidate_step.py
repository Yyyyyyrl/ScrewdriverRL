from __future__ import annotations

import pytest

from tools.run_g20_ring_candidate_step import build_ring_command


def _state() -> list[int]:
    state = [255] * 20
    state[3] = 253
    state[8] = 125
    state[11:15] = [0, 0, 0, 0]
    return state


def _frame(state: list[int]) -> list[int]:
    return [state[8], state[13], state[3], state[14], 0, state[18]]


def test_pip_step_changes_only_slot18_and_holds_pitch_and_side():
    state = _state()
    after, frame, diff = build_ring_command(
        state,
        _frame(state),
        joint="ring_pip",
        expected_start_raw=255,
        target_raw=240,
        side_hold_raw=125,
        other_flex_hold_raw=255,
    )
    assert diff == [
        {"slot": 3, "before": 253, "after": 255},
        {"slot": 18, "before": 255, "after": 240},
    ]
    assert frame == [125, 0, 255, 0, 0, 240]
    assert after[18] == 240


def test_pitch_step_changes_only_slot3_and_holds_pip_and_side():
    state = _state()
    state[3] = 240
    after, frame, diff = build_ring_command(
        state,
        _frame(state),
        joint="ring_mcp_pitch",
        expected_start_raw=240,
        target_raw=224,
        side_hold_raw=125,
        other_flex_hold_raw=255,
    )
    assert diff == [{"slot": 3, "before": 240, "after": 224}]
    assert frame == [125, 0, 224, 0, 0, 255]
    assert after[18] == 255


def test_reverse_bridge_uses_actual_readback_for_step_limit():
    state = _state()
    state[18] = 78
    _, _, diff = build_ring_command(
        state,
        _frame(state),
        joint="ring_pip",
        expected_start_raw=80,
        target_raw=94,
        side_hold_raw=125,
        start_tolerance_raw=2,
    )
    assert diff[-1] == {"slot": 18, "before": 78, "after": 94}


def test_rejects_step_larger_than_17_from_actual_readback():
    state = _state()
    state[18] = 77
    with pytest.raises(ValueError, match="step must be 1..17"):
        build_ring_command(
            state,
            _frame(state),
            joint="ring_pip",
            expected_start_raw=79,
            target_raw=95,
            side_hold_raw=125,
            start_tolerance_raw=2,
        )


def test_rejects_fresh_frame_mapping_mismatch():
    state = _state()
    frame = _frame(state)
    frame[2] = 200
    with pytest.raises(ValueError, match="fresh ring frame"):
        build_ring_command(
            state,
            frame,
            joint="ring_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )


def test_rejects_nonzero_reserved_slot():
    state = _state()
    state[13] = 1
    with pytest.raises(ValueError, match="fresh ring frame|reserved"):
        build_ring_command(
            state,
            _frame(state),
            joint="ring_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )


def test_rejects_side_hold_drift():
    state = _state()
    with pytest.raises(ValueError, match="hold slot 8"):
        build_ring_command(
            state,
            _frame(state),
            joint="ring_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=130,
            hold_tolerance_raw=2,
        )


def test_fifth_reserved_byte_is_not_aliased_from_thumb_slot15():
    state = _state()
    state[15] = 77
    after, frame, _ = build_ring_command(
        state,
        _frame(state),
        joint="ring_pip",
        expected_start_raw=255,
        target_raw=240,
        side_hold_raw=125,
    )
    assert after[15] == 77
    assert frame[4] == 0

    bad = _frame(state)
    bad[4] = 77
    with pytest.raises(ValueError, match="reserved byte 4"):
        build_ring_command(
            state,
            bad,
            joint="ring_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )
