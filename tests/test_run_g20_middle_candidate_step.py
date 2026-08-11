from __future__ import annotations

import pytest

from tools.run_g20_middle_candidate_step import build_middle_command


def _state() -> list[int]:
    state = [255] * 20
    state[2] = 253
    state[7] = 125
    state[11:15] = [0, 0, 0, 0]
    return state


def _frame(state: list[int]) -> list[int]:
    return [state[7], state[12], state[2], state[13], state[14], state[17]]


def test_pip_step_changes_only_slot17_and_holds_pitch_and_side():
    state = _state()
    after, frame, diff = build_middle_command(
        state,
        _frame(state),
        joint="middle_pip",
        expected_start_raw=255,
        target_raw=240,
        side_hold_raw=125,
        other_flex_hold_raw=255,
    )
    assert diff == [
        {"slot": 2, "before": 253, "after": 255},
        {"slot": 17, "before": 255, "after": 240},
    ]
    assert frame == [125, 0, 255, 0, 0, 240]
    assert after[17] == 240


def test_pitch_step_changes_only_slot2_and_holds_pip_and_side():
    state = _state()
    state[2] = 240
    after, frame, diff = build_middle_command(
        state,
        _frame(state),
        joint="middle_mcp_pitch",
        expected_start_raw=240,
        target_raw=224,
        side_hold_raw=125,
        other_flex_hold_raw=255,
    )
    assert diff == [{"slot": 2, "before": 240, "after": 224}]
    assert frame == [125, 0, 224, 0, 0, 255]
    assert after[17] == 255


def test_reverse_bridge_uses_actual_readback_for_step_limit():
    state = _state()
    state[17] = 78
    _, _, diff = build_middle_command(
        state,
        _frame(state),
        joint="middle_pip",
        expected_start_raw=80,
        target_raw=94,
        side_hold_raw=125,
        start_tolerance_raw=2,
    )
    assert diff[-1] == {"slot": 17, "before": 78, "after": 94}


def test_rejects_step_larger_than_17_from_actual_readback():
    state = _state()
    state[17] = 77
    with pytest.raises(ValueError, match="step must be 1..17"):
        build_middle_command(
            state,
            _frame(state),
            joint="middle_pip",
            expected_start_raw=79,
            target_raw=95,
            side_hold_raw=125,
            start_tolerance_raw=2,
        )


def test_rejects_fresh_frame_mapping_mismatch():
    state = _state()
    frame = _frame(state)
    frame[2] = 200
    with pytest.raises(ValueError, match="fresh middle frame"):
        build_middle_command(
            state,
            frame,
            joint="middle_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )


def test_rejects_nonzero_reserved_slot():
    state = _state()
    state[13] = 1
    with pytest.raises(ValueError, match="fresh middle frame|reserved"):
        build_middle_command(
            state,
            _frame(state),
            joint="middle_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=125,
        )


def test_rejects_side_hold_drift():
    state = _state()
    with pytest.raises(ValueError, match="hold slot 7"):
        build_middle_command(
            state,
            _frame(state),
            joint="middle_pip",
            expected_start_raw=255,
            target_raw=240,
            side_hold_raw=130,
            hold_tolerance_raw=2,
        )
