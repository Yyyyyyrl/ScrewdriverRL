from __future__ import annotations

import pytest

from tools.run_g20_index_clearance_step import build_index_clearance_command


def _state() -> list[int]:
    state = [255] * 20
    state[1] = 247
    state[6] = 135
    state[16] = 128
    state[11:15] = [0, 0, 0, 0]
    return state


def _frame(state: list[int]) -> list[int]:
    return [state[6], state[11], state[1], state[13], state[14], state[16]]


def test_three_axis_step_maps_exact_index_frame():
    state = _state()
    after, frame, diff = build_index_clearance_command(
        state,
        _frame(state),
        expected_start_side_raw=135,
        expected_start_pitch_raw=247,
        expected_start_pip_raw=128,
        target_side_raw=119,
        target_pitch_raw=231,
        target_pip_raw=112,
    )
    assert diff == [
        {"slot": 1, "before": 247, "after": 231},
        {"slot": 6, "before": 135, "after": 119},
        {"slot": 16, "before": 128, "after": 112},
    ]
    assert frame == [119, 0, 231, 0, 0, 112]
    assert after[11:15] == [0, 0, 0, 0]


def test_allows_axis_to_remain_at_target():
    state = _state()
    state[16] = 0
    _, frame, _ = build_index_clearance_command(
        state,
        _frame(state),
        expected_start_side_raw=135,
        expected_start_pitch_raw=247,
        expected_start_pip_raw=0,
        target_side_raw=119,
        target_pitch_raw=231,
        target_pip_raw=0,
    )
    assert frame[-1] == 0


def test_rejects_any_axis_step_above_17():
    state = _state()
    with pytest.raises(ValueError, match="slot 1 step"):
        build_index_clearance_command(
            state,
            _frame(state),
            expected_start_side_raw=135,
            expected_start_pitch_raw=247,
            expected_start_pip_raw=128,
            target_side_raw=119,
            target_pitch_raw=229,
            target_pip_raw=112,
        )


def test_rejects_reserved_frame_mismatch():
    state = _state()
    frame = _frame(state)
    frame[1] = 1
    with pytest.raises(ValueError, match="fresh index frame"):
        build_index_clearance_command(
            state,
            frame,
            expected_start_side_raw=135,
            expected_start_pitch_raw=247,
            expected_start_pip_raw=128,
            target_side_raw=119,
            target_pitch_raw=231,
            target_pip_raw=112,
        )
