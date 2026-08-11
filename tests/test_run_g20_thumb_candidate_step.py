from __future__ import annotations

import pytest

from tools.run_g20_thumb_candidate_step import build_thumb_command


def _state20(
    pitch: int = 255,
    roll: int = 67,
    yaw: int = 113,
    mcp: int = 255,
) -> list[int]:
    return [
        pitch, 254, 253, 253, 252,
        roll, 134, 126, 125, 135,
        yaw, 0, 0, 0, 0,
        mcp, 255, 255, 254, 255,
    ]


def _thumb6(state: list[int]) -> list[int]:
    return [state[5], state[10], state[0], 0, 0, state[15]]


def test_mcp_step_changes_only_slot15_and_holds_view_and_pitch():
    before = _state20(pitch=254)
    after, thumb6, diff = build_thumb_command(
        before,
        _thumb6(before),
        joint="thumb_mcp",
        expected_start_raw=255,
        target_raw=240,
        roll_hold_raw=67,
        yaw_hold_raw=113,
    )
    assert thumb6 == [67, 113, 255, 0, 0, 240]
    assert diff == [
        {"slot": 0, "before": 254, "after": 255},
        {"slot": 15, "before": 255, "after": 240},
    ]
    assert all(
        after[i] == before[i] for i in range(20) if i not in (0, 15)
    )


def test_pitch_step_changes_only_slot0_and_holds_view_and_mcp():
    before = _state20()
    after, thumb6, diff = build_thumb_command(
        before,
        _thumb6(before),
        joint="thumb_cmc_pitch",
        expected_start_raw=255,
        target_raw=240,
        roll_hold_raw=67,
        yaw_hold_raw=113,
    )
    assert thumb6 == [67, 113, 240, 0, 0, 255]
    assert diff == [{"slot": 0, "before": 255, "after": 240}]
    assert all(after[i] == before[i] for i in range(20) if i != 0)


def test_rejects_non_adjacent_step():
    before = _state20()
    with pytest.raises(ValueError, match="1..17"):
        build_thumb_command(
            before,
            _thumb6(before),
            joint="thumb_mcp",
            expected_start_raw=255,
            target_raw=224,
            roll_hold_raw=67,
            yaw_hold_raw=113,
        )


def test_accepts_non_table_bridge_target_when_actual_step_is_safe():
    before = _state20(mcp=78)
    after, thumb6, diff = build_thumb_command(
        before,
        _thumb6(before),
        joint="thumb_mcp",
        expected_start_raw=78,
        target_raw=94,
        roll_hold_raw=67,
        yaw_hold_raw=113,
    )
    assert after[15] == 94
    assert thumb6 == [67, 113, 255, 0, 0, 94]
    assert diff == [{"slot": 15, "before": 78, "after": 94}]


def test_rejects_out_of_range_bridge_target():
    before = _state20()
    with pytest.raises(ValueError, match="0..255"):
        build_thumb_command(
            before,
            _thumb6(before),
            joint="thumb_mcp",
            expected_start_raw=255,
            target_raw=256,
            roll_hold_raw=67,
            yaw_hold_raw=113,
        )


def test_rejects_wrong_fresh_thumb_mapping():
    before = _state20()
    wrong = _thumb6(before)
    wrong[1] = 100
    with pytest.raises(ValueError, match="disagrees"):
        build_thumb_command(
            before,
            wrong,
            joint="thumb_mcp",
            expected_start_raw=255,
            target_raw=240,
            roll_hold_raw=67,
            yaw_hold_raw=113,
        )


def test_rejects_view_hold_outside_tolerance():
    before = _state20(roll=63)
    with pytest.raises(ValueError, match="hold slot 5"):
        build_thumb_command(
            before,
            _thumb6(before),
            joint="thumb_mcp",
            expected_start_raw=255,
            target_raw=240,
            roll_hold_raw=67,
            yaw_hold_raw=113,
        )


def test_rejects_nonzero_reserved_slots():
    before = _state20()
    before[11] = 1
    thumb = _thumb6(before)
    thumb[3] = 1
    with pytest.raises(ValueError, match="reserved"):
        build_thumb_command(
            before,
            thumb,
            joint="thumb_mcp",
            expected_start_raw=255,
            target_raw=240,
            roll_hold_raw=67,
            yaw_hold_raw=113,
        )
