from __future__ import annotations

import pytest

from tools.run_g20_index_mcp_pitch_candidate_step import (
    build_index_mcp_pitch_command,
)


def _state20(root: int = 249, pip: int = 255) -> list[int]:
    return [
        254,
        root,
        254,
        254,
        254,
        253,
        127,
        128,
        129,
        127,
        254,
        0,
        0,
        0,
        0,
        254,
        pip,
        255,
        255,
        255,
    ]


def test_mcp_step_changes_only_root_and_holds_pip():
    before = _state20()
    after20, index6, diff = build_index_mcp_pitch_command(
        before,
        [127, 19, 249, 21, 22, 255],
        expected_start_raw=249,
        target_raw=255,
    )
    assert after20[1] == 255
    assert after20[16] == 255
    assert after20[:1] + after20[2:] == before[:1] + before[2:]
    assert index6 == [127, 19, 255, 21, 22, 255]
    assert diff == [{"slot": 1, "before": 249, "after": 255}]


def test_mcp_step_can_correct_one_raw_pip_readback_with_explicit_hold():
    before = _state20(pip=254)
    after20, index6, diff = build_index_mcp_pitch_command(
        before,
        [127, 19, 249, 21, 22, 254],
        expected_start_raw=249,
        target_raw=255,
        pip_tolerance_raw=1,
    )
    assert after20[16] == 255
    assert index6[5] == 255
    assert diff == [
        {"slot": 1, "before": 249, "after": 255},
        {"slot": 16, "before": 254, "after": 255},
    ]


def test_mcp_step_rejects_non_adjacent_target():
    with pytest.raises(ValueError, match="1..17"):
        build_index_mcp_pitch_command(
            _state20(),
            [127, 19, 249, 21, 22, 255],
            expected_start_raw=249,
            target_raw=224,
        )


def test_mcp_step_rejects_pip_outside_isolation_reference():
    with pytest.raises(ValueError, match="PIP"):
        build_index_mcp_pitch_command(
            _state20(pip=250),
            [127, 19, 249, 21, 22, 250],
            expected_start_raw=249,
            target_raw=255,
            pip_tolerance_raw=1,
        )


def test_mcp_step_rejects_fresh_frame_mapping_disagreement():
    with pytest.raises(ValueError, match="disagrees"):
        build_index_mcp_pitch_command(
            _state20(),
            [127, 19, 248, 21, 22, 255],
            expected_start_raw=249,
            target_raw=255,
        )
