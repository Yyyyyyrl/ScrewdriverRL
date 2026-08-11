from __future__ import annotations

import pytest

from tools.run_g20_index_pip_first_step import build_index_pip_command


def _state20() -> list[int]:
    return [
        254, 254, 254, 254, 254,
        253, 127, 128, 129, 127,
        254, 0, 0, 0, 0,
        254, 255, 255, 255, 255,
    ]


def test_first_step_changes_only_slot16_and_preserves_index_reserved_values():
    after20, index6, diff = build_index_pip_command(
        _state20(),
        [127, 19, 254, 21, 22, 255],
    )
    assert after20[:16] == _state20()[:16]
    assert after20[16:] == [240, 255, 255, 255]
    assert index6 == [127, 19, 254, 21, 22, 240]
    assert diff == [{"slot": 16, "before": 255, "after": 240}]


def test_first_step_rejects_non_255_start():
    state = _state20()
    state[16] = 254
    with pytest.raises(ValueError, match="must start at raw 255"):
        build_index_pip_command(state, [127, 0, 254, 0, 0, 254])


def test_first_step_rejects_any_other_target():
    with pytest.raises(ValueError, match="permits only target raw 240"):
        build_index_pip_command(
            _state20(),
            [127, 0, 254, 0, 0, 255],
            target_raw=239,
        )


def test_first_step_rejects_index_frame_mapping_disagreement():
    with pytest.raises(ValueError, match="disagrees with raw20"):
        build_index_pip_command(
            _state20(),
            [128, 0, 254, 0, 0, 255],
        )
