from __future__ import annotations

import pytest

from tools.recover_g20_index_pip_fault64 import build_recovery_command


def _state20() -> list[int]:
    return [
        254, 249, 254, 254, 254,
        253, 127, 128, 129, 127,
        254, 0, 0, 0, 0,
        254, 4, 255, 255, 255,
    ]


def test_recovery_changes_only_index_root_and_pip():
    before = _state20()
    after, index6, diff = build_recovery_command(
        before,
        [127, 0, 249, 0, 0, 4],
    )
    assert after == [
        254, 250, 254, 254, 254,
        253, 127, 128, 129, 127,
        254, 0, 0, 0, 0,
        254, 12, 255, 255, 255,
    ]
    assert index6 == [127, 0, 250, 0, 0, 12]
    assert diff == [
        {"slot": 1, "before": 249, "after": 250},
        {"slot": 16, "before": 4, "after": 12},
    ]


def test_recovery_rejects_unexpected_start_raw():
    state = _state20()
    state[16] = 5
    with pytest.raises(RuntimeError, match="must be raw 4"):
        build_recovery_command(state, [127, 0, 249, 0, 0, 5])


def test_recovery_rejects_index_frame_mapping_disagreement():
    with pytest.raises(RuntimeError, match="disagrees with raw20"):
        build_recovery_command(_state20(), [127, 0, 250, 0, 0, 4])


def test_recovery_rejects_nonzero_index_reserved_elements():
    with pytest.raises(RuntimeError, match="reserved elements"):
        build_recovery_command(_state20(), [127, 1, 249, 0, 0, 4])
