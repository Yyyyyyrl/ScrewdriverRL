from __future__ import annotations

import numpy as np
import pytest

from tools.generate_linker_l20_topdown_thumb_pad_candidate_bank import (
    THUMB_PAD_NORMAL_LOCAL,
    _diverse_selection,
    _palmward_direction_world_xy,
)


def test_thumb_pad_normal_is_unit_and_not_a_side_face() -> None:
    assert np.linalg.norm(THUMB_PAD_NORMAL_LOCAL) == pytest.approx(1.0)
    assert THUMB_PAD_NORMAL_LOCAL[1] == pytest.approx(0.0)
    assert abs(THUMB_PAD_NORMAL_LOCAL[0]) > 0.7
    assert abs(THUMB_PAD_NORMAL_LOCAL[2]) > 0.5


def test_diverse_selection_enforces_requested_count() -> None:
    rows = [
        {
            "_normalized_parameters": [float(index) / 9.0, 0.0],
            "_quality": float(index),
            "index": index,
        }
        for index in range(10)
    ]

    selected = _diverse_selection(rows, 4)

    assert len(selected) == 4
    assert len({row["index"] for row in selected}) == 4


def test_diverse_selection_rejects_undersized_pool() -> None:
    with pytest.raises(ValueError, match="only 1 candidates"):
        _diverse_selection(
            [{"_normalized_parameters": [0.0], "_quality": 1.0}],
            2,
        )


def test_palmward_direction_is_in_plane_and_points_from_axis_to_hand() -> None:
    direction = _palmward_direction_world_xy(
        np.asarray((0.0, 0.20, 1.45)),
        np.asarray((-0.01, 0.0, 1.30)),
    )

    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction[2] == pytest.approx(0.0)
    assert direction[0] > 0.0
    assert direction[1] > 0.0
