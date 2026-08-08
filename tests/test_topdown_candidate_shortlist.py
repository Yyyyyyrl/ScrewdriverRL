from __future__ import annotations

import pytest

from tools.build_linker_l20_topdown_candidate_shortlist import (
    _select_candidates,
    _select_replicated_candidate_groups,
)


def _row(candidate_index: int, gate: bool, joint: float) -> dict:
    return {
        "candidate_index": candidate_index,
        "physics_contact_gate": gate,
        "root_pos_w": [0.0, 0.0, 0.0],
        "joint_positions_independent": {"joint": joint},
    }


def test_select_candidates_uses_rank_order_and_unique_candidate_ids() -> None:
    rows = [
        _row(3, True, 0.1),
        _row(4, True, 0.1),
        _row(8, False, 0.2),
        _row(5, True, 0.3),
    ]

    selected = _select_candidates(
        rows,
        2,
        require_physics_contact_gate=True,
    )

    assert [row["candidate_index"] for row in selected] == [3, 5]


def test_select_candidates_fails_closed_when_gate_has_too_few() -> None:
    with pytest.raises(ValueError, match="only 1 unique candidates"):
        _select_candidates(
            [
                _row(0, True, 0.1),
                _row(1, False, 0.2),
            ],
            2,
            require_physics_contact_gate=True,
        )


def test_select_candidates_can_use_functional_gate_field() -> None:
    rows = [
        {**_row(0, False, 0.1), "functional_physics_gate": True},
        {**_row(1, True, 0.2), "functional_physics_gate": False},
    ]

    selected = _select_candidates(
        rows,
        1,
        require_physics_contact_gate=True,
        gate_field="functional_physics_gate",
    )

    assert [row["candidate_index"] for row in selected] == [0]


def _replica_row(
    group: int,
    replica: int,
    *,
    gate: bool,
    role_fraction: float,
    cost: float,
) -> dict:
    return {
        "candidate_index": group + 8 * replica,
        "candidate_group_index": group,
        "physics_contact_gate": gate,
        "root_pos_w": [float(group), 0.0, 0.0],
        "joint_positions_independent": {"joint": float(group)},
        "role_contact_fraction": {"ring": role_fraction},
        "cost": cost,
    }


def test_replicated_selection_ranks_whole_groups_not_lucky_replicas() -> None:
    rows = []
    rows.extend(
        _replica_row(
            0, replica, gate=replica != 3, role_fraction=1.0, cost=0.1
        )
        for replica in range(4)
    )
    rows.extend(
        _replica_row(
            1, replica, gate=True, role_fraction=0.96, cost=2.0
        )
        for replica in range(4)
    )
    rows.extend(
        _replica_row(
            2, replica, gate=True, role_fraction=1.0, cost=3.0
        )
        for replica in range(4)
    )

    selected = _select_replicated_candidate_groups(
        rows,
        2,
        replicas_per_candidate=4,
        minimum_passing_replicas=4,
    )

    assert [
        row["_source_group_diagnostics"]["candidate_group_index"]
        for row in selected
    ] == [2, 1]
    assert all(
        row["_source_group_diagnostics"]["physics_contact_gate_passes"] == 4
        for row in selected
    )


def test_replicated_selection_fails_closed_on_incomplete_group() -> None:
    with pytest.raises(ValueError, match="has 3 replicas; expected 4"):
        _select_replicated_candidate_groups(
            [
                _replica_row(
                    0, replica, gate=True, role_fraction=1.0, cost=1.0
                )
                for replica in range(3)
            ],
            1,
            replicas_per_candidate=4,
            minimum_passing_replicas=4,
        )

