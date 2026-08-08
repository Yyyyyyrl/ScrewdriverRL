from __future__ import annotations

import pytest

from tools.audit_linker_l20_topdown_search_settled_geometry import (
    _select_rows,
    _settled_posture,
    _summarize_groups,
)


def test_select_rows_filters_candidate_groups() -> None:
    summary = {
        "top_candidates": [
            {"candidate_group_index": 2},
            {"candidate_group_index": 3},
            {"candidate_group_index": 2},
        ]
    }

    selected = _select_rows(summary, {2})

    assert [row["candidate_group_index"] for row in selected] == [2, 2]


def test_select_rows_rejects_missing_group() -> None:
    with pytest.raises(ValueError, match=r"not found.*\[9\]"):
        _select_rows({"top_candidates": [{"candidate_group_index": 2}]}, {9})


def test_settled_posture_uses_settled_joint_states() -> None:
    row = {
        "candidate_index": 7,
        "root_pos_w": [1.0, 2.0, 3.0],
        "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "settled_joint_positions_independent": {"index_pip": 0.5},
        "settled_screwdriver_joint_positions": {"table_screwdriver_joint_1": 0.1},
    }

    posture = _settled_posture(row)

    assert posture["joint_positions_independent"] == {"index_pip": 0.5}
    assert posture["screwdriver_joint_positions"] == {
        "table_screwdriver_joint_1": 0.1
    }


def test_group_summary_requires_every_replica_to_pass_both_gates() -> None:
    rows = [
        {
            "candidate_group_index": 5,
            "label": "candidate51:replica0",
            "physics_contact_gate": True,
            "settled_release_geometry_pass": True,
            "combined_release_gate_pass": True,
        },
        {
            "candidate_group_index": 5,
            "label": "candidate51:replica1",
            "physics_contact_gate": True,
            "settled_release_geometry_pass": False,
            "combined_release_gate_pass": False,
        },
    ]

    summary = _summarize_groups(rows)

    assert summary == [
        {
            "candidate_group_index": 5,
            "label": "candidate51:replica0",
            "replicas": 2,
            "physics_contact_gate_passes": 2,
            "settled_release_geometry_passes": 1,
            "combined_release_gate_passes": 1,
            "promotion_replica_gate_pass": False,
        }
    ]
