#!/usr/bin/env python3
"""Build a reproducible, unpromoted candidate bank from a physics search."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_candidates(
    rows: list[dict],
    count: int,
    *,
    require_physics_contact_gate: bool,
    gate_field: str = "physics_contact_gate",
) -> list[dict]:
    if count < 1:
        raise ValueError("--num-candidates must be positive")
    selected = []
    seen_candidate_indices: set[int] = set()
    seen_postures: set[tuple] = set()
    for row in rows:
        candidate_index = int(row["candidate_index"])
        posture = (
            tuple(float(value) for value in row["root_pos_w"]),
            tuple(
                (name, float(value))
                for name, value in sorted(
                    row["joint_positions_independent"].items()
                )
            ),
        )
        if candidate_index in seen_candidate_indices or posture in seen_postures:
            continue
        if require_physics_contact_gate and not row.get(gate_field, False):
            continue
        selected.append(row)
        seen_candidate_indices.add(candidate_index)
        seen_postures.add(posture)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} unique candidates satisfy the shortlist gate; "
            f"need {count}"
        )
    return selected


def _select_replicated_candidate_groups(
    rows: list[dict],
    count: int,
    *,
    replicas_per_candidate: int,
    minimum_passing_replicas: int,
    gate_field: str = "physics_contact_gate",
) -> list[dict]:
    """Rank replicated source groups without selecting a lucky replica."""
    if count < 1:
        raise ValueError("--num-candidates must be positive")
    if replicas_per_candidate < 2:
        raise ValueError("replicated group selection requires at least 2 replicas")
    if not 1 <= minimum_passing_replicas <= replicas_per_candidate:
        raise ValueError(
            "--minimum-passing-replicas must be in "
            f"[1, {replicas_per_candidate}]"
        )

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["candidate_group_index"])].append(row)

    ranked = []
    for group_index, group_rows in grouped.items():
        if len(group_rows) != replicas_per_candidate:
            raise ValueError(
                f"candidate group {group_index} has {len(group_rows)} replicas; "
                f"expected {replicas_per_candidate}"
            )
        postures = {
            (
                tuple(float(value) for value in row["root_pos_w"]),
                tuple(
                    (name, float(value))
                    for name, value in sorted(
                        row["joint_positions_independent"].items()
                    )
                ),
            )
            for row in group_rows
        }
        if len(postures) != 1:
            raise ValueError(
                f"candidate group {group_index} contains multiple postures"
            )
        passing = sum(bool(row.get(gate_field, False)) for row in group_rows)
        if passing < minimum_passing_replicas:
            continue
        role_fractions = [
            float(fraction)
            for row in group_rows
            for fraction in row.get("role_contact_fraction", {}).values()
        ]
        worst_role_fraction = min(role_fractions) if role_fractions else 0.0
        mean_cost = sum(float(row["cost"]) for row in group_rows) / len(group_rows)
        representative = min(
            group_rows,
            key=lambda row: (float(row["cost"]), int(row["candidate_index"])),
        )
        ranked.append(
            (
                -passing,
                -worst_role_fraction,
                mean_cost,
                group_index,
                representative,
                {
                    "candidate_group_index": group_index,
                    "replicas": replicas_per_candidate,
                    "physics_contact_gate_passes": passing,
                    "gate_field": gate_field,
                    "worst_role_contact_fraction": worst_role_fraction,
                    "mean_cost": mean_cost,
                },
            )
        )

    ranked.sort(key=lambda item: item[:4])
    selected = []
    seen_postures: set[tuple] = set()
    for *_, representative, diagnostics in ranked:
        posture = (
            tuple(float(value) for value in representative["root_pos_w"]),
            tuple(
                (name, float(value))
                for name, value in sorted(
                    representative["joint_positions_independent"].items()
                )
            ),
        )
        if posture in seen_postures:
            continue
        seen_postures.add(posture)
        selected.append(
            {
                **representative,
                "_source_group_diagnostics": diagnostics,
            }
        )
        if len(selected) == count:
            return selected
    raise ValueError(
        f"only {len(selected)} unique candidate groups satisfy the replicated "
        f"source gate; need {count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument(
        "--require-physics-contact-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--minimum-passing-replicas",
        type=int,
        help=(
            "For a replicated source, require at least this many passing "
            "replicas in every selected candidate group. Defaults to all "
            "source replicas."
        ),
    )
    parser.add_argument(
        "--gate-field",
        choices=("physics_contact_gate", "functional_physics_gate"),
        default="physics_contact_gate",
        help="Candidate record boolean used for shortlist admission.",
    )
    args = parser.parse_args()

    summary = json.loads(args.search_summary.read_text())
    replicas_per_candidate = int(summary.get("replicas_per_candidate", 1))
    rows = summary.get("top_candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("search summary must contain a non-empty top_candidates list")
    if replicas_per_candidate == 1:
        if args.minimum_passing_replicas not in (None, 1):
            raise ValueError(
                "--minimum-passing-replicas must be 1 for a single-replica source"
            )
        selected = _select_candidates(
            rows,
            args.num_candidates,
            require_physics_contact_gate=args.require_physics_contact_gate,
            gate_field=args.gate_field,
        )
    else:
        minimum_passing_replicas = (
            replicas_per_candidate
            if args.minimum_passing_replicas is None
            else args.minimum_passing_replicas
        )
        selected = _select_replicated_candidate_groups(
            rows,
            args.num_candidates,
            replicas_per_candidate=replicas_per_candidate,
            minimum_passing_replicas=minimum_passing_replicas,
            gate_field=args.gate_field,
        )

    source_bank_path = summary.get("candidate_bank")
    source_bank = None
    if source_bank_path is not None:
        source_bank_path = Path(source_bank_path)
        if not source_bank_path.is_absolute():
            source_bank_path = Path.cwd() / source_bank_path
        source_bank = json.loads(source_bank_path.read_text())

    candidates = []
    for rank, row in enumerate(selected):
        candidate_index = int(row["candidate_index"])
        candidate = {
            "root_pos_w": row["root_pos_w"],
            "joint_positions_independent": row["joint_positions_independent"],
            "label": (
                f"source_candidate_{candidate_index}:"
                f"{row.get('label', 'physics_search')}:shortlist_rank={rank}"
            ),
            "source_candidate_index": candidate_index,
            "source_cost": float(row["cost"]),
            "source_physics_contact_gate": bool(row["physics_contact_gate"]),
            "source_selected_gate_field": args.gate_field,
            "source_selected_gate_pass": bool(row[args.gate_field]),
        }
        if "_source_group_diagnostics" in row:
            candidate["source_group_diagnostics"] = row[
                "_source_group_diagnostics"
            ]
        if source_bank is not None:
            bank_rows = source_bank.get("candidates", [])
            if candidate_index >= len(bank_rows):
                raise ValueError(
                    f"candidate_index {candidate_index} is outside source bank"
                )
            bank_row = bank_rows[candidate_index]
            if bank_row["root_pos_w"] != row["root_pos_w"]:
                raise ValueError(
                    f"source bank root mismatch for candidate {candidate_index}"
                )
            if (
                bank_row["joint_positions_independent"]
                != row["joint_positions_independent"]
            ):
                raise ValueError(
                    f"source bank joint mismatch for candidate {candidate_index}"
                )
            candidate["source_bank_diagnostics"] = {
                key: value
                for key, value in bank_row.items()
                if key
                not in {
                    "root_pos_w",
                    "joint_positions_independent",
                    "label",
                }
            }
        candidates.append(candidate)

    output = {
        "schema_version": 1,
        "status": "UNPROMOTED_CANDIDATE_SHORTLIST",
        "source": str(args.search_summary),
        "source_sha256": _sha256(args.search_summary),
        "source_candidate_bank": (
            None if source_bank_path is None else str(source_bank_path)
        ),
        "require_physics_contact_gate": args.require_physics_contact_gate,
        "gate_field": args.gate_field,
        "source_replicas_per_candidate": replicas_per_candidate,
        "minimum_passing_replicas": (
            1
            if replicas_per_candidate == 1
            else (
                replicas_per_candidate
                if args.minimum_passing_replicas is None
                else args.minimum_passing_replicas
            )
        ),
        "candidate_indices": [
            int(row["candidate_index"]) for row in selected
        ],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_indices": output["candidate_indices"],
                "num_candidates": len(candidates),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
