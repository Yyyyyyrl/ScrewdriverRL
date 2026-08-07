#!/usr/bin/env python3
"""Re-evaluate a posture-search artifact with the functional contact gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from screwdriver_rl.utils.linker_topdown_contact_gate import (  # noqa: E402
    CRITICAL_ROLE_NAMES,
    FUNCTIONAL_ACTIVE_ROLE_FRACTION,
    FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT,
    FUNCTIONAL_MIN_CRITICAL_FRACTION,
    functional_physics_gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-group-index", type=int, action="append")
    args = parser.parse_args()

    source = json.loads(args.search_summary.read_text())
    rows = source.get("top_candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("search summary must contain top_candidates")
    if args.candidate_group_index is not None:
        selected = set(args.candidate_group_index)
        rows = [
            row
            for row in rows
            if int(row.get("candidate_group_index", 0)) in selected
        ]
        if not rows:
            raise ValueError("selected candidate groups are absent")

    max_drift = float(source["max_zero_action_drift_rad_s"])
    audited = []
    for row in rows:
        result = functional_physics_gate(
            row,
            max_zero_action_drift_rad_s=max_drift,
        )
        audited.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "candidate_group_index": int(
                    row.get("candidate_group_index", 0)
                ),
                "replica_index": int(row.get("replica_index", 0)),
                "strict_all_finger_physics_gate": bool(
                    row.get("physics_contact_gate", False)
                ),
                "functional_physics_gate": bool(result["pass"]),
                "functional_gate_diagnostics": result,
            }
        )

    output = {
        "schema_version": 1,
        "search_summary": str(args.search_summary),
        "search_summary_sha256": _sha256(args.search_summary),
        "gate_spec": {
            "critical_roles": list(CRITICAL_ROLE_NAMES),
            "minimum_critical_role_fraction": (
                FUNCTIONAL_MIN_CRITICAL_FRACTION
            ),
            "active_role_fraction": FUNCTIONAL_ACTIVE_ROLE_FRACTION,
            "minimum_active_role_count": FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT,
            "mean_role_fraction_is_diagnostic_only": True,
            "safety_limits_unchanged_from_strict_gate": True,
        },
        "replicas": len(audited),
        "strict_all_finger_physics_gate_passes": sum(
            row["strict_all_finger_physics_gate"] for row in audited
        ),
        "functional_physics_gate_passes": sum(
            row["functional_physics_gate"] for row in audited
        ),
        "all_functional_physics_gates_pass": all(
            row["functional_physics_gate"] for row in audited
        ),
        "rows": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "replicas": output["replicas"],
                "strict_passes": output[
                    "strict_all_finger_physics_gate_passes"
                ],
                "functional_passes": output[
                    "functional_physics_gate_passes"
                ],
                "all_functional_pass": output[
                    "all_functional_physics_gates_pass"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(
        0 if output["all_functional_physics_gates_pass"] else 1
    )


if __name__ == "__main__":
    main()
