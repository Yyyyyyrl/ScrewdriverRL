#!/usr/bin/env python3
"""Apply the release geometry gate to settled posture-search replicas.

The physics search ranks candidates with imported PhysX colliders and contact
forces.  The release gate additionally checks the corresponding settled state
against the source URDF meshes.  This tool joins those two pieces of evidence
without treating the controller target itself as a collision-free pose: a
penetrating controller target is allowed to create preload, while the settled
state, non-distal links, PIP margins, and self-collisions remain gated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    UrdfGeometry,
)
import tools.validate_linker_l20_screwdriver_topdown as nominal_validator  # noqa: E402
import tools.validate_linker_l20_screwdriver_topdown_diameter_bank as diameter_validator  # noqa: E402


MANIFEST = REPO_ROOT / "assets/screwdriver/topdown_variants/manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_for_diameter(diameter_mm: int) -> tuple[int, Path]:
    manifest = json.loads(MANIFEST.read_text())
    diameter_m = float(diameter_mm) / 1000.0
    matches = [
        row
        for row in manifest["variants"]
        if abs(float(row["diameter"]) - diameter_m) <= 1.0e-9
    ]
    if len(matches) != 1:
        supported = sorted(round(float(row["diameter"]) * 1000) for row in manifest["variants"])
        raise ValueError(
            f"unsupported diameter {diameter_mm} mm; expected one of {supported}"
        )
    variant = matches[0]
    asset = MANIFEST.parent.parent / variant["file"]
    return int(variant["bucket"]), asset


def _select_rows(
    search_summary: dict,
    candidate_group_indices: set[int] | None,
) -> list[dict]:
    rows = search_summary.get("top_candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("search summary must contain a non-empty top_candidates list")
    if candidate_group_indices is None:
        return rows
    selected = [
        row
        for row in rows
        if int(row["candidate_group_index"]) in candidate_group_indices
    ]
    present = {int(row["candidate_group_index"]) for row in selected}
    missing = sorted(candidate_group_indices - present)
    if missing:
        raise ValueError(f"candidate groups not found in search summary: {missing}")
    return selected


def _settled_posture(row: dict) -> dict:
    required = (
        "root_pos_w",
        "root_quat_wxyz",
        "settled_joint_positions_independent",
        "settled_screwdriver_joint_positions",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(
            f"candidate {row.get('candidate_index')} is missing settled fields: {missing}"
        )
    return {
        "root_pos_w": row["root_pos_w"],
        "root_quat_wxyz": row["root_quat_wxyz"],
        "joint_positions_independent": row["settled_joint_positions_independent"],
        "screwdriver_joint_positions": row["settled_screwdriver_joint_positions"],
    }


def _minimum_link_distance(validation: dict) -> float:
    links = validation["non_distal_clearance"]["links"]
    return min(
        float(row["minimum_surface_distance_m"]) for row in links.values()
    )


def _maximum_link_penetration(validation: dict) -> float:
    links = validation["non_distal_clearance"]["links"]
    return max(float(row["maximum_penetration_m"]) for row in links.values())


def _validation_digest(validation: dict) -> dict:
    pose = validation["pose_and_joint_limits"]
    fingertip = validation["fingertip_contacts"]["fingers"]
    non_distal_links = validation["non_distal_clearance"]["links"]
    return {
        "pass": bool(validation["pass"]),
        "minimum_scoped_joint_limit_margin_rad": float(
            pose["minimum_scoped_joint_limit_margin_rad"]
        ),
        "minimum_non_distal_surface_distance_m": _minimum_link_distance(validation),
        "maximum_non_distal_penetration_m": _maximum_link_penetration(validation),
        "minimum_unfiltered_self_clearance_m": float(
            validation["unfiltered_self_collision"][
                "minimum_unfiltered_pair_clearance_m"
            ]
        ),
        "component_pass": {
            "pose_and_joint_limits": bool(pose["pass"]),
            "non_distal_clearance": bool(
                validation["non_distal_clearance"]["pass"]
            ),
            "unfiltered_self_collision": bool(
                validation["unfiltered_self_collision"]["pass"]
            ),
        },
        "failing_non_distal_links": {
            name: {
                "minimum_surface_distance_m": float(
                    result["minimum_surface_distance_m"]
                ),
                "maximum_penetration_m": float(result["maximum_penetration_m"]),
            }
            for name, result in non_distal_links.items()
            if not result["pass"]
        },
        "fingertip_mesh_diagnostic": {
            finger: {
                "pass_under_nominal_contact_thresholds": bool(result["pass"]),
                "penetration_m": float(result["penetration_m"]),
                "surface_distance_m": float(result["surface_distance_m"]),
            }
            for finger, result in fingertip.items()
        },
    }


def _summarize_groups(rows: list[dict]) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["candidate_group_index"])].append(row)
    summaries = []
    for group_index in sorted(groups):
        replicas = groups[group_index]
        combined = sum(bool(row["combined_release_gate_pass"]) for row in replicas)
        physics = sum(bool(row["physics_contact_gate"]) for row in replicas)
        geometry = sum(bool(row["settled_release_geometry_pass"]) for row in replicas)
        summaries.append(
            {
                "candidate_group_index": group_index,
                "label": replicas[0]["label"],
                "replicas": len(replicas),
                "physics_contact_gate_passes": physics,
                "settled_release_geometry_passes": geometry,
                "combined_release_gate_passes": combined,
                "promotion_replica_gate_pass": combined == len(replicas),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diameter-mm", type=int, default=64)
    parser.add_argument(
        "--max-physics-non-fingertip-force-n",
        type=float,
        default=0.05,
        help="Must match the search/release proximal-force threshold.",
    )
    parser.add_argument(
        "--candidate-group-index",
        type=int,
        action="append",
        default=None,
        help="Audit only this group; repeat to select more than one group.",
    )
    parser.add_argument(
        "--physics-gate-field",
        choices=("physics_contact_gate", "functional_physics_gate"),
        default="physics_contact_gate",
        help="Search-record boolean combined with settled release geometry.",
    )
    args = parser.parse_args()

    source = json.loads(args.search_summary.read_text())
    selected_groups = (
        None
        if args.candidate_group_index is None
        else set(args.candidate_group_index)
    )
    rows = _select_rows(source, selected_groups)
    variant_index, screwdriver_asset = _variant_for_diameter(args.diameter_mm)
    hand_model = UrdfGeometry(nominal_validator.HAND_URDF)
    screwdriver_model = UrdfGeometry(screwdriver_asset)

    audited_rows = []
    for ordinal, row in enumerate(rows):
        validation = diameter_validator._validate_posture(
            hand_model,
            screwdriver_model,
            _settled_posture(row),
            require_fingertip_contact=False,
            settled_state=True,
        )
        proximal_force_max_n = max(
            map(float, row["proximal_force_max_n"].values()),
            default=0.0,
        )
        authority_gate = diameter_validator._non_distal_authority_gate(
            validation,
            proximal_force_max_n,
            args.max_physics_non_fingertip_force_n,
        )
        physics_pass = bool(row.get(args.physics_gate_field, False))
        geometry_pass = bool(authority_gate["pass"])
        audited_rows.append(
            {
                "source_row_ordinal": ordinal,
                "candidate_group_index": int(row["candidate_group_index"]),
                "candidate_index": int(row["candidate_index"]),
                "replica_index": int(row["replica_index"]),
                "label": row["label"],
                "physics_contact_gate": physics_pass,
                "physics_gate_field": args.physics_gate_field,
                "source_strict_all_finger_physics_gate": bool(
                    row.get("physics_contact_gate", False)
                ),
                "settled_release_geometry_pass": geometry_pass,
                "combined_release_gate_pass": physics_pass and geometry_pass,
                "non_distal_authority_gate": authority_gate,
                "settled_validation": _validation_digest(validation),
            }
        )

    group_summaries = _summarize_groups(audited_rows)
    output = {
        "schema_version": 1,
        "search_summary": str(args.search_summary),
        "search_summary_sha256": _sha256(args.search_summary),
        "diameter_mm": args.diameter_mm,
        "geometry_variant_index": variant_index,
        "physics_gate_field": args.physics_gate_field,
        "screwdriver_asset": str(screwdriver_asset),
        "screwdriver_asset_sha256": _sha256(screwdriver_asset),
        "validation_model": {
            "controller_target_collision_free_required": False,
            "settled_fingertip_mesh_contact_enforced": False,
            "settled_physx_role_contact_required": True,
            "settled_non_distal_clearance_required_m": (
                diameter_validator.DIAMETER_BANK_MIN_NON_DISTAL_CLEARANCE_M
            ),
            "maximum_reconcilable_offline_penetration_m": (
                diameter_validator.MAX_OFFLINE_PHYSICS_DISCREPANCY_PENETRATION_M
            ),
            "maximum_physics_non_fingertip_force_n": (
                args.max_physics_non_fingertip_force_n
            ),
            "settled_pip_joint_margin_required_rad": nominal_validator.MIN_JOINT_MARGIN_RAD,
            "settled_unfiltered_self_collision_clearance_required_m": (
                nominal_validator.MIN_SELF_CLEARANCE_M
            ),
        },
        "rows": audited_rows,
        "groups": group_summaries,
        "all_selected_groups_pass": all(
            row["promotion_replica_gate_pass"] for row in group_summaries
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "all_selected_groups_pass": output["all_selected_groups_pass"],
                "groups": group_summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    raise SystemExit(0 if output["all_selected_groups_pass"] else 1)


if __name__ == "__main__":
    main()
