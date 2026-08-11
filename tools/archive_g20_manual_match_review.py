#!/usr/bin/env python3
"""Archive a user-approved G20 five-point visual review as candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--production-urdf", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    review_dir = args.review_dir.resolve()
    archive_dir = (
        args.archive_dir.resolve()
        if args.archive_dir is not None
        else review_dir / "archive"
    )
    source_manifest_path = review_dir / "five_point_spotcheck_manifest.json"
    source_sheet = review_dir / "five_point_spotcheck_contact_sheet.png"
    source_panels = review_dir / "panels"
    for path in (
        source_manifest_path, source_sheet, source_panels,
        args.summary, args.production_urdf,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if archive_dir.exists() and any(archive_dir.iterdir()):
        raise RuntimeError(f"archive directory is not empty: {archive_dir}")

    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    dataset = json.loads(Path(manifest["dataset"]).read_text(encoding="utf-8"))
    if manifest["joint"] != summary["joint"] or manifest["joint"] != dataset["joint"]:
        raise ValueError("joint mismatch across review, summary, and dataset")
    expected_production_hash = manifest.get("production_og_urdf_sha256")
    actual_production_hash = sha256(args.production_urdf)
    if expected_production_hash != actual_production_hash:
        raise RuntimeError("production OG URDF hash changed before archive")

    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_sheet = archive_dir / source_sheet.name
    shutil.copy2(source_sheet, archived_sheet)
    archived_panels = archive_dir / "panels"
    shutil.copytree(source_panels, archived_panels)

    review = summary["review"]
    observed = summary["observed_absolute_urdf_local_q_range"]
    limit = summary["original_og_limit_assessment"]
    outside_count = int(limit["photo_count_outside"])
    q_grid = dataset["q_grid_rad"]
    selected_q = [float(row["q_urdf_local_rad"]) for row in manifest["rows"]]
    grid_edge_saturated = any(
        abs(q - float(q_grid["lower"])) <= 1.0e-9
        or abs(q - float(q_grid["upper"])) <= 1.0e-9
        for q in selected_q
    )
    has_fault_endpoint = any(
        "fault_endpoint" in str(row.get("sample_id", ""))
        for row in manifest["rows"]
    )
    focus_finger = manifest.get("focus_finger", "none")
    hidden_body_names = list(manifest.get("hidden_body_names", []))

    criteria = [
        "Projected target-joint rotation direction and ordering agree at all five checkpoints.",
        "Normal operating samples and physical-limit samples form a continuous monotonic mapping.",
        "No sign inversion, discontinuity, or diagnostic-grid-edge saturation is present.",
        "The medium-gray Isaac background preserves a clear white-hand silhouette.",
    ]
    if dataset.get("candidate_mode") == "per_sample":
        criteria.insert(
            2,
            "Each follower render uses the independently matched parent local q for the same physical photo.",
        )
    if focus_finger != "none":
        criteria.append(
            f"Focused rendering retains the palm and complete {focus_finger} chain while hiding only non-target bodies."
        )

    limitations = [
        "This pass validates candidate calibration evidence, not production runtime deployment.",
        (
            f"Manual matching resolution is {float(review['candidate_step_rad']):.2f} rad, "
            f"implying at least +/-{float(review['quantization_half_step_rad']):.2f} rad quantization uncertainty."
        ),
    ]
    if has_fault_endpoint:
        limitations.append(
            "The fault-trigger endpoint is offline evidence only and must not become a runtime command target."
        )
    if outside_count:
        limitations.append(
            f"{outside_count} matched photo(s) lie outside the original production OG limit."
        )
    if dataset.get("candidate_mode") == "per_sample":
        limitations.append(
            "The target mimic was removed only in an isolated diagnostic URDF; the production OG mimic remains unchanged."
        )
    if focus_finger != "none":
        limitations.append(
            "Focus changes USD visibility only; it does not alter joint states, URDF geometry, physics, or runtime behavior."
        )

    manifest["status"] = "visual_review_pass_candidate_only_not_promoted"
    manifest["contact_sheet"] = str(archived_sheet)
    for row in manifest["rows"]:
        row["panel"] = str(
            archived_panels / Path(str(row["panel"])).name
        )
    manifest["visual_review"] = {
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "result": "pass",
        "scope": f"{manifest['joint']} raw-to-absolute-OG-URDF-local-q candidate mapping",
        "criteria": criteria,
        "limitations": limitations,
        "grid_edge_saturated": grid_edge_saturated,
        "production_og_urdf_sha256_verified": actual_production_hash,
    }
    archived_manifest = archive_dir / source_manifest_path.name
    archived_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    has_parent = any(
        row.get("parent_q_urdf_rad") is not None for row in manifest["rows"]
    )
    table_rows = "\n".join(
        (
            f"| {int(row['stable_readback_raw'])} | "
            f"{float(row['q_urdf_local_rad']):.3f} | "
            f"{float(row['q_urdf_local_deg']):.2f} | "
            f"{float(row['parent_q_urdf_rad']):.3f} |"
            if has_parent
            else
            f"| {int(row['stable_readback_raw'])} | "
            f"{float(row['q_urdf_local_rad']):.3f} | "
            f"{float(row['q_urdf_local_deg']):.2f} |"
        )
        for row in manifest["rows"]
    )
    parent_header = dataset.get("parent_joint", "Parent q")
    table_header = (
        f"| Stable raw | OG URDF local q (rad) | Degrees | Fixed {parent_header} q (rad) |\n"
        "|---:|---:|---:|---:|"
        if has_parent
        else
        "| Stable raw | OG URDF local q (rad) | Degrees |\n"
        "|---:|---:|---:|"
    )
    readme = f"""# {manifest['joint']} five-point same-view spot check

Status: **visual review passed for the candidate mapping; not promoted**.
This archive does not modify runtime config or the production OG URDF.

{table_header}
{table_rows}

Full manual review coverage: {int(review['matched_count'])}/{int(review['sample_count'])}
photos matched, {int(review['unique_stable_raw_count'])} unique stable raw values,
{int(review['monotonic_violation_count'])} monotonicity violations, and
grid-edge saturation = {str(grid_edge_saturated).lower()}. The observed absolute
local-q range is {float(observed['q_min_rad']):.2f}--{float(observed['q_max_rad']):.2f} rad.
{outside_count} matched photo(s) exceed the original OG limit
[{float(limit['declared_limit_rad'][0]):.2f}, {float(limit['declared_limit_rad'][1]):.2f}] rad.

Candidate step is {float(review['candidate_step_rad']):.2f} rad. Production OG
SHA-256 was reverified at archive time as `{actual_production_hash}`. The
candidate remains unpromoted until all 21 joints complete the same review.

- Contact sheet: `{archived_sheet.name}`
- Manifest: `{archived_manifest.name}`
"""
    if dataset.get("candidate_mode") == "per_sample":
        readme += (
            "\nFollower-specific method: each photo fixes the independently matched "
            f"`{dataset['parent_joint']}` local q and scans only `{dataset['joint']}`. "
            "The mimic tag is removed only in the isolated diagnostic URDF.\n"
        )
    if focus_finger != "none":
        readme += (
            f"\nIsaac visual focus: `{focus_finger}`; hidden non-target body count: "
            f"`{len(hidden_body_names)}`. Palm and target-finger links remain visible.\n"
        )
    (archive_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({
        "archive": str(archive_dir),
        "joint": manifest["joint"],
        "status": manifest["status"],
        "contact_sheet": str(archived_sheet),
        "panels": len(manifest["rows"]),
        "production_og_sha256": actual_production_hash,
        "grid_edge_saturated": grid_edge_saturated,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
