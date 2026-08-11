#!/usr/bin/env python3
"""Build an isolated G20 raw <-> OG-URDF-local-q calibration candidate.

The source of truth is the set of 21 manually reviewed archive manifests under
``records/g20_urdf_local_q_visual_registration_20260804``.  The runtime overlay
contains only the 16 independently actuated SDK joints.  A separate provenance
sidecar retains all 21 results, including fitted relationships for the five
mechanical follower joints.

This tool never edits a production calibration or URDF and never activates the
candidate in a live process.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from screwdriver_rl.deploy import linker_sdk_map as sdkmap  # noqa: E402


DEFAULT_ARCHIVE_ROOT = (
    ROOT / "records" / "g20_urdf_local_q_visual_registration_20260804"
)
DEFAULT_OUTPUT = (
    ROOT
    / "assets"
    / "calibrations"
    / "linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805.json"
)
DEFAULT_PROVENANCE = DEFAULT_OUTPUT.with_name(
    DEFAULT_OUTPUT.stem + "_provenance.json"
)
EXPECTED_PRODUCTION_URDF_SHA256 = (
    "697fe08490c957e4c9fa595ac0750cc80512b1f28dd3f53382db0d24ed202b2f"
)
FOLLOWERS = {
    "index_dip": "index_pip",
    "middle_dip": "middle_pip",
    "ring_dip": "ring_pip",
    "pinky_dip": "pinky_pip",
    "thumb_ip": "thumb_mcp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def interpolate(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for index, (left, right) in enumerate(zip(xs, xs[1:])):
        if left <= x <= right:
            fraction = (x - left) / (right - left)
            return float(ys[index] + fraction * (ys[index + 1] - ys[index]))
    raise AssertionError("unreachable interpolation interval")


def coalesce_flat_q_knots(
    raw: Sequence[float], q: Sequence[float]
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    """Collapse consecutive equal-q grid selections to one mean-raw knot."""

    out_raw: list[float] = []
    out_q: list[float] = []
    groups: list[dict[str, Any]] = []
    start = 0
    while start < len(q):
        stop = start + 1
        while stop < len(q) and q[stop] == q[start]:
            stop += 1
        members = [float(value) for value in raw[start:stop]]
        representative = sum(members) / len(members)
        out_raw.append(representative)
        out_q.append(float(q[start]))
        if len(members) > 1:
            groups.append(
                {
                    "q_urdf_local_rad": float(q[start]),
                    "source_raw": members,
                    "representative_raw_mean": representative,
                    "reason": "manual q-grid quantization plateau",
                }
            )
        start = stop
    return out_raw, out_q, groups


def monotonic_direction(values: Sequence[float]) -> str:
    if all(left < right for left, right in zip(values, values[1:])):
        return "increasing"
    if all(left > right for left, right in zip(values, values[1:])):
        return "decreasing"
    raise ValueError("values are not strictly monotonic")


def fit_follower(
    follower: str,
    parent: str,
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    follower_lut = summaries[follower]["physical_lut_urdf_local_q"]
    parent_lut = summaries[parent]["physical_lut_urdf_local_q"]
    parent_q = [
        interpolate(raw, parent_lut["raw"], parent_lut["q_urdf_local_rad"])
        for raw in follower_lut["raw"]
    ]
    follower_q = [float(value) for value in follower_lut["q_urdf_local_rad"]]
    count = len(parent_q)
    mean_x = sum(parent_q) / count
    mean_y = sum(follower_q) / count
    denominator = sum((value - mean_x) ** 2 for value in parent_q)
    multiplier = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(parent_q, follower_q)
    ) / denominator
    offset = mean_y - multiplier * mean_x
    residuals = [
        y - (multiplier * x + offset) for x, y in zip(parent_q, follower_q)
    ]
    rmse = math.sqrt(sum(value * value for value in residuals) / count)
    origin_multiplier = sum(x * y for x, y in zip(parent_q, follower_q)) / sum(
        x * x for x in parent_q
    )
    origin_residuals = [
        y - origin_multiplier * x for x, y in zip(parent_q, follower_q)
    ]
    origin_rmse = math.sqrt(
        sum(value * value for value in origin_residuals) / count
    )
    return {
        "joint": follower,
        "source": parent,
        "sample_count": count,
        "parent_q_method": "linear interpolation of independently matched parent raw-to-q LUT",
        "affine_fit": {
            "formula": "q_follower = multiplier * q_source + offset",
            "multiplier": multiplier,
            "offset": offset,
            "rmse_rad": rmse,
            "max_abs_residual_rad": max(abs(value) for value in residuals),
        },
        "through_origin_diagnostic": {
            "multiplier": origin_multiplier,
            "offset": 0.0,
            "rmse_rad": origin_rmse,
        },
        "status": "candidate_for_sim_mimic_review_not_applied_to_production_urdf",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()

    manifests = sorted(
        args.archive_root.glob("**/archive/five_point_spotcheck_manifest.json")
    )
    if len(manifests) != 21:
        raise ValueError(f"expected 21 archived joints, found {len(manifests)}")

    by_joint: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    archive_rows: dict[str, dict[str, Any]] = {}
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        joint = str(manifest["joint"])
        if joint in by_joint:
            raise ValueError(f"duplicate archived joint {joint}")
        if manifest.get("status") != "visual_review_pass_candidate_only_not_promoted":
            raise ValueError(f"{joint}: visual review is not approved candidate-only")
        if manifest.get("production_og_urdf_sha256") != EXPECTED_PRODUCTION_URDF_SHA256:
            raise ValueError(f"{joint}: production OG URDF digest drift")
        summary_path = resolve_recorded_path(str(manifest["summary"]))
        if sha256(summary_path) != manifest["summary_sha256"]:
            raise ValueError(f"{joint}: summary digest mismatch")
        summary = read_json(summary_path)
        if summary.get("joint") != joint:
            raise ValueError(f"{joint}: summary joint mismatch")
        if summary["review"]["monotonic_violation_count"] != 0:
            raise ValueError(f"{joint}: manual summary has monotonic violations")
        by_joint[joint] = manifest
        summaries[joint] = summary
        archive_rows[joint] = {
            "archive_manifest": str(manifest_path.relative_to(ROOT)),
            "archive_manifest_sha256": sha256(manifest_path),
            "summary": str(summary_path.relative_to(ROOT)),
            "summary_sha256": sha256(summary_path),
            "camera": str(resolve_recorded_path(str(manifest["camera"])).relative_to(ROOT)),
            "camera_sha256": manifest["camera_sha256"],
            "unique_stable_raw_count": summary["review"]["unique_stable_raw_count"],
            "observed_q_range_rad": [
                summary["observed_absolute_urdf_local_q_range"]["q_min_rad"],
                summary["observed_absolute_urdf_local_q_range"]["q_max_rad"],
            ],
            "production_og_limit_rad": summary["original_og_limit_assessment"][
                "declared_limit_rad"
            ],
            "unique_raw_count_outside_production_limit": summary[
                "original_og_limit_assessment"
            ]["unique_raw_count_outside"],
            "affine_diagnostic_rmse_rad": summary["affine_diagnostic"]["rmse_rad"],
        }

    expected_names = {joint.name for joint in sdkmap.DEFAULT_JOINTS} | set(FOLLOWERS)
    if set(by_joint) != expected_names:
        raise ValueError(
            f"archive joint set mismatch: missing={sorted(expected_names - set(by_joint))}, "
            f"extra={sorted(set(by_joint) - expected_names)}"
        )

    overlay_joints: dict[str, dict[str, Any]] = {}
    runtime_luts: dict[str, dict[str, Any]] = {}
    for spec in sdkmap.DEFAULT_JOINTS:
        source = summaries[spec.name]["physical_lut_urdf_local_q"]
        original_raw = [float(value) for value in source["raw"]]
        original_q = [float(value) for value in source["q_urdf_local_rad"]]
        raw, q, coalesced = coalesce_flat_q_knots(original_raw, original_q)
        direction = monotonic_direction(q)
        if any(left >= right for left, right in zip(raw, raw[1:])):
            raise ValueError(f"{spec.name}: runtime raw knots are not increasing")
        entry = {
            "slot": spec.slot,
            "lo": min(q),
            "hi": max(q),
            "flip": False,
            "physical_lut": {"raw": raw, "rad": q},
        }
        overlay_joints[spec.name] = entry
        runtime_luts[spec.name] = {
            "direction_q_with_increasing_raw": direction,
            "source_raw": original_raw,
            "source_q_urdf_local_rad": original_q,
            "runtime_raw": raw,
            "runtime_q_urdf_local_rad": q,
            "coalesced_flat_q_groups": coalesced,
        }

    overlay = {
        "version": 4,
        "note": (
            "CANDIDATE ONLY, not auto-loaded. LHT20-010-415-L-B-1-D. "
            "SDK stable raw to manually registered OG URDF local joint q, 2026-08-05. "
            "Sixteen actuated joints only; five follower joints are in the provenance sidecar."
        ),
        "joints": overlay_joints,
    }
    # Validate with the same strict loader used by deployment, without activating it.
    sdkmap.build_joint_table(overlay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(overlay, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_digest = sha256(args.output)
    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_not_promoted",
        "hand": {
            "model": "LinkerHand G20 left",
            "serial": "LHT20-010-415-L-B-1-D",
        },
        "coordinate_contract": {
            "input": "physical SDK stable readback raw in [0,255]",
            "output": "linker_hand_l20_OG URDF local revolute joint q in radians",
            "association": "human same-view matching of formal D435 photo to Isaac render",
        },
        "production_og_urdf": "assets/linker_hand_l20_OG/linkerhand_l20_left.urdf",
        "production_og_urdf_sha256": EXPECTED_PRODUCTION_URDF_SHA256,
        "candidate_overlay": str(args.output.relative_to(ROOT)),
        "candidate_overlay_sha256": output_digest,
        "active_joint_count": len(overlay_joints),
        "follower_joint_count": len(FOLLOWERS),
        "archive_joint_count": len(by_joint),
        "active_runtime_luts": runtime_luts,
        "follower_mimic_candidates": [
            fit_follower(follower, parent, summaries)
            for follower, parent in FOLLOWERS.items()
        ],
        "joint_archives": archive_rows,
        "promotion_gates": [
            "candidate overlay static and round-trip tests pass",
            "same-view Isaac versus physical still-image A/B passes at multiple poses",
            "short continuous-motion A/B shows no direction, coupling, or self-collision anomaly",
            "five follower mimic fits are reviewed before any URDF or semantic schema update",
            "production config and URDF digests are explicitly changed only after approval",
        ],
    }
    args.provenance.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output.relative_to(ROOT)} sha256={output_digest}")
    print(f"wrote {args.provenance.relative_to(ROOT)} sha256={sha256(args.provenance)}")
    print("validated 16 active LUTs + archived 5 follower calibrations")


if __name__ == "__main__":
    main()
