#!/usr/bin/env python3
"""Prepare per-photo local-q grids for an unlocked G20 mimic follower."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEMANTIC_ORDER = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-joint", required=True)
    parser.add_argument("--parent-joint", required=True, choices=SEMANTIC_ORDER)
    parser.add_argument("--parent-dataset", type=Path, required=True)
    parser.add_argument("--parent-matches", type=Path, required=True)
    parser.add_argument("--base-pose", type=Path, required=True)
    parser.add_argument("--render-urdf", type=Path, required=True)
    parser.add_argument("--production-urdf", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--q-lower", type=float, required=True)
    parser.add_argument("--q-upper", type=float, required=True)
    parser.add_argument("--q-step", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_single_pose(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if len(payload) != 1:
            raise ValueError("--base-pose must contain exactly one named pose")
        payload = next(iter(payload.values()))
    if not isinstance(payload, list) or len(payload) != len(SEMANTIC_ORDER):
        raise ValueError("base pose must resolve to 16 semantic joint values")
    return [float(value) for value in payload]


def joint_info(path: Path, joint_name: str) -> tuple[tuple[float, float], dict | None]:
    root = ET.parse(path).getroot()
    joint = next(
        (item for item in root.findall("joint") if item.get("name") == joint_name),
        None,
    )
    if joint is None:
        raise ValueError(f"{joint_name} not found in {path}")
    limit = joint.find("limit")
    if limit is None:
        raise ValueError(f"{joint_name} has no finite limit")
    mimic = joint.find("mimic")
    mimic_info = None if mimic is None else {
        "joint": str(mimic.get("joint")),
        "multiplier": float(mimic.get("multiplier", "1")),
        "offset": float(mimic.get("offset", "0")),
    }
    return (
        (float(limit.get("lower")), float(limit.get("upper"))),
        mimic_info,
    )


def q_grid(lower: float, upper: float, step: float) -> list[float]:
    if upper <= lower:
        raise ValueError("--q-upper must exceed --q-lower")
    if step <= 0.0:
        raise ValueError("--q-step must be positive")
    count = int(math.floor((upper - lower) / step + 1.0e-9))
    values = [round(lower + index * step, 10) for index in range(count + 1)]
    if values[-1] < upper - 1.0e-9:
        values.append(float(upper))
    else:
        values[-1] = float(upper)
    return values


def safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def main() -> int:
    args = parse_args()
    for path in (
        args.parent_dataset, args.parent_matches, args.base_pose,
        args.render_urdf, args.production_urdf, args.camera, args.intrinsics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent_dataset = json.loads(
        args.parent_dataset.read_text(encoding="utf-8")
    )
    parent_matches = json.loads(
        args.parent_matches.read_text(encoding="utf-8")
    )
    if parent_dataset.get("joint") != args.parent_joint:
        raise ValueError("parent dataset joint does not match --parent-joint")
    if parent_matches.get("dataset_sha256") != sha256(args.parent_dataset):
        raise ValueError("parent matches do not belong to parent dataset")
    matches_by_id = {
        item["sample_id"]: item for item in parent_matches.get("matches", [])
    }
    missing_matches = [
        sample["sample_id"] for sample in parent_dataset["samples"]
        if sample["sample_id"] not in matches_by_id
        or matches_by_id[sample["sample_id"]].get("q_urdf_rad") is None
    ]
    if missing_matches:
        raise ValueError(
            "parent samples are not all matched: " + ", ".join(missing_matches)
        )

    production_limit, production_mimic = joint_info(
        args.production_urdf, args.follower_joint
    )
    render_limit, render_mimic = joint_info(
        args.render_urdf, args.follower_joint
    )
    if production_mimic is None:
        raise ValueError("production follower has no mimic tag")
    if production_mimic["joint"] != args.parent_joint:
        raise ValueError("production mimic parent does not match --parent-joint")
    if render_mimic is not None:
        raise ValueError("diagnostic render URDF still contains follower mimic")
    tolerance = 1.0e-9
    if args.q_lower < render_limit[0] - tolerance:
        raise ValueError("q grid lower bound is outside render URDF limit")
    if args.q_upper > render_limit[1] + tolerance:
        raise ValueError("q grid upper bound is outside render URDF limit")

    values = q_grid(args.q_lower, args.q_upper, args.q_step)
    base = load_single_pose(args.base_pose)
    parent_index = SEMANTIC_ORDER.index(args.parent_joint)
    poses: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    for sample in parent_dataset["samples"]:
        sample_id = str(sample["sample_id"])
        parent_q = float(matches_by_id[sample_id]["q_urdf_rad"])
        semantic = list(base)
        semantic[parent_index] = parent_q
        candidates: list[dict[str, Any]] = []
        for index, q in enumerate(values):
            sign = "p" if q >= 0.0 else "m"
            token = f"{abs(q):.4f}".replace(".", "p")
            name = (
                f"{args.follower_joint}__{safe_token(sample_id)}__"
                f"q_{index:03d}_{sign}{token}"
            )
            poses[name] = {
                "semantic": semantic,
                "joint_overrides": {args.follower_joint: q},
            }
            candidates.append({
                "index": index,
                "name": name,
                "q_urdf_rad": q,
                "q_urdf_deg": math.degrees(q),
                "parent_q_urdf_rad": parent_q,
                "outside_production_og_limit": not (
                    production_limit[0] - tolerance
                    <= q <= production_limit[1] + tolerance
                ),
            })
        output_sample = dict(sample)
        output_sample["parent_joint"] = args.parent_joint
        output_sample["parent_q_urdf_rad"] = parent_q
        output_sample["candidates"] = candidates
        output_sample["match_status"] = "unreviewed"
        samples.append(output_sample)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    poses_path = args.out_dir / "poses_urdf_local_q.json"
    dataset_path = args.out_dir / "manual_match_dataset.json"
    poses_path.write_text(json.dumps(poses, indent=2) + "\n", encoding="utf-8")
    dataset = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "per_photo_parent_fixed_mimic_follower_local_q_registration",
        "status": "unreviewed_candidate_only",
        "joint": args.follower_joint,
        "parent_joint": args.parent_joint,
        "candidate_mode": "per_sample",
        "coordinate_contract": {
            "ground_truth_coordinate": "OG URDF follower local q in radians",
            "parent_coordinate": (
                "parent q is fixed per photo to its independently human-matched "
                "OG URDF local q"
            ),
            "raw_coordinate": "physical stable SDK readback [0,255]",
            "selection_rule": (
                "human chooses the same-view unlocked-follower render that best "
                "matches each formal D435 photo"
            ),
            "diagnostic_asset_rule": (
                "only the target mimic tag is removed and only its diagnostic "
                "limit is extended; production OG URDF is unchanged"
            ),
        },
        "production_urdf": str(args.production_urdf.resolve()),
        "production_urdf_sha256": sha256(args.production_urdf),
        "production_urdf_limit_rad": list(production_limit),
        "urdf_limit_rad": list(production_limit),
        "production_mimic": production_mimic,
        "render_urdf": str(args.render_urdf.resolve()),
        "render_urdf_sha256": sha256(args.render_urdf),
        "urdf": str(args.render_urdf.resolve()),
        "urdf_sha256": sha256(args.render_urdf),
        "render_urdf_limit_rad": list(render_limit),
        "render_mimic": render_mimic,
        "diagnostic_extended_limit_only": True,
        "camera": str(args.camera.resolve()),
        "camera_sha256": sha256(args.camera),
        "intrinsics": str(args.intrinsics.resolve()),
        "intrinsics_sha256": sha256(args.intrinsics),
        "base_pose": str(args.base_pose.resolve()),
        "base_pose_sha256": sha256(args.base_pose),
        "poses_json": str(poses_path.resolve()),
        "semantic_order": list(SEMANTIC_ORDER),
        "parent_semantic_index": parent_index,
        "q_grid_rad": {
            "lower": args.q_lower,
            "upper": args.q_upper,
            "step": args.q_step,
            "margin_below_production_lower": production_limit[0] - args.q_lower,
            "margin_above_production_upper": args.q_upper - production_limit[1],
        },
        "candidate_count_per_sample": len(values),
        "total_render_pose_count": len(poses),
        "sample_count": len(samples),
        "samples": samples,
        "parent_dataset": str(args.parent_dataset.resolve()),
        "parent_dataset_sha256": sha256(args.parent_dataset),
        "parent_matches": str(args.parent_matches.resolve()),
        "parent_matches_sha256": sha256(args.parent_matches),
    }
    dataset_path.write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "dataset": str(dataset_path),
        "poses": str(poses_path),
        "samples": len(samples),
        "candidates_per_sample": len(values),
        "total_render_poses": len(poses),
        "q_grid_rad": [args.q_lower, args.q_upper, args.q_step],
        "production_limit_rad": list(production_limit),
        "production_mimic": production_mimic,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
