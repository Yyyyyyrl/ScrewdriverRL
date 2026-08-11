#!/usr/bin/env python3
"""Prepare the formal thumb-roll photos and a dense OG-URDF local-q grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEMANTIC_ORDER = (
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
    "thumb_cmc_yaw",
    "thumb_cmc_roll",
    "thumb_cmc_pitch",
    "thumb_mcp",
)
TARGET_JOINT = "thumb_cmc_roll"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base-pose", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
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
        values = next(iter(payload.values()))
    else:
        values = payload
    if not isinstance(values, list) or len(values) != len(SEMANTIC_ORDER):
        raise ValueError("base pose must resolve to 16 semantic joint values")
    return [float(value) for value in values]


def joint_limit(urdf: Path, joint_name: str) -> tuple[float, float]:
    root = ET.parse(urdf).getroot()
    joint = next(
        item for item in root.findall("joint")
        if item.get("name") == joint_name
    )
    limit = joint.find("limit")
    if limit is None:
        raise ValueError(f"{joint_name} has no URDF limit")
    return float(limit.get("lower")), float(limit.get("upper"))


def q_grid(lower: float, upper: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("--q-step must be positive")
    count = int(math.floor((upper - lower) / step + 1.0e-9))
    values = [round(lower + index * step, 10) for index in range(count + 1)]
    if values[-1] < upper - 1.0e-9:
        values.append(float(upper))
    else:
        values[-1] = float(upper)
    return values


def color_from_summary(summary_path: Path) -> Path:
    suffix = "_summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"unexpected summary filename: {summary_path}")
    return summary_path.with_name(
        summary_path.name.removesuffix(suffix) + "_color.png"
    )


def row(
    *,
    direction: str,
    command_raw: int,
    stable_raw: int,
    summary_path: Path,
    color_path: Path,
    old_measurement_rad: float | None,
    suffix: str = "",
) -> dict[str, Any]:
    token = f"{direction}_cmd{command_raw:03d}_read{stable_raw:03d}{suffix}"
    return {
        "sample_id": token,
        "direction": direction,
        "command_raw": command_raw,
        "stable_readback_raw": stable_raw,
        "real_color": str(color_path.resolve()),
        "camera_summary": str(summary_path.resolve()),
        "old_visual_measurement_rad_context_only": old_measurement_rad,
        "match_status": "unreviewed",
    }


def build_samples(session: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for direction, key in (
        ("forward", "forward_points"),
        ("reverse", "reverse_points"),
    ):
        for point in summary[key]:
            command = int(point["command_raw"])
            stable = int(point["stable_readback_raw"])
            old_rad = (
                float(point["physical_rad"])
                if point.get("physical_rad") is not None
                else None
            )
            source = str(point["camera_source"])
            if direction == "forward" and source == ".":
                for label in ("a", "b"):
                    summary_path = (
                        session
                        / f"reference_raw058_camera_{label}_fixed_exp_180f_summary.json"
                    )
                    samples.append(
                        row(
                            direction="forward_reference",
                            command_raw=command,
                            stable_raw=stable,
                            summary_path=summary_path,
                            color_path=color_from_summary(summary_path),
                            old_measurement_rad=old_rad,
                            suffix=f"_{label}",
                        )
                    )
                continue
            summary_path = session / source
            samples.append(
                row(
                    direction=direction,
                    command_raw=command,
                    stable_raw=stable,
                    summary_path=summary_path,
                    color_path=color_from_summary(summary_path),
                    old_measurement_rad=old_rad,
                )
            )

    missing = [
        path
        for sample in samples
        for path in (
            Path(sample["real_color"]),
            Path(sample["camera_summary"]),
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing formal evidence:\n" + "\n".join(map(str, missing))
        )
    # Present semantic zero first, then progress monotonically toward max q.
    return sorted(
        samples,
        key=lambda item: (
            -int(item["stable_readback_raw"]),
            str(item["direction"]),
            str(item["sample_id"]),
        ),
    )


def main() -> int:
    args = parse_args()
    for path in (
        args.summary,
        args.base_pose,
        args.urdf,
        args.camera,
        args.intrinsics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.session.is_dir():
        raise NotADirectoryError(args.session)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    samples = build_samples(args.session, summary)
    lower, upper = joint_limit(args.urdf, TARGET_JOINT)
    values = q_grid(lower, upper, args.q_step)
    base = load_single_pose(args.base_pose)
    target_index = SEMANTIC_ORDER.index(TARGET_JOINT)
    poses: dict[str, list[float]] = {}
    candidates: list[dict[str, Any]] = []
    for index, q in enumerate(values):
        name = f"thumb_cmc_roll_q_{index:03d}_{q:.4f}".replace(".", "p")
        pose = list(base)
        pose[target_index] = q
        poses[name] = pose
        candidates.append(
            {
                "index": index,
                "name": name,
                "q_urdf_rad": q,
                "q_urdf_deg": math.degrees(q),
            }
        )

    poses_path = args.out_dir / "poses_urdf_local_q.json"
    dataset_path = args.out_dir / "manual_match_dataset.json"
    poses_path.write_text(
        json.dumps(poses, indent=2) + "\n", encoding="utf-8"
    )
    dataset = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "formal_photo_to_og_urdf_local_q_manual_registration",
        "joint": TARGET_JOINT,
        "coordinate_contract": {
            "ground_truth_coordinate": "OG URDF local joint q in radians",
            "raw_coordinate": "physical stable SDK readback [0,255]",
            "old_visual_measurement_role": (
                "context only; hidden from the matcher to avoid anchoring bias"
            ),
            "selection_rule": (
                "human chooses the same-view OG-URDF render that best matches "
                "each formal D435 photo"
            ),
        },
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": sha256(args.urdf),
        "camera": str(args.camera.resolve()),
        "camera_sha256": sha256(args.camera),
        "intrinsics": str(args.intrinsics.resolve()),
        "intrinsics_sha256": sha256(args.intrinsics),
        "base_pose": str(args.base_pose.resolve()),
        "base_pose_sha256": sha256(args.base_pose),
        "poses_json": str(poses_path.resolve()),
        "semantic_order": list(SEMANTIC_ORDER),
        "target_semantic_index": target_index,
        "urdf_limit_rad": [lower, upper],
        "q_step_rad": args.q_step,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "sample_count": len(samples),
        "samples": samples,
    }
    dataset_path.write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "poses": str(poses_path),
                "samples": len(samples),
                "candidates": len(candidates),
                "urdf_limit_rad": [lower, upper],
                "q_step_rad": args.q_step,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
