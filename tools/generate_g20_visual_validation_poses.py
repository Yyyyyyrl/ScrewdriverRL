#!/usr/bin/env python3
"""Generate deterministic, conservative G20 poses for visual sim/real checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ORDER = [
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
]

RANGES = {
    "index_mcp_roll": (-0.075, 0.075),
    "middle_mcp_roll": (-0.075, 0.075),
    "ring_mcp_roll": (-0.075, 0.075),
    "pinky_mcp_roll": (-0.075, 0.075),
    "index_mcp_pitch": (0.12, 0.78),
    "middle_mcp_pitch": (0.12, 0.78),
    "ring_mcp_pitch": (0.12, 0.78),
    "pinky_mcp_pitch": (0.12, 0.78),
    "index_pip": (0.16, 1.05),
    "middle_pip": (0.16, 1.05),
    "ring_pip": (0.16, 1.05),
    "pinky_pip": (0.16, 1.05),
    "thumb_cmc_yaw": (0.28, 1.00),
    "thumb_cmc_roll": (0.48, 1.20),
    "thumb_cmc_pitch": (0.08, 0.52),
    "thumb_mcp": (0.10, 0.68),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument("--poses-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def vector(values: dict[str, float]) -> list[float]:
    return [float(values[name]) for name in ORDER]


def main() -> int:
    args = parse_args()
    if args.poses_out.exists() or args.manifest_out.exists():
        raise FileExistsError("refusing to overwrite pose outputs")
    rng = np.random.default_rng(args.seed)
    poses = {
        "diag_flex_wave": vector({
            "index_mcp_roll": 0.02, "index_mcp_pitch": 0.18, "index_pip": 0.22,
            "middle_mcp_roll": 0.01, "middle_mcp_pitch": 0.34, "middle_pip": 0.42,
            "ring_mcp_roll": -0.01, "ring_mcp_pitch": 0.50, "ring_pip": 0.62,
            "pinky_mcp_roll": -0.02, "pinky_mcp_pitch": 0.66, "pinky_pip": 0.82,
            "thumb_cmc_yaw": 0.42, "thumb_cmc_roll": 0.92,
            "thumb_cmc_pitch": 0.18, "thumb_mcp": 0.24,
        }),
        "diag_roll_fan": vector({
            "index_mcp_roll": 0.075, "index_mcp_pitch": 0.24, "index_pip": 0.30,
            "middle_mcp_roll": 0.035, "middle_mcp_pitch": 0.24, "middle_pip": 0.30,
            "ring_mcp_roll": -0.035, "ring_mcp_pitch": 0.24, "ring_pip": 0.30,
            "pinky_mcp_roll": -0.075, "pinky_mcp_pitch": 0.24, "pinky_pip": 0.30,
            "thumb_cmc_yaw": 0.36, "thumb_cmc_roll": 0.88,
            "thumb_cmc_pitch": 0.14, "thumb_mcp": 0.18,
        }),
        "diag_thumb_opposition": vector({
            "index_mcp_roll": 0.0, "index_mcp_pitch": 0.34, "index_pip": 0.44,
            "middle_mcp_roll": 0.0, "middle_mcp_pitch": 0.34, "middle_pip": 0.44,
            "ring_mcp_roll": 0.0, "ring_mcp_pitch": 0.34, "ring_pip": 0.44,
            "pinky_mcp_roll": 0.0, "pinky_mcp_pitch": 0.34, "pinky_pip": 0.44,
            "thumb_cmc_yaw": 0.90, "thumb_cmc_roll": 0.66,
            "thumb_cmc_pitch": 0.40, "thumb_mcp": 0.46,
        }),
    }
    for index in range(args.random_count):
        values = {
            name: float(rng.uniform(low, high))
            for name, (low, high) in RANGES.items()
        }
        poses[f"random_{index:02d}"] = vector(values)

    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.poses_out.write_text(json.dumps(poses, indent=2) + "\n")
    args.manifest_out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": args.seed,
                "random_count": args.random_count,
                "joint_order16": ORDER,
                "sampling_ranges_rad": RANGES,
                "curated_poses": [
                    "diag_flex_wave", "diag_roll_fan", "diag_thumb_opposition"
                ],
                "poses_path": str(args.poses_out),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.poses_out}")
    print(f"wrote {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
