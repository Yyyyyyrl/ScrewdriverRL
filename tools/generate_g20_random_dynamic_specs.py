#!/usr/bin/env python3
"""Generate deterministic, bounded large-amplitude G20 validation trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Callable


JOINTS = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
)

# Deliberately inside the 2026-08-05 production limits.  These bounds expose
# substantially more travel than the first dynamic gate while retaining room
# for LUT quantization, tracking lag, and uncertain fingertip self-contact.
SAFE_ENVELOPE = {
    "index_mcp_roll": (-0.12, 0.12),
    "index_mcp_pitch": (0.14, 0.96),
    "index_pip": (0.16, 1.10),
    "middle_mcp_roll": (-0.12, 0.12),
    "middle_mcp_pitch": (0.14, 0.94),
    "middle_pip": (0.16, 1.10),
    "ring_mcp_roll": (-0.12, 0.12),
    "ring_mcp_pitch": (0.14, 0.94),
    "ring_pip": (0.16, 1.10),
    "pinky_mcp_roll": (-0.12, 0.12),
    "pinky_mcp_pitch": (0.14, 0.88),
    "pinky_pip": (0.16, 1.10),
    "thumb_cmc_yaw": (0.18, 0.94),
    "thumb_cmc_roll": (0.56, 1.08),
    "thumb_cmc_pitch": (0.08, 0.60),
    "thumb_mcp": (0.08, 0.72),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finger_pose(
    rng: random.Random,
    *,
    pitch_hi: float,
    deep: bool = False,
    open_: bool = False,
) -> list[float]:
    roll = rng.uniform(-0.115, 0.115)
    if deep:
        pitch = rng.uniform(min(0.78, pitch_hi), pitch_hi)
        pip = rng.uniform(0.78, 1.02)
    elif open_:
        pitch = rng.uniform(0.15, 0.29)
        pip = rng.uniform(0.17, 0.34)
    else:
        pitch = rng.uniform(0.28, min(0.86, pitch_hi))
        # Keep the most flexed MCP/PIP combinations away from the palm.
        pip_hi = 0.92 if pitch > 0.74 else 1.08
        pip = rng.uniform(0.24, pip_hi)
    return [roll, pitch, pip]


def _thumb_pose(
    rng: random.Random, *, opposed: bool = False, retracted: bool = False
) -> list[float]:
    if opposed:
        return [
            rng.uniform(0.78, 0.93), rng.uniform(0.61, 0.82),
            rng.uniform(0.38, 0.58), rng.uniform(0.42, 0.68),
        ]
    if retracted:
        return [
            rng.uniform(0.19, 0.34), rng.uniform(0.90, 1.07),
            rng.uniform(0.09, 0.24), rng.uniform(0.09, 0.25),
        ]
    return [
        rng.uniform(0.27, 0.88), rng.uniform(0.61, 1.02),
        rng.uniform(0.13, 0.55), rng.uniform(0.14, 0.65),
    ]


def _random_pose(
    rng: random.Random,
    *,
    deep_finger: int | None = None,
    open_finger: int | None = None,
    thumb_mode: str = "random",
) -> list[float]:
    pose: list[float] = []
    for finger in range(4):
        pitch_hi = (0.94, 0.92, 0.92, 0.87)[finger]
        pose.extend(
            _finger_pose(
                rng,
                pitch_hi=pitch_hi,
                deep=finger == deep_finger,
                open_=finger == open_finger,
            )
        )
    pose.extend(
        _thumb_pose(
            rng,
            opposed=thumb_mode == "opposed",
            retracted=thumb_mode == "retracted",
        )
    )
    return pose


def _wave(rng: random.Random) -> list[list[float]]:
    order = [0, 2, 1, 3, 0, 3, 1]
    return [
        _random_pose(
            rng,
            deep_finger=finger,
            open_finger=(finger + 2) % 4,
            thumb_mode="random",
        )
        for finger in order
    ]


def _roll_flex(rng: random.Random) -> list[list[float]]:
    poses = [_random_pose(rng, deep_finger=k % 4) for k in range(6)]
    for k, pose in enumerate(poses):
        for finger in range(4):
            magnitude = rng.uniform(0.085, 0.118)
            base = 3 * finger
            pose[base] = magnitude if (k + finger) % 2 == 0 else -magnitude
            # The first hardware attempt showed a transient PIP fault when a
            # finger combined large roll with both MCP and PIP near 1 rad.
            # Preserve the wide roll excursion but bound that coupled flexion
            # corner for every finger, not only the one that raised the fault.
            if pose[base + 1] > 0.84 and pose[base + 2] > 0.90:
                pose[base + 1] = 0.82
                pose[base + 2] = 0.84
    return poses


def _thumb(rng: random.Random) -> list[list[float]]:
    modes = ["retracted", "random", "opposed", "random", "retracted", "opposed"]
    poses = []
    for k, mode in enumerate(modes):
        pose = _random_pose(
            rng,
            deep_finger=None if k % 2 == 0 else rng.randrange(4),
            open_finger=rng.randrange(4) if k % 2 == 0 else None,
            thumb_mode=mode,
        )
        poses.append(pose)
    return poses


def _full(rng: random.Random) -> list[list[float]]:
    poses = []
    for k in range(7):
        poses.append(
            _random_pose(
                rng,
                deep_finger=rng.randrange(4) if k in (1, 3, 5) else None,
                open_finger=rng.randrange(4) if k in (0, 2, 6) else None,
                thumb_mode=rng.choice(("random", "random", "opposed", "retracted")),
            )
        )
    return poses


def _apply_collision_constraints(pose: list[float]) -> list[float]:
    """Keep large single-axis travel while excluding observed coupled jams."""

    result = list(pose)
    for finger in range(4):
        base = 3 * finger
        if result[base + 1] > 0.82 and result[base + 2] > 0.82:
            result[base + 2] = 0.80

    thumb_yaw, thumb_pitch, thumb_mcp = result[12], result[14], result[15]
    if thumb_yaw > 0.72:
        # High yaw is retained, but the thumb does not close into the index.
        result[14] = min(thumb_pitch, 0.34)
        result[15] = min(thumb_mcp, 0.36)
        result[1] = min(result[1], 0.55)
        result[2] = min(result[2], 0.55)
    elif thumb_pitch > 0.38 or thumb_mcp > 0.42:
        # Large thumb flexion remains available at a yaw with clear separation.
        result[12] = min(thumb_yaw, 0.68)
        result[1] = min(result[1], 0.65)
        result[2] = min(result[2], 0.70)
    return result


def _validate_pose(pose: list[float]) -> None:
    if len(pose) != len(JOINTS):
        raise ValueError(f"pose has {len(pose)} joints")
    for name, value in zip(JOINTS, pose):
        lo, hi = SAFE_ENVELOPE[name]
        if not lo <= value <= hi:
            raise ValueError(f"{name}={value:.6f} outside safe envelope {lo}..{hi}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080504)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    families: list[tuple[str, str, Callable[[random.Random], list[list[float]]]]] = [
        ("01_random_four_finger_wave_large", "Randomized large four-finger wave", _wave),
        ("02_random_roll_flex_mix_large", "Random roll and flexion mixture", _roll_flex),
        ("03_random_thumb_opposition_large", "Random thumb opposition with finger counter-motion", _thumb),
        ("04_random_full_hand_smooth_large", "Random coordinated full-hand motion", _full),
    ]
    manifest = {
        "schema_version": 2,
        "generator": str(Path(__file__).resolve()),
        "seed": args.seed,
        "calibration": str(args.calib.resolve()),
        "calibration_sha256": _sha256(args.calib),
        "joint_order16": list(JOINTS),
        "safe_envelope_rad": {name: list(bounds) for name, bounds in SAFE_ENVELOPE.items()},
        "specs": [],
    }
    all_waypoints: dict[str, list[float]] = {}
    for family_index, (name, description, factory) in enumerate(families):
        family_seed = args.seed + 1009 * family_index
        waypoints = [
            _apply_collision_constraints(pose)
            for pose in factory(random.Random(family_seed))
        ]
        for pose in waypoints:
            _validate_pose(pose)
        rows = [
            {
                "name": f"wp_{index + 1:02d}",
                "semantic": [round(value, 8) for value in pose],
                "duration_s": round(3.60 + 0.15 * ((index + family_index) % 4), 2),
                "hold_s": 0.18,
            }
            for index, pose in enumerate(waypoints)
        ]
        spec = {
            "schema_version": 2,
            "name": name,
            "description": description,
            "seed": family_seed,
            "joint_order16": list(JOINTS),
            "safety_envelope_rad": manifest["safe_envelope_rad"],
            "preposition_s": 3.0,
            "return_s": 3.0,
            "waypoints": rows,
        }
        path = args.out_dir / f"{name}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        spans = {
            joint: max(pose[j] for pose in waypoints) - min(pose[j] for pose in waypoints)
            for j, joint in enumerate(JOINTS)
        }
        manifest["specs"].append(
            {
                "name": name,
                "path": str(path),
                "seed": family_seed,
                "waypoint_count": len(rows),
                "nominal_waypoint_duration_s": sum(
                    row["duration_s"] + row["hold_s"] for row in rows
                ),
                "joint_span_rad": spans,
            }
        )
        for row in rows:
            all_waypoints[f"{name}__{row['name']}"] = row["semantic"]

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "waypoint_poses.json").write_text(
        json.dumps(all_waypoints, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[generate] wrote {len(families)} specs and {len(all_waypoints)} waypoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
