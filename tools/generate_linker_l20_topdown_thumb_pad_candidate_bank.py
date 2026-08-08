#!/usr/bin/env python3
"""Generate D64 candidates with palm-plane palmward offset and thumb-pad contact.

"Toward the palm" is the in-plane direction from the screwdriver axis toward
the hand root/wrist, not the palm-surface normal.  The mounted screwdriver stays
fixed and the generator applies the equivalent opposite XY hand-root shift, so
table clearance is preserved.  The generated candidates reduce thumb CMC roll, retune
thumb CMC pitch/MCP in the URDF actual sign convention, and score the broad
distal pad face against the handle radial direction.  The output is a candidate bank for
``search_linker_l20_screwdriver_topdown_physics.py``; it is not a promoted
posture and still requires replicated PhysX/domain-randomisation gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from screwdriver_rl.utils.linker_topdown_diameter_postures import (  # noqa: E402
    TOPDOWN_SCREWDRIVER_TILT_XY_BUCKETS,
)
from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    SCREWDRIVER_ROOT_POS_W,
    UrdfGeometry,
)


HAND_URDF = REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
SCREWDRIVER_URDF = (
    REPO_ROOT / "assets/screwdriver/topdown_variants/screwdriver_topdown_d064.urdf"
)
JOINT_NAMES = (
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
# Broad palmar face of thumb_distal.STL.  This is the large +X/+Z face, not
# either +/-Y edge.  Values are the normalized mesh face normal.
THUMB_PAD_NORMAL_LOCAL = np.asarray(
    (0.788425087928772, 0.0, 0.6151307821273804),
    dtype=np.float64,
)
THUMB_PAD_NORMAL_LOCAL /= np.linalg.norm(THUMB_PAD_NORMAL_LOCAL)
THUMB_SIDE_NORMAL_LOCAL = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_candidate(path: Path, candidate_index: int | None) -> dict:
    data = json.loads(path.read_text())
    if "top_candidates" not in data:
        return data
    rows = data["top_candidates"]
    if candidate_index is None:
        return rows[0]
    matches = [
        row for row in rows if int(row["candidate_index"]) == candidate_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one candidate_index={candidate_index}, found {len(matches)}"
        )
    return matches[0]


def _guarded_joint_bounds(
    model: UrdfGeometry,
    margin_rad: float,
) -> dict[str, tuple[float, float]]:
    limits = {
        name: [float(lo), float(hi)]
        for name, (lo, hi) in model.joint_limits(margin_rad).items()
        if name in JOINT_NAMES
    }
    for follower in model.joints.values():
        if follower.mimic is None or follower.mimic.source not in limits:
            continue
        if follower.lower is None or follower.upper is None:
            continue
        multiplier = follower.mimic.multiplier
        offset = follower.mimic.offset
        guarded = (
            (follower.lower + margin_rad - offset) / multiplier,
            (follower.upper - margin_rad - offset) / multiplier,
        )
        source = follower.mimic.source
        limits[source][0] = max(limits[source][0], min(guarded))
        limits[source][1] = min(limits[source][1], max(guarded))
    return {name: tuple(limits[name]) for name in JOINT_NAMES}


def _thumb_alignment(
    hand_model: UrdfGeometry,
    screwdriver_axis_origin_w: np.ndarray,
    screwdriver_axis_w: np.ndarray,
    root_pos_w: list[float],
    root_quat_wxyz: list[float],
    joints: dict[str, float],
) -> dict[str, float]:
    transforms = hand_model.forward_kinematics(
        joints,
        root_pos_w,
        root_quat_wxyz,
    )
    thumb_tip_w = transforms["thumb_tip"][:3, 3]
    relative = thumb_tip_w - screwdriver_axis_origin_w
    radial = relative - screwdriver_axis_w * float(
        np.dot(relative, screwdriver_axis_w)
    )
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1.0e-12:
        raise ValueError("thumb tip lies on the screwdriver axis")
    toward_axis_w = -radial / radial_norm
    thumb_rotation_w = transforms["thumb_distal"][:3, :3]
    pad_alignment = float(
        np.dot(thumb_rotation_w @ THUMB_PAD_NORMAL_LOCAL, toward_axis_w)
    )
    side_alignment = float(
        np.dot(thumb_rotation_w @ THUMB_SIDE_NORMAL_LOCAL, toward_axis_w)
    )
    return {
        "thumb_pad_alignment": pad_alignment,
        "thumb_side_alignment": side_alignment,
        "thumb_pad_face_advantage": pad_alignment - abs(side_alignment),
        "thumb_tip_radial_distance_to_handle_axis_m": radial_norm,
    }


def _minimum_joint_margin(
    model: UrdfGeometry,
    joints: dict[str, float],
) -> float:
    expanded = model.expanded_positions(joints)
    margins = []
    for joint in model.joints.values():
        if joint.lower is None or joint.upper is None:
            continue
        value = float(expanded[joint.name])
        margins.append(min(value - joint.lower, joint.upper - value))
    return min(margins)


def _palmward_direction_world_xy(
    hand_root_w: np.ndarray,
    screwdriver_axis_origin_w: np.ndarray,
) -> np.ndarray:
    direction = np.asarray(hand_root_w, dtype=np.float64).copy()
    direction[:2] -= np.asarray(screwdriver_axis_origin_w, dtype=np.float64)[:2]
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("hand root and screwdriver axis coincide in world XY")
    return direction / norm


def _diverse_selection(
    eligible: list[dict],
    count: int,
) -> list[dict]:
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} candidates passed the pad-face gate; need {count}"
        )
    parameters = np.asarray([row["_normalized_parameters"] for row in eligible])
    quality = np.asarray([row["_quality"] for row in eligible])
    quality = (quality - quality.min()) / max(float(np.ptp(quality)), 1.0e-12)
    selected = [int(np.argmax(quality))]
    minimum_distance_sq = np.sum(
        (parameters - parameters[selected[0]]) ** 2,
        axis=1,
    )
    minimum_distance_sq[selected[0]] = -np.inf
    while len(selected) < count:
        score = minimum_distance_sq + 0.10 * quality
        index = int(np.argmax(score))
        selected.append(index)
        distance_sq = np.sum((parameters - parameters[index]) ** 2, axis=1)
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)
        minimum_distance_sq[selected] = -np.inf
    return [eligible[index] for index in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-search", type=Path, required=True)
    parser.add_argument("--initial-candidate-index", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--joint-margin-rad", type=float, default=0.105)
    parser.add_argument("--screwdriver-palmward-min-m", type=float, default=0.001)
    parser.add_argument("--screwdriver-palmward-max-m", type=float, default=0.008)
    parser.add_argument("--thumb-yaw-delta-min-rad", type=float, default=-0.15)
    parser.add_argument("--thumb-yaw-delta-max-rad", type=float, default=0.02)
    parser.add_argument("--thumb-roll-delta-min-rad", type=float, default=-0.30)
    parser.add_argument("--thumb-roll-delta-max-rad", type=float, default=-0.05)
    parser.add_argument("--thumb-pitch-delta-min-rad", type=float, default=-0.10)
    parser.add_argument("--thumb-pitch-delta-max-rad", type=float, default=0.04)
    parser.add_argument("--thumb-mcp-delta-min-rad", type=float, default=-0.25)
    parser.add_argument("--thumb-mcp-delta-max-rad", type=float, default=0.04)
    parser.add_argument("--min-thumb-pad-alignment", type=float, default=0.65)
    parser.add_argument("--min-thumb-pad-face-advantage", type=float, default=0.05)
    args = parser.parse_args()

    if args.pool_size < args.num_candidates:
        raise ValueError("--pool-size must be >= --num-candidates")
    ranges = np.asarray(
        [
            [args.screwdriver_palmward_min_m, args.screwdriver_palmward_max_m],
            [args.thumb_yaw_delta_min_rad, args.thumb_yaw_delta_max_rad],
            [args.thumb_roll_delta_min_rad, args.thumb_roll_delta_max_rad],
            [args.thumb_pitch_delta_min_rad, args.thumb_pitch_delta_max_rad],
            [args.thumb_mcp_delta_min_rad, args.thumb_mcp_delta_max_rad],
        ],
        dtype=np.float64,
    )
    if np.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError("every candidate range minimum must be <= its maximum")

    base = _load_candidate(args.initial_search, args.initial_candidate_index)
    root_quat = [
        float(value) for value in base["root_quat_wxyz"]
    ]
    base_root = np.asarray(base["root_pos_w"], dtype=np.float64)
    base_joints = {
        name: float(base["joint_positions_independent"][name])
        for name in JOINT_NAMES
    }
    hand_model = UrdfGeometry(HAND_URDF)
    bounds = _guarded_joint_bounds(hand_model, args.joint_margin_rad)
    screwdriver_model = UrdfGeometry(SCREWDRIVER_URDF)
    tilt_xy = TOPDOWN_SCREWDRIVER_TILT_XY_BUCKETS[1]
    screwdriver_transforms = screwdriver_model.forward_kinematics(
        {
            "table_screwdriver_joint_1": float(tilt_xy[0]),
            "table_screwdriver_joint_2": float(tilt_xy[1]),
        },
        SCREWDRIVER_ROOT_POS_W,
        (1.0, 0.0, 0.0, 0.0),
    )
    body_transform = screwdriver_transforms["screwdriver_body"]
    screwdriver_axis_origin_w = body_transform[:3, 3]
    screwdriver_axis_w = body_transform[:3, 2]
    palmward_direction_w = _palmward_direction_world_xy(
        base_root,
        screwdriver_axis_origin_w,
    )

    engine = torch.quasirandom.SobolEngine(5, scramble=True, seed=args.seed)
    unit_samples = engine.draw(args.pool_size).to(torch.float64).numpy()
    samples = ranges[:, 0] + unit_samples * (ranges[:, 1] - ranges[:, 0])
    eligible = []
    for pool_index, (unit, sample) in enumerate(
        zip(unit_samples, samples, strict=True)
    ):
        root = base_root.copy()
        root -= sample[0] * palmward_direction_w
        joints = dict(base_joints)
        for name, delta in zip(
            (
                "thumb_cmc_yaw",
                "thumb_cmc_roll",
                "thumb_cmc_pitch",
                "thumb_mcp",
            ),
            sample[1:],
            strict=True,
        ):
            lower, upper = bounds[name]
            joints[name] = float(np.clip(joints[name] + delta, lower, upper))
        alignment = _thumb_alignment(
            hand_model,
            screwdriver_axis_origin_w,
            screwdriver_axis_w,
            root.tolist(),
            root_quat,
            joints,
        )
        if (
            alignment["thumb_pad_alignment"] < args.min_thumb_pad_alignment
            or alignment["thumb_pad_face_advantage"]
            < args.min_thumb_pad_face_advantage
        ):
            continue
        quality = (
            alignment["thumb_pad_alignment"]
            + 0.5 * alignment["thumb_pad_face_advantage"]
        )
        eligible.append(
            {
                "_normalized_parameters": unit.tolist(),
                "_quality": quality,
                "pool_index": pool_index,
                "root_pos_w": [float(value) for value in root],
                "joint_positions_independent": joints,
                "minimum_joint_limit_margin_rad": _minimum_joint_margin(
                    hand_model,
                    joints,
                ),
                "parameter_deltas": {
                    "equivalent_screwdriver_toward_palm_m": float(sample[0]),
                    "equivalent_hand_root_world_xyz_m": [
                        float(value) for value in -sample[0] * palmward_direction_w
                    ],
                    "thumb_cmc_yaw_rad": float(sample[1]),
                    "thumb_cmc_roll_rad": float(sample[2]),
                    "thumb_cmc_pitch_rad": float(sample[3]),
                    "thumb_mcp_rad": float(sample[4]),
                },
                **alignment,
            }
        )

    selected = _diverse_selection(eligible, args.num_candidates)
    candidates = []
    for rank, row in enumerate(selected):
        candidates.append(
            {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
        )
        candidates[-1]["label"] = f"thumb_pad_palmward_rank_{rank:03d}"

    output = {
        "schema_version": 1,
        "status": "UNPROMOTED_CANDIDATE_BANK",
        "source": str(args.initial_search),
        "source_sha256": _sha256(args.initial_search),
        "initial_candidate_index": args.initial_candidate_index,
        "seed": args.seed,
        "pool_size": args.pool_size,
        "eligible_pool_count": len(eligible),
        "num_candidates": len(candidates),
        "joint_margin_rad": args.joint_margin_rad,
        "palmward_direction_world_xyz": palmward_direction_w.tolist(),
        "parameter_ranges": {
            "equivalent_screwdriver_toward_palm_m": ranges[0].tolist(),
            "thumb_cmc_yaw_delta_rad": ranges[1].tolist(),
            "thumb_cmc_roll_delta_rad": ranges[2].tolist(),
            "thumb_cmc_pitch_delta_rad": ranges[3].tolist(),
            "thumb_mcp_delta_rad": ranges[4].tolist(),
        },
        "thumb_pad_gate": {
            "pad_normal_local": THUMB_PAD_NORMAL_LOCAL.tolist(),
            "side_normal_local": THUMB_SIDE_NORMAL_LOCAL.tolist(),
            "minimum_pad_alignment": args.min_thumb_pad_alignment,
            "minimum_pad_face_advantage": args.min_thumb_pad_face_advantage,
        },
        "base_candidate": {
            "root_pos_w": base_root.tolist(),
            "root_quat_wxyz": root_quat,
            "joint_positions_independent": base_joints,
            **_thumb_alignment(
                hand_model,
                screwdriver_axis_origin_w,
                screwdriver_axis_w,
                base_root.tolist(),
                root_quat,
                base_joints,
            ),
        },
        "selected_summary": {
            "minimum_joint_limit_margin_rad": min(
                row["minimum_joint_limit_margin_rad"] for row in candidates
            ),
            "thumb_pad_alignment": {
                "min": min(row["thumb_pad_alignment"] for row in candidates),
                "max": max(row["thumb_pad_alignment"] for row in candidates),
            },
            "thumb_pad_face_advantage": {
                "min": min(
                    row["thumb_pad_face_advantage"] for row in candidates
                ),
                "max": max(
                    row["thumb_pad_face_advantage"] for row in candidates
                ),
            },
            "equivalent_screwdriver_toward_palm_m": {
                "min": min(
                    row["parameter_deltas"][
                        "equivalent_screwdriver_toward_palm_m"
                    ]
                    for row in candidates
                ),
                "max": max(
                    row["parameter_deltas"][
                        "equivalent_screwdriver_toward_palm_m"
                    ]
                    for row in candidates
                ),
            },
        },
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "eligible_pool_count": len(eligible),
                "num_candidates": len(candidates),
                "base_thumb_pad_alignment": output["base_candidate"][
                    "thumb_pad_alignment"
                ],
                "selected_pad_alignment_min": min(
                    row["thumb_pad_alignment"] for row in candidates
                ),
                "selected_pad_face_advantage_min": min(
                    row["thumb_pad_face_advantage"] for row in candidates
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
