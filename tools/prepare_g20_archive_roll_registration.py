#!/usr/bin/env python3
"""Prepare archived MCP-roll blue-marker sweeps for same-view Isaac rendering."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from tools.prepare_g20_archive_chain_registration import (
    SEMANTIC_ORDER,
    _camera_payload,
    _circle_center,
    _deproject,
    _unit,
    _zero_fk,
)


def _director_from_zero(
    zero: np.ndarray, vector: np.ndarray, axis: np.ndarray
) -> float:
    zero = _unit(zero - axis * float(zero @ axis))
    vector = _unit(vector - axis * float(vector @ axis))
    angle = math.atan2(
        float(axis @ np.cross(zero, vector)), float(zero @ vector)
    )
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--finger", required=True)
    parser.add_argument("--joint", choices=SEMANTIC_ORDER, required=True)
    parser.add_argument("--zero-command-raw", type=int, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--select-command-raw", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for source in csv.DictReader(args.points_csv.open(newline="", encoding="utf-8")):
        relative = source["source"]
        summary_path = args.session / relative
        if not summary_path.is_file():
            summary_path = summary_path / "camera_a_fixed_exp_180f_summary.json"
        if not summary_path.is_file():
            continue
        payload = json.loads(summary_path.read_text())
        moving = payload["last_markers"][args.finger]
        if moving.get("axis3_xyz") is None:
            continue
        intr = payload["camera"]["intrinsics"]
        rows.append(
            {
                "command_raw": int(round(float(source["command_raw"]))),
                "stable_readback_raw": float(source["stable_readback_raw"]),
                "old_visual_q_rad": float(source["abduction_rad"]),
                "summary": str(summary_path.resolve()),
                "color": payload.get("outputs", {}).get("color_png"),
                "intrinsics": intr,
                "moving_axis_camera": _unit(np.asarray(moving["axis3_xyz"])),
                "moving_centroid_camera": _deproject(moving, intr),
            }
        )
    if len(rows) < 8:
        raise RuntimeError(f"only {len(rows)} usable observations")

    points = np.stack([row["moving_centroid_camera"] for row in rows])
    _, _, vh = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    axis_camera = _unit(vh[-1])
    zero_vectors = [
        row["moving_axis_camera"]
        for row in rows
        if row["command_raw"] == args.zero_command_raw
    ]
    if not zero_vectors:
        raise RuntimeError("zero-command camera observation is missing")
    zero_camera = _unit(np.median(np.stack(zero_vectors), axis=0))
    q_reconstructed = np.asarray(
        [
            _director_from_zero(zero_camera, row["moving_axis_camera"], axis_camera)
            for row in rows
        ]
    )
    old_q = np.asarray([row["old_visual_q_rad"] for row in rows])
    if np.corrcoef(q_reconstructed, old_q)[0, 1] < 0.0:
        axis_camera = -axis_camera
        q_reconstructed = -q_reconstructed

    joint_origin_base, axis_base, child_zero_base = _zero_fk(
        args.urdf, args.joint
    )
    child_zero_base = _unit(
        child_zero_base - axis_base * float(child_zero_base @ axis_base)
    )
    zero_camera = _unit(
        zero_camera - axis_camera * float(zero_camera @ axis_camera)
    )
    base_frame = np.column_stack(
        (np.cross(axis_base, child_zero_base), axis_base, child_zero_base)
    )
    camera_frame = np.column_stack(
        (np.cross(axis_camera, zero_camera), axis_camera, zero_camera)
    )
    rotation_camera_from_base = camera_frame @ base_frame.T
    center_camera, radius = _circle_center(points, axis_camera)
    translation_camera_from_base = (
        center_camera - rotation_camera_from_base @ joint_origin_base
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_camera_from_base
    transform[:3, 3] = translation_camera_from_base

    residual = old_q - q_reconstructed
    for row, q, delta in zip(rows, q_reconstructed, residual):
        row["reconstructed_local_q_rad"] = float(q)
        row["old_minus_reconstructed_rad"] = float(delta)
        row["moving_axis_camera"] = row["moving_axis_camera"].tolist()
        row["moving_centroid_camera"] = row["moving_centroid_camera"].tolist()

    selected = {
        int(value)
        for value in args.select_command_raw.split(",")
        if value.strip()
    }
    semantic_index = SEMANTIC_ORDER.index(args.joint)
    poses = {}
    used_commands = set()
    for row in sorted(rows, key=lambda item: abs(item["old_visual_q_rad"])):
        raw = row["command_raw"]
        if raw not in selected or raw in used_commands:
            continue
        used_commands.add(raw)
        pose = [0.0] * len(SEMANTIC_ORDER)
        pose[semantic_index] = row["old_visual_q_rad"]
        poses[f"raw{raw:03d}_oldq_{row['old_visual_q_rad']:.6f}"] = pose

    camera_path = args.out_dir / "camera_archive_roll.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )
    first_intr = rows[0]["intrinsics"]
    intrinsics_path = args.out_dir / "intrinsics.json"
    intrinsics_path.write_text(
        json.dumps(
            {
                "width": 1280,
                "height": 720,
                **{key: float(first_intr[key]) for key in ("fx", "fy", "ppx", "ppy")},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    poses_path = args.out_dir / "poses_old_visual_q.json"
    poses_path.write_text(json.dumps(poses, indent=2) + "\n", encoding="utf-8")
    output = {
        "schema_version": 1,
        "method": "archive_roll_moving_centroid_plane_and_zero_axis",
        "eligible_for_deployment_lut": False,
        "session": str(args.session.resolve()),
        "joint": args.joint,
        "finger_marker": args.finger,
        "zero_command_raw": args.zero_command_raw,
        "axis_camera": axis_camera.tolist(),
        "axis_from_camera_optical_deg": float(
            np.degrees(math.acos(np.clip(abs(axis_camera[2]), 0.0, 1.0)))
        ),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "joint_center_camera_estimate_m": center_camera.tolist(),
        "marker_circle_radius_m": float(radius),
        "old_vs_reconstructed": {
            "count": len(rows),
            "mean_rad": float(np.mean(residual)),
            "rmse_rad": float(np.sqrt(np.mean(residual * residual))),
            "max_abs_rad": float(np.max(np.abs(residual))),
        },
        "camera_json": str(camera_path.resolve()),
        "intrinsics_json": str(intrinsics_path.resolve()),
        "poses_json": str(poses_path.resolve()),
        "rows": rows,
    }
    (args.out_dir / "archive_roll_registration.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["old_vs_reconstructed"], indent=2))
    print(
        f"[archive-roll] optical-axis difference "
        f"{output['axis_from_camera_optical_deg']:.3f} deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
