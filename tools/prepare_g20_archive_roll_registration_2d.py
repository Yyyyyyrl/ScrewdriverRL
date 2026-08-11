#!/usr/bin/env python3
"""Prepare MCP-roll sweeps using their primary 2-D measurement contract.

The camera optical axis is constrained to the URDF roll axis.  A 2-D circle fit
to the moving marker centroid supplies the joint pivot; no depth-PCA rotation
axis is fitted.
"""

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
    _unit,
    _zero_fk,
)


def _director_delta_deg(value: float, zero: float) -> float:
    return (value - zero + 90.0) % 180.0 - 90.0


def _circle_2d(points: np.ndarray) -> tuple[np.ndarray, float]:
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    rhs = x * x + y * y
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    center = solution[:2]
    radius = math.sqrt(max(0.0, solution[2] + center @ center))
    return center, radius


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
        summary_path = args.session / source["source"]
        if not summary_path.is_file():
            summary_path = summary_path / "camera_a_fixed_exp_180f_summary.json"
        if not summary_path.is_file():
            continue
        payload = json.loads(summary_path.read_text())
        marker = payload["last_markers"][args.finger]
        intr = payload["camera"]["intrinsics"]
        rows.append(
            {
                "command_raw": int(round(float(source["command_raw"]))),
                "stable_readback_raw": float(source["stable_readback_raw"]),
                "old_visual_q_rad": float(source["abduction_rad"]),
                "summary": str(summary_path.resolve()),
                "color": payload.get("outputs", {}).get("color_png"),
                "intrinsics": intr,
                "heading_deg_2d": float(marker["heading2_deg"]),
                "centroid_uv": np.asarray(marker["centroid_uv"], dtype=float),
                "depth_m": float(marker["depth_median_m"]),
            }
        )
    if len(rows) < 8:
        raise RuntimeError(f"only {len(rows)} usable observations")

    zero_rows = [
        row for row in rows if row["command_raw"] == args.zero_command_raw
    ]
    if not zero_rows:
        raise RuntimeError("zero-command camera observation is missing")
    zero_heading = float(np.median([row["heading_deg_2d"] for row in zero_rows]))
    reconstructed = np.radians(
        [
            _director_delta_deg(row["heading_deg_2d"], zero_heading)
            for row in rows
        ]
    )
    old_q = np.asarray([row["old_visual_q_rad"] for row in rows])
    if np.corrcoef(reconstructed, old_q)[0, 1] < 0.0:
        reconstructed = -reconstructed
        heading_sign = -1.0
    else:
        heading_sign = 1.0

    uv = np.stack([row["centroid_uv"] for row in rows])
    center_uv, radius_px = _circle_2d(uv)
    depth = float(np.median([row["depth_m"] for row in rows]))
    intr = rows[0]["intrinsics"]
    center_camera = np.asarray(
        [
            (center_uv[0] - intr["ppx"]) / intr["fx"] * depth,
            (center_uv[1] - intr["ppy"]) / intr["fy"] * depth,
            depth,
        ]
    )
    axis_camera = np.asarray([0.0, 0.0, heading_sign])
    theta = math.radians(zero_heading)
    zero_camera = np.asarray([math.cos(theta), math.sin(theta), 0.0])

    joint_origin_base, axis_base, child_zero_base = _zero_fk(
        args.urdf, args.joint
    )
    child_zero_base = _unit(
        child_zero_base - axis_base * float(child_zero_base @ axis_base)
    )
    base_frame = np.column_stack(
        (np.cross(axis_base, child_zero_base), axis_base, child_zero_base)
    )
    camera_frame = np.column_stack(
        (np.cross(axis_camera, zero_camera), axis_camera, zero_camera)
    )
    rotation_camera_from_base = camera_frame @ base_frame.T
    translation_camera_from_base = (
        center_camera - rotation_camera_from_base @ joint_origin_base
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_camera_from_base
    transform[:3, 3] = translation_camera_from_base

    residual = old_q - reconstructed
    for row, q, delta in zip(rows, reconstructed, residual):
        row["reconstructed_local_q_rad"] = float(q)
        row["old_minus_reconstructed_rad"] = float(delta)
        row["centroid_uv"] = row["centroid_uv"].tolist()

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

    camera_path = args.out_dir / "camera_archive_roll_2d.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )
    intrinsics_path = args.out_dir / "intrinsics.json"
    intrinsics_path.write_text(
        json.dumps(
            {
                "width": 1280,
                "height": 720,
                **{key: float(intr[key]) for key in ("fx", "fy", "ppx", "ppy")},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    poses_path = args.out_dir / "poses_old_visual_q.json"
    poses_path.write_text(json.dumps(poses, indent=2) + "\n", encoding="utf-8")
    output = {
        "schema_version": 2,
        "method": "archive_roll_primary_2d_optical_axis_constrained",
        "eligible_for_deployment_lut": False,
        "session": str(args.session.resolve()),
        "joint": args.joint,
        "finger_marker": args.finger,
        "zero_command_raw": args.zero_command_raw,
        "zero_heading_deg_2d": zero_heading,
        "axis_camera": axis_camera.tolist(),
        "joint_center_uv": center_uv.tolist(),
        "joint_center_depth_m": depth,
        "marker_circle_radius_px": radius_px,
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
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
    (args.out_dir / "archive_roll_registration_2d.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["old_vs_reconstructed"], indent=2))
    print("[archive-roll-2d] camera optical axis constrained to URDF roll axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
