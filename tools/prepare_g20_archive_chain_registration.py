#!/usr/bin/env python3
"""Prepare an archived blue-marker planar joint for same-view Isaac rendering.

The archived visual q is never used to fit the camera.  Camera rotation comes
from the moving marker centroid plane plus the fixed parent marker direction;
translation comes from the moving marker's fitted rotation centre.  The output
pose JSON writes only the active semantic joint, leaving mimic resolution to
the selected URDF inside the renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_kinematics as pk
import torch
from scipy.spatial.transform import Rotation


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--camera-source-column", required=True)
    parser.add_argument("--q-column", required=True)
    parser.add_argument("--command-raw-column", default="command_raw")
    parser.add_argument("--stable-raw-column")
    parser.add_argument("--parent-marker", required=True)
    parser.add_argument("--moving-marker", required=True)
    parser.add_argument("--joint", choices=SEMANTIC_ORDER, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--zero-command-raw", type=int, default=255)
    parser.add_argument(
        "--select-command-raw",
        default="255,224,192,160,128,96",
        help="comma-separated archived poses to put in poses.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _director_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle of two undirected marker axes, modulo pi."""
    a = _unit(a - axis * float(a @ axis))
    b = _unit(b - axis * float(b @ axis))
    value = math.atan2(float(axis @ np.cross(a, b)), float(a @ b))
    return (value + math.pi / 2.0) % math.pi - math.pi / 2.0


def _deproject(marker: dict[str, Any], intr: dict[str, float]) -> np.ndarray:
    u, v = marker["centroid_uv"]
    z = float(marker["depth_median_m"])
    return np.asarray(
        [
            (u - intr["ppx"]) / intr["fx"] * z,
            (v - intr["ppy"]) / intr["fy"] * z,
            z,
        ]
    )


def _circle_center(
    points: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, float]:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    radial_u = _unit(vh[0] - axis * float(vh[0] @ axis))
    radial_v = np.cross(axis, radial_u)
    xy = np.column_stack((points @ radial_u, points @ radial_v))
    x, y = xy[:, 0], xy[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    rhs = x * x + y * y
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    center_2d = solution[:2]
    radius = math.sqrt(max(0.0, solution[2] + center_2d @ center_2d))
    axial = float(np.median(points @ axis))
    center = center_2d[0] * radial_u + center_2d[1] * radial_v + axial * axis
    return center, radius


def _joint_contract(
    urdf: Path, joint_name: str
) -> tuple[str, str, np.ndarray]:
    root = ET.parse(urdf).getroot()
    joint = next(item for item in root.findall("joint") if item.get("name") == joint_name)
    parent = joint.find("parent").get("link")
    child = joint.find("child").get("link")
    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
    return parent, child, axis


def _zero_fk(
    urdf: Path, joint_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent, child, local_axis = _joint_contract(urdf, joint_name)
    chain = pk.build_chain_from_urdf(urdf.read_bytes())
    names = chain.get_joint_parameter_names()
    transforms = chain.forward_kinematics(torch.zeros((1, len(names))))
    parent_matrix = transforms[parent].get_matrix()[0].detach().cpu().numpy()
    child_matrix = transforms[child].get_matrix()[0].detach().cpu().numpy()
    joint_axis_base = _unit(parent_matrix[:3, :3] @ local_axis)
    parent_longitudinal_base = _unit(parent_matrix[:3, 2])
    joint_origin_base = child_matrix[:3, 3]
    return joint_origin_base, joint_axis_base, parent_longitudinal_base


def _camera_payload(transform_camera_from_base: np.ndarray) -> list[dict[str, Any]]:
    rotation_camera_from_base = transform_camera_from_base[:3, :3]
    translation_camera_from_base = transform_camera_from_base[:3, 3]
    rotation_base_from_camera = rotation_camera_from_base.T
    position_base = -rotation_base_from_camera @ translation_camera_from_base
    x, y, z, w = Rotation.from_matrix(rotation_base_from_camera).as_quat()
    return [
        {
            "name": "archive_planar_chain",
            "pos": position_base.tolist(),
            "quat": [float(w), float(x), float(y), float(z)],
        }
    ]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for table_row in csv.DictReader(args.points_csv.open(newline="", encoding="utf-8")):
        source = table_row.get(args.camera_source_column, "")
        q_text = table_row.get(args.q_column, "")
        if not source or not q_text:
            continue
        summary_path = Path(source)
        if not summary_path.is_absolute():
            summary_path = args.session / summary_path
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        markers = payload["last_markers"]
        parent = markers[args.parent_marker]
        moving = markers[args.moving_marker]
        if parent.get("axis3_xyz") is None or moving.get("axis3_xyz") is None:
            continue
        intr = payload["camera"]["intrinsics"]
        command_raw = int(round(float(table_row[args.command_raw_column])))
        stable_raw = (
            float(table_row[args.stable_raw_column])
            if args.stable_raw_column and table_row.get(args.stable_raw_column)
            else None
        )
        rows.append(
            {
                "command_raw": command_raw,
                "stable_readback_raw": stable_raw,
                "old_visual_q_rad": float(q_text),
                "summary": str(summary_path.resolve()),
                "color": payload.get("outputs", {}).get("color_png"),
                "intrinsics": intr,
                "parent_axis_camera": _unit(np.asarray(parent["axis3_xyz"])),
                "moving_axis_camera": _unit(np.asarray(moving["axis3_xyz"])),
                "moving_centroid_camera": _deproject(moving, intr),
            }
        )
    if len(rows) < 6:
        raise RuntimeError(f"only {len(rows)} usable observations")

    points = np.stack([row["moving_centroid_camera"] for row in rows])
    _, _, vh = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    axis_camera = _unit(vh[-1])
    raw_angles = np.asarray(
        [
            _director_angle(
                row["parent_axis_camera"], row["moving_axis_camera"], axis_camera
            )
            for row in rows
        ]
    )
    zero_mask = np.asarray(
        [row["command_raw"] == args.zero_command_raw for row in rows]
    )
    if not np.any(zero_mask):
        raise RuntimeError("no zero-command observation")
    raw_angles -= float(np.median(raw_angles[zero_mask]))
    old_q = np.asarray([row["old_visual_q_rad"] for row in rows])
    if np.corrcoef(raw_angles, old_q)[0, 1] < 0.0:
        axis_camera = -axis_camera
        raw_angles = -raw_angles

    parent_camera = _unit(
        np.median(np.stack([row["parent_axis_camera"] for row in rows]), axis=0)
    )
    parent_camera = _unit(
        parent_camera - axis_camera * float(parent_camera @ axis_camera)
    )
    joint_origin_base, axis_base, parent_base = _zero_fk(args.urdf, args.joint)
    parent_base = _unit(parent_base - axis_base * float(parent_base @ axis_base))

    # Build right-handed orthonormal frames and map base frame to camera frame.
    base_frame = np.column_stack(
        (np.cross(axis_base, parent_base), axis_base, parent_base)
    )
    camera_frame = np.column_stack(
        (np.cross(axis_camera, parent_camera), axis_camera, parent_camera)
    )
    rotation_camera_from_base = camera_frame @ base_frame.T
    center_camera, circle_radius = _circle_center(points, axis_camera)
    translation_camera_from_base = (
        center_camera - rotation_camera_from_base @ joint_origin_base
    )
    transform_camera_from_base = np.eye(4)
    transform_camera_from_base[:3, :3] = rotation_camera_from_base
    transform_camera_from_base[:3, 3] = translation_camera_from_base

    delta = old_q - raw_angles
    for row, q_local, residual in zip(rows, raw_angles, delta):
        row["reconstructed_local_q_rad"] = float(q_local)
        row["old_minus_reconstructed_rad"] = float(residual)
        for key in (
            "parent_axis_camera",
            "moving_axis_camera",
            "moving_centroid_camera",
        ):
            row[key] = row[key].tolist()

    selected = {
        int(value)
        for value in args.select_command_raw.split(",")
        if value.strip()
    }
    semantic_index = SEMANTIC_ORDER.index(args.joint)
    poses: dict[str, list[float]] = {}
    for row in rows:
        if row["command_raw"] not in selected:
            continue
        pose = [0.0] * len(SEMANTIC_ORDER)
        pose[semantic_index] = row["old_visual_q_rad"]
        poses[
            f"raw{row['command_raw']:03d}_oldq_{row['old_visual_q_rad']:.6f}"
        ] = pose

    camera_path = args.out_dir / "camera_archive_chain.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform_camera_from_base), indent=2) + "\n",
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
        "method": "archive_fixed_parent_plus_moving_centroid_plane",
        "eligible_for_deployment_lut": False,
        "session": str(args.session.resolve()),
        "points_csv": str(args.points_csv.resolve()),
        "urdf": str(args.urdf.resolve()),
        "joint": args.joint,
        "parent_marker": args.parent_marker,
        "moving_marker": args.moving_marker,
        "observation_count": len(rows),
        "axis_camera": axis_camera.tolist(),
        "axis_from_camera_optical_deg": float(
            np.degrees(math.acos(np.clip(abs(axis_camera[2]), 0.0, 1.0)))
        ),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "joint_center_camera_estimate_m": center_camera.tolist(),
        "marker_circle_radius_m": float(circle_radius),
        "old_vs_reconstructed": {
            "mean_rad": float(np.mean(delta)),
            "rmse_rad": float(np.sqrt(np.mean(delta * delta))),
            "max_abs_rad": float(np.max(np.abs(delta))),
        },
        "camera_json": str(camera_path.resolve()),
        "intrinsics_json": str(intrinsics_path.resolve()),
        "poses_json": str(poses_path.resolve()),
        "rows": rows,
    }
    (args.out_dir / "archive_chain_registration.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["old_vs_reconstructed"], indent=2))
    print(
        f"[archive-chain] optical-axis difference "
        f"{output['axis_from_camera_optical_deg']:.3f} deg"
    )
    print(f"[archive-chain] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
