#!/usr/bin/env python3
"""Prepare archived thumb-CMC-yaw photographs for same-view Isaac rendering.

The formal yaw sweep measured a 2-D director-angle change relative to a rigid
blue palm L.  This tool reconstructs one camera from that same rigid L, maps
the physical reference pose to URDF local q=0, and emits representative
archived poses without using the moving thumb marker to move the camera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from tools.measure_g20_thumb_yaw import (
    _blue,
    _major_extent,
    _pair_in_depth_band,
)
from tools.prepare_g20_archive_chain_registration import _camera_payload, _unit


def _points_camera(
    pixels: np.ndarray, depth_m: np.ndarray, intr: dict
) -> np.ndarray:
    rows = pixels[:, 1].astype(int)
    cols = pixels[:, 0].astype(int)
    z = depth_m[rows, cols]
    valid = (z > 0.05) & (z < 3.0)
    u = pixels[valid, 0]
    v = pixels[valid, 1]
    z = z[valid]
    return np.column_stack(
        (
            (u - intr["ppx"]) / intr["fx"] * z,
            (v - intr["ppy"]) / intr["fy"] * z,
            z,
        )
    )


def _axis3(points: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(points - np.median(points, axis=0), full_matrices=False)
    return _unit(vh[0])


def _source_summary(session: Path, source: str) -> Path:
    if source == "reference_raw251":
        return session / "reference_raw251/camera_a_fixed_exp_180f_summary.json"
    path = session / source
    if path.is_file():
        return path
    raise FileNotFoundError(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--points-summary", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--sdk-snapshot", type=Path, required=True)
    parser.add_argument(
        "--select-readback-raw",
        default="251,121,17",
        help="comma-separated stable readbacks; nearest formal point is used",
    )
    parser.add_argument(
        "--palm-long-base",
        nargs=3,
        type=float,
        default=(-0.0373939909, -0.030, 0.125),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--palm-cross-base",
        nargs=3,
        type=float,
        default=(-0.0373939909, 0.000, 0.065),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    intr = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    reference_summary = (
        args.session / "reference_raw251/camera_a_fixed_exp_180f_summary.json"
    )
    reference = json.loads(reference_summary.read_text(encoding="utf-8"))
    color_path = Path(reference["outputs"]["color_png"])
    depth_path = Path(reference["outputs"]["depth_raw_npy"])
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f"failed to read {color_path}")
    depth_raw = np.load(depth_path)
    depth_scale = float(reference["camera"]["depth_scale_m_per_unit"])
    depth_m = depth_raw.astype(np.float64) * depth_scale

    detector = SimpleNamespace(
        hsv_lower=(75, 40, 20),
        hsv_upper=(165, 255, 255),
    )
    mask = _blue(color, detector)
    palm_pair = _pair_in_depth_band(
        mask,
        depth_m,
        (0.305, 0.350),
        (440, 430, 900, 660),
        500,
        "palm",
    )
    palm_pair.sort(key=lambda pixels: -_major_extent(pixels))
    palm_long_pixels, palm_cross_pixels = palm_pair
    palm_long_camera = _points_camera(palm_long_pixels, depth_m, intr)
    palm_cross_camera = _points_camera(palm_cross_pixels, depth_m, intr)
    long_axis_camera = _axis3(palm_long_camera)
    cross_axis_camera = _axis3(palm_cross_camera)
    if long_axis_camera[0] < 0.0:
        long_axis_camera = -long_axis_camera
    cross_axis_camera = _unit(
        cross_axis_camera
        - long_axis_camera * float(cross_axis_camera @ long_axis_camera)
    )
    if cross_axis_camera[1] < 0.0:
        cross_axis_camera = -cross_axis_camera

    # URDF columns are base x, y, z in the camera frame.  The two arms of the
    # physical palm L are aligned with base z (longitudinal) and base y.
    base_z_camera = long_axis_camera
    base_y_camera = cross_axis_camera
    base_x_camera = _unit(np.cross(base_y_camera, base_z_camera))
    rotation_camera_from_base = np.column_stack(
        (base_x_camera, base_y_camera, base_z_camera)
    )

    observed_centers = np.stack(
        (
            np.median(palm_long_camera, axis=0),
            np.median(palm_cross_camera, axis=0),
        )
    )
    base_centers = np.stack(
        (
            np.asarray(args.palm_long_base, dtype=float),
            np.asarray(args.palm_cross_base, dtype=float),
        )
    )
    translations = (
        observed_centers
        - base_centers @ rotation_camera_from_base.T
    )
    translation_camera_from_base = translations.mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotation_camera_from_base
    transform[:3, 3] = translation_camera_from_base

    camera_path = args.out_dir / "camera_thumb_yaw_palm_l_initializer.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )
    intrinsics_path = args.out_dir / "intrinsics.json"
    intrinsics_path.write_text(
        json.dumps(intr, indent=2) + "\n", encoding="utf-8"
    )

    formal = json.loads(args.points_summary.read_text(encoding="utf-8"))
    formal_rows = formal["points"]
    desired = [
        int(value)
        for value in args.select_readback_raw.split(",")
        if value.strip()
    ]
    selected = []
    for target in desired:
        row = min(
            formal_rows,
            key=lambda item: abs(int(item["stable_readback_raw"]) - target),
        )
        if row not in selected:
            selected.append(row)

    snapshot = json.loads(args.sdk_snapshot.read_text(encoding="utf-8"))
    pose_template = np.median(
        np.asarray(
            [sample["joint_rad16"] for sample in snapshot["samples"]],
            dtype=float,
        ),
        axis=0,
    )
    poses = {}
    rows = []
    for row in selected:
        summary_path = _source_summary(args.session, row["source"])
        frame = json.loads(summary_path.read_text(encoding="utf-8"))
        pose = pose_template.copy()
        pose[12] = float(row["yaw_rad"])
        name = (
            f"raw{int(row['stable_readback_raw']):03d}"
            f"_oldq_{float(row['yaw_rad']):.6f}"
        )
        poses[name] = pose.tolist()
        rows.append(
            {
                "command_raw": int(row["command_raw"]),
                "stable_readback_raw": int(row["stable_readback_raw"]),
                "old_visual_q_rad": float(row["yaw_rad"]),
                "old_visual_q_deg": float(row["yaw_deg"]),
                "summary": str(summary_path.resolve()),
                "color": frame["outputs"]["color_png"],
                "pose_name": name,
                "palm_reference_mode": row["palm_reference_mode"],
                "lut_source": row["lut_source"],
            }
        )
    poses_path = args.out_dir / "poses_old_visual_q.json"
    poses_path.write_text(
        json.dumps(poses, indent=2) + "\n", encoding="utf-8"
    )

    predicted = base_centers @ rotation_camera_from_base.T + translation_camera_from_base
    anchor_error = predicted - observed_centers
    payload = {
        "schema_version": 1,
        "method": "thumb_yaw_formal_2d_with_rigid_palm_l_camera",
        "eligible_for_deployment_lut": False,
        "joint": "thumb_cmc_yaw",
        "measurement_contract": formal["measurement_contract"],
        "camera_initializer_json": str(camera_path.resolve()),
        "intrinsics_json": str(intrinsics_path.resolve()),
        "poses_json": str(poses_path.resolve()),
        "reference_summary": str(reference_summary.resolve()),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "palm_long_axis_camera": long_axis_camera.tolist(),
        "palm_cross_axis_camera": cross_axis_camera.tolist(),
        "palm_anchor_base_m": base_centers.tolist(),
        "palm_anchor_observed_camera_m": observed_centers.tolist(),
        "palm_anchor_fit_error_mm": (anchor_error * 1000.0).tolist(),
        "palm_anchor_fit_rmse_mm": float(
            np.sqrt(np.mean(anchor_error * anchor_error)) * 1000.0
        ),
        "rows": rows,
    }
    (args.out_dir / "archive_thumb_yaw_registration.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "palm_anchor_fit_rmse_mm": payload[
                    "palm_anchor_fit_rmse_mm"
                ],
                "selected": [
                    {
                        "stable_raw": row["stable_readback_raw"],
                        "q_rad": row["old_visual_q_rad"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )
    print("[thumb-yaw] wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
