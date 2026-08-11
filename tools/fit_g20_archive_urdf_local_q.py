#!/usr/bin/env python3
"""Fit URDF-local joint q from archived blue-marker RGB-D observations.

The first supported contract is the index MCP pitch session produced by
``measure_g20_index_mcp_pitch``.  The fit uses the archived RGB/depth frames,
not a live hand pose:

1. re-detect the rigid metacarpal and moving proximal blue markers;
2. fit the moving marker's 3-D rotation plane to recover the physical pitch
   axis in the D435 frame;
3. measure signed rotation about that axis relative to the archived raw-255
   zero; and
4. construct an initial Isaac camera transform by aligning the URDF joint axis,
   rigid metacarpal direction, and fitted 3-D rotation centre.

The result is an auditable initializer for Isaac rendering.  A later
silhouette/depth registration may refine camera translation, but it must keep
the target joint q fixed to the value recovered here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tools.measure_g20_index_mcp_pitch import measure_mcp_pitch_frame


RAW_RE = re.compile(r"(?:flex_|camera_raw)(\d{3})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        help="old projected-angle summary used only for comparison columns",
    )
    return parser.parse_args()


def _canonical(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    vector /= np.linalg.norm(vector)
    if vector[0] < 0.0:
        vector = -vector
    return vector


def _raw_from_path(path: Path) -> int | None:
    text = str(path)
    if "raw255_zero" in text:
        return 255
    match = RAW_RE.search(text)
    return int(match.group(1)) if match else None


def _direction(path: Path) -> str:
    return "return" if "return_to_raw255" in str(path) else "forward"


def _deproject(
    uv: tuple[float, float], depth_m: float, intr: dict[str, float]
) -> np.ndarray:
    u, v = uv
    return np.asarray(
        [
            (u - intr["ppx"]) / intr["fx"] * depth_m,
            (v - intr["ppy"]) / intr["fy"] * depth_m,
            depth_m,
        ],
        dtype=float,
    )


def _find_observations(session: Path) -> list[Path]:
    summaries = sorted(session.glob("flex_raw*/camera_fixed_exp_180f_summary.json"))
    summaries += sorted(
        session.glob("raw255_zero/camera_*_fixed_exp_180f_summary.json")
    )
    summaries += sorted(
        session.glob("return_to_raw255/camera_raw*_return_fixed_exp_180f_summary.json")
    )
    return summaries


def _old_rad_by_raw(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, float] = {}
    for anchor in payload.get("anchors", []):
        raw = int(round(float(anchor["stable_readback_raw"])))
        result[raw] = float(anchor["physical_rad"])
    return result


def _joint_origins_axes(
    urdf: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(urdf).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    roll = joints["index_mcp_roll"]
    pitch = joints["index_mcp_pitch"]
    roll_origin = np.fromstring(roll.find("origin").get("xyz"), sep=" ")
    pitch_origin = np.fromstring(pitch.find("origin").get("xyz"), sep=" ")
    pitch_axis = np.fromstring(pitch.find("axis").get("xyz"), sep=" ")
    pitch_origin_base = roll_origin + pitch_origin
    return pitch_origin_base, pitch_axis, np.asarray([0.0, 0.0, 1.0])


def _fit_circle_2d(points: np.ndarray) -> tuple[np.ndarray, float]:
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    rhs = x * x + y * y
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    center = solution[:2]
    radius = math.sqrt(max(0.0, solution[2] + center @ center))
    return center, radius


def _camera_json(transform_camera_from_base: np.ndarray) -> list[dict[str, Any]]:
    rotation_camera_from_base = transform_camera_from_base[:3, :3]
    translation_camera_from_base = transform_camera_from_base[:3, 3]
    rotation_base_from_camera = rotation_camera_from_base.T
    position_base = -rotation_base_from_camera @ translation_camera_from_base
    x, y, z, w = Rotation.from_matrix(rotation_base_from_camera).as_quat()
    return [
        {
            "name": "archive_blue_marker_initializer",
            "pos": position_base.tolist(),
            "quat": [float(w), float(x), float(y), float(z)],
        }
    ]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    old_rad = _old_rad_by_raw(args.summary)
    observations: list[dict[str, Any]] = []

    for summary_path in _find_observations(args.session):
        raw = _raw_from_path(summary_path)
        if raw is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        outputs = summary["outputs"]
        color_path = Path(outputs["color_png"])
        depth_path = Path(outputs["depth_raw_npy"])
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise RuntimeError(f"failed to read {color_path}")
        depth = np.load(depth_path)
        camera = summary["camera"]
        intr = camera["intrinsics"]
        detector = summary["detector"]
        markers, _, diagnostics = measure_mcp_pitch_frame(
            color,
            depth,
            intr,
            float(camera["depth_scale_m_per_unit"]),
            tuple(detector["roi_xyxy"]),
            tuple(detector["hsv_lower"]),
            tuple(detector["hsv_upper"]),
            int(detector["component_min_area_px"]),
            int(detector["palm_min_area_px"]),
            3.0,
            15.0,
            int(detector["proximal_min_area_px"]),
            8.0,
            float(detector["proximal_along_min_px"]),
            float(detector["proximal_along_max_px"]),
            float(detector["proximal_radius_max_px"]),
            float(detector["segmentation_depth_min_m"]),
            float(detector["segmentation_depth_max_m"]),
            3,
        )
        metacarpal = markers["metacarpal"]
        proximal = markers["proximal"]
        if metacarpal.axis3_xyz is None or proximal.axis3_xyz is None:
            continue
        observations.append(
            {
                "summary": str(summary_path.resolve()),
                "color": str(color_path.resolve()),
                "depth": str(depth_path.resolve()),
                "raw": raw,
                "direction": _direction(summary_path),
                "intrinsics": intr,
                "metacarpal_axis_camera": _canonical(
                    np.asarray(metacarpal.axis3_xyz)
                ),
                "proximal_axis_camera": _canonical(
                    np.asarray(proximal.axis3_xyz)
                ),
                "proximal_centroid_camera": _deproject(
                    proximal.centroid_uv,
                    float(proximal.depth_median_m),
                    intr,
                ),
                "palm_endpoint_uv": diagnostics["palm_endpoint_uv"],
                "old_projected_rad": old_rad.get(raw),
            }
        )
    if len(observations) < 6:
        raise RuntimeError(f"only {len(observations)} usable archived observations")

    vectors = np.stack(
        [item["proximal_axis_camera"] for item in observations]
    )
    # Robustly fit the normal of the rotation plane.  The first fit identifies
    # high-residual depth-PCA frames; the second excludes those outliers.
    centered = vectors - np.mean(vectors, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[-1]
    offsets = vectors @ axis
    residual = np.abs(offsets - np.median(offsets))
    cutoff = max(0.025, float(np.quantile(residual, 0.80)))
    inlier = residual <= cutoff
    centered = vectors[inlier] - np.mean(vectors[inlier], axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[-1]
    axis /= np.linalg.norm(axis)

    zero_vectors = np.stack(
        [
            item["proximal_axis_camera"]
            for item in observations
            if item["raw"] >= 254
        ]
    )
    zero = _canonical(np.median(zero_vectors, axis=0))
    zero_projected = zero - axis * float(zero @ axis)
    zero_projected /= np.linalg.norm(zero_projected)

    def signed_angle(vector: np.ndarray) -> float:
        projected = vector - axis * float(vector @ axis)
        projected /= np.linalg.norm(projected)
        return math.atan2(
            float(axis @ np.cross(zero_projected, projected)),
            float(zero_projected @ projected),
        )

    angles = np.asarray([signed_angle(vector) for vector in vectors])
    raw_values = np.asarray([item["raw"] for item in observations], dtype=float)
    if np.corrcoef(angles, 255.0 - raw_values)[0, 1] < 0.0:
        axis = -axis
        angles = -angles
    zero_offset = float(np.median(angles[raw_values >= 254]))
    angles -= zero_offset

    for item, q, plane_residual, keep in zip(
        observations, angles, residual, inlier
    ):
        item["q_urdf_local_rad"] = float(q)
        item["q_urdf_local_deg"] = float(np.degrees(q))
        item["axis_plane_residual"] = float(plane_residual)
        item["axis_fit_inlier"] = bool(keep)
        old = item["old_projected_rad"]
        item["delta_old_projected_minus_q_rad"] = (
            None if old is None else float(old - q)
        )
        for key in (
            "metacarpal_axis_camera",
            "proximal_axis_camera",
            "proximal_centroid_camera",
        ):
            item[key] = item[key].tolist()

    metacarpal_axes = np.stack(
        [np.asarray(item["metacarpal_axis_camera"]) for item in observations]
    )
    z_camera = _canonical(np.median(metacarpal_axes, axis=0))
    y_camera = axis
    z_camera -= y_camera * float(z_camera @ y_camera)
    z_camera /= np.linalg.norm(z_camera)
    x_camera = np.cross(y_camera, z_camera)
    x_camera /= np.linalg.norm(x_camera)
    rotation_camera_from_base = np.column_stack(
        (x_camera, y_camera, z_camera)
    )
    if np.linalg.det(rotation_camera_from_base) < 0.0:
        x_camera = -x_camera
        rotation_camera_from_base = np.column_stack(
            (x_camera, y_camera, z_camera)
        )

    # Fit the rotation centre of the proximal marker centroid.  Its unknown
    # tape offset along the joint axis remains as an axial camera-translation
    # uncertainty and is reported explicitly.
    points = np.stack(
        [
            np.asarray(item["proximal_centroid_camera"])
            for item in observations
            if item["axis_fit_inlier"]
        ]
    )
    basis_u = zero_projected
    basis_v = np.cross(axis, basis_u)
    plane_xy = np.column_stack((points @ basis_u, points @ basis_v))
    circle_center_2d, circle_radius = _fit_circle_2d(plane_xy)
    axial_coordinate = float(np.median(points @ axis))
    joint_center_camera = (
        circle_center_2d[0] * basis_u
        + circle_center_2d[1] * basis_v
        + axial_coordinate * axis
    )
    joint_origin_base, urdf_axis, urdf_link_z = _joint_origins_axes(args.urdf)
    axis_alignment_deg = float(
        np.degrees(
            math.acos(
                np.clip(
                    abs(
                        float(
                            (rotation_camera_from_base @ urdf_axis)
                            @ axis
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
        )
    )
    translation_camera_from_base = (
        joint_center_camera - rotation_camera_from_base @ joint_origin_base
    )
    transform_camera_from_base = np.eye(4)
    transform_camera_from_base[:3, :3] = rotation_camera_from_base
    transform_camera_from_base[:3, 3] = translation_camera_from_base

    camera_path = args.out_dir / "camera_archive_initializer.json"
    camera_path.write_text(
        json.dumps(_camera_json(transform_camera_from_base), indent=2) + "\n",
        encoding="utf-8",
    )
    rows_path = args.out_dir / "archive_urdf_local_q_points.csv"
    fields = [
        "raw",
        "direction",
        "q_urdf_local_rad",
        "q_urdf_local_deg",
        "old_projected_rad",
        "delta_old_projected_minus_q_rad",
        "axis_plane_residual",
        "axis_fit_inlier",
        "summary",
        "color",
        "depth",
    ]
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(observations, key=lambda item: (-item["raw"], item["direction"]))
        )

    comparable = [
        item
        for item in observations
        if item["old_projected_rad"] is not None
        and item["axis_fit_inlier"]
    ]
    differences = np.asarray(
        [item["delta_old_projected_minus_q_rad"] for item in comparable],
        dtype=float,
    )
    payload = {
        "schema_version": 1,
        "method": "archived_blue_marker_rgbd_rotation_axis_fit",
        "eligible_for_deployment_lut": False,
        "eligibility_note": (
            "camera initializer and q fit must be verified by Isaac render "
            "overlay before any LUT is produced"
        ),
        "session": str(args.session.resolve()),
        "urdf": str(args.urdf.resolve()),
        "joint": "index_mcp_pitch",
        "observation_count": len(observations),
        "axis_fit_inlier_count": int(np.sum(inlier)),
        "pitch_axis_camera": axis.tolist(),
        "rigid_metacarpal_axis_camera": z_camera.tolist(),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "transform_camera_from_base": transform_camera_from_base.tolist(),
        "joint_center_camera_estimate_m": joint_center_camera.tolist(),
        "joint_center_axial_uncertainty": (
            "unknown tape offset along the pitch axis; refine translation "
            "against rigid CAD depth before judging silhouettes"
        ),
        "marker_circle_radius_m": float(circle_radius),
        "urdf_axis_alignment_error_deg": axis_alignment_deg,
        "old_projected_comparison": {
            "count": len(comparable),
            "mean_old_minus_q_rad": (
                float(np.mean(differences)) if len(differences) else None
            ),
            "rmse_rad": (
                float(np.sqrt(np.mean(differences * differences)))
                if len(differences)
                else None
            ),
            "max_abs_rad": (
                float(np.max(np.abs(differences)))
                if len(differences)
                else None
            ),
        },
        "camera_json": str(camera_path.resolve()),
        "points_csv": str(rows_path.resolve()),
        "observations": observations,
    }
    (args.out_dir / "archive_urdf_local_q_fit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["old_projected_comparison"], indent=2))
    print(f"[archive-fit] axis inliers {int(np.sum(inlier))}/{len(inlier)}")
    print(f"[archive-fit] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
