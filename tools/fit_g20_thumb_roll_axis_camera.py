#!/usr/bin/env python3
"""Fit the formal thumb-roll side-view camera from the measured roll axis.

The replacement white marker proves the camera is fixed and supplies an
in-image longitudinal direction.  The three selected blue-pair vectors supply
the physical roll axis.  The white marker normal is deliberately not treated
as a hand-base normal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tools.prepare_g20_archive_chain_registration import _camera_payload, _unit


def _director_align(values: list[np.ndarray]) -> np.ndarray:
    reference = _unit(values[0])
    aligned = [
        value if float(value @ reference) >= 0.0 else -value
        for value in values
    ]
    return _unit(np.median(np.stack(aligned), axis=0))


def _signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    a_perp = _unit(a - axis * float(a @ axis))
    b_perp = _unit(b - axis * float(b @ axis))
    return math.atan2(
        float(axis @ np.cross(a_perp, b_perp)),
        float(a_perp @ b_perp),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for source in registration["rows"]:
        frame = json.loads(Path(source["summary"]).read_text(encoding="utf-8"))
        markers = frame["last_markers"]
        fk = manifest["poses"][source["pose_name"]]["fk"]
        w, x, y, z = fk["thumb_metacarpals_base1"]["quat_wxyz"]
        base1_rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
        rows.append(
            {
                "source": source,
                "q": float(source["old_visual_q_rad"]),
                "vector_camera": _unit(
                    np.asarray(markers["thumb_pair_vector_xyz"], dtype=float)
                ),
                "midpoint_camera": 0.5
                * (
                    np.asarray(
                        markers["thumb_upper"]["centroid_xyz_m"], dtype=float
                    )
                    + np.asarray(
                        markers["thumb_lower"]["centroid_xyz_m"], dtype=float
                    )
                ),
                "palm_long_axis": _unit(
                    np.asarray(markers["palm_long_axis_xyz"], dtype=float)
                ),
                "base1_rotation": base1_rotation,
                "base1_pos": np.asarray(
                    fk["thumb_metacarpals_base1"]["pos"], dtype=float
                ),
                "metacarpals": np.asarray(
                    fk["thumb_metacarpals"]["pos"], dtype=float
                ),
                "proximal": np.asarray(
                    fk["thumb_proximal"]["pos"], dtype=float
                ),
            }
        )
    rows.sort(key=lambda row: row["q"])
    if len(rows) < 3:
        raise ValueError("at least three poses are required")

    directions = np.stack([row["vector_camera"] for row in rows])
    _, _, vh = np.linalg.svd(
        directions - directions.mean(axis=0), full_matrices=False
    )
    axis_camera = _unit(vh[-1])
    measured_angles = np.asarray(
        [
            _signed_angle(directions[0], direction, axis_camera)
            for direction in directions
        ]
    )
    q = np.asarray([row["q"] - rows[0]["q"] for row in rows])
    if np.corrcoef(q, measured_angles)[0, 1] < 0.0:
        axis_camera = -axis_camera
        measured_angles = -measured_angles

    # Derive the URDF active axis in the hand-base frame from endpoint child
    # rotations.  This automatically includes the held thumb-yaw posture.
    delta = (
        rows[-1]["base1_rotation"]
        @ rows[0]["base1_rotation"].T
    )
    rotvec = Rotation.from_matrix(delta).as_rotvec()
    axis_base = _unit(rotvec)
    if float(np.linalg.norm(rotvec)) * q[-1] < 0.0:
        axis_base = -axis_base

    palm_long_camera = _director_align(
        [row["palm_long_axis"] for row in rows]
    )
    # +base-z points from wrist toward fingertips, which is image-left here.
    target_long = -palm_long_camera
    rotation, rssd = Rotation.align_vectors(
        np.stack((axis_camera, target_long)),
        np.stack((axis_base, np.asarray([0.0, 0.0, 1.0]))),
        weights=np.asarray([4.0, 1.0]),
    )
    rotation_camera_from_base = rotation.as_matrix()

    # Anchor translation on the q=0 blue-pair midpoint.  The physical tape
    # pair straddles the metacarpal/proximal segment; their URDF-origin midpoint
    # is the least-assumptive corresponding point.
    marker_midpoint_base = 0.5 * (
        rows[0]["metacarpals"] + rows[0]["proximal"]
    )
    translation_camera_from_base = (
        rows[0]["midpoint_camera"]
        - rotation_camera_from_base @ marker_midpoint_base
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_camera_from_base
    transform[:3, 3] = translation_camera_from_base
    camera_path = args.out_dir / "camera_thumb_roll_axis_fit.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )

    angle_residual = q - measured_angles
    diagnostics = []
    for row, measured, residual in zip(rows, measured_angles, angle_residual):
        marker_midpoint = 0.5 * (row["metacarpals"] + row["proximal"])
        predicted_midpoint = (
            rotation_camera_from_base @ marker_midpoint
            + translation_camera_from_base
        )
        position_error = predicted_midpoint - row["midpoint_camera"]
        diagnostics.append(
            {
                "stable_readback_raw": row["source"]["stable_readback_raw"],
                "old_visual_q_rad": row["q"],
                "measured_rotation_about_fitted_axis_rad": float(measured),
                "old_q_minus_axis_rotation_rad": float(residual),
                "midpoint_error_mm": (position_error * 1000.0).tolist(),
                "midpoint_error_norm_mm": float(
                    np.linalg.norm(position_error) * 1000.0
                ),
            }
        )
    axis_dot = directions @ axis_camera
    payload = {
        "schema_version": 1,
        "method": "thumb_roll_3d_axis_plus_white_long_camera",
        "eligible_for_deployment_lut": False,
        "camera_json": str(camera_path.resolve()),
        "axis_camera": axis_camera.tolist(),
        "axis_base": axis_base.tolist(),
        "palm_long_camera": palm_long_camera.tolist(),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "orientation_align_rssd": float(rssd),
        "axis_dot_peak_to_peak": float(np.ptp(axis_dot)),
        "q_vs_axis_rotation": {
            "count": len(rows),
            "mean_rad": float(np.mean(angle_residual)),
            "rmse_rad": float(
                np.sqrt(np.mean(angle_residual * angle_residual))
            ),
            "max_abs_rad": float(np.max(np.abs(angle_residual))),
        },
        "diagnostics": diagnostics,
    }
    (args.out_dir / "axis_camera_fit.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "axis_dot_peak_to_peak": payload["axis_dot_peak_to_peak"],
                "q_vs_axis_rotation": payload["q_vs_axis_rotation"],
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )
    print("[thumb-roll-axis-camera] wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
