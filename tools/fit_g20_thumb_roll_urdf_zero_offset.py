#!/usr/bin/env python3
"""Fit the URDF-local zero offset for the formal thumb-roll measurement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from tools.calibrate_g20_camera_extrinsic import base_to_camera, camera_pose
from tools.prepare_g20_archive_chain_registration import _unit


def _wrap(value: np.ndarray) -> np.ndarray:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-sweep", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sweep = json.loads(args.full_sweep.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    camera_data = json.loads(args.camera.read_text(encoding="utf-8"))
    view = camera_data[0] if isinstance(camera_data, list) else camera_data
    position, rotation_base_from_camera = camera_pose(view)
    rotation_camera_from_base = base_to_camera(
        position, rotation_base_from_camera
    )[:3, :3]

    zero_pose = manifest["poses"]["raw248_oldq_0.000000"]["fk"]
    metacarpals = np.asarray(
        zero_pose["thumb_metacarpals"]["pos"], dtype=float
    )
    proximal = np.asarray(zero_pose["thumb_proximal"]["pos"], dtype=float)
    vector_zero_base = _unit(proximal - metacarpals)

    def base1_rotation(name: str) -> np.ndarray:
        quat = manifest["poses"][name]["fk"][
            "thumb_metacarpals_base1"
        ]["quat_wxyz"]
        w, x, y, z = quat
        return Rotation.from_quat([x, y, z, w]).as_matrix()

    rotation_zero = base1_rotation("raw248_oldq_0.000000")
    rotation_max = base1_rotation("raw004_oldq_1.439939")
    delta = rotation_max @ rotation_zero.T
    rotvec = Rotation.from_matrix(delta).as_rotvec()
    axis_base = _unit(rotvec)
    # The renderer wrote +q; align the recovered axis with that direction.
    if np.linalg.norm(rotvec) < 0.5:
        raise RuntimeError("endpoint URDF rotations do not span the roll sweep")

    rows = sweep["rows"]
    q = np.asarray([float(row["old_visual_q_rad"]) for row in rows])
    vectors_camera = np.stack(
        [np.asarray(row["vector_camera"], dtype=float) for row in rows]
    )
    measured_azimuth = np.arctan2(
        vectors_camera[:, 1], vectors_camera[:, 0]
    )

    def predict(offset: float) -> np.ndarray:
        vectors_base = np.stack(
            [
                Rotation.from_rotvec(axis_base * (value + offset)).apply(
                    vector_zero_base
                )
                for value in q
            ]
        )
        vectors = vectors_base @ rotation_camera_from_base.T
        return np.arctan2(vectors[:, 1], vectors[:, 0])

    candidates = []
    for marker_sign in (1.0, -1.0):
        measured = (
            measured_azimuth
            if marker_sign > 0.0
            else _wrap(measured_azimuth + math.pi)
        )

        def objective(offset: float) -> float:
            residual = _wrap(measured - predict(offset))
            return float(np.mean(residual * residual))

        result = minimize_scalar(
            objective,
            bounds=(-math.pi, math.pi),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        candidates.append((float(result.fun), marker_sign, float(result.x)))
    _, marker_sign, offset = min(candidates)
    measured = (
        measured_azimuth
        if marker_sign > 0.0
        else _wrap(measured_azimuth + math.pi)
    )
    predicted = predict(offset)
    residual = _wrap(measured - predicted)
    for row, m, p, r in zip(rows, measured, predicted, residual):
        row["measured_marker_azimuth_rad"] = float(m)
        row["predicted_urdf_azimuth_rad"] = float(p)
        row["measured_minus_predicted_azimuth_rad"] = float(r)
        row["fitted_urdf_local_q_rad"] = float(
            row["old_visual_q_rad"] + offset
        )

    payload = {
        "schema_version": 1,
        "method": "formal_thumb_roll_camera_azimuth_to_urdf_local_zero",
        "eligible_for_deployment_lut": False,
        "marker_vector_contract": "thumb_metacarpals_to_thumb_proximal",
        "marker_sign": marker_sign,
        "urdf_axis_base": axis_base.tolist(),
        "visual_q_to_urdf_local_q": {
            "formula": "q_urdf_local = q_visual_semantic + offset_rad",
            "offset_rad": offset,
            "offset_deg": math.degrees(offset),
        },
        "azimuth_residual": {
            "count": len(rows),
            "mean_rad": float(np.mean(residual)),
            "rmse_rad": float(np.sqrt(np.mean(residual * residual))),
            "max_abs_rad": float(np.max(np.abs(residual))),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "visual_q_to_urdf_local_q": payload[
                    "visual_q_to_urdf_local_q"
                ],
                "azimuth_residual": payload["azimuth_residual"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
