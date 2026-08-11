#!/usr/bin/env python3
"""Fit the thumb-roll side-view camera from endpoint archive poses.

The formal blue-marker pair lies along the metacarpal-to-proximal direction.
Only q=0 and q=max endpoints fit the single rigid camera.  The middle pose is
held out, so its direction/position residual tests the URDF local-q convention
instead of being absorbed into the camera.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tools.prepare_g20_archive_chain_registration import _camera_payload, _unit


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(np.clip(float(_unit(a) @ _unit(b)), -1.0, 1.0))
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--fit-readback-raw", default="248,4",
        help="comma-separated endpoint stable readbacks",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fit_raw = {
        int(value)
        for value in args.fit_readback_raw.split(",")
        if value.strip()
    }
    rows = []
    for source in registration["rows"]:
        frame = json.loads(Path(source["summary"]).read_text(encoding="utf-8"))
        markers = frame["last_markers"]
        midpoint_camera = 0.5 * (
            np.asarray(markers["thumb_upper"]["centroid_xyz_m"], dtype=float)
            + np.asarray(markers["thumb_lower"]["centroid_xyz_m"], dtype=float)
        )
        vector_camera = _unit(
            np.asarray(markers["thumb_pair_vector_xyz"], dtype=float)
        )
        fk = manifest["poses"][source["pose_name"]]["fk"]
        metacarpals = np.asarray(fk["thumb_metacarpals"]["pos"], dtype=float)
        proximal = np.asarray(fk["thumb_proximal"]["pos"], dtype=float)
        vector_base = _unit(proximal - metacarpals)
        base1_pos = np.asarray(
            fk["thumb_metacarpals_base1"]["pos"], dtype=float
        )
        w, x, y, z = fk["thumb_metacarpals_base1"]["quat_wxyz"]
        base1_rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
        rows.append(
            {
                "source": source,
                "midpoint_camera": midpoint_camera,
                "vector_camera": vector_camera,
                "vector_base": vector_base,
                "base1_pos": base1_pos,
                "base1_rotation": base1_rotation,
                "fit": int(source["stable_readback_raw"]) in fit_raw,
            }
        )
    fit_rows = [row for row in rows if row["fit"]]
    if len(fit_rows) != 2:
        raise ValueError(f"expected exactly 2 fit endpoints, got {len(fit_rows)}")

    source_vectors = np.stack([row["vector_base"] for row in fit_rows])
    measured_vectors = np.stack([row["vector_camera"] for row in fit_rows])
    candidates = []
    for sign in (1.0, -1.0):
        rotation, rssd = Rotation.align_vectors(
            sign * measured_vectors, source_vectors
        )
        matrix = rotation.as_matrix()
        endpoint_errors = [
            _angle_deg(
                sign * row["vector_camera"],
                matrix @ row["vector_base"],
            )
            for row in fit_rows
        ]
        candidates.append((max(endpoint_errors), float(rssd), sign, matrix))
    _, rssd, marker_sign, rotation_camera_from_base = min(candidates)

    # The marker-pair midpoint is a fixed point of the held downstream thumb
    # assembly.  Solve that local point and camera translation from endpoints.
    design = []
    rhs = []
    for row in fit_rows:
        camera_from_base1 = (
            rotation_camera_from_base @ row["base1_rotation"]
        )
        design.append(
            np.column_stack((camera_from_base1, np.eye(3)))
        )
        rhs.append(
            row["midpoint_camera"]
            - rotation_camera_from_base @ row["base1_pos"]
        )
    design_matrix = np.vstack(design)
    rhs_vector = np.concatenate(rhs)
    solution, *_ = np.linalg.lstsq(design_matrix, rhs_vector, rcond=None)
    marker_midpoint_base1 = solution[:3]
    translation_camera_from_base = solution[3:]

    transform = np.eye(4)
    transform[:3, :3] = rotation_camera_from_base
    transform[:3, 3] = translation_camera_from_base
    camera_path = args.out_dir / "camera_thumb_roll_endpoint_fit.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )

    diagnostics = []
    for row in rows:
        predicted_vector = (
            rotation_camera_from_base @ row["vector_base"]
        )
        predicted_midpoint = (
            rotation_camera_from_base
            @ (
                row["base1_rotation"] @ marker_midpoint_base1
                + row["base1_pos"]
            )
            + translation_camera_from_base
        )
        angle_error = _angle_deg(
            marker_sign * row["vector_camera"], predicted_vector
        )
        position_error = predicted_midpoint - row["midpoint_camera"]
        diagnostics.append(
            {
                "stable_readback_raw": row["source"]["stable_readback_raw"],
                "old_visual_q_rad": row["source"]["old_visual_q_rad"],
                "role": "camera_fit_endpoint" if row["fit"] else "held_out",
                "direction_error_deg": angle_error,
                "midpoint_error_mm": (position_error * 1000.0).tolist(),
                "midpoint_error_norm_mm": float(
                    np.linalg.norm(position_error) * 1000.0
                ),
                "measured_marker_vector_camera": row[
                    "vector_camera"
                ].tolist(),
                "predicted_urdf_vector_camera": predicted_vector.tolist(),
            }
        )
    heldout = [row for row in diagnostics if row["role"] == "held_out"]
    payload = {
        "schema_version": 1,
        "method": "thumb_roll_endpoint_camera_fit_middle_pose_held_out",
        "eligible_for_deployment_lut": False,
        "camera_json": str(camera_path.resolve()),
        "fit_readback_raw": sorted(fit_raw),
        "marker_vector_contract": "thumb_metacarpals_to_thumb_proximal",
        "marker_vector_sign": marker_sign,
        "endpoint_align_rssd": rssd,
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "marker_midpoint_base1_m": marker_midpoint_base1.tolist(),
        "linear_position_solve_condition": float(
            np.linalg.cond(design_matrix)
        ),
        "heldout_direction_rmse_rad": float(
            np.sqrt(
                np.mean(
                    np.radians(
                        [row["direction_error_deg"] for row in heldout]
                    )
                    ** 2
                )
            )
        ),
        "heldout_direction_max_abs_rad": float(
            np.max(
                np.abs(
                    np.radians(
                        [row["direction_error_deg"] for row in heldout]
                    )
                )
            )
        ),
        "diagnostics": diagnostics,
    }
    (args.out_dir / "endpoint_camera_fit.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "marker_vector_sign": marker_sign,
                "linear_position_solve_condition": payload[
                    "linear_position_solve_condition"
                ],
                "heldout_direction_rmse_rad": payload[
                    "heldout_direction_rmse_rad"
                ],
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )
    print("[thumb-roll-endpoint-camera] wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
