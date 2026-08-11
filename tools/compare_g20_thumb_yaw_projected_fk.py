#!/usr/bin/env python3
"""Compare archived thumb-yaw visual radians with projected URDF link heading."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.calibrate_g20_camera_extrinsic import base_to_camera, camera_pose


def _director_delta_rad(value: float, zero: float) -> float:
    return (value - zero + math.pi / 2.0) % math.pi - math.pi / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    intr = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    camera_data = json.loads(args.camera.read_text(encoding="utf-8"))
    view = camera_data[0] if isinstance(camera_data, list) else camera_data
    position, rotation_base_from_camera = camera_pose(view)
    transform = base_to_camera(position, rotation_base_from_camera)

    rows = []
    for source in registration["rows"]:
        name = source["pose_name"]
        fk = manifest["poses"][name]["fk"]
        proximal = np.asarray(fk["thumb_proximal"]["pos"], dtype=float)
        distal = np.asarray(fk["thumb_distal"]["pos"], dtype=float)
        points = np.stack((proximal, distal))
        points_camera = (
            points @ transform[:3, :3].T + transform[:3, 3]
        )
        uv = np.column_stack(
            (
                intr["fx"] * points_camera[:, 0] / points_camera[:, 2]
                + intr["ppx"],
                intr["fy"] * points_camera[:, 1] / points_camera[:, 2]
                + intr["ppy"],
            )
        )
        vector = uv[1] - uv[0]
        heading = math.atan2(float(vector[1]), float(vector[0]))
        rows.append(
            {
                "pose_name": name,
                "stable_readback_raw": source["stable_readback_raw"],
                "old_visual_q_rad": source["old_visual_q_rad"],
                "projected_link_heading_rad": heading,
                "projected_link_heading_deg": math.degrees(heading),
                "projected_proximal_uv": uv[0].tolist(),
                "projected_distal_uv": uv[1].tolist(),
            }
        )

    rows.sort(key=lambda row: row["old_visual_q_rad"])
    visual_zero = float(rows[0]["old_visual_q_rad"])
    heading_zero = float(rows[0]["projected_link_heading_rad"])
    visual = np.asarray(
        [float(row["old_visual_q_rad"]) - visual_zero for row in rows]
    )
    projected = np.asarray(
        [
            _director_delta_rad(
                float(row["projected_link_heading_rad"]), heading_zero
            )
            for row in rows
        ]
    )
    if np.corrcoef(visual, projected)[0, 1] < 0.0:
        projected = -projected
        projection_sign = -1.0
    else:
        projection_sign = 1.0
    residual = visual - projected
    for row, v, p, r in zip(rows, visual, projected, residual):
        row["visual_delta_from_zero_rad"] = float(v)
        row["projected_delta_from_zero_rad"] = float(p)
        row["visual_minus_projected_rad"] = float(r)

    payload = {
        "schema_version": 1,
        "method": "same_camera_projected_thumb_proximal_to_distal_director",
        "eligible_for_deployment_lut": False,
        "projection_sign": projection_sign,
        "count": len(rows),
        "mean_rad": float(np.mean(residual)),
        "rmse_rad": float(np.sqrt(np.mean(residual * residual))),
        "max_abs_rad": float(np.max(np.abs(residual))),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "count", "mean_rad", "rmse_rad", "max_abs_rad"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
