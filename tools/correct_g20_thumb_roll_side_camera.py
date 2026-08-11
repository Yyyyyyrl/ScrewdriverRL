#!/usr/bin/env python3
"""Build the thumb-roll side camera with the white marker's correct axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.prepare_g20_archive_chain_registration import _camera_payload, _unit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument(
        "--white-marker-base",
        nargs=3,
        type=float,
        default=(-0.010, 0.0554660112, 0.090),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.registration.read_text(encoding="utf-8"))
    long_axis = _unit(np.asarray(source["palm_long_axis_camera"], dtype=float))
    short_axis = _unit(
        np.asarray(source["palm_short_axis_camera"], dtype=float)
    )
    normal = _unit(np.asarray(source["palm_normal_camera"], dtype=float))

    # White marker plane = hand side plane (base x-z), not dorsal plane.
    base_z_camera = _unit(-long_axis)
    base_x_camera = _unit(
        short_axis - base_z_camera * float(short_axis @ base_z_camera)
    )
    base_y_camera = _unit(np.cross(base_z_camera, base_x_camera))
    if float(base_y_camera @ normal) < 0.0:
        base_x_camera = -base_x_camera
        base_y_camera = -base_y_camera
    rotation_camera_from_base = np.column_stack(
        (base_x_camera, base_y_camera, base_z_camera)
    )

    marker_base = np.asarray(args.white_marker_base, dtype=float)
    marker_camera = np.asarray(source["palm_center_camera_m"], dtype=float)
    translation_camera_from_base = (
        marker_camera - rotation_camera_from_base @ marker_base
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_camera_from_base
    transform[:3, 3] = translation_camera_from_base
    camera_path = args.out_dir / "camera_thumb_roll_true_side.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "method": "white_marker_xz_side_plane_camera",
        "camera_json": str(camera_path.resolve()),
        "white_marker_base_m": marker_base.tolist(),
        "white_marker_camera_m": marker_camera.tolist(),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "base_x_camera_dot_white_short": float(
            base_x_camera @ short_axis
        ),
        "base_y_camera_dot_white_normal": float(
            base_y_camera @ normal
        ),
        "base_z_camera_dot_negative_white_long": float(
            base_z_camera @ (-long_axis)
        ),
    }
    (args.out_dir / "side_camera_registration.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
