#!/usr/bin/env python3
"""Build the thumb-roll camera as a four-finger-axis end view.

In the formal white-marker photographs the four fingers point approximately
toward the D435.  The white marker therefore lies in the hand-base x-y plane:
its short image axis maps base x, its long image axis maps base y, and its
normal maps base z.  This is intentionally different from the rejected side
view interpretation.
"""

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
        default=(-0.010, 0.0, 0.150),
        metavar=("X", "Y", "Z"),
        help="approximate rigid palm-marker center in the URDF hand-base frame",
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

    base_x_camera = _unit(short_axis)
    base_y_camera = _unit(
        long_axis - base_x_camera * float(long_axis @ base_x_camera)
    )
    base_z_camera = _unit(np.cross(base_x_camera, base_y_camera))
    if float(base_z_camera @ normal) < 0.0:
        base_y_camera = -base_y_camera
        base_z_camera = -base_z_camera
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

    camera_path = args.out_dir / "camera_thumb_roll_finger_axis.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "method": "white_marker_xy_plane_four_finger_axis_end_view",
        "camera_json": str(camera_path.resolve()),
        "white_marker_base_m": marker_base.tolist(),
        "white_marker_camera_m": marker_camera.tolist(),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "axis_checks": {
            "base_x_dot_white_short": float(base_x_camera @ short_axis),
            "base_y_dot_white_long": float(base_y_camera @ long_axis),
            "base_z_dot_white_normal": float(base_z_camera @ normal),
        },
    }
    (args.out_dir / "finger_axis_camera_registration.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
