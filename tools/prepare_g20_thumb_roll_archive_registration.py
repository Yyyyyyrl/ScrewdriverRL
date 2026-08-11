#!/usr/bin/env python3
"""Prepare the formal white-palm-marker thumb-CMC-roll archive for Isaac.

Only the replacement-white-marker recalibration is accepted.  The rigid palm
plane supplies one camera for all poses; the moving blue thumb pair is reserved
for the physical angle measurement and never moves the camera.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.prepare_g20_archive_chain_registration import _camera_payload, _unit


def _director_align(values: list[np.ndarray]) -> np.ndarray:
    reference = _unit(values[0])
    aligned = [
        value if float(value @ reference) >= 0.0 else -value
        for value in values
    ]
    return _unit(np.median(np.stack(aligned), axis=0))


def _color_for_summary(path: Path, payload: dict) -> Path:
    outputs = payload.get("outputs") or {}
    if outputs.get("color_png"):
        return Path(outputs["color_png"])
    candidate = path.with_name(path.name.replace("_summary.json", "_color.png"))
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--sdk-snapshot", type=Path, required=True)
    parser.add_argument(
        "--select-readback-raw",
        default="248,130,4",
        help="comma-separated formal stable readbacks",
    )
    parser.add_argument(
        "--palm-anchor-base",
        nargs=3,
        type=float,
        default=(-0.0373939909, 0.0, 0.080),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    formal = json.loads(args.summary.read_text(encoding="utf-8"))
    point_rows = {
        int(row["stable_readback_raw"]): row
        for row in csv.DictReader(
            args.points_csv.open(newline="", encoding="utf-8")
        )
    }
    evidence = {}
    for direction in ("forward_points", "reverse_points"):
        for row in formal[direction]:
            stable = int(row["stable_readback_raw"])
            evidence.setdefault(stable, []).append(row)

    selected_raw = [
        int(value)
        for value in args.select_readback_raw.split(",")
        if value.strip()
    ]
    selected = []
    for stable in selected_raw:
        if stable not in point_rows or stable not in evidence:
            raise KeyError(f"stable raw {stable} missing from formal evidence")
        candidates = evidence[stable]
        # Prefer the forward pass in overlap, matching the visible progression;
        # endpoints naturally fall back to their only available pass.
        source = candidates[0]
        camera_source = source["camera_source"]
        if camera_source == ".":
            raise ValueError("raw58 reference is not a semantic endpoint")
        summary_path = args.session / camera_source
        frame = json.loads(summary_path.read_text(encoding="utf-8"))
        selected.append(
            {
                "stable_readback_raw": stable,
                "command_raw": int(source["command_raw"]),
                "q_rad": float(point_rows[stable]["physical_rad"]),
                "q_deg": float(point_rows[stable]["physical_deg"]),
                "lut_source": point_rows[stable]["lut_source"],
                "summary_path": summary_path,
                "frame": frame,
                "color": _color_for_summary(summary_path, frame),
            }
        )

    long_axes = []
    short_axes = []
    normals = []
    centers = []
    for item in selected:
        markers = item["frame"]["last_markers"]
        long_axes.append(
            _unit(np.asarray(markers["palm_long_axis_xyz"], dtype=float))
        )
        short_axes.append(
            _unit(np.asarray(markers["palm_short_axis_xyz"], dtype=float))
        )
        normals.append(
            _unit(np.asarray(markers["palm_normal_xyz"], dtype=float))
        )
        centers.append(
            np.asarray(markers["palm"]["centroid_xyz_m"], dtype=float)
        )
    long_axis = _director_align(long_axes)
    short_axis = _director_align(short_axes)
    normal = _director_align(normals)

    # In this side view the robot wrist is to image-right and fingertips are
    # to image-left, so +base-z (wrist -> fingers) is -palm-long.  The camera
    # sees the negative-x dorsal surface; its outward normal is -base-x.
    base_x_camera = _unit(-normal)
    base_z_camera = _unit(
        -long_axis - base_x_camera * float((-long_axis) @ base_x_camera)
    )
    base_y_camera = _unit(np.cross(base_z_camera, base_x_camera))
    if float(base_y_camera @ short_axis) < 0.0:
        base_y_camera = -base_y_camera
        base_x_camera = -base_x_camera
    rotation_camera_from_base = np.column_stack(
        (base_x_camera, base_y_camera, base_z_camera)
    )

    palm_center_camera = np.median(np.stack(centers), axis=0)
    palm_anchor_base = np.asarray(args.palm_anchor_base, dtype=float)
    translation_camera_from_base = (
        palm_center_camera
        - rotation_camera_from_base @ palm_anchor_base
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_camera_from_base
    transform[:3, 3] = translation_camera_from_base
    camera_path = args.out_dir / "camera_thumb_roll_white_palm.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )

    first_frame = selected[0]["frame"]
    intr = {
        "width": 1280,
        "height": 720,
        **{
            key: float(first_frame["camera"]["intrinsics"][key])
            for key in ("fx", "fy", "ppx", "ppy")
        },
    }
    intrinsics_path = args.out_dir / "intrinsics.json"
    intrinsics_path.write_text(
        json.dumps(intr, indent=2) + "\n", encoding="utf-8"
    )

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
    for item in selected:
        pose = pose_template.copy()
        pose[13] = item["q_rad"]
        name = (
            f"raw{item['stable_readback_raw']:03d}"
            f"_oldq_{item['q_rad']:.6f}"
        )
        poses[name] = pose.tolist()
        rows.append(
            {
                "command_raw": item["command_raw"],
                "stable_readback_raw": item["stable_readback_raw"],
                "old_visual_q_rad": item["q_rad"],
                "old_visual_q_deg": item["q_deg"],
                "pose_name": name,
                "summary": str(item["summary_path"].resolve()),
                "color": str(item["color"]),
                "lut_source": item["lut_source"],
            }
        )
    poses_path = args.out_dir / "poses_old_visual_q.json"
    poses_path.write_text(
        json.dumps(poses, indent=2) + "\n", encoding="utf-8"
    )

    centers_array = np.stack(centers)
    payload = {
        "schema_version": 1,
        "method": "thumb_roll_formal_white_palm_shared_camera",
        "eligible_for_deployment_lut": False,
        "joint": "thumb_cmc_roll",
        "measurement_contract": formal["measurement_contract"],
        "rejected_evidence": formal["rejected_evidence"],
        "camera_json": str(camera_path.resolve()),
        "intrinsics_json": str(intrinsics_path.resolve()),
        "poses_json": str(poses_path.resolve()),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "palm_anchor_base_m": palm_anchor_base.tolist(),
        "palm_center_camera_m": palm_center_camera.tolist(),
        "palm_center_peak_to_peak_mm": (
            np.ptp(centers_array, axis=0) * 1000.0
        ).tolist(),
        "palm_long_axis_camera": long_axis.tolist(),
        "palm_short_axis_camera": short_axis.tolist(),
        "palm_normal_camera": normal.tolist(),
        "rows": rows,
    }
    (args.out_dir / "archive_thumb_roll_registration.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "palm_center_peak_to_peak_mm": payload[
                    "palm_center_peak_to_peak_mm"
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
    print("[thumb-roll] wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
