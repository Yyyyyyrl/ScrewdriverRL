#!/usr/bin/env python3
"""Fit one camera for the four archived MCP-roll calibration sessions.

The roll sessions share a fixed D435 and a rigid palm marker.  Per-finger
camera fits are under-constrained because a moving finger marker can explain a
camera translation.  This tool instead:

1. pools the rigid palm marker across all four sessions;
2. estimates the image direction of the URDF finger-row axis from all four MCP
   centers;
3. anchors the palm tape to the visible back surface of the URDF hand base;
4. emits one camera JSON to reuse for every roll render.

Depth on the moving tape is used only for deprojection.  The reported angle
comparison remains the archived 2-D director-angle measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.prepare_g20_archive_chain_registration import (
    _camera_payload,
    _unit,
    _zero_fk,
)


def _deproject(uv: np.ndarray, depth: float, intr: dict) -> np.ndarray:
    return np.asarray(
        [
            (uv[0] - intr["ppx"]) / intr["fx"] * depth,
            (uv[1] - intr["ppy"]) / intr["fy"] * depth,
            depth,
        ],
        dtype=float,
    )


def _director_align(values: list[np.ndarray]) -> np.ndarray:
    reference = _unit(values[0])
    aligned = [
        value if float(value @ reference) >= 0.0 else -value
        for value in values
    ]
    return _unit(np.median(np.stack(aligned), axis=0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration",
        type=Path,
        action="append",
        required=True,
        help="archive_roll_registration_2d.json; repeat for four fingers",
    )
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--visible-base-x-m",
        type=float,
        default=-0.03739399090409279,
        help="URDF x coordinate of the visible dorsal hand-base surface",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.registration) < 3:
        raise ValueError("at least three roll registrations are required")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = []
    palm_centers_all: list[np.ndarray] = []
    palm_axes_all: list[np.ndarray] = []
    intrinsics = None
    for path in args.registration:
        payload = json.loads(path.read_text(encoding="utf-8"))
        intr = json.loads(Path(payload["intrinsics_json"]).read_text())
        if intrinsics is None:
            intrinsics = intr
        elif any(
            abs(float(intr[key]) - float(intrinsics[key])) > 1.0e-6
            for key in ("fx", "fy", "ppx", "ppy")
        ):
            raise ValueError(f"intrinsics differ in {path}")

        palm_centers = []
        palm_axes = []
        for row in payload["rows"]:
            frame = json.loads(Path(row["summary"]).read_text(encoding="utf-8"))
            marker = frame["last_markers"]["palm"]
            palm_centers.append(
                _deproject(
                    np.asarray(marker["centroid_uv"], dtype=float),
                    float(marker["depth_median_m"]),
                    intr,
                )
            )
            if marker.get("axis3_xyz") is not None:
                palm_axes.append(_unit(np.asarray(marker["axis3_xyz"], dtype=float)))
        if not palm_centers or not palm_axes:
            raise RuntimeError(f"missing rigid palm observations in {path}")

        joint_center_camera = _deproject(
            np.asarray(payload["joint_center_uv"], dtype=float),
            float(payload["joint_center_depth_m"]),
            intr,
        )
        joint_origin_base, _, _ = _zero_fk(args.urdf, payload["joint"])
        palm_center = np.median(np.stack(palm_centers), axis=0)
        palm_axis = _director_align(palm_axes)
        palm_centers_all.extend(palm_centers)
        palm_axes_all.extend(palm_axes)
        sessions.append(
            {
                "registration": str(path.resolve()),
                "joint": payload["joint"],
                "joint_origin_base": joint_origin_base,
                "joint_center_camera": joint_center_camera,
                "joint_center_uv": np.asarray(payload["joint_center_uv"], dtype=float),
                "palm_center_camera": palm_center,
                "palm_axis_camera": palm_axis,
            }
        )

    assert intrinsics is not None
    base_centers = np.stack([item["joint_origin_base"] for item in sessions])
    camera_centers = np.stack([item["joint_center_camera"] for item in sessions])

    # The four MCP origins give a strong estimate of the finger-row direction.
    base_y = base_centers[:, 1]
    centered_y = base_y - base_y.mean()
    camera_slope = (
        centered_y[:, None]
        * (camera_centers - camera_centers.mean(axis=0))
    ).sum(axis=0) / float(centered_y @ centered_y)
    base_y_camera = _unit(camera_slope)

    # Resolve the remaining rotation by observing the dorsal surface.  The
    # camera is on the negative-URDF-x side, so +base-x points approximately
    # along +camera-z.  Orthogonalization preserves the measured row direction.
    optical_hint = np.asarray([0.0, 0.0, 1.0])
    base_x_camera = _unit(
        optical_hint - base_y_camera * float(optical_hint @ base_y_camera)
    )
    base_z_camera = _unit(np.cross(base_x_camera, base_y_camera))
    rotation_camera_from_base = np.column_stack(
        (base_x_camera, base_y_camera, base_z_camera)
    )

    # Infer the palm tape's in-plane y/z coordinates independently from every
    # MCP center.  Its x coordinate is fixed to the visible CAD surface because
    # median marker depth belongs to the tape surface, not the revolute pivot.
    tape_base_candidates = []
    for item in sessions:
        delta_camera = (
            item["joint_center_camera"] - item["palm_center_camera"]
        )
        delta_base = rotation_camera_from_base.T @ delta_camera
        tape_base_candidates.append(item["joint_origin_base"] - delta_base)
    tape_base_candidates = np.stack(tape_base_candidates)
    tape_base = np.median(tape_base_candidates, axis=0)
    tape_base[0] = args.visible_base_x_m

    palm_center_camera = np.median(np.stack(palm_centers_all), axis=0)
    translation_camera_from_base = (
        palm_center_camera - rotation_camera_from_base @ tape_base
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_camera_from_base
    transform[:3, 3] = translation_camera_from_base

    camera_path = args.out_dir / "camera_archive_roll_shared_palm.json"
    camera_path.write_text(
        json.dumps(_camera_payload(transform), indent=2) + "\n",
        encoding="utf-8",
    )

    row_axis_marker = _director_align(palm_axes_all)
    if float(row_axis_marker @ base_y_camera) < 0.0:
        row_axis_marker = -row_axis_marker
    diagnostics = []
    errors = []
    for item in sessions:
        predicted = (
            rotation_camera_from_base @ item["joint_origin_base"]
            + translation_camera_from_base
        )
        uv = np.asarray(
            [
                intrinsics["fx"] * predicted[0] / predicted[2] + intrinsics["ppx"],
                intrinsics["fy"] * predicted[1] / predicted[2] + intrinsics["ppy"],
            ]
        )
        error = uv - item["joint_center_uv"]
        errors.append(error)
        diagnostics.append(
            {
                "joint": item["joint"],
                "observed_joint_center_uv": item["joint_center_uv"].tolist(),
                "predicted_joint_origin_uv": uv.tolist(),
                "pixel_error_uv": error.tolist(),
                "pixel_error_norm": float(np.linalg.norm(error)),
                "session_palm_center_camera_m": item[
                    "palm_center_camera"
                ].tolist(),
                "tape_base_candidate_m": tape_base_candidates[
                    len(diagnostics)
                ].tolist(),
            }
        )
    errors_array = np.stack(errors)
    palm_array = np.stack(palm_centers_all)
    payload = {
        "schema_version": 1,
        "method": "shared_roll_camera_rigid_palm_and_four_mcp_rows",
        "eligible_for_deployment_lut": False,
        "camera_json": str(camera_path.resolve()),
        "urdf": str(args.urdf.resolve()),
        "rotation_camera_from_base": rotation_camera_from_base.tolist(),
        "translation_camera_from_base": translation_camera_from_base.tolist(),
        "tape_base_m": tape_base.tolist(),
        "tape_base_candidates_m": tape_base_candidates.tolist(),
        "palm_center_camera_m": palm_center_camera.tolist(),
        "palm_center_peak_to_peak_mm": (
            np.ptp(palm_array, axis=0) * 1000.0
        ).tolist(),
        "base_y_camera_from_four_mcp_centers": base_y_camera.tolist(),
        "palm_marker_axis_camera": row_axis_marker.tolist(),
        "axis_agreement_deg": float(
            np.degrees(
                np.arccos(
                    np.clip(float(base_y_camera @ row_axis_marker), -1.0, 1.0)
                )
            )
        ),
        "joint_origin_reprojection": {
            "rmse_px": float(np.sqrt(np.mean(errors_array * errors_array))),
            "mean_norm_px": float(
                np.mean(np.linalg.norm(errors_array, axis=1))
            ),
            "max_norm_px": float(
                np.max(np.linalg.norm(errors_array, axis=1))
            ),
            "rows": diagnostics,
        },
    }
    (args.out_dir / "shared_camera_registration.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["joint_origin_reprojection"], indent=2))
    print(
        "[shared-roll-camera] rigid palm peak-to-peak mm:",
        payload["palm_center_peak_to_peak_mm"],
    )
    print("[shared-roll-camera] wrote", camera_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
