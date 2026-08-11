#!/usr/bin/env python3
"""Audit formal thumb-roll azimuth radians against a fitted 3-D rotation axis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from tools.prepare_g20_archive_chain_registration import _unit


def _signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    a_perp = _unit(a - axis * float(a @ axis))
    b_perp = _unit(b - axis * float(b @ axis))
    return math.atan2(
        float(axis @ np.cross(a_perp, b_perp)),
        float(a_perp @ b_perp),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    formal = json.loads(args.summary.read_text(encoding="utf-8"))
    q_by_raw = {
        int(row["stable_readback_raw"]): float(row["physical_rad"])
        for row in csv.DictReader(
            args.points_csv.open(newline="", encoding="utf-8")
        )
    }
    rows = []
    for pass_name in ("forward_points", "reverse_points"):
        for source in formal[pass_name]:
            raw = int(source["stable_readback_raw"])
            if raw not in q_by_raw or source["camera_source"] == ".":
                continue
            summary_path = args.session / source["camera_source"]
            frame = json.loads(summary_path.read_text(encoding="utf-8"))
            markers = frame["last_markers"]
            rows.append(
                {
                    "pass": pass_name.replace("_points", ""),
                    "stable_readback_raw": raw,
                    "old_visual_q_rad": q_by_raw[raw],
                    "summary": str(summary_path.resolve()),
                    "vector_camera": _unit(
                        np.asarray(
                            markers["thumb_pair_vector_xyz"], dtype=float
                        )
                    ),
                    "midpoint_camera": 0.5
                    * (
                        np.asarray(
                            markers["thumb_upper"]["centroid_xyz_m"],
                            dtype=float,
                        )
                        + np.asarray(
                            markers["thumb_lower"]["centroid_xyz_m"],
                            dtype=float,
                        )
                    ),
                    "palm_center_camera": np.asarray(
                        markers["palm"]["centroid_xyz_m"], dtype=float
                    ),
                }
            )
    if len(rows) < 12:
        raise RuntimeError(f"only {len(rows)} formal camera rows")

    directions = np.stack([row["vector_camera"] for row in rows])
    _, singular_values, vh = np.linalg.svd(
        directions - directions.mean(axis=0), full_matrices=False
    )
    axis = _unit(vh[-1])
    zero_candidates = [
        row for row in rows if row["stable_readback_raw"] == 248
    ]
    if not zero_candidates:
        raise RuntimeError("semantic zero raw248 camera row missing")
    zero = zero_candidates[0]["vector_camera"]
    measured = np.asarray(
        [_signed_angle(zero, row["vector_camera"], axis) for row in rows]
    )
    q = np.asarray([row["old_visual_q_rad"] for row in rows])
    if np.corrcoef(q, measured)[0, 1] < 0.0:
        axis = -axis
        measured = -measured
    residual = q - measured
    for row, angle, delta in zip(rows, measured, residual):
        row["vector_camera"] = row["vector_camera"].tolist()
        row["midpoint_camera"] = row["midpoint_camera"].tolist()
        row["palm_center_camera"] = row["palm_center_camera"].tolist()
        row["rotation_about_fitted_axis_rad"] = float(angle)
        row["old_visual_q_minus_axis_rotation_rad"] = float(delta)

    axis_dot = directions @ axis
    palm_centers = np.stack(
        [np.asarray(row["palm_center_camera"], dtype=float) for row in rows]
    )
    payload = {
        "schema_version": 1,
        "method": "formal_white_palm_full_sweep_fitted_3d_axis_audit",
        "eligible_for_deployment_lut": False,
        "count": len(rows),
        "axis_camera": axis.tolist(),
        "direction_singular_values": singular_values.tolist(),
        "axis_dot_mean": float(np.mean(axis_dot)),
        "axis_dot_peak_to_peak": float(np.ptp(axis_dot)),
        "palm_center_peak_to_peak_mm": (
            np.ptp(palm_centers, axis=0) * 1000.0
        ).tolist(),
        "old_visual_q_vs_axis_rotation": {
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
                "count": payload["count"],
                "axis_dot_peak_to_peak": payload["axis_dot_peak_to_peak"],
                "palm_center_peak_to_peak_mm": payload[
                    "palm_center_peak_to_peak_mm"
                ],
                "old_visual_q_vs_axis_rotation": payload[
                    "old_visual_q_vs_axis_rotation"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
