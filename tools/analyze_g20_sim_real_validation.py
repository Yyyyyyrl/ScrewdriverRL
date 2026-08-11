#!/usr/bin/env python3
"""Quantify G20 SDK readback and D435/Isaac silhouette agreement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROI = (480, 90, 1000, 550)  # x0, y0, x1, y1; excludes the robot arm and table.


def estimate_background_bgr(image: np.ndarray, patch: int = 24) -> np.ndarray:
    """Estimate the uniform Isaac background from four corner patches."""

    height, width = image.shape[:2]
    corners = np.concatenate(
        [
            image[:patch, :patch].reshape(-1, 3),
            image[:patch, width - patch :].reshape(-1, 3),
            image[height - patch :, :patch].reshape(-1, 3),
            image[height - patch :, width - patch :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(corners, axis=0).astype(np.uint8)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros_like(mask)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= 150:
            result[labels == component] = 255
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(result)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def sim_mask(image: np.ndarray) -> np.ndarray:
    background = estimate_background_bgr(image).astype(np.int16)
    delta = np.max(np.abs(image.astype(np.int16) - background), axis=2)
    return clean_mask(delta > 3)


def real_mask(depth_raw: np.ndarray) -> np.ndarray:
    # At the locked camera pose the hand lies at ~0.30-0.55 m and the white
    # calibration wall is ~0.71 m.  The ROI removes the nearer robot arm.
    return clean_mask((depth_raw >= 250) & (depth_raw <= 600))


def crop(mask: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = ROI
    return mask[y0:y1, x0:x1]


def edge(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))


def mask_metrics(real: np.ndarray, sim: np.ndarray) -> dict:
    real = crop(real)
    sim = crop(sim)
    intersection = int(np.logical_and(real > 0, sim > 0).sum())
    union = int(np.logical_or(real > 0, sim > 0).sum())
    re = edge(real)
    se = edge(sim)
    real_distance = cv2.distanceTransform(255 - re, cv2.DIST_L2, 3)
    sim_distance = cv2.distanceTransform(255 - se, cv2.DIST_L2, 3)
    sim_to_real = real_distance[se > 0]
    real_to_sim = sim_distance[re > 0]
    distances = np.concatenate([sim_to_real, real_to_sim])
    real_points = np.argwhere(real > 0)
    sim_points = np.argwhere(sim > 0)
    centroid_delta = np.linalg.norm(real_points.mean(0) - sim_points.mean(0))
    return {
        "roi_xyxy": list(ROI),
        "silhouette_iou": intersection / union if union else 0.0,
        "symmetric_contour_mean_px": float(distances.mean()),
        "symmetric_contour_median_px": float(np.median(distances)),
        "symmetric_contour_p95_px": float(np.percentile(distances, 95)),
        "centroid_delta_px": float(centroid_delta),
        "real_mask_pixels": int((real > 0).sum()),
        "sim_mask_pixels": int((sim > 0).sum()),
    }


def visual(real_image: np.ndarray, real: np.ndarray, sim: np.ndarray) -> np.ndarray:
    result = real_image.copy()
    tint = np.zeros_like(result)
    tint[:, :, 0] = 255
    result[sim > 0] = cv2.addWeighted(
        result[sim > 0], 0.55, tint[sim > 0], 0.45, 0.0
    )
    result[edge(real) > 0] = (40, 220, 40)
    result[edge(sim) > 0] = (30, 30, 245)
    x0, y0, x1, y1 = ROI
    cv2.rectangle(result, (x0, y0), (x1, y1), (255, 255, 255), 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-summary", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--corrected-sim-dir", type=Path, required=True)
    parser.add_argument("--sim-render-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.hardware_summary.read_text())
    render = json.loads(args.sim_render_manifest.read_text())
    rows: list[dict] = []
    joint_rows: list[dict] = []
    for name, record in summary["poses"].items():
        real_path = args.real_dir / f"{name}_real_color.png"
        depth_path = args.real_dir / f"{name}_real_depth_raw.npy"
        sim_path = args.corrected_sim_dir / f"{name}_sim_intrinsic_corrected.png"
        real_image = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        sim_image = cv2.imread(str(sim_path), cv2.IMREAD_COLOR)
        depth = np.load(depth_path)
        rm = real_mask(depth)
        sm = sim_mask(sim_image)
        metrics = mask_metrics(rm, sm)
        overlay_path = args.out_dir / f"{name}_depth_silhouette_overlay.png"
        cv2.imwrite(str(overlay_path), visual(real_image, rm, sm))
        row = {
            "pose": name,
            **metrics,
            "fault_free": not any(record["faults20_after"]),
            "max_temperature_c": max(record["temperature20_after"]),
            "worst_abs_joint_error_rad": max(
                abs(item["error_rad"]) for item in record["per_joint"]
            ),
            "overlay": str(overlay_path),
        }
        rows.append(row)
        for item in record["per_joint"]:
            joint_rows.append(
                {
                    "pose": name,
                    "joint": item["name"],
                    "target_rad": item["target_rad"],
                    "settled_rad": item["settled_rad"],
                    "error_rad": item["error_rad"],
                    "abs_error_rad": abs(item["error_rad"]),
                    "raw_error": item["raw_error"],
                }
            )

    errors = np.array([item["abs_error_rad"] for item in joint_rows])
    contours = np.array([item["symmetric_contour_mean_px"] for item in rows])
    render_drifts = [
        float(item["max_abs_joint_drift_rad"]) for item in render["poses"].values()
    ]
    payload = {
        "pose_count": len(rows),
        "joint_observation_count": len(joint_rows),
        "all_fault_free": all(item["fault_free"] for item in rows),
        "max_temperature_c": max(item["max_temperature_c"] for item in rows),
        "joint_readback_abs_error_rad": {
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
            "max": float(errors.max()),
        },
        "isaac_settle_max_abs_joint_drift_rad": max(render_drifts),
        "image_metric_scope": {
            "description": (
                "Indicative depth-vs-render silhouette metric inside a fixed hand ROI; "
                "not a calibrated metrology result because real and URDF meshes differ."
            ),
            "roi_xyxy": list(ROI),
            "mean_symmetric_contour_distance_px": float(contours.mean()),
            "min_symmetric_contour_distance_px": float(contours.min()),
            "max_symmetric_contour_distance_px": float(contours.max()),
        },
        "poses": rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.out_dir / "pose_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[key for key in rows[0] if key != "overlay"] + ["overlay"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "joint_readback_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=joint_rows[0].keys())
        writer.writeheader()
        writer.writerows(joint_rows)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
