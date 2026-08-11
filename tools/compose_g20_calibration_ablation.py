#!/usr/bin/env python3
"""Compare calibrated-LUT and legacy-affine Isaac renders against one real pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.analyze_g20_sim_real_validation import edge, sim_mask


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 48), (18, 22, 29), -1)
    cv2.putText(
        result, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (245, 247, 250), 2, cv2.LINE_AA,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--calibrated-dir", type=Path, required=True)
    parser.add_argument("--default-dir", type=Path, required=True)
    parser.add_argument("--calibrated-metrics", type=Path, required=True)
    parser.add_argument("--default-metrics", type=Path, required=True)
    parser.add_argument("--calibrated-label", default="CALIBRATED LUT")
    parser.add_argument("--default-label", default="DEFAULT AFFINE")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    calibrated_metrics = {
        row["pose"]: row
        for row in json.loads(args.calibrated_metrics.read_text())["poses"]
    }
    default_metrics = {
        row["pose"]: row
        for row in json.loads(args.default_metrics.read_text())["poses"]
    }
    rows = []
    previews = []
    for real_path in sorted(args.real_dir.glob("*_real_color.png")):
        name = real_path.name.removesuffix("_real_color.png")
        real = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        calibrated = cv2.imread(
            str(args.calibrated_dir / f"{name}_sim_intrinsic_corrected.png"),
            cv2.IMREAD_COLOR,
        )
        default = cv2.imread(
            str(args.default_dir / f"{name}_sim_intrinsic_corrected.png"),
            cv2.IMREAD_COLOR,
        )
        combined = real.copy()
        combined[edge(sim_mask(calibrated)) > 0] = (60, 220, 70)
        combined[edge(sim_mask(default)) > 0] = (230, 60, 230)
        cm = calibrated_metrics[name]["symmetric_contour_mean_px"]
        dm = default_metrics[name]["symmetric_contour_mean_px"]
        panel = np.hstack(
            [
                label(real, f"REAL | {name}"),
                label(calibrated, f"{args.calibrated_label} | {cm:.1f}px"),
                label(default, f"{args.default_label} | {dm:.1f}px"),
                label(
                    combined,
                    f"OVERLAY | green={args.calibrated_label} magenta={args.default_label}",
                ),
            ]
        )
        path = args.out_dir / f"{name}_calibration_ablation.png"
        cv2.imwrite(str(path), panel)
        previews.append(cv2.resize(panel, (1920, 270)))
        rows.append(
            {
                "pose": name,
                "calibrated_contour_mean_px": cm,
                "default_contour_mean_px": dm,
                "calibrated_improvement_percent": 100.0 * (dm - cm) / dm,
                "panel": str(path),
            }
        )
    contact = np.vstack(previews)
    contact_path = args.out_dir / "calibration_ablation_contact_sheet.png"
    cv2.imwrite(str(contact_path), contact)
    means = {
        "calibrated_contour_mean_px": float(
            np.mean([row["calibrated_contour_mean_px"] for row in rows])
        ),
        "default_contour_mean_px": float(
            np.mean([row["default_contour_mean_px"] for row in rows])
        ),
    }
    means["calibrated_improvement_percent"] = (
        100.0
        * (
            means["default_contour_mean_px"]
            - means["calibrated_contour_mean_px"]
        )
        / means["default_contour_mean_px"]
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pose_count": len(rows),
                "calibrated_label": args.calibrated_label,
                "default_label": args.default_label,
                "means": means,
                "poses": rows,
                "contact_sheet": str(contact_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(means, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
