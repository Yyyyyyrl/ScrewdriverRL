#!/usr/bin/env python3
"""Create auditable real/sim/edge-overlay camera-comparison panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 46), (18, 24, 32), -1)
    cv2.putText(
        result,
        text,
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (245, 248, 252),
        2,
        cv2.LINE_AA,
    )
    return result


def intrinsic_correct(sim: np.ndarray, intr: dict) -> tuple[np.ndarray, dict]:
    h, w = sim.shape[:2]
    averaged_focal = (float(intr["fx"]) + float(intr["fy"])) / 2.0
    sx = float(intr["fx"]) / averaged_focal
    sy = float(intr["fy"]) / averaged_focal
    matrix = np.array(
        [
            [sx, 0.0, float(intr["ppx"]) - sx * w / 2.0],
            [0.0, sy, float(intr["ppy"]) - sy * h / 2.0],
        ],
        dtype=np.float64,
    )
    corrected = cv2.warpAffine(
        sim,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(233, 233, 233),
    )
    return corrected, {
        "isaac_effective_focal_px": averaged_focal,
        "scale_x": sx,
        "scale_y": sy,
        "translate_x_px": float(matrix[0, 2]),
        "translate_y_px": float(matrix[1, 2]),
        "matrix_source_sim_to_measured_d435": matrix.tolist(),
    }


def edge_overlay(real: np.ndarray, sim: np.ndarray) -> np.ndarray:
    background = np.full_like(sim, 233)
    foreground = np.max(np.abs(sim.astype(np.int16) - background.astype(np.int16)), axis=2)
    mask = (foreground > 8).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges = cv2.Canny(mask, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    result = real.copy()
    result[edges > 0] = (20, 40, 255)
    return result


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    real = cv2.imread(str(args.real), cv2.IMREAD_COLOR)
    if real is None:
        raise RuntimeError(f"failed to read {args.real}")
    intr = json.loads(args.intrinsics.read_text())
    rows = []
    manifest = {"real": str(args.real), "rows": []}
    for sim_path in sorted(args.sim_dir.glob("*_sim.png")):
        sim = cv2.imread(str(sim_path), cv2.IMREAD_COLOR)
        if sim is None:
            continue
        corrected, correction = intrinsic_correct(sim, intr)
        corrected_path = args.out_dir / sim_path.name.replace("_sim.png", "_sim_intrinsic_corrected.png")
        cv2.imwrite(str(corrected_path), corrected)
        overlay = edge_overlay(real, corrected)
        name = sim_path.stem.removeprefix("current_readback_pose_").removesuffix("_sim")
        panel = np.hstack(
            (
                label(real, f"REAL D435 | current pose"),
                label(corrected, f"ISAAC | {name}"),
                label(overlay, f"OVERLAY | red = sim silhouette"),
            )
        )
        panel = cv2.resize(panel, (1920, 360), interpolation=cv2.INTER_AREA)
        row_path = args.out_dir / f"ab_{name}.png"
        cv2.imwrite(str(row_path), panel)
        rows.append(label(panel, f"CAMERA CANDIDATE: {name}"))
        manifest["rows"].append(
            {
                "name": name,
                "source_sim": str(sim_path),
                "corrected_sim": str(corrected_path),
                "panel": str(row_path),
                "intrinsic_correction": correction,
            }
        )
    if not rows:
        raise RuntimeError(f"no *_sim.png files found under {args.sim_dir}")
    contact = np.vstack(rows)
    contact_path = args.out_dir / "camera_candidate_ab_contact.png"
    cv2.imwrite(str(contact_path), contact)
    manifest["contact_sheet"] = str(contact_path)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {contact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
