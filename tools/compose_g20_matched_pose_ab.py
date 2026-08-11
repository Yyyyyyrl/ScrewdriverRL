#!/usr/bin/env python3
"""Compose name-matched G20 real/Isaac/outline comparison panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def estimate_background_bgr(image: np.ndarray, patch: int = 24) -> np.ndarray:
    """Estimate the uniform Isaac dome background from four corner patches."""

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


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 48), (18, 22, 29), -1)
    cv2.putText(
        result, text, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (245, 247, 250), 2, cv2.LINE_AA,
    )
    return result


def intrinsic_correct(sim: np.ndarray, intr: dict) -> tuple[np.ndarray, dict]:
    height, width = sim.shape[:2]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    ppx = float(intr["ppx"])
    ppy = float(intr["ppy"])
    focal = (fx + fy) / 2.0
    matrix = np.array(
        [
            [fx / focal, 0.0, ppx - (fx / focal) * width / 2.0],
            [0.0, fy / focal, ppy - (fy / focal) * height / 2.0],
        ],
        dtype=np.float32,
    )
    background = estimate_background_bgr(sim)
    corrected = cv2.warpAffine(
        sim, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(value) for value in background),
    )
    return corrected, {
        "matrix_source_sim_to_measured_d435": matrix.tolist(),
        "inferred_sim_background_bgr": background.tolist(),
    }


def sim_contour(sim: np.ndarray) -> np.ndarray:
    background = np.broadcast_to(estimate_background_bgr(sim), sim.shape)
    foreground = np.max(
        np.abs(sim.astype(np.int16) - background.astype(np.int16)), axis=2
    )
    mask = (foreground > 18).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def overlay(real: np.ndarray, corrected_sim: np.ndarray) -> np.ndarray:
    result = real.copy()
    contour = sim_contour(corrected_sim)
    result[contour > 0] = (30, 30, 245)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    intr = json.loads(args.intrinsics.read_text())
    rows: list[dict] = []
    panels: list[np.ndarray] = []

    for real_path in sorted(args.real_dir.glob("*_real_color.png")):
        name = real_path.name.removesuffix("_real_color.png")
        sim_path = args.sim_dir / f"{name}_sim.png"
        if not sim_path.exists():
            raise FileNotFoundError(f"missing matched Isaac image: {sim_path}")
        real = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        sim = cv2.imread(str(sim_path), cv2.IMREAD_COLOR)
        if real is None or sim is None:
            raise RuntimeError(f"failed to read pair {name}")
        if sim.shape[:2] != real.shape[:2]:
            sim = cv2.resize(sim, (real.shape[1], real.shape[0]))
        corrected, correction = intrinsic_correct(sim, intr)
        outlined = overlay(real, corrected)
        panel = np.hstack(
            [
                label(real, f"REAL D435 | {name}"),
                label(corrected, f"ISAAC (settled readback) | {name}"),
                label(outlined, "OVERLAY | red = Isaac silhouette"),
            ]
        )
        panel_path = args.out_dir / f"{name}_ab.png"
        corrected_path = args.out_dir / f"{name}_sim_intrinsic_corrected.png"
        cv2.imwrite(str(panel_path), panel)
        cv2.imwrite(str(corrected_path), corrected)
        panels.append(cv2.resize(panel, (1920, 360)))
        rows.append(
            {
                "pose": name,
                "real": str(real_path),
                "sim": str(sim_path),
                "sim_intrinsic_corrected": str(corrected_path),
                "panel": str(panel_path),
                "intrinsic_correction": correction,
            }
        )

    if not panels:
        raise RuntimeError(f"no *_real_color.png files found under {args.real_dir}")
    contact_sheet = np.vstack(panels)
    contact_path = args.out_dir / "matched_pose_contact_sheet.png"
    cv2.imwrite(str(contact_path), contact_sheet)
    manifest = {
        "pair_count": len(rows),
        "contact_sheet": str(contact_path),
        "pairs": rows,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} matched panels and {contact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
