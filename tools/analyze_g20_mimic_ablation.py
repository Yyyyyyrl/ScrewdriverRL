#!/usr/bin/env python3
"""Compare candidate and OG mimic contours only where their renders differ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.analyze_g20_sim_real_validation import ROI, crop, edge, real_mask, sim_mask


def directed_mean(source_edge: np.ndarray, target_edge: np.ndarray, focus: np.ndarray) -> float:
    distance = cv2.distanceTransform(255 - target_edge, cv2.DIST_L2, 3)
    samples = distance[(source_edge > 0) & focus]
    return float(samples.mean()) if samples.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--og-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--focus-dilation-px", type=int, default=17)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    kernel_size = 2 * args.focus_dilation_px + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    rows = []
    for depth_path in sorted(args.real_dir.glob("*_real_depth_raw.npy")):
        name = depth_path.name.removesuffix("_real_depth_raw.npy")
        real_image = cv2.imread(str(args.real_dir / f"{name}_real_color.png"))
        candidate_image = cv2.imread(
            str(args.candidate_dir / f"{name}_sim_intrinsic_corrected.png")
        )
        og_image = cv2.imread(str(args.og_dir / f"{name}_sim_intrinsic_corrected.png"))
        if real_image is None or candidate_image is None or og_image is None:
            raise FileNotFoundError(f"missing real/candidate/OG image for {name}")

        real = crop(real_mask(np.load(depth_path)))
        candidate = crop(sim_mask(candidate_image))
        og = crop(sim_mask(og_image))
        real_edge = edge(real)
        candidate_edge = edge(candidate)
        og_edge = edge(og)

        changed = cv2.absdiff(candidate, og) > 0
        focus = cv2.dilate(changed.astype(np.uint8), kernel) > 0
        candidate_to_real = directed_mean(candidate_edge, real_edge, focus)
        og_to_real = directed_mean(og_edge, real_edge, focus)
        real_to_candidate = directed_mean(real_edge, candidate_edge, focus)
        real_to_og = directed_mean(real_edge, og_edge, focus)
        candidate_symmetric = float(np.nanmean([candidate_to_real, real_to_candidate]))
        og_symmetric = float(np.nanmean([og_to_real, real_to_og]))

        x0, y0, x1, y1 = ROI
        visual = real_image.copy()
        candidate_full = np.zeros(real_image.shape[:2], np.uint8)
        og_full = np.zeros(real_image.shape[:2], np.uint8)
        focus_full = np.zeros(real_image.shape[:2], np.uint8)
        candidate_full[y0:y1, x0:x1] = candidate_edge
        og_full[y0:y1, x0:x1] = og_edge
        focus_full[y0:y1, x0:x1] = focus.astype(np.uint8) * 255
        dimmed = cv2.addWeighted(visual, 0.35, np.zeros_like(visual), 0.65, 0.0)
        visual[focus_full > 0] = dimmed[focus_full > 0]
        visual[og_full > 0] = (230, 60, 230)
        visual[candidate_full > 0] = (60, 220, 70)
        overlay_path = args.out_dir / f"{name}_mimic_change_band.png"
        cv2.imwrite(str(overlay_path), visual)

        rows.append(
            {
                "pose": name,
                "changed_mask_pixels": int(changed.sum()),
                "focus_pixels": int(focus.sum()),
                "candidate_sim_to_real_px": candidate_to_real,
                "og_sim_to_real_px": og_to_real,
                "candidate_real_to_sim_px": real_to_candidate,
                "og_real_to_sim_px": real_to_og,
                "candidate_symmetric_px": candidate_symmetric,
                "og_symmetric_px": og_symmetric,
                "candidate_minus_og_symmetric_px": candidate_symmetric - og_symmetric,
                "candidate_improves": candidate_symmetric < og_symmetric,
                "overlay": str(overlay_path),
            }
        )

    if not rows:
        raise RuntimeError(f"no depth captures found in {args.real_dir}")
    candidate_mean = float(np.mean([row["candidate_symmetric_px"] for row in rows]))
    og_mean = float(np.mean([row["og_symmetric_px"] for row in rows]))
    payload = {
        "scope": (
            "Diagnostic contour distance restricted to a dilated band around pixels "
            "changed by candidate-vs-OG mimic renders. It isolates follower sensitivity "
            "but remains indicative because real and URDF meshes differ."
        ),
        "roi_xyxy": list(ROI),
        "focus_dilation_px": args.focus_dilation_px,
        "pose_count": len(rows),
        "candidate_better_pose_count": sum(row["candidate_improves"] for row in rows),
        "candidate_mean_symmetric_px": candidate_mean,
        "og_mean_symmetric_px": og_mean,
        "candidate_minus_og_mean_px": candidate_mean - og_mean,
        "candidate_improvement_percent": 100.0 * (og_mean - candidate_mean) / og_mean,
        "poses": rows,
    }
    output = args.out_dir / "metrics.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
