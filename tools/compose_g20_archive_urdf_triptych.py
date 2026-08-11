#!/usr/bin/env python3
"""Compose archived RGB, OG URDF, calibrated URDF, and dual-edge overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.compose_g20_sim_real_ab import intrinsic_correct, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--og-sim-dir", type=Path, required=True)
    parser.add_argument("--calibrated-sim-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _edges(sim: np.ndarray) -> np.ndarray:
    background = np.full_like(sim, 233)
    foreground = np.max(
        np.abs(sim.astype(np.int16) - background.astype(np.int16)), axis=2
    )
    mask = (foreground > 8).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.dilate(cv2.Canny(mask, 50, 150), np.ones((3, 3), np.uint8))


def _crop(image: np.ndarray, crop: list[int] | None) -> np.ndarray:
    if crop is None:
        return image
    x1, y1, x2, y2 = crop
    return image[y1:y2, x1:x2]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registration = json.loads(args.registration.read_text())
    intr = json.loads(args.intrinsics.read_text())
    panels: list[np.ndarray] = []
    manifest: dict = {"rows": []}

    for row in registration["rows"]:
        raw = int(row["command_raw"])
        q = float(row["old_visual_q_rad"])
        stem = f"raw{raw:03d}_oldq_{q:.6f}"
        og_path = args.og_sim_dir / f"{stem}_sim.png"
        calibrated_path = args.calibrated_sim_dir / f"{stem}_sim.png"
        real_path = Path(row["color"]) if row.get("color") else None
        if (
            real_path is None
            or not real_path.exists()
            or not og_path.exists()
            or not calibrated_path.exists()
        ):
            continue
        real = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        og = cv2.imread(str(og_path), cv2.IMREAD_COLOR)
        calibrated = cv2.imread(str(calibrated_path), cv2.IMREAD_COLOR)
        if real is None or og is None or calibrated is None:
            continue
        og, correction = intrinsic_correct(og, intr)
        calibrated, _ = intrinsic_correct(calibrated, intr)
        overlay = real.copy()
        overlay[_edges(og) > 0] = (20, 60, 240)
        overlay[_edges(calibrated) > 0] = (40, 190, 40)

        images = [
            label(_crop(real, args.crop), f"ARCHIVE RGB | raw {raw}"),
            label(_crop(og, args.crop), f"OG URDF | active q={q:.3f} rad"),
            label(_crop(calibrated, args.crop), "CALIBRATED URDF | same active q"),
            label(
                _crop(overlay, args.crop),
                "OVERLAY | orange=OG, green=calibrated",
            ),
        ]
        target_height = 360
        resized = [
            cv2.resize(
                image,
                (int(round(image.shape[1] * target_height / image.shape[0])), target_height),
                interpolation=cv2.INTER_AREA,
            )
            for image in images
        ]
        panel = np.hstack(resized)
        panel_path = args.out_dir / f"{stem}_archive_og_calibrated.png"
        cv2.imwrite(str(panel_path), panel)
        panels.append(panel)
        manifest["rows"].append(
            {
                "command_raw": raw,
                "old_visual_q_rad": q,
                "real": str(real_path),
                "og_sim": str(og_path),
                "calibrated_sim": str(calibrated_path),
                "panel": str(panel_path),
                "intrinsic_correction": correction,
            }
        )

    if not panels:
        raise RuntimeError("no matching real/OG/calibrated rows")
    width = min(panel.shape[1] for panel in panels)
    panels = [
        cv2.resize(
            panel,
            (width, int(round(panel.shape[0] * width / panel.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
        for panel in panels
    ]
    contact = np.vstack(panels)
    contact_path = args.out_dir / "archive_og_calibrated_contact_sheet.png"
    cv2.imwrite(str(contact_path), contact)
    manifest["contact_sheet"] = str(contact_path)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {contact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
