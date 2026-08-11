#!/usr/bin/env python3
"""Compose one formal G20 photo with a focused Isaac render and overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from tools.manual_match_g20_joint_q import (
    fit,
    intrinsic_correct,
    label,
    overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--focus-finger", required=True)
    parser.add_argument("--stable-raw", type=int, required=True)
    parser.add_argument("--q-rad", type=float, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument(
        "--roi", nargs=4, type=int, default=(260, 70, 1080, 650),
        metavar=("X1", "Y1", "X2", "Y2"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ROI {roi} is outside image {width}x{height}")
    return image[y1:y2, x1:x2]


def main() -> int:
    args = parse_args()
    for path in (args.real, args.sim, args.intrinsics, args.render_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    render = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    if render.get("focus_finger") != args.focus_finger:
        raise ValueError("render manifest focus_finger does not match request")
    hidden = list(render.get("hidden_body_names", []))
    if "hand_base_link" in hidden:
        raise RuntimeError("focused render unexpectedly hides the palm")
    if any(name.startswith(f"{args.focus_finger}_") for name in hidden):
        raise RuntimeError("focused render unexpectedly hides a target body")

    real = cv2.imread(str(args.real), cv2.IMREAD_COLOR)
    sim = cv2.imread(str(args.sim), cv2.IMREAD_COLOR)
    if real is None or sim is None:
        raise RuntimeError("failed to read real or sim image")
    intr = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    corrected = intrinsic_correct(sim, intr)
    if corrected.shape[:2] != real.shape[:2]:
        raise RuntimeError(
            f"shape mismatch real={real.shape} sim={corrected.shape}"
        )
    outlined = overlay(real, corrected)
    roi = tuple(args.roi)
    panel_width = 560
    panel_height = 418
    body_height = panel_height - 58
    panels = [
        label(
            fit(crop(real, roi), panel_width, body_height),
            f"FORMAL D435 | stable raw {args.stable_raw:03d}",
            f"view gate for {args.joint}",
        ),
        label(
            fit(crop(corrected, roi), panel_width, body_height),
            f"FOCUSED ISAAC | q={args.q_rad:.3f} rad",
            f"palm + {args.focus_finger} only | diagnostic pose",
        ),
        label(
            fit(crop(outlined, roi), panel_width, body_height),
            "SAME-PIXEL FOCUS OVERLAY",
            "orange=edge, blue=open fill",
        ),
    ]
    contact = np.hstack(panels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contact_path = args.out_dir / "focus_view_gate_ab.png"
    cv2.imwrite(str(contact_path), contact)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending_user_view_approval",
        "joint": args.joint,
        "focus_finger": args.focus_finger,
        "stable_readback_raw": args.stable_raw,
        "diagnostic_pose_q_rad": args.q_rad,
        "roi_xyxy": list(roi),
        "real": str(args.real.resolve()),
        "real_sha256": sha256(args.real),
        "sim": str(args.sim.resolve()),
        "sim_sha256": sha256(args.sim),
        "intrinsics": str(args.intrinsics.resolve()),
        "intrinsics_sha256": sha256(args.intrinsics),
        "render_manifest": str(args.render_manifest.resolve()),
        "render_manifest_sha256": sha256(args.render_manifest),
        "hidden_body_names": hidden,
        "contact_sheet": str(contact_path.resolve()),
    }
    manifest_path = args.out_dir / "focus_view_gate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "contact_sheet": str(contact_path),
        "manifest": str(manifest_path),
        "hidden_body_count": len(hidden),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
