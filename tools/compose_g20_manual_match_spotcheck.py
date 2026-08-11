#!/usr/bin/env python3
"""Compose selected physical-raw / URDF-local-q spot-check archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--registration-manifest", type=Path)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--raw-values", nargs="+", type=int, required=True)
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        default=(260, 70, 1080, 650),
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
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"ROI {roi} is outside image {w}x{h}")
    return image[y1:y2, x1:x2]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.out_dir / "panels"
    panels_dir.mkdir(exist_ok=True)

    matches_payload = json.loads(args.matches.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    registration = (
        json.loads(args.registration_manifest.read_text(encoding="utf-8"))
        if args.registration_manifest is not None
        else {"targets": []}
    )
    render = json.loads(
        args.render_manifest.read_text(encoding="utf-8")
    )
    focus_finger = render.get("focus_finger", "none")
    hidden_body_names = list(render.get("hidden_body_names", []))
    intr = json.loads(
        Path(dataset["intrinsics"]).read_text(encoding="utf-8")
    )
    render_dir = args.render_manifest.resolve().parent
    roi = tuple(args.roi)

    lut = summary["physical_lut_urdf_local_q"]
    q_by_raw = dict(
        zip(
            map(int, lut["raw"]),
            map(float, lut["q_urdf_local_rad"]),
        )
    )
    targets = list(registration["targets"])
    target_by_q: dict[float, dict[str, Any]] = {
        round(float(target["q_urdf_rad"]), 8): target
        for target in targets
    }
    matches_by_raw: dict[int, list[dict[str, Any]]] = {}
    for item in matches_payload["matches"]:
        if item["status"] != "matched":
            continue
        matches_by_raw.setdefault(
            int(item["stable_readback_raw"]), []
        ).append(item)

    contact_rows: list[np.ndarray] = []
    archive_rows: list[dict[str, Any]] = []
    for raw in args.raw_values:
        if raw not in q_by_raw:
            raise KeyError(f"raw {raw} is absent from the unique LUT")
        if raw not in matches_by_raw:
            raise KeyError(f"raw {raw} has no matched photo")
        q = q_by_raw[raw]
        for item in matches_by_raw[raw]:
            target = target_by_q.get(round(q, 8))
            render_name = item.get("candidate_name")
            if render_name is None:
                if target is None:
                    raise KeyError(f"no spot-check render target at q={q}")
                render_name = target["name"]
            render_entry = render["poses"][render_name]
            sim_name = next(iter(render_entry["images"].values()))
            sim_path = render_dir / sim_name
            sim = cv2.imread(str(sim_path), cv2.IMREAD_COLOR)
            if sim is None:
                raise RuntimeError(f"failed to read {sim_path}")
            corrected = intrinsic_correct(sim, intr)
            real_path = Path(item["real_color"])
            real = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
            if real is None:
                raise RuntimeError(f"failed to read {real_path}")
            if real.shape[:2] != corrected.shape[:2]:
                raise RuntimeError(
                    f"shape mismatch real={real.shape} sim={corrected.shape}"
                )
            outlined = overlay(real, corrected)
            panel_width = 560
            panel_height = 418
            body_height = panel_height - 58
            real_panel = label(
                fit(crop(real, roi), panel_width, body_height),
                f"FORMAL D435 | stable raw {raw:03d}",
                f"{item['direction']} | command {item['command_raw']:03d}",
            )
            production_limit = dataset.get(
                "production_urdf_limit_rad", [0.0, 1.22]
            )
            outside_original_limit = not (
                float(production_limit[0]) - 1.0e-9
                <= q
                <= float(production_limit[1]) + 1.0e-9
            )
            sim_panel = label(
                fit(crop(corrected, roi), panel_width, body_height),
                (
                    f"FOCUSED OG URDF q={q:.3f} rad"
                    if focus_finger != "none"
                    else f"OG URDF LOCAL q={q:.3f} rad"
                ),
                (
                    f"{np.degrees(q):.2f} deg"
                    + (
                        f" | palm + {focus_finger} only"
                        if focus_finger != "none" else ""
                    )
                    + (
                        " | OUTSIDE ORIGINAL LIMIT"
                        if outside_original_limit
                        else ""
                    )
                ),
                border=(36, 92, 235)
                if outside_original_limit
                else (35, 176, 225),
            )
            overlay_panel = label(
                fit(crop(outlined, roi), panel_width, body_height),
                "SAME-PIXEL OVERLAY",
                "orange=edge, blue=open fill",
            )
            panel = np.hstack((real_panel, sim_panel, overlay_panel))
            panel_path = panels_dir / f"{item['sample_id']}_spotcheck.png"
            cv2.imwrite(str(panel_path), panel)
            contact_rows.append(panel)
            archive_rows.append(
                {
                    "stable_readback_raw": raw,
                    "q_urdf_local_rad": q,
                    "q_urdf_local_deg": float(np.degrees(q)),
                    "sample_id": item["sample_id"],
                    "direction": item["direction"],
                    "command_raw": item["command_raw"],
                    "real_color": str(real_path.resolve()),
                    "sim_render": str(sim_path.resolve()),
                    "panel": str(panel_path.resolve()),
                    "parent_joint": item.get("parent_joint"),
                    "parent_q_urdf_rad": item.get("parent_q_urdf_rad"),
                    "candidate_name": render_name,
                    "outside_original_og_limit": outside_original_limit,
                }
            )

    if not contact_rows:
        raise RuntimeError("no spot-check rows were composed")
    width = min(panel.shape[1] for panel in contact_rows)
    normalized = [
        cv2.resize(
            panel,
            (
                width,
                round(panel.shape[0] * width / panel.shape[1]),
            ),
            interpolation=cv2.INTER_AREA,
        )
        for panel in contact_rows
    ]
    contact = np.vstack(normalized)
    contact_path = args.out_dir / "five_point_spotcheck_contact_sheet.png"
    cv2.imwrite(str(contact_path), contact)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending_visual_review",
        "method": "selected_raw_same_view_real_og_urdf_overlay",
        "joint": dataset["joint"],
        "unique_raw_values": args.raw_values,
        "evidence_panel_count": len(archive_rows),
        "roi_xyxy": list(roi),
        "matches": str(args.matches.resolve()),
        "matches_sha256": sha256(args.matches),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "summary": str(args.summary.resolve()),
        "summary_sha256": sha256(args.summary),
        "registration_manifest": (
            str(args.registration_manifest.resolve())
            if args.registration_manifest is not None else None
        ),
        "registration_manifest_sha256": (
            sha256(args.registration_manifest)
            if args.registration_manifest is not None else None
        ),
        "render_manifest": str(args.render_manifest.resolve()),
        "render_manifest_sha256": sha256(args.render_manifest),
        "focus_finger": focus_finger,
        "hidden_body_names": hidden_body_names,
        "camera": dataset["camera"],
        "camera_sha256": dataset["camera_sha256"],
        "render_urdf": dataset["urdf"],
        "render_urdf_sha256": dataset["urdf_sha256"],
        "production_og_urdf": dataset.get(
            "production_og_urdf", dataset.get("production_urdf")
        ),
        "production_og_urdf_sha256": dataset.get(
            "production_og_urdf_sha256", dataset.get("production_urdf_sha256")
        ),
        "contact_sheet": str(contact_path.resolve()),
        "rows": archive_rows,
    }
    manifest_path = args.out_dir / "five_point_spotcheck_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    report_path = args.out_dir / "README.md"
    table_rows = "\n".join(
        (
            f"| {raw} | {q_by_raw[raw]:.3f} | "
            f"{np.degrees(q_by_raw[raw]):.2f} | "
            f"{len(matches_by_raw[raw])} |"
        )
        for raw in args.raw_values
    )
    report_path.write_text(
        f"""# {dataset['joint']} five-point same-view spot check

Status: **pending visual review**. This archive does not modify runtime config.

| Stable raw | OG URDF local q (rad) | Degrees | Evidence photos |
|---:|---:|---:|---:|
{table_rows}

Each row uses its matched candidate render, preserving any per-photo fixed
parent-joint q. Renders use the camera and intrinsics recorded in the dataset.
Any value outside the production OG limit is explicitly marked; the archive is
candidate evidence only and does not promote a runtime mapping or URDF change.

Isaac visual focus: `{focus_finger}`. Hidden body count:
`{len(hidden_body_names)}`. Focus changes USD visibility only and does not alter
joint states, URDF geometry, or production runtime behavior.

- Contact sheet: `{contact_path.name}`
- Manifest: `{manifest_path.name}`
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "contact_sheet": str(contact_path),
                "manifest": str(manifest_path),
                "report": str(report_path),
                "unique_raw_values": len(args.raw_values),
                "evidence_panels": len(archive_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
