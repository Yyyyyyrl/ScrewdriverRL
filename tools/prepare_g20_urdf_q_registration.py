#!/usr/bin/env python3
"""Prepare traceable URDF-local-q targets for visual/raw registration.

This tool is deliberately offline.  It never imports the LinkerHand SDK and
never opens a camera.  It creates:

* a pose JSON consumable by ``tools/render_g20_hand_pose.py``;
* a manifest that makes the chosen URDF-local joint coordinate explicit; and
* optionally, an index of the old calibration RGB evidence keyed by raw value.

The old projected tape angles are not imported.  Old images and SDK raw values
remain useful evidence, but the local joint coordinate must come from the URDF
render that visually matches each image.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEMANTIC_ORDER = (
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
    "thumb_cmc_yaw",
    "thumb_cmc_roll",
    "thumb_cmc_pitch",
    "thumb_mcp",
)
RAW_RE = re.compile(r"raw[_-]?(\d{1,3})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", required=True, choices=SEMANTIC_ORDER)
    parser.add_argument(
        "--q-values",
        required=True,
        nargs="+",
        type=float,
        help="URDF-local joint coordinates in radians",
    )
    parser.add_argument(
        "--base-pose",
        type=Path,
        help=(
            "optional JSON containing one 16-value semantic pose; if omitted, "
            "all non-target active joints are zero"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--archive-session",
        type=Path,
        help="optional old calibration session to index without interpreting angles",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("assets/linker_hand_l20_OG/linkerhand_l20_left.urdf"),
        help="URDF defining the local joint coordinate used by the targets",
    )
    return parser.parse_args()


def _load_pose(path: Path | None) -> tuple[list[float], str | None]:
    if path is None:
        return [0.0] * len(SEMANTIC_ORDER), None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if len(payload) != 1:
            raise ValueError("--base-pose object must contain exactly one named pose")
        values = next(iter(payload.values()))
    else:
        values = payload
    if not isinstance(values, list) or len(values) != len(SEMANTIC_ORDER):
        raise ValueError(
            f"--base-pose must resolve to {len(SEMANTIC_ORDER)} semantic values"
        )
    return [float(value) for value in values], str(path.resolve())


def _pose_name(joint: str, q: float) -> str:
    sign = "p" if q >= 0.0 else "m"
    token = f"{abs(q):.4f}".replace(".", "p")
    return f"{joint}_q_{sign}{token}"


def _extract_raw(path: Path, session: Path) -> int | None:
    relative = path.relative_to(session)
    for part in reversed(relative.parts):
        match = RAW_RE.search(part)
        if match:
            raw = int(match.group(1))
            if 0 <= raw <= 255:
                return raw
    return None


def _classify_direction(path: Path) -> str:
    lower = str(path).lower()
    if "return" in lower or "reverse" in lower or "upward" in lower:
        return "return"
    if "zero" in lower or "reference" in lower:
        return "reference"
    return "forward"


def _find_nearby_motion(image: Path, session: Path) -> str | None:
    candidates: list[Path] = []
    for directory in (image.parent, image.parent.parent):
        if directory == session.parent:
            continue
        candidates.extend(directory.glob("*.json"))
    motion = [
        path
        for path in candidates
        if "motion" in path.name.lower()
        or re.search(r"raw\d+_to_raw\d+", path.name.lower())
    ]
    if not motion:
        return None
    return str(sorted(set(motion))[0].resolve())


def _index_archive(session: Path) -> list[dict[str, Any]]:
    if not session.is_dir():
        raise ValueError(f"archive session is not a directory: {session}")
    rows: list[dict[str, Any]] = []
    for image in sorted(session.rglob("*_color.png")):
        raw = _extract_raw(image, session)
        if raw is None:
            continue
        rows.append(
            {
                "raw": raw,
                "direction": _classify_direction(image),
                "color_image": str(image.resolve()),
                "motion_json": _find_nearby_motion(image, session),
                "q_urdf_rad": None,
                "match_status": "unreviewed",
                "note": "Do not copy a projected tape angle into q_urdf_rad.",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if not args.q_values:
        raise ValueError("at least one --q-values entry is required")
    if len(set(args.q_values)) != len(args.q_values):
        raise ValueError("--q-values contains duplicates")
    if not args.urdf.is_file():
        raise ValueError(f"URDF does not exist: {args.urdf}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base, base_source = _load_pose(args.base_pose)
    joint_index = SEMANTIC_ORDER.index(args.joint)
    poses: dict[str, list[float]] = {}
    targets: list[dict[str, Any]] = []
    for q in args.q_values:
        pose = list(base)
        pose[joint_index] = float(q)
        name = _pose_name(args.joint, q)
        poses[name] = pose
        targets.append(
            {
                "name": name,
                "joint": args.joint,
                "semantic_index": joint_index,
                "q_urdf_rad": float(q),
                "q_urdf_deg": float(q * 180.0 / 3.141592653589793),
            }
        )

    poses_path = args.out_dir / "poses_urdf_local_q.json"
    poses_path.write_text(json.dumps(poses, indent=2) + "\n", encoding="utf-8")

    archive_rows: list[dict[str, Any]] = []
    archive_json: str | None = None
    archive_csv: str | None = None
    if args.archive_session is not None:
        archive_rows = _index_archive(args.archive_session)
        archive_json_path = args.out_dir / "old_image_raw_index.json"
        archive_csv_path = args.out_dir / "old_image_raw_index.csv"
        archive_json_path.write_text(
            json.dumps(archive_rows, indent=2) + "\n", encoding="utf-8"
        )
        with archive_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(archive_rows[0]) if archive_rows else [
                "raw",
                "direction",
                "color_image",
                "motion_json",
                "q_urdf_rad",
                "match_status",
                "note",
            ])
            writer.writeheader()
            writer.writerows(archive_rows)
        archive_json = str(archive_json_path.resolve())
        archive_csv = str(archive_csv_path.resolve())

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "visual_registration_to_urdf_local_joint_coordinate",
        "coordinate_contract": {
            "ground_truth_coordinate": "URDF local revolute joint q in radians",
            "raw_coordinate": "physical SDK stable readback in [0,255]",
            "image_role": "visual evidence used to associate raw with q_URDF",
            "forbidden_shortcut": (
                "camera-frame projected tape angles are not URDF local joint q"
            ),
        },
        "urdf": str(args.urdf.resolve()),
        "joint": args.joint,
        "semantic_order": list(SEMANTIC_ORDER),
        "base_pose_source": base_source,
        "base_pose": base,
        "targets": targets,
        "poses_json": str(poses_path.resolve()),
        "archive_session": (
            str(args.archive_session.resolve())
            if args.archive_session is not None
            else None
        ),
        "archive_image_count": len(archive_rows),
        "archive_index_json": archive_json,
        "archive_index_csv": archive_csv,
    }
    manifest_path = args.out_dir / "registration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[prepare] wrote {poses_path}")
    print(f"[prepare] wrote {manifest_path}")
    if args.archive_session is not None:
        print(f"[prepare] indexed {len(archive_rows)} old color images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
