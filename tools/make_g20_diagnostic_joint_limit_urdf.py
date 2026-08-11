#!/usr/bin/env python3
"""Create an isolated URDF copy with one diagnostic joint-limit override."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--lower", type=float, required=True)
    parser.add_argument("--upper", type=float, required=True)
    parser.add_argument("--out-urdf", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def limits(path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(path).getroot()
    result: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            continue
        result[str(joint.get("name"))] = (float(lower), float(upper))
    return result


def main() -> int:
    args = parse_args()
    if not args.source_urdf.is_file():
        raise FileNotFoundError(args.source_urdf)
    if args.upper <= args.lower:
        raise ValueError("--upper must exceed --lower")

    source_limits = limits(args.source_urdf)
    if args.joint not in source_limits:
        raise ValueError(f"joint has no finite limit: {args.joint}")

    text = args.source_urdf.read_text(encoding="utf-8")
    joint_pattern = re.compile(
        rf'(<joint\s+name="{re.escape(args.joint)}"\s+type="revolute">)'
        rf'(.*?)'
        rf'(</joint>)',
        re.DOTALL,
    )
    matches = list(joint_pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one joint block for {args.joint}, got {len(matches)}"
        )
    block = matches[0].group(0)
    limit_pattern = re.compile(r"(<limit\s+)(.*?)(/>)", re.DOTALL)
    limit_matches = list(limit_pattern.finditer(block))
    if len(limit_matches) != 1:
        raise ValueError(
            f"expected exactly one limit in {args.joint}, got {len(limit_matches)}"
        )
    attrs = limit_matches[0].group(2)
    attrs, lower_count = re.subn(
        r'lower="[^"]+"', f'lower="{args.lower:.10g}"', attrs, count=1
    )
    attrs, upper_count = re.subn(
        r'upper="[^"]+"', f'upper="{args.upper:.10g}"', attrs, count=1
    )
    if lower_count != 1 or upper_count != 1:
        raise ValueError("joint limit is missing lower or upper attribute")
    changed_block = (
        block[: limit_matches[0].start(2)]
        + attrs
        + block[limit_matches[0].end(2) :]
    )
    changed_text = (
        text[: matches[0].start()]
        + changed_block
        + text[matches[0].end() :]
    )

    args.out_urdf.parent.mkdir(parents=True, exist_ok=True)
    args.out_urdf.write_text(changed_text, encoding="utf-8")
    output_limits = limits(args.out_urdf)
    changed_joints = [
        name
        for name in sorted(source_limits)
        if source_limits[name] != output_limits.get(name)
    ]
    if changed_joints != [args.joint]:
        raise RuntimeError(
            f"unexpected changed joint limits: {changed_joints}"
        )

    source_meshes = (args.source_urdf.parent / "meshes").resolve()
    output_meshes = args.out_urdf.parent / "meshes"
    if not source_meshes.is_dir():
        raise FileNotFoundError(source_meshes)
    if output_meshes.exists() or output_meshes.is_symlink():
        if output_meshes.resolve() != source_meshes:
            raise RuntimeError(f"unexpected existing mesh path: {output_meshes}")
    else:
        output_meshes.symlink_to(source_meshes, target_is_directory=True)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only_not_for_runtime",
        "source_urdf": str(args.source_urdf.resolve()),
        "source_urdf_sha256": sha256(args.source_urdf),
        "output_urdf": str(args.out_urdf.resolve()),
        "output_urdf_sha256": sha256(args.out_urdf),
        "joint": args.joint,
        "source_limit_rad": list(source_limits[args.joint]),
        "diagnostic_limit_rad": list(output_limits[args.joint]),
        "changed_joint_limits": changed_joints,
        "mesh_link": str(output_meshes),
        "mesh_target": str(source_meshes),
    }
    manifest_path = args.out_urdf.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
