#!/usr/bin/env python3
"""Create an isolated OG URDF copy with candidate follower mimic fits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mimic_signatures(path: Path) -> dict[str, dict[str, float | str]]:
    result = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        result[str(joint.get("name"))] = {
            "source": str(mimic.get("joint")),
            "multiplier": float(mimic.get("multiplier", "1")),
            "offset": float(mimic.get("offset", "0")),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--out-urdf", type=Path, required=True)
    parser.add_argument("--fit", choices=("affine", "through-origin"), default="affine")
    args = parser.parse_args()

    source = mimic_signatures(args.source_urdf)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    fits = provenance["follower_mimic_candidates"]
    expected = {str(item["joint"]) for item in fits}
    if expected != set(source):
        raise ValueError(
            f"mimic set mismatch: source={sorted(source)}, candidate={sorted(expected)}"
        )

    text = args.source_urdf.read_text(encoding="utf-8")
    selected = {}
    for item in fits:
        joint = str(item["joint"])
        parent = str(item["source"])
        fit = (
            item["affine_fit"]
            if args.fit == "affine"
            else item["through_origin_diagnostic"]
        )
        multiplier = float(fit["multiplier"])
        offset = float(fit["offset"])
        block_pattern = re.compile(
            rf'(<joint\s+name="{re.escape(joint)}"\s+type="revolute">)(.*?)(</joint>)',
            re.DOTALL,
        )
        matches = list(block_pattern.finditer(text))
        if len(matches) != 1:
            raise ValueError(f"{joint}: expected one joint block, got {len(matches)}")
        block = matches[0].group(0)
        mimic_pattern = re.compile(r"<mimic\s+.*?\s*/>", re.DOTALL)
        mimic_matches = list(mimic_pattern.finditer(block))
        if len(mimic_matches) != 1:
            raise ValueError(f"{joint}: expected one mimic tag")
        replacement = (
            f'<mimic\n      joint="{parent}"\n'
            f'      multiplier="{multiplier:.12g}"\n'
            f'      offset="{offset:.12g}" />'
        )
        changed_block = (
            block[: mimic_matches[0].start()]
            + replacement
            + block[mimic_matches[0].end() :]
        )
        text = text[: matches[0].start()] + changed_block + text[matches[0].end() :]
        selected[joint] = {
            "source": parent,
            # Compare against the values exactly as serialized into the URDF.
            "multiplier": float(f"{multiplier:.12g}"),
            "offset": float(f"{offset:.12g}"),
        }

    args.out_urdf.parent.mkdir(parents=True, exist_ok=True)
    args.out_urdf.write_text(text, encoding="utf-8")
    output = mimic_signatures(args.out_urdf)
    if output != selected:
        raise RuntimeError(f"output mimic signatures differ: {output!r}")
    changed = [name for name in sorted(source) if source[name] != output[name]]
    if changed != sorted(expected):
        raise RuntimeError(f"unexpected changed mimic set: {changed}")

    source_meshes = (args.source_urdf.parent / "meshes").resolve()
    output_meshes = args.out_urdf.parent / "meshes"
    if output_meshes.exists() or output_meshes.is_symlink():
        if output_meshes.resolve() != source_meshes:
            raise RuntimeError(f"unexpected mesh path {output_meshes}")
    else:
        output_meshes.symlink_to(source_meshes, target_is_directory=True)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_not_promoted",
        "source_urdf": str(args.source_urdf.resolve()),
        "source_urdf_sha256": sha256(args.source_urdf),
        "provenance": str(args.provenance.resolve()),
        "provenance_sha256": sha256(args.provenance),
        "fit": args.fit,
        "output_urdf": str(args.out_urdf.resolve()),
        "output_urdf_sha256": sha256(args.out_urdf),
        "source_mimics": source,
        "candidate_mimics": output,
        "changed_mimics": changed,
        "unchanged_joint_limits": True,
        "mesh_link": str(output_meshes),
        "mesh_target": str(source_meshes),
    }
    manifest_path = args.out_urdf.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
