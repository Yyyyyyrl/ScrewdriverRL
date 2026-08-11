#!/usr/bin/env python3
"""Derive a candidate-only thumb-roll absolute-zero correction overlay.

The physical sweep establishes direction, travel, hysteresis, and local slope,
but it cannot by itself tie the hardware endpoint to the CAD/URDF zero.  This
tool preserves every measured raw knot and adds an independently validated
constant visual offset to the semantic radians.  The commandable semantic range
is restricted to the intersection with the existing URDF range.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


JOINT = "thumb_cmc_roll"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offset-deg", type=float, required=True)
    parser.add_argument("--urdf-upper-rad", type=float, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite an existing output or manifest")
    source = json.loads(args.input.read_text())
    result = copy.deepcopy(source)
    joint = result["joints"][JOINT]
    lut = joint["physical_lut"]
    old_rad = [float(value) for value in lut["rad"]]
    offset_rad = math.radians(args.offset_deg)
    shifted_rad = [value + offset_rad for value in old_rad]
    old_lo = float(joint["lo"])
    old_hi = float(joint["hi"])
    semantic_lo = old_lo + offset_rad
    semantic_hi = min(old_hi, float(args.urdf_upper_rad))
    if not semantic_lo < semantic_hi:
        raise ValueError(
            f"visual offset leaves no URDF overlap: [{semantic_lo}, {semantic_hi}]"
        )
    joint["lo"] = semantic_lo
    joint["hi"] = semantic_hi
    lut["rad"] = shifted_rad
    result["version"] = max(int(result.get("version", 0)) + 1, 4)
    result["note"] = (
        "CANDIDATE ONLY; not promoted. Derived from "
        f"{args.input.name}. thumb_cmc_roll physical sweep shape and raw knots "
        f"are preserved, while an independently validated visual CAD zero "
        f"offset of +{args.offset_deg:.6f} deg ({offset_rad:.12f} rad) is added. "
        f"Commandable semantic range is the URDF intersection "
        f"[{semantic_lo:.12f}, {semantic_hi:.12f}] rad; hardware states outside "
        "that range remain decodable but are not command targets."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "candidate_only": True,
        "promoted": False,
        "joint": JOINT,
        "source": {
            "path": str(args.input),
            "sha256": sha256(args.input),
            "lo_rad": old_lo,
            "hi_rad": old_hi,
        },
        "correction": {
            "type": "constant_absolute_visual_zero_offset",
            "offset_deg": args.offset_deg,
            "offset_rad": offset_rad,
            "raw_knots_preserved": True,
            "rad_knots_shifted_by_constant": True,
        },
        "semantic_command_intersection_rad": [semantic_lo, semantic_hi],
        "shifted_readback_span_rad": [shifted_rad[-1], shifted_rad[0]],
        "evidence": [
            {"path": str(path), "sha256": sha256(path)} for path in args.evidence
        ],
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
