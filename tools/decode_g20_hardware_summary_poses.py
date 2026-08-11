#!/usr/bin/env python3
"""Decode settled G20 raw20 samples into semantic Isaac poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from screwdriver_rl.deploy import linker_sdk_map as sdkmap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--calib",
        default="default",
        help="physical-LUT overlay path, or 'default' for the legacy affine map",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.calib == "default":
        sdkmap.reset_calibration()
    else:
        sdkmap.apply_calibration(args.calib)
    summary = json.loads(args.summary.read_text())
    poses = {
        name: sdkmap.sdk_range_to_joints16(record["settled_raw20"])
        for name, record in summary["poses"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(poses, indent=2) + "\n")
    print(f"wrote {args.output} ({len(poses)} poses, calib={args.calib})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
