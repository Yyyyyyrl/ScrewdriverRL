#!/usr/bin/env python3
"""Extract a name-to-16-DoF map from G20 hardware result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    rows = summary["poses"]
    if isinstance(rows, dict):
        poses = {
            name: [float(value) for value in row["settled_semantic"]]
            for name, row in rows.items()
        }
    else:
        poses = {
            row.get("name", row["pose"]): [
                float(value) for value in row["settled_semantic"]
            ]
            for row in rows
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(poses, indent=2) + "\n")
    print(f"wrote {args.output} ({len(poses)} poses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
