#!/usr/bin/env python3
"""Extract ordered Isaac semantic poses from a G20 dynamic command log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.trajectory_log.read_text(encoding="utf-8"))
    commands = data["commands"]
    poses = {
        f"frame_{index:04d}": [float(value) for value in command["target_semantic"]]
        for index, command in enumerate(commands)
    }
    if any(len(values) != 16 for values in poses.values()):
        raise ValueError("trajectory log contains a non-16-joint command")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(poses, indent=2) + "\n", encoding="utf-8")
    print(f"[extract] wrote {len(poses)} poses to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
