#!/usr/bin/env python3
"""Compare archived passive-joint measurements with URDF mimic contracts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--raw-column", default="command_raw")
    parser.add_argument("--parent-column", required=True)
    parser.add_argument("--child-column", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=MULTIPLIER",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates: dict[str, float] = {}
    for item in args.candidate:
        name, value = item.split("=", 1)
        candidates[name] = float(value)

    rows = []
    for source in csv.DictReader(args.points_csv.open(newline="", encoding="utf-8")):
        if not source.get(args.parent_column) or not source.get(args.child_column):
            continue
        parent = float(source[args.parent_column])
        child = float(source[args.child_column])
        row = {
            "raw": float(source[args.raw_column]),
            "measured_parent_rad": parent,
            "measured_child_rad": child,
            "candidates": {},
        }
        for name, multiplier in candidates.items():
            predicted = multiplier * parent
            row["candidates"][name] = {
                "multiplier": multiplier,
                "predicted_child_rad": predicted,
                "measured_minus_predicted_rad": child - predicted,
            }
        rows.append(row)

    metrics = {}
    for name in candidates:
        residuals = [
            row["candidates"][name]["measured_minus_predicted_rad"] for row in rows
        ]
        metrics[name] = {
            "count": len(residuals),
            "mean_measured_minus_predicted_rad": sum(residuals) / len(residuals),
            "rmse_rad": math.sqrt(
                sum(value * value for value in residuals) / len(residuals)
            ),
            "max_abs_rad": max(abs(value) for value in residuals),
        }

    payload = {
        "schema_version": 1,
        "points_csv": str(args.points_csv.resolve()),
        "parent_column": args.parent_column,
        "child_column": args.child_column,
        "metrics": metrics,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
