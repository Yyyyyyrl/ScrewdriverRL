#!/usr/bin/env python3
"""Unwrap pi-periodic marker director angles in an archive-chain result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def _director(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    a = _unit(a - axis * float(a @ axis))
    b = _unit(b - axis * float(b @ axis))
    angle = math.atan2(float(axis @ np.cross(a, b)), float(a @ b))
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    rows = payload["rows"]
    axis = _unit(np.asarray(payload["axis_camera"], dtype=float))
    order = np.argsort([row["old_visual_q_rad"] for row in rows])
    wrapped = np.asarray(
        [
            _director(
                np.asarray(rows[index]["parent_axis_camera"], dtype=float),
                np.asarray(rows[index]["moving_axis_camera"], dtype=float),
                axis,
            )
            for index in order
        ]
    )
    unwrapped = np.unwrap(wrapped, period=math.pi)
    old = np.asarray([rows[index]["old_visual_q_rad"] for index in order])
    zero = float(np.median(unwrapped[np.isclose(old, np.min(old))]))
    unwrapped -= zero
    if np.corrcoef(unwrapped, old)[0, 1] < 0.0:
        unwrapped = -unwrapped
        axis = -axis
    residual = old - unwrapped
    for index, q, delta in zip(order, unwrapped, residual):
        rows[index]["reconstructed_local_q_rad"] = float(q)
        rows[index]["old_minus_reconstructed_rad"] = float(delta)

    payload["schema_version"] = max(int(payload.get("schema_version", 1)), 2)
    payload["method"] += "_pi_period_unwrapped"
    payload["axis_camera"] = axis.tolist()
    payload["old_vs_reconstructed"] = {
        "mean_rad": float(np.mean(residual)),
        "rmse_rad": float(np.sqrt(np.mean(residual * residual))),
        "max_abs_rad": float(np.max(np.abs(residual))),
    }
    payload["supersedes"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["old_vs_reconstructed"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
