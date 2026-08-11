#!/usr/bin/env python3
"""Compose labeled random-trajectory waypoint renders into a review sheet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--tile-height", type=int, default=330)
    args = parser.parse_args()
    data = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    render_dir = args.render_manifest.parent
    tiles: list[np.ndarray] = []
    for name, entry in data["poses"].items():
        image_path = render_dir / next(iter(entry["images"].values()))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        header = 46
        body_height = args.tile_height - header
        scale = min(args.tile_width / image.shape[1], body_height / image.shape[0])
        resized = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.full((args.tile_height, args.tile_width, 3), 224, dtype=np.uint8)
        x = (args.tile_width - resized.shape[1]) // 2
        y = header + (body_height - resized.shape[0]) // 2
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.rectangle(tile, (0, 0), (args.tile_width, header), (23, 28, 37), -1)
        title = name.replace("_random_", " | ").replace("_large", "")
        cv2.putText(
            tile, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (246, 248, 251), 1, cv2.LINE_AA,
        )
        tiles.append(tile)
    rows = math.ceil(len(tiles) / args.columns)
    sheet = np.full(
        (rows * args.tile_height, args.columns * args.tile_width, 3),
        210,
        dtype=np.uint8,
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, args.columns)
        y = row * args.tile_height
        x = column * args.tile_width
        sheet[y : y + args.tile_height, x : x + args.tile_width] = tile
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), sheet):
        raise RuntimeError(f"failed to write {args.out}")
    print(f"[waypoint-sheet] wrote {args.out} ({len(tiles)} poses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
