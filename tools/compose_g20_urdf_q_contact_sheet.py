#!/usr/bin/env python3
"""Compose Isaac URDF-local-q renders into a labeled review sheet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--registration-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=560)
    parser.add_argument("--tile-height", type=int, default=360)
    return parser.parse_args()


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def main() -> int:
    args = parse_args()
    if args.columns <= 0 or args.tile_width <= 0 or args.tile_height <= 0:
        raise ValueError("columns and tile dimensions must be positive")
    render = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    registration = json.loads(
        args.registration_manifest.read_text(encoding="utf-8")
    )
    target_by_name = {
        target["name"]: target for target in registration["targets"]
    }
    render_dir = args.render_manifest.parent
    tiles: list[np.ndarray] = []
    for name, entry in render["poses"].items():
        target = target_by_name.get(name)
        if target is None:
            continue
        image_name = next(iter(entry["images"].values()))
        image = cv2.imread(str(render_dir / image_name), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read render: {render_dir / image_name}")
        body_height = args.tile_height - 62
        tile = np.full(
            (args.tile_height, args.tile_width, 3), 245, dtype=np.uint8
        )
        tile[62:] = _fit(image, args.tile_width, body_height)
        cv2.rectangle(tile, (0, 0), (args.tile_width - 1, 61), (24, 30, 40), -1)
        title = f"{target['joint']}  q={target['q_urdf_rad']:.3f} rad"
        subtitle = f"{target['q_urdf_deg']:.1f} deg   URDF local axis"
        cv2.putText(
            tile,
            title,
            (14, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (250, 250, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            subtitle,
            (14, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (190, 215, 245),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        raise ValueError("no render poses matched registration targets")

    rows = math.ceil(len(tiles) / args.columns)
    sheet = np.full(
        (rows * args.tile_height, args.columns * args.tile_width, 3),
        220,
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
    print(f"[contact-sheet] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
