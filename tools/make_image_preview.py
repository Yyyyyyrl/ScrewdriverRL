#!/usr/bin/env python3
"""Create a compact JPEG preview for visual QA of large local PNG artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=85)
    args = parser.parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {args.input}")
    if image.shape[1] > args.max_width:
        scale = args.max_width / image.shape[1]
        image = cv2.resize(
            image,
            (args.max_width, int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(args.output), image, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    ):
        raise RuntimeError(f"failed to write {args.output}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
