#!/usr/bin/env python3
"""Extract quarter/mid/three-quarter frames from G20 A/B videos for QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=800)
    args = parser.parse_args()
    tiles = []
    for path in args.video:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open {path}")
        count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        row = []
        for fraction in (0.25, 0.50, 0.75):
            index = min(count - 1, round((count - 1) * fraction))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read {path} frame {index}")
            height = round(frame.shape[0] * args.tile_width / frame.shape[1])
            frame = cv2.resize(
                frame, (args.tile_width, height), interpolation=cv2.INTER_AREA
            )
            cv2.putText(
                frame,
                f"{path.stem} | t={index / fps:.1f}s",
                (14, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (30, 30, 30),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{path.stem} | t={index / fps:.1f}s",
                (14, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            row.append(frame)
        cap.release()
        tiles.append(np.hstack(row))
    sheet = np.vstack(tiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"failed to write {args.out}")
    print(f"[qa-sheet] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
