#!/usr/bin/env python3
"""Record a fixed-exposure D435 colour video with per-frame wall timestamps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        default=None,
        help="optional RealSense serial; by default use the available device",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--exposure", type=float, default=300.0)
    parser.add_argument("--gain", type=float, default=32.0)
    parser.add_argument("--white-balance", type=float, default=4600.0)
    parser.add_argument("--out-video", type=Path, required=True)
    parser.add_argument("--out-meta", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="optional sentinel; stop cleanly after it appears",
    )
    args = parser.parse_args()
    if args.duration_s <= 0 or args.fps <= 0 or args.warmup_frames < 0:
        parser.error("duration/fps must be positive and warmup non-negative")

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    args.ready.parent.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    # Keep the stream combination identical to the proven still-capture path.
    # This D435 runs older firmware and has previously wedged the UVC driver
    # while starting a color-only pipeline.
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )
    print("[camera] starting color+depth pipeline", flush=True)
    profile = pipeline.start(config)
    print("[camera] pipeline started", flush=True)
    writer = None
    records: list[dict] = []
    try:
        device = profile.get_device()
        resolved_serial = device.get_info(rs.camera_info.serial_number)
        print(f"[camera] device serial {resolved_serial}", flush=True)
        color_sensor = next(
            sensor
            for sensor in device.query_sensors()
            if sensor.supports(rs.option.exposure)
            and sensor.supports(rs.option.gain)
            and sensor.supports(rs.option.enable_auto_exposure)
            and sensor.supports(rs.option.white_balance)
            and sensor.supports(rs.option.enable_auto_white_balance)
        )
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, args.exposure)
        color_sensor.set_option(rs.option.gain, args.gain)
        if color_sensor.supports(rs.option.enable_auto_white_balance):
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, args.white_balance)

        print(
            f"[camera] controls set; warming up {args.warmup_frames} frames",
            flush=True,
        )
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(3000)
        print("[camera] warmup complete", flush=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(args.out_video), fourcc, float(args.fps), (args.width, args.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer {args.out_video}")

        started_wall_ns = time.time_ns()
        started_monotonic = time.monotonic()
        args.ready.write_text(
            json.dumps(
                {
                    "ready": True,
                    "started_wall_ns": started_wall_ns,
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "video": str(args.out_video),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[camera] READY {args.ready}", flush=True)

        stopped_by_file = False
        while time.monotonic() - started_monotonic < args.duration_s:
            if args.stop_file is not None and args.stop_file.exists():
                stopped_by_file = True
                print(f"[camera] stop sentinel observed: {args.stop_file}", flush=True)
                break
            frames = pipeline.wait_for_frames(3000)
            color = frames.get_color_frame()
            if not color:
                continue
            wall_ns = time.time_ns()
            image = np.asanyarray(color.get_data())
            if image.shape != (args.height, args.width, 3):
                raise RuntimeError(f"unexpected color shape {image.shape}")
            writer.write(image)
            records.append(
                {
                    "index": len(records),
                    "wall_time_ns": wall_ns,
                    "device_timestamp_ms": float(color.get_timestamp()),
                    "frame_number": int(color.get_frame_number()),
                }
            )
    finally:
        if writer is not None:
            writer.release()
        pipeline.stop()

    if not records:
        raise RuntimeError("no D435 frames recorded")
    payload = {
        "schema_version": 1,
        "serial": resolved_serial,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "duration_requested_s": args.duration_s,
        "stop_file": str(args.stop_file) if args.stop_file is not None else None,
        "stopped_by_file": stopped_by_file,
        "controls": {
            "exposure": args.exposure,
            "gain": args.gain,
            "white_balance": args.white_balance,
            "auto_exposure": False,
            "auto_white_balance": False,
        },
        "video": str(args.out_video),
        "frame_count": len(records),
        "frames": records,
    }
    args.out_meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[camera] wrote {args.out_video} and {args.out_meta} "
        f"({len(records)} frames)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
