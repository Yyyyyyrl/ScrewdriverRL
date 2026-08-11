#!/usr/bin/env python3
"""Overlay a fixed Isaac hand silhouette on a live RealSense color stream.

The tool is camera-only: it never imports the hand SDK and cannot command the
robot.  Move the physical camera until the rigid palm, four fingers, and thumb
all coincide with the fixed Isaac outline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from compose_g20_sim_real_ab import intrinsic_correct
from realsense_capture import configure_rgb_controls, grab, start_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rgb-exposure", type=float, default=166)
    parser.add_argument("--rgb-gain", type=float, default=32)
    parser.add_argument("--rgb-white-balance", type=float, default=4600)
    parser.add_argument("--control-warmup-frames", type=int, default=60)
    parser.add_argument("--edge-thickness", type=int, default=2)
    return parser.parse_args()


def silhouette(sim: np.ndarray, thickness: int) -> tuple[np.ndarray, np.ndarray]:
    border = np.concatenate(
        (
            sim[:20].reshape(-1, 3),
            sim[-20:].reshape(-1, 3),
            sim[:, :20].reshape(-1, 3),
            sim[:, -20:].reshape(-1, 3),
        ),
        axis=0,
    )
    background_color = np.median(border, axis=0).astype(np.int16)
    difference = np.max(
        np.abs(sim.astype(np.int16) - background_color[None, None, :]), axis=2
    )
    mask = (difference > 8).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges = cv2.Canny(mask, 50, 150)
    if thickness > 1:
        edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8))
    return mask, edges


def compose(
    color: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    show_fill: bool,
) -> np.ndarray:
    result = color.copy()
    if show_fill:
        fill = np.zeros_like(result)
        fill[:, :, 1] = 255
        region = mask > 0
        result[region] = cv2.addWeighted(
            result[region], 0.82, fill[region], 0.18, 0.0
        )
    result[edges > 0] = (0, 0, 255)
    cv2.rectangle(result, (0, 0), (result.shape[1], 58), (18, 24, 32), -1)
    cv2.putText(
        result,
        "MOVE HAND/CAMERA: match RED Isaac outline | f: fill | s: save | q/Esc: quit",
        (14, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 248, 252),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim = cv2.imread(str(args.sim), cv2.IMREAD_COLOR)
    if sim is None:
        raise RuntimeError(f"failed to read {args.sim}")
    if sim.shape[:2] != (args.height, args.width):
        raise ValueError(
            f"sim is {sim.shape[1]}x{sim.shape[0]}, expected "
            f"{args.width}x{args.height}"
        )
    intr = json.loads(args.intrinsics.read_text())
    sim_corrected, correction = intrinsic_correct(sim, intr)
    mask, edges = silhouette(sim_corrected, args.edge_thickness)

    pipeline, align, depth_scale = start_pipeline(
        args.width, args.height, args.fps, True
    )
    rgb_controls = configure_rgb_controls(pipeline, args)
    profile = pipeline.get_active_profile()
    device = profile.get_device()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    live_intr = color_profile.get_intrinsics()
    device_serial = device.get_info(rs.camera_info.serial_number)
    session = {
        "schema_version": 1,
        "camera_only": True,
        "hand_motion_sent": False,
        "sim_image": str(args.sim),
        "intrinsics_file": str(args.intrinsics),
        "intrinsic_correction": correction,
        "stream": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "serial": device_serial,
            "depth_scale_m_per_unit": depth_scale,
            "intrinsics": {
                "width": live_intr.width,
                "height": live_intr.height,
                "fx": live_intr.fx,
                "fy": live_intr.fy,
                "ppx": live_intr.ppx,
                "ppy": live_intr.ppy,
                "model": str(live_intr.model),
                "coeffs": list(live_intr.coeffs),
            },
            "rgb_controls": rgb_controls,
        },
        "saves": [],
    }
    (args.out_dir / "live_overlay_session.json").write_text(
        json.dumps(session, indent=2) + "\n"
    )
    cv2.imwrite(str(args.out_dir / "fixed_sim_intrinsic_corrected.png"), sim_corrected)
    cv2.imwrite(str(args.out_dir / "fixed_sim_silhouette.png"), mask)

    window = "D435 LIVE + FIXED ISAAC SILHOUETTE"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)
    show_fill = False
    print("camera-only live overlay started")
    print("move the D435; f toggles fill, s saves, q/Esc quits")
    try:
        while True:
            color, depth = grab(pipeline, align)
            overlay = compose(color, mask, edges, show_fill)
            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                show_fill = not show_fill
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                color_path = args.out_dir / f"aligned_{stamp}_color.png"
                overlay_path = args.out_dir / f"aligned_{stamp}_overlay.png"
                depth_path = args.out_dir / f"aligned_{stamp}_depth_raw.npy"
                cv2.imwrite(str(color_path), color)
                cv2.imwrite(str(overlay_path), overlay)
                np.save(depth_path, depth)
                session["saves"].append(
                    {
                        "timestamp": stamp,
                        "color": str(color_path),
                        "overlay": str(overlay_path),
                        "depth_raw": str(depth_path),
                    }
                )
                (args.out_dir / "live_overlay_session.json").write_text(
                    json.dumps(session, indent=2) + "\n"
                )
                print(f"saved {overlay_path}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
