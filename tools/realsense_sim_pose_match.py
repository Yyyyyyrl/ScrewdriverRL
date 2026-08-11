#!/usr/bin/env python3
"""Match a real hand pose to one fixed Isaac render, camera-only.

The tool never imports the LinkerHand SDK.  Pressing S saves color, depth, and
overlay evidence and exits, so a separate read-only SDK snapshot can be taken
after the RealSense pipeline has released the camera.
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
from realsense_sim_silhouette_overlay import silhouette


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--q-urdf-rad", required=True, type=float)
    parser.add_argument("--expected-camera-serial", default="143322073091")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rgb-exposure", type=float, default=166)
    parser.add_argument("--rgb-gain", type=float, default=32)
    parser.add_argument("--rgb-white-balance", type=float, default=4600)
    parser.add_argument("--control-warmup-frames", type=int, default=60)
    parser.add_argument("--edge-thickness", type=int, default=2)
    return parser.parse_args()


def _compose(
    color: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    joint: str,
    q: float,
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
    cv2.rectangle(result, (0, 0), (result.shape[1], 78), (18, 24, 32), -1)
    cv2.putText(
        result,
        f"POSE MATCH: {joint}  q_URDF={q:.3f} rad ({np.degrees(q):.1f} deg)",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 248, 252),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "Move the HAND only | f: fill | s: save+exit | q/Esc: cancel",
        (14, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (190, 215, 245),
        1,
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
    intr = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    sim_corrected, correction = intrinsic_correct(sim, intr)
    mask, edges = silhouette(sim_corrected, args.edge_thickness)

    pipeline, align, depth_scale = start_pipeline(
        args.width, args.height, args.fps, True
    )
    saved = False
    session: dict = {}
    try:
        rgb_controls = configure_rgb_controls(pipeline, args)
        profile = pipeline.get_active_profile()
        device = profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)
        if serial != args.expected_camera_serial:
            raise RuntimeError(
                f"camera serial mismatch: got {serial}, "
                f"expected {args.expected_camera_serial}"
            )
        color_profile = profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        live_intr = color_profile.get_intrinsics()
        session = {
            "schema_version": 1,
            "camera_only": True,
            "hand_motion_sent": False,
            "joint": args.joint,
            "q_urdf_rad": args.q_urdf_rad,
            "q_urdf_deg": float(np.degrees(args.q_urdf_rad)),
            "sim_image": str(args.sim.resolve()),
            "intrinsics_file": str(args.intrinsics.resolve()),
            "intrinsic_correction": correction,
            "camera": {
                "serial": serial,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "depth_scale_m_per_unit": depth_scale,
                "intrinsics": {
                    "fx": live_intr.fx,
                    "fy": live_intr.fy,
                    "ppx": live_intr.ppx,
                    "ppy": live_intr.ppy,
                },
                "rgb_controls": rgb_controls,
            },
            "status": "running",
            "save": None,
        }
        manifest_path = args.out_dir / "pose_match_session.json"
        manifest_path.write_text(
            json.dumps(session, indent=2) + "\n", encoding="utf-8"
        )
        cv2.imwrite(
            str(args.out_dir / "fixed_sim_intrinsic_corrected.png"), sim_corrected
        )
        cv2.imwrite(str(args.out_dir / "fixed_sim_silhouette.png"), mask)

        window = "D435 LIVE + ISAAC URDF LOCAL-Q TARGET"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.width, args.height)
        show_fill = False
        print("[pose-match] camera-only window started", flush=True)
        print("[pose-match] move HAND only; S saves and exits", flush=True)
        while True:
            color, depth = grab(pipeline, align)
            overlay = _compose(
                color,
                mask,
                edges,
                args.joint,
                args.q_urdf_rad,
                show_fill,
            )
            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                session["status"] = "cancelled"
                break
            if key == ord("f"):
                show_fill = not show_fill
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                color_path = args.out_dir / f"matched_{stamp}_color.png"
                overlay_path = args.out_dir / f"matched_{stamp}_overlay.png"
                depth_path = args.out_dir / f"matched_{stamp}_depth_raw.npy"
                cv2.imwrite(str(color_path), color)
                cv2.imwrite(str(overlay_path), overlay)
                np.save(depth_path, depth)
                session["status"] = "matched"
                session["save"] = {
                    "timestamp": stamp,
                    "color": str(color_path.resolve()),
                    "overlay": str(overlay_path.resolve()),
                    "depth_raw": str(depth_path.resolve()),
                }
                saved = True
                break
        manifest_path.write_text(
            json.dumps(session, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    print(f"[pose-match] {'saved' if saved else 'cancelled'}", flush=True)
    return 0 if saved else 2


if __name__ == "__main__":
    raise SystemExit(main())
