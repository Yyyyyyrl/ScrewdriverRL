"""Intel RealSense D435 capture utility.

Examples:
    python tools/realsense_capture.py --info
    python tools/realsense_capture.py --snapshot outputs/rs_shot
    python tools/realsense_capture.py --live
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def print_device_info() -> None:
    devices = list(rs.context().query_devices())
    if not devices:
        raise RuntimeError("no RealSense device found (check USB cable/port)")
    for dev in devices:
        print(f"{dev.get_info(rs.camera_info.name)}")
        print(f"  serial   : {dev.get_info(rs.camera_info.serial_number)}")
        print(f"  firmware : {dev.get_info(rs.camera_info.firmware_version)}")
        print(f"  usb      : {dev.get_info(rs.camera_info.usb_type_descriptor)}")
        for sensor in dev.sensors:
            name = sensor.get_info(rs.camera_info.name)
            profiles = sorted(
                {
                    (
                        p.stream_name(),
                        p.as_video_stream_profile().width(),
                        p.as_video_stream_profile().height(),
                        p.fps(),
                    )
                    for p in sensor.get_stream_profiles()
                    if p.is_video_stream_profile()
                }
            )
            print(f"  sensor   : {name} ({len(profiles)} video profiles)")
            for stream, w, h, fps in profiles:
                print(f"      {stream:<8} {w}x{h} @ {fps}")


def start_pipeline(width: int, height: int, fps: int, align_to_color: bool):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color) if align_to_color else None

    # The auto-exposure needs a few frames to settle before the image is usable.
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, depth_scale


def grab(pipeline, align, timeout_ms: int = 5000):
    frames = pipeline.wait_for_frames(timeout_ms)
    if align is not None:
        frames = align.process(frames)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        raise RuntimeError("incomplete frameset")
    return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())


def configure_rgb_controls(pipeline, args) -> dict[str, float]:
    device = pipeline.get_active_profile().get_device()
    color_sensor = next(
        sensor
        for sensor in device.sensors
        if sensor.supports(rs.option.exposure)
        and sensor.supports(rs.option.gain)
        and sensor.supports(rs.option.enable_auto_exposure)
        and sensor.supports(rs.option.white_balance)
        and sensor.supports(rs.option.enable_auto_white_balance)
    )
    if args.rgb_exposure is not None or args.rgb_gain is not None:
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    if args.rgb_exposure is not None:
        color_sensor.set_option(rs.option.exposure, args.rgb_exposure)
    if args.rgb_gain is not None:
        color_sensor.set_option(rs.option.gain, args.rgb_gain)
    if args.rgb_white_balance is not None:
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, args.rgb_white_balance)
    for _ in range(args.control_warmup_frames):
        pipeline.wait_for_frames()
    return {
        "enable_auto_exposure": color_sensor.get_option(
            rs.option.enable_auto_exposure
        ),
        "exposure": color_sensor.get_option(rs.option.exposure),
        "gain": color_sensor.get_option(rs.option.gain),
        "enable_auto_white_balance": color_sensor.get_option(
            rs.option.enable_auto_white_balance
        ),
        "white_balance": color_sensor.get_option(rs.option.white_balance),
    }


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)


def snapshot(prefix: str, args) -> None:
    pipeline, align, depth_scale = start_pipeline(
        args.width, args.height, args.fps, not args.no_align
    )
    try:
        rgb_controls = configure_rgb_controls(pipeline, args)
        color, depth = grab(pipeline, align)
    finally:
        pipeline.stop()

    out = Path(prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{out}_color.png", color)
    cv2.imwrite(f"{out}_depth.png", colorize_depth(depth))
    np.save(f"{out}_depth_raw.npy", depth)
    Path(f"{out}_meta.json").write_text(
        json.dumps(
            {
                "camera_only": True,
                "hand_motion_sent": False,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "depth_scale_m_per_unit": depth_scale,
                "rgb_controls": rgb_controls,
            },
            indent=2,
        )
        + "\n"
    )

    valid = depth[depth > 0]
    center = depth[depth.shape[0] // 2, depth.shape[1] // 2] * depth_scale
    print(f"color  : {color.shape} -> {out}_color.png")
    print(f"depth  : {depth.shape} -> {out}_depth.png, {out}_depth_raw.npy")
    print(f"scale  : {depth_scale} m/unit")
    print(f"valid  : {valid.size / depth.size:.1%} of pixels")
    if valid.size:
        print(f"range  : {valid.min() * depth_scale:.3f} - {valid.max() * depth_scale:.3f} m")
    print(f"center : {center:.3f} m")


def live(args) -> None:
    pipeline, align, _ = start_pipeline(args.width, args.height, args.fps, not args.no_align)
    configure_rgb_controls(pipeline, args)
    print("press q or ESC to quit, s to save a frame to outputs/")
    try:
        while True:
            color, depth = grab(pipeline, align)
            cv2.imshow("RealSense D435 (color | depth)", np.hstack([color, colorize_depth(depth)]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = Path("outputs") / f"rs_{stamp}_color.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), color)
                print(f"saved {path}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", action="store_true", help="list devices and stream profiles")
    parser.add_argument("--snapshot", metavar="PREFIX", help="save one color+depth frame")
    parser.add_argument("--live", action="store_true", help="open a live preview window")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rgb-exposure", type=float, default=None)
    parser.add_argument("--rgb-gain", type=float, default=None)
    parser.add_argument("--rgb-white-balance", type=float, default=None)
    parser.add_argument("--control-warmup-frames", type=int, default=45)
    parser.add_argument("--no-align", action="store_true", help="skip depth-to-color alignment")
    args = parser.parse_args()

    if args.info:
        print_device_info()
    if args.snapshot:
        snapshot(args.snapshot, args)
    if args.live:
        live(args)
    if not (args.info or args.snapshot or args.live):
        parser.print_help()


if __name__ == "__main__":
    main()
