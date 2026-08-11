#!/usr/bin/env python3
"""Measure thumb CMC yaw from the palm tape and the thumb proximal tape.

Camera-only, 2D in-plane, the same structure as the Phase B roll detector: the
palm tape gives a fixed reference heading and the thumb proximal tape gives the
moving one. Per-marker 3D axes are not fitted; the palm tape carries usable
depth in this pose but a single stripe still cannot define a plane.

Two things differ from the roll detector and both matter:

* The measured thumb-minus-palm heading sits near +76 deg in this pose, only
  about 14 deg from the +-90 wrap boundary of the unoriented-line convention.
  A yaw sweep the size of a Phase B roll sweep would cross it and the sign
  would flip mid-curve. ``--reference-offset-deg`` is therefore subtracted
  *before* normalizing, which recentres the reported angle near zero and puts
  the wrap boundary 90 deg away on both sides.
* The thumb and palm tapes are much less elongated than the Phase B finger
  stripes (PCA elongation about 13 and 16 against 50 to 90), but they are also
  far larger, and the measured 60-frame block stability is about 0.06 deg. The
  projected area and major-axis extent of the thumb tape are reported every
  frame so that out-of-plane motion, which foreshortens them, can be detected:
  a 2D angle is only valid while the yaw axis stays near the optical axis.

Physical yaw is the change from a separately recorded reference capture, not
the absolute angle in any single frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import numpy as np

from tools.measure_g20_tape_angles import (
    _normalize_axis_heading,
    _parse_roi,
    _pca_axis,
    _summary,
)

ANGLE_KEYS = (
    "thumb_cmc_yaw_deg_2d",
    "palm_heading_deg_2d",
    "thumb_heading_deg_2d",
    "thumb_cross_heading_deg_2d",
    "palm_cross_heading_deg_2d",
    "thumb_l_angle_deg",
    "palm_l_angle_deg",
    "thumb_marker_area_px",
    "thumb_major_extent_px",
)


def _major_extent(pixels: np.ndarray) -> float:
    axis, _ = _pca_axis(pixels)
    return float(np.ptp((pixels - pixels.mean(axis=0)) @ axis))


def yaw_angle(palm_heading_deg: float, thumb_heading_deg: float,
              reference_offset_deg: float) -> float:
    """Signed in-plane thumb yaw, recentred so the wrap boundary is far away.

    The offset is removed before normalizing. Doing it afterwards would leave
    the raw difference sitting 14 deg from +90 in this pose and the sweep would
    wrap. The sign is never removed with abs(): a reversed convention has to be
    corrected explicitly once the small-amplitude direction check fixes it.
    """

    return _normalize_axis_heading(
        thumb_heading_deg - palm_heading_deg - reference_offset_deg
    )


def _blue(color: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(args.hsv_lower, dtype=np.uint8),
                       np.array(args.hsv_upper, dtype=np.uint8)) > 0


def _pair_in_depth_band(mask: np.ndarray, depth_m: np.ndarray,
                        band: tuple[float, float], roi: tuple[int, int, int, int],
                        min_area: int, label: str) -> list[np.ndarray]:
    """The two stripes of one L, isolated by depth before connectivity.

    Depth is applied *before* connected components on purpose. The thumb sweeps
    across the palm, so at one end of the travel the thumb tape and the palm
    tape touch in the image and would fuse into a single component. They sit
    about 45 mm apart in depth with an almost empty valley between them, so
    depth-first segmentation splits a touching pair automatically, which no ROI
    rectangle can do.
    """
    region = np.zeros(mask.shape, dtype=bool)
    x0, y0, x1, y1 = roi
    region[y0:y1, x0:x1] = True
    near = (depth_m >= band[0]) & (depth_m <= band[1])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask & region & near).astype(np.uint8), 8)
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(keep) != 2:
        raise ValueError(
            f"expected exactly 2 {label} markers in depth band "
            f"{band[0]:.3f}..{band[1]:.3f} m, found {len(keep)}")
    out = []
    for i in keep:
        ys, xs = np.where(labels == i)
        out.append(np.column_stack([xs.astype(np.float64), ys.astype(np.float64)]))
    return out


def measure_yaw_frame(color: np.ndarray, depth: np.ndarray,
                      depth_scale: float,
                      args: argparse.Namespace) -> tuple[dict[str, float], dict]:
    blue = _blue(color, args)
    depth_m = depth.astype(np.float64) * depth_scale

    palm_pair = _pair_in_depth_band(blue, depth_m, args.palm_depth_band,
                                    args.palm_roi, args.palm_min_area, "palm")
    thumb_pair = _pair_in_depth_band(blue, depth_m, args.thumb_depth_band,
                                     args.thumb_roi, args.thumb_min_area, "thumb")

    # Palm: the long stripe is the reference direction, the short one only
    # exists to give the palm a second axis. Length separates them by 1.6x.
    palm_pair.sort(key=lambda p: -_major_extent(p))
    palm, palm_cross = palm_pair
    palm_centroid = palm.mean(axis=0)
    # Thumb: the longitudinal stripe is consistently about 9 mm nearer
    # the camera than the cross stripe in this fixed pose (roughly 0.283 vs
    # 0.292 m). Pixel distance to the palm reverses near raw31 and swaps the
    # labels, while depth ordering remains unchanged across the full sweep.
    thumb_pair.sort(key=lambda p: float(np.median(
        depth_m[p[:, 1].astype(int), p[:, 0].astype(int)])))
    thumb, thumb_cross = thumb_pair

    out: dict[str, float] = {}
    detail: dict[str, Any] = {}
    for name, pixels in (("palm", palm), ("palm_cross", palm_cross),
                         ("thumb", thumb), ("thumb_cross", thumb_cross)):
        axis, elongation = _pca_axis(pixels)
        heading = _normalize_axis_heading(
            math.degrees(math.atan2(axis[1], axis[0])))
        if elongation < args.min_elongation2:
            raise ValueError(
                f"{name} marker elongation {elongation:.2f} below "
                f"{args.min_elongation2}; heading would be ill conditioned")
        out[f"{name}_heading_deg_2d"] = heading
        centred = pixels - pixels.mean(axis=0)
        extent = float(np.ptp(centred @ axis))
        rows = pixels[:, 1].astype(int)
        cols = pixels[:, 0].astype(int)
        values = depth_m[rows, cols]
        valid = (values > 0.05) & (values < 3.0)
        detail[name] = {
            "area_px": int(len(pixels)),
            "elongation2": elongation,
            "major_extent_px": extent,
            "centroid_uv": [float(v) for v in pixels.mean(axis=0)],
            "depth_median_m": (float(np.median(values[valid]))
                               if valid.any() else None),
            "depth_valid_fraction": float(valid.mean()),
        }

    out["thumb_marker_area_px"] = float(detail["thumb"]["area_px"])
    out["thumb_major_extent_px"] = detail["thumb"]["major_extent_px"]
    # The projected angle inside each L is the out-of-plane witness: it stays at
    # its true value only while that segment faces the camera, and it does not
    # depend on marker area, so ROI clipping cannot fake it.
    out["thumb_l_angle_deg"] = abs(_normalize_axis_heading(
        out["thumb_heading_deg_2d"] - out["thumb_cross_heading_deg_2d"]))
    out["palm_l_angle_deg"] = abs(_normalize_axis_heading(
        out["palm_heading_deg_2d"] - out["palm_cross_heading_deg_2d"]))
    out["thumb_cmc_yaw_deg_2d"] = yaw_angle(
        out["palm_heading_deg_2d"], out["thumb_heading_deg_2d"],
        args.reference_offset_deg)
    return out, detail


def _blocks(rows: list[dict[str, Any]], key: str, size: int) -> list[float]:
    values = [row[key] for row in rows if row.get(key) is not None]
    return [float(np.median(values[i:i + size]))
            for i in range(0, len(values) - size + 1, size)]


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import pyrealsense2 as rs

    context = rs.context()
    matches = [d for d in context.query_devices()
               if d.get_info(rs.camera_info.serial_number) == args.serial]
    if len(matches) != 1:
        raise RuntimeError(f"expected one D435 {args.serial}, found {len(matches)}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height,
                         rs.format.z16, args.fps)
    profile = pipeline.start(config)
    try:
        sensor = next(s for s in profile.get_device().query_sensors()
                      if "RGB" in s.get_info(rs.camera_info.name))
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, args.rgb_exposure)
        sensor.set_option(rs.option.gain, args.rgb_gain)
        sensor.set_option(rs.option.enable_auto_white_balance, 0)
        sensor.set_option(rs.option.white_balance, args.rgb_white_balance)
        rgb_controls = {
            "exposure": sensor.get_option(rs.option.exposure),
            "gain": sensor.get_option(rs.option.gain),
            "white_balance": sensor.get_option(rs.option.white_balance),
            "enable_auto_exposure": sensor.get_option(
                rs.option.enable_auto_exposure),
            "enable_auto_white_balance": sensor.get_option(
                rs.option.enable_auto_white_balance),
        }
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(5000)

        started = time.time()
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        last: dict[str, Any] = {}
        for index in range(args.frames):
            frames = align.process(pipeline.wait_for_frames(5000))
            color = np.asanyarray(frames.get_color_frame().get_data())
            depth = np.asanyarray(frames.get_depth_frame().get_data())
            try:
                angles, detail = measure_yaw_frame(color, depth, depth_scale, args)
            except ValueError as exc:
                failures.append({"frame": index, "reason": str(exc)})
                continue
            rows.append({"frame": index, **angles})
            last = {"color": color, "depth": depth, "angles": angles,
                    "detail": detail}
    finally:
        pipeline.stop()

    if not rows:
        raise RuntimeError(f"no usable frames; first failures={failures[:5]}")

    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{prefix}_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", *ANGLE_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(f"{prefix}_color.png", last["color"])
    np.save(f"{prefix}_depth_raw.npy", last["depth"])

    canvas = last["color"].copy()
    for roi, shade in ((args.thumb_roi, (0, 200, 255)),
                       (args.palm_roi, (255, 200, 0))):
        cv2.rectangle(canvas, (roi[0], roi[1]), (roi[2], roi[3]), shade, 1)
    cv2.putText(canvas, f"yaw {last['angles']['thumb_cmc_yaw_deg_2d']:+.3f} deg",
                (args.thumb_roi[0], max(14, args.thumb_roi[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(f"{prefix}_annotated.png", canvas)

    summary = {
        "schema_version": 1,
        "camera_only": True,
        "hand_motion_sent": False,
        "camera": {"serial": args.serial, "rgb_controls": rgb_controls,
                   "depth_scale_m_per_unit": depth_scale},
        "detector": {
            "mode": "thumb_cmc_yaw_palm_reference_plus_thumb_proximal",
            "thumb_roi_xyxy": list(args.thumb_roi),
            "palm_roi_xyxy": list(args.palm_roi),
            "hsv_lower": list(args.hsv_lower),
            "hsv_upper": list(args.hsv_upper),
            "thumb_min_area_px": args.thumb_min_area,
            "palm_min_area_px": args.palm_min_area,
            "min_elongation2": args.min_elongation2,
            "reference_offset_deg": args.reference_offset_deg,
            "primary_value": "2D in-plane; valid only while the yaw axis stays "
                             "near the optical axis, which the thumb marker "
                             "area and major extent are recorded to check",
        },
        "capture": {
            "requested_frames": args.frames,
            "usable_frames": len(rows),
            "failure_count": len(failures),
            "failure_fraction": len(failures) / float(args.frames),
            "started_wall_time_s": started,
            "ended_wall_time_s": time.time(),
        },
        "angle_summary_deg": {
            key: _summary([float(r[key]) for r in rows])
            for key in ANGLE_KEYS
        },
        "block_medians_deg": {
            key: _blocks(rows, key, args.stability_block_frames)
            for key in ANGLE_KEYS
        },
        "last_markers": last["detail"],
        "failures": failures,
        "interpretation": (
            "Physical thumb CMC yaw is the change from a separately recorded "
            "reference capture in this same camera pose and isolation posture. "
            "thumb_marker_area_px and thumb_major_extent_px are diagnostics for "
            "out-of-plane motion, not angles."
        ),
        "outputs": {
            "samples_csv": str(Path(f"{prefix}_samples.csv")),
            "color_png": str(Path(f"{prefix}_color.png")),
            "depth_raw_npy": str(Path(f"{prefix}_depth_raw.npy")),
            "annotated_png": str(Path(f"{prefix}_annotated.png")),
        },
    }
    Path(f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="143322073091")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--rgb-exposure", type=float, default=166.0)
    parser.add_argument("--rgb-gain", type=float, default=32.0)
    parser.add_argument("--rgb-white-balance", type=float, default=4600.0)
    parser.add_argument(
        "--thumb-roi", type=_parse_roi, default=(330, 170, 900, 470),
        help="must contain the whole thumb tape across the entire sweep. A ROI "
             "sized for the rest pose clipped the marker mid-sweep on "
             "2026-08-03 and biased the angle by 9.5 deg while every other "
             "gate still passed.")
    parser.add_argument("--palm-roi", type=_parse_roi, default=(440, 430, 900, 660))
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=(75, 40, 20))
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=(165, 255, 255))
    parser.add_argument("--thumb-depth-band", type=float, nargs=2,
                        default=(0.265, 0.305),
                        help="depth band holding the thumb L; the valley "
                             "between the thumb and palm tapes is at 0.30-0.32 m")
    parser.add_argument("--palm-depth-band", type=float, nargs=2,
                        default=(0.305, 0.350))
    parser.add_argument("--thumb-min-area", type=int, default=500)
    parser.add_argument("--palm-min-area", type=int, default=500)
    parser.add_argument("--min-elongation2", type=float, default=4.0)
    parser.add_argument("--reference-offset-deg", type=float, default=118.0,
                        help="removed before normalizing to keep the reported "
                             "angle away from the +-90 wrap boundary. 118 "
                             "centres the measured ~80 deg travel on zero: the "
                             "rest pose reads about -40 and the far end +40.")
    parser.add_argument("--stability-block-frames", type=int, default=60)
    parser.add_argument("--max-failure-fraction", type=float, default=0.1)
    parser.add_argument("--out-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = capture(args)
    print(json.dumps({
        "usable_frames": summary["capture"]["usable_frames"],
        "failure_fraction": summary["capture"]["failure_fraction"],
        "angle_summary_deg": summary["angle_summary_deg"],
        "block_medians_deg": summary["block_medians_deg"],
    }, indent=2))
    if summary["capture"]["failure_fraction"] > args.max_failure_fraction:
        print("FAIL: failure fraction above gate")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
