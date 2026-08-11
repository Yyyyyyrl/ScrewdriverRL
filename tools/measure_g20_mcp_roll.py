#!/usr/bin/env python3
"""Measure four-finger MCP roll (abduction) from the Phase B roll-view camera.

Camera-only. The roll view looks along the palm normal, so abduction is an
in-plane rotation and the primary value is 2D, exactly like the Phase A MCP
pitch tool. The palm marker in this pose carries almost no valid depth, so no
palm reference *plane* is reconstructed; the palm tape supplies a 2D reference
heading only and per-finger 3D axes are kept as diagnostics.

Marker layout expected in this pose:

- one long palm tape, roughly perpendicular to the fingers,
- one proximal tape per long finger, roughly along the finger.

Finger identity is positional: sorted by image row, bottom to top is
index, middle, ring, pinky. A frame in which the finger ROI does not contain
exactly four separated components is a failure, never a best guess.

The reported angles include fixed tape installation offsets. Physical roll is
the change from a separately recorded reference capture, not the absolute angle
in any single frame.
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
    MarkerMeasurement,
    _deproject_mask_points,
    _line_difference,
    _normalize_axis_heading,
    _parse_roi,
    _pca_axis,
    _summary,
)

# Bottom to top in the roll view. Confirmed against the hand on 2026-08-03.
FINGERS_BOTTOM_TO_TOP = ("index", "middle", "ring", "pinky")

ANGLE_KEYS = (
    *(f"{finger}_abduction_deg_2d" for finger in FINGERS_BOTTOM_TO_TOP),
    "palm_reference_heading_deg_2d",
    *(f"{finger}_heading_deg_2d" for finger in FINGERS_BOTTOM_TO_TOP),
)


def palm_reference_heading(palm_heading_deg: float) -> float:
    """Palm tape heading rotated into the finger-pointing direction.

    The palm tape is near-vertical in this pose, so its own heading sits on the
    +-90 wrap boundary of the unoriented-line convention and flips sign between
    frames. Rotating by 90 degrees before normalizing moves the reference to
    approximately zero, away from the discontinuity: both +90 and -90 map to 0.
    """

    return _normalize_axis_heading(palm_heading_deg + 90.0)


def abduction_angle(palm_heading_deg: float, finger_heading_deg: float) -> float:
    """Signed in-plane abduction of one finger against the palm reference.

    Positive means the finger heading is rotated counter-clockwise in image
    coordinates relative to the palm reference. Image y points down, so the
    mapping from this sign to anatomical abduction/adduction is fixed by the
    small-amplitude direction check required before any sweep, and is recorded
    in the session rather than assumed here. The sign is never removed with
    abs(): a reversed convention must be corrected explicitly.
    """

    return _line_difference(palm_reference_heading(palm_heading_deg),
                            finger_heading_deg)


def _marker_measurement(
    name: str,
    pixels: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_scale: float,
) -> MarkerMeasurement:
    axis2, elongation2 = _pca_axis(pixels)
    heading = _normalize_axis_heading(
        math.degrees(math.atan2(axis2[1], axis2[0]))
    )
    points3, valid_fraction, depth_median = _deproject_mask_points(
        pixels, depth, intrinsics, depth_scale
    )
    axis3: tuple[float, float, float] | None = None
    elongation3: float | None = None
    if len(points3) >= 20:
        axis3_value, elongation3 = _pca_axis(points3)
        axis3 = tuple(float(value) for value in axis3_value)
    min_xy = np.floor(np.min(pixels, axis=0)).astype(int)
    max_xy = np.ceil(np.max(pixels, axis=0)).astype(int)
    center = np.mean(pixels, axis=0)
    return MarkerMeasurement(
        name=name,
        area_px=len(pixels),
        bbox_xywh=(
            int(min_xy[0]),
            int(min_xy[1]),
            int(max_xy[0] - min_xy[0] + 1),
            int(max_xy[1] - min_xy[1] + 1),
        ),
        centroid_uv=tuple(float(value) for value in center),
        axis2_uv=tuple(float(value) for value in axis2),
        heading2_deg=heading,
        elongation2=elongation2,
        depth_valid_fraction=valid_fraction,
        depth_median_m=depth_median,
        axis3_xyz=axis3,
        elongation3=elongation3,
    )


def _parse_row_band(text: str) -> tuple[float, float]:
    try:
        low, high = (float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("row band must be Y0,Y1") from exc
    if high <= low:
        raise argparse.ArgumentTypeError("row band must have Y1 > Y0")
    return low, high


def _blue_mask(color: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(args.hsv_lower, dtype=np.uint8),
                       np.array(args.hsv_upper, dtype=np.uint8)) > 0


def _roi_mask(shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = roi
    mask[y0:y1, x0:x1] = True
    return mask


def _components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    out: list[np.ndarray] = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(labels == index)
        out.append(np.column_stack([xs.astype(np.float64), ys.astype(np.float64)]))
    return out


def measure_roll_frame(
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_scale: float,
    args: argparse.Namespace,
) -> tuple[dict[str, MarkerMeasurement], dict[str, float], dict[str, Any]]:
    """Locate the palm and the four proximal tapes and derive abduction angles."""

    blue = _blue_mask(color, args)
    depth_m = depth.astype(np.float64) * depth_scale
    near = (depth_m >= args.segmentation_depth_min_m) & (
        depth_m <= args.segmentation_depth_max_m
    )

    # Palm: no depth gate. The palm tape sits on a specular housing that the
    # D435 cannot range, so requiring depth here would drop every frame.
    palm_blobs = _components(
        blue & _roi_mask(color.shape[:2], args.palm_roi), args.palm_min_area
    )
    if not palm_blobs:
        raise ValueError("no palm marker in palm ROI")
    palm_pixels = max(palm_blobs, key=len)

    finger_blobs = _components(
        blue & _roi_mask(color.shape[:2], args.roi) & near, args.component_min_area
    )

    if args.target_finger is None:
        # Reference mode: all four fingers extended and visible, identity is
        # positional bottom to top.
        if len(finger_blobs) != len(FINGERS_BOTTOM_TO_TOP):
            raise ValueError(
                f"expected {len(FINGERS_BOTTOM_TO_TOP)} finger markers in ROI, "
                f"found {len(finger_blobs)}"
            )
        finger_blobs.sort(key=lambda pixels: float(np.mean(pixels[:, 1])),
                          reverse=True)
        rows = [float(np.mean(pixels[:, 1])) for pixels in finger_blobs]
        gaps = [rows[i] - rows[i + 1] for i in range(len(rows) - 1)]
        if min(gaps) < args.min_marker_row_gap_px:
            raise ValueError(
                f"finger markers too close to separate by row: gaps={gaps}"
            )
        measured = list(zip(FINGERS_BOTTOM_TO_TOP, finger_blobs))
    else:
        # Sweep mode: the non-target fingers are pitched out of the way, so
        # only the target's proximal tape has to be visible. Identity is not
        # inferred from "whichever blob is left": exactly one blob may remain
        # in the ROI and it must sit inside the target's expected row band.
        if len(finger_blobs) != 1:
            raise ValueError(
                f"expected exactly 1 finger marker in ROI for target "
                f"{args.target_finger}, found {len(finger_blobs)}"
            )
        row = float(np.mean(finger_blobs[0][:, 1]))
        low, high = args.target_row_band
        if not low <= row <= high:
            raise ValueError(
                f"{args.target_finger} marker row {row:.1f} outside its "
                f"expected band {low}..{high}"
            )
        rows, gaps = [row], []
        measured = [(args.target_finger, finger_blobs[0])]

    markers = {
        "palm": _marker_measurement(
            "palm", palm_pixels, depth, intrinsics, depth_scale
        )
    }
    for finger, pixels in measured:
        markers[finger] = _marker_measurement(
            finger, pixels, depth, intrinsics, depth_scale
        )
        if markers[finger].elongation2 < args.min_elongation2:
            raise ValueError(
                f"{finger} marker not elongated enough for a stable heading: "
                f"{markers[finger].elongation2:.2f}"
            )

    palm_ref = palm_reference_heading(markers["palm"].heading2_deg)
    angles: dict[str, float] = {"palm_reference_heading_deg_2d": palm_ref}
    for finger, _ in measured:
        angles[f"{finger}_heading_deg_2d"] = markers[finger].heading2_deg
        angles[f"{finger}_abduction_deg_2d"] = abduction_angle(
            markers["palm"].heading2_deg, markers[finger].heading2_deg
        )

    diagnostics = {
        "palm_depth_valid_fraction": markers["palm"].depth_valid_fraction,
        "palm_heading_raw_deg": markers["palm"].heading2_deg,
        "marker_rows_bottom_to_top": rows,
        "marker_row_gaps_px": gaps,
        "measured_fingers": [finger for finger, _ in measured],
        "finger_axis3_xyz": {
            finger: (
                None
                if markers[finger].axis3_xyz is None
                else list(markers[finger].axis3_xyz)
            )
            for finger, _ in measured
        },
        "finger_depth_median_m": {
            finger: markers[finger].depth_median_m
            for finger, _ in measured
        },
    }
    return markers, angles, diagnostics


def _annotate(
    color: np.ndarray,
    markers: dict[str, MarkerMeasurement],
    angles: dict[str, float],
    args: argparse.Namespace,
) -> np.ndarray:
    canvas = color.copy()
    for roi, shade in ((args.roi, (0, 200, 255)), (args.palm_roi, (255, 200, 0))):
        cv2.rectangle(canvas, (roi[0], roi[1]), (roi[2], roi[3]), shade, 1)
    for name, marker in markers.items():
        x, y, w, h = marker.bbox_xywh
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cx, cy = marker.centroid_uv
        axis = marker.axis2_uv
        length = 40.0
        cv2.line(
            canvas,
            (int(cx - axis[0] * length), int(cy - axis[1] * length)),
            (int(cx + axis[0] * length), int(cy + axis[1] * length)),
            (0, 0, 255),
            1,
        )
        label = name
        if name != "palm":
            label = f"{name} {angles[f'{name}_abduction_deg_2d']:+.2f}"
        cv2.putText(canvas, label, (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"palm ref {angles['palm_reference_heading_deg_2d']:+.2f}",
                (args.palm_roi[0], max(12, args.palm_roi[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
    return canvas


def _block_medians(rows: list[dict[str, Any]], key: str, block: int) -> list[float]:
    """Median of each consecutive block, for the 60-frame stability gate."""
    values = [row[key] for row in rows if row.get(key) is not None]
    return [
        float(np.median(values[start:start + block]))
        for start in range(0, len(values) - block + 1, block)
    ]


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import pyrealsense2 as rs

    context = rs.context()
    matches = [
        device
        for device in context.query_devices()
        if device.get_info(rs.camera_info.serial_number) == args.serial
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one D435 serial {args.serial}, found {len(matches)}"
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height,
                         rs.format.z16, args.fps)
    profile = pipeline.start(config)
    try:
        color_sensor = next(
            sensor
            for sensor in profile.get_device().query_sensors()
            if "RGB" in sensor.get_info(rs.camera_info.name)
        )
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, args.rgb_exposure)
        color_sensor.set_option(rs.option.gain, args.rgb_gain)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, args.rgb_white_balance)
        rgb_controls = {
            "enable_auto_exposure": color_sensor.get_option(
                rs.option.enable_auto_exposure),
            "exposure": color_sensor.get_option(rs.option.exposure),
            "gain": color_sensor.get_option(rs.option.gain),
            "enable_auto_white_balance": color_sensor.get_option(
                rs.option.enable_auto_white_balance),
            "white_balance": color_sensor.get_option(rs.option.white_balance),
        }
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsics = {
            "width": int(intr.width), "height": int(intr.height),
            "fx": float(intr.fx), "fy": float(intr.fy),
            "ppx": float(intr.ppx), "ppy": float(intr.ppy),
            "model": str(intr.model),
            "coeffs": [float(value) for value in intr.coeffs],
        }
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
                markers, angles, diagnostics = measure_roll_frame(
                    color, depth, intrinsics, depth_scale, args
                )
            except ValueError as exc:
                failures.append({"frame": index, "reason": str(exc)})
                continue
            rows.append({"frame": index, **angles})
            last = {"color": color, "depth": depth, "markers": markers,
                    "angles": angles, "diagnostics": diagnostics}
    finally:
        pipeline.stop()

    failure_fraction = len(failures) / float(args.frames)
    if not rows:
        raise RuntimeError(f"no usable frames; first failures={failures[:5]}")

    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{prefix}_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        present = [key for key in ANGLE_KEYS if key in rows[0]]
        writer = csv.DictWriter(fh, fieldnames=["frame", *present])
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(f"{prefix}_color.png", last["color"])
    np.save(f"{prefix}_depth_raw.npy", last["depth"])
    cv2.imwrite(f"{prefix}_annotated.png",
                _annotate(last["color"], last["markers"], last["angles"], args))

    summary = {
        "schema_version": 1,
        "camera_only": True,
        "hand_motion_sent": False,
        "camera": {
            "serial": args.serial,
            "profile": {"width": args.width, "height": args.height,
                        "fps": args.fps},
            "intrinsics": intrinsics,
            "depth_scale_m_per_unit": depth_scale,
            "rgb_controls": rgb_controls,
        },
        "detector": {
            "mode": "phase_b_roll_palm_reference_plus_four_proximal",
            "finger_order_bottom_to_top": list(FINGERS_BOTTOM_TO_TOP),
            "target_finger": args.target_finger,
            "target_row_band": (None if args.target_row_band is None
                                else list(args.target_row_band)),
            "roi_xyxy": list(args.roi),
            "palm_roi_xyxy": list(args.palm_roi),
            "hsv_lower": list(args.hsv_lower),
            "hsv_upper": list(args.hsv_upper),
            "component_min_area_px": args.component_min_area,
            "palm_min_area_px": args.palm_min_area,
            "min_marker_row_gap_px": args.min_marker_row_gap_px,
            "min_elongation2": args.min_elongation2,
            "segmentation_depth_min_m": args.segmentation_depth_min_m,
            "segmentation_depth_max_m": args.segmentation_depth_max_m,
            "palm_depth_gated": False,
        },
        "capture": {
            "requested_frames": args.frames,
            "usable_frames": len(rows),
            "failure_count": len(failures),
            "failure_fraction": failure_fraction,
            "started_wall_time_s": started,
            "ended_wall_time_s": time.time(),
        },
        "angle_summary_deg": {
            key: _summary([float(row[key]) for row in rows
                           if row.get(key) is not None])
            for key in ANGLE_KEYS
            if any(row.get(key) is not None for row in rows)
        },
        "block_medians_deg": {
            key: _block_medians(rows, key, args.stability_block_frames)
            for key in ANGLE_KEYS
            if any(row.get(key) is not None for row in rows)
        },
        "last_markers": {
            name: {
                "area_px": marker.area_px,
                "bbox_xywh": list(marker.bbox_xywh),
                "centroid_uv": list(marker.centroid_uv),
                "heading2_deg": marker.heading2_deg,
                "elongation2": marker.elongation2,
                "depth_valid_fraction": marker.depth_valid_fraction,
                "depth_median_m": marker.depth_median_m,
                "axis3_xyz": (None if marker.axis3_xyz is None
                              else list(marker.axis3_xyz)),
                "elongation3": marker.elongation3,
            }
            for name, marker in last["markers"].items()
        },
        "last_diagnostics": last["diagnostics"],
        "failures": failures,
        "interpretation": (
            "Physical MCP roll is the abduction change from a separately "
            "recorded reference capture in this same camera pose. The 2D value "
            "is primary; per-finger 3D axes are diagnostic only because the "
            "palm tape carries too little depth in this pose to reconstruct a "
            "palm plane. Finger identity is positional, bottom to top is "
            "index, middle, ring, pinky."
        ),
        "outputs": {
            "samples_csv": str(Path(f"{prefix}_samples.csv")),
            "color_png": str(Path(f"{prefix}_color.png")),
            "depth_raw_npy": str(Path(f"{prefix}_depth_raw.npy")),
            "annotated_png": str(Path(f"{prefix}_annotated.png")),
        },
    }
    Path(f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
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
    parser.add_argument("--roi", type=_parse_roi, default=(600, 150, 1000, 560),
                        help="finger ROI x0,y0,x1,y1; must exclude the palm tape")
    parser.add_argument("--palm-roi", type=_parse_roi, default=(500, 150, 610, 560),
                        help="palm ROI x0,y0,x1,y1; must exclude the finger tapes")
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=(75, 40, 20))
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=(165, 255, 255))
    parser.add_argument("--component-min-area", type=int, default=15)
    parser.add_argument("--palm-min-area", type=int, default=1000)
    parser.add_argument("--min-marker-row-gap-px", type=float, default=20.0)
    parser.add_argument("--min-elongation2", type=float, default=4.0)
    parser.add_argument("--segmentation-depth-min-m", type=float, default=0.245)
    parser.add_argument("--segmentation-depth-max-m", type=float, default=0.280)
    parser.add_argument(
        "--target-finger", choices=FINGERS_BOTTOM_TO_TOP, default=None,
        help=(
            "sweep mode: only this finger's proximal tape must be visible, "
            "because the other fingers are pitched out of the way. Omit for "
            "the four-finger reference mode."
        ))
    parser.add_argument(
        "--target-row-band", type=_parse_row_band, default=None,
        metavar="Y0,Y1",
        help="image row band the target marker must fall inside, required "
             "with --target-finger so identity is never inferred by elimination")
    parser.add_argument("--stability-block-frames", type=int, default=60)
    parser.add_argument("--max-failure-fraction", type=float, default=0.1)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.target_finger is None) != (args.target_row_band is None):
        parser.error(
            "--target-finger and --target-row-band must be given together")
    return args


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
