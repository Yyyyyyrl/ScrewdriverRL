#!/usr/bin/env python3
"""Measure index MCP pitch from palm and proximal blue-tape markers.

This camera-only tool is intentionally narrower than
``measure_g20_tape_angles``.  It identifies the large metacarpal marker, finds
the fingertip-facing end of that marker, and merges blue components in the
nearby proximal-link region.  PIP/middle/distal markers are not required, so
their fragmentation or motion cannot change the MCP association.

The reported tape-to-tape angle includes fixed tape installation offsets.
Physical MCP pitch is the angle change from a separately recorded raw-255
reference, not the absolute angle in any single frame.
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


def measure_mcp_pitch_frame(
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_scale: float,
    roi: tuple[int, int, int, int],
    hsv_lower: tuple[int, int, int] = (75, 40, 20),
    hsv_upper: tuple[int, int, int] = (165, 255, 255),
    component_min_area: int = 15,
    palm_min_area: int = 1000,
    palm_dominance_ratio: float = 3.0,
    palm_min_elongation: float = 15.0,
    proximal_min_area: int = 100,
    proximal_min_elongation: float = 8.0,
    proximal_along_min_px: float = 5.0,
    proximal_along_max_px: float = 125.0,
    proximal_radius_max_px: float = 125.0,
    segmentation_depth_min_m: float = 0.330,
    segmentation_depth_max_m: float = 0.390,
    morph_open_kernel: int = 3,
) -> tuple[dict[str, MarkerMeasurement], np.ndarray, dict[str, Any]]:
    """Return fail-closed palm/proximal association for one aligned frame."""

    height, width = color.shape[:2]
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"ROI {roi} is outside image {width}x{height}")
    if not (
        0.0 < segmentation_depth_min_m < segmentation_depth_max_m
    ):
        raise ValueError("segmentation depth bounds must satisfy 0 < min < max")

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray(hsv_lower, dtype=np.uint8),
        np.asarray(hsv_upper, dtype=np.uint8),
    )
    depth_m = depth.astype(np.float64) * depth_scale
    depth_gate = (
        (depth > 0)
        & (depth_m >= segmentation_depth_min_m)
        & (depth_m <= segmentation_depth_max_m)
    )
    mask = cv2.bitwise_and(
        mask, np.where(depth_gate, 255, 0).astype(np.uint8)
    )
    roi_mask = np.zeros_like(mask)
    roi_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    if morph_open_kernel > 1:
        roi_mask = cv2.morphologyEx(
            roi_mask,
            cv2.MORPH_OPEN,
            np.ones(
                (morph_open_kernel, morph_open_kernel), dtype=np.uint8
            ),
        )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        roi_mask
    )
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < component_min_area:
            continue
        pixels = np.column_stack(np.where(labels == label))[
            :, ::-1
        ].astype(np.float64)
        axis, elongation = _pca_axis(pixels)
        components.append(
            {
                "label": label,
                "pixels": pixels,
                "area": area,
                "centroid": np.asarray(centroids[label], dtype=np.float64),
                "axis": axis,
                "elongation": elongation,
            }
        )
    if len(components) < 2:
        raise ValueError(
            f"need palm and proximal components, found {len(components)}"
        )

    ranked = sorted(components, key=lambda item: item["area"], reverse=True)
    palm = ranked[0]
    if palm["area"] < palm_min_area:
        raise ValueError(
            f"palm marker area {palm['area']} is below {palm_min_area}"
        )
    if palm["area"] < palm_dominance_ratio * ranked[1]["area"]:
        raise ValueError("palm marker area identity is ambiguous")
    if palm["elongation"] < palm_min_elongation:
        raise ValueError(
            f"palm marker elongation {palm['elongation']:.1f} is below "
            f"{palm_min_elongation:.1f}"
        )

    palm_axis = palm["axis"]
    # The fixed D435 view has the index finger on the positive image-x side.
    # _pca_axis already orients its x component nonnegative.
    palm_center = np.mean(palm["pixels"], axis=0)
    palm_projection = (palm["pixels"] - palm_center) @ palm_axis
    palm_endpoint = palm_center + np.max(palm_projection) * palm_axis

    selected_parts: list[dict[str, Any]] = []
    rejected_parts: list[dict[str, Any]] = []
    for component in ranked[1:]:
        delta = component["centroid"] - palm_endpoint
        along = float(delta @ palm_axis)
        radius = float(np.linalg.norm(delta))
        detail = {
            "area_px": component["area"],
            "centroid_uv": [
                float(component["centroid"][0]),
                float(component["centroid"][1]),
            ],
            "along_from_palm_endpoint_px": along,
            "radius_from_palm_endpoint_px": radius,
            "elongation2": float(component["elongation"]),
        }
        if (
            proximal_along_min_px <= along <= proximal_along_max_px
            and radius <= proximal_radius_max_px
            and component["elongation"] >= 3.0
        ):
            selected_parts.append(component)
        else:
            rejected_parts.append(detail)
    if not selected_parts:
        raise ValueError("no component lies in the proximal-link region")

    proximal_pixels = np.concatenate(
        [item["pixels"] for item in selected_parts], axis=0
    )
    proximal_axis, proximal_elongation = _pca_axis(proximal_pixels)
    if len(proximal_pixels) < proximal_min_area:
        raise ValueError(
            f"merged proximal area {len(proximal_pixels)} is below "
            f"{proximal_min_area}"
        )
    if proximal_elongation < proximal_min_elongation:
        raise ValueError(
            f"merged proximal elongation {proximal_elongation:.1f} is below "
            f"{proximal_min_elongation:.1f}"
        )
    proximal_center = np.mean(proximal_pixels, axis=0)
    proximal_delta = proximal_center - palm_endpoint
    proximal_along = float(proximal_delta @ palm_axis)
    proximal_radius = float(np.linalg.norm(proximal_delta))
    if not (
        proximal_along_min_px
        <= proximal_along
        <= proximal_along_max_px
        and proximal_radius <= proximal_radius_max_px
    ):
        raise ValueError("merged proximal marker left the allowed region")

    markers = {
        "metacarpal": _marker_measurement(
            "metacarpal",
            palm["pixels"],
            depth,
            intrinsics,
            depth_scale,
        ),
        "proximal": _marker_measurement(
            "proximal",
            proximal_pixels,
            depth,
            intrinsics,
            depth_scale,
        ),
    }
    diagnostics = {
        "palm_endpoint_uv": [
            float(palm_endpoint[0]),
            float(palm_endpoint[1]),
        ],
        "selected_proximal_component_count": len(selected_parts),
        "selected_proximal_component_areas_px": [
            int(item["area"]) for item in selected_parts
        ],
        "proximal_along_from_palm_endpoint_px": proximal_along,
        "proximal_radius_from_palm_endpoint_px": proximal_radius,
        "rejected_components": rejected_parts,
    }
    return markers, roi_mask, diagnostics


def mcp_pitch_angles(
    markers: dict[str, MarkerMeasurement],
) -> dict[str, float | None]:
    palm = markers["metacarpal"]
    proximal = markers["proximal"]
    projected = _line_difference(palm.heading2_deg, proximal.heading2_deg)
    output: dict[str, float | None] = {
        "mcp_projected_deg_2d": projected,
        "mcp_projected_deg_3d": None,
    }
    if palm.axis3_xyz is None or proximal.axis3_xyz is None:
        return output
    palm3 = np.asarray(palm.axis3_xyz)
    proximal3 = np.asarray(proximal.axis3_xyz)
    dot = float(np.clip(abs(np.dot(palm3, proximal3)), 0.0, 1.0))
    unsigned = math.degrees(math.acos(dot))
    output["mcp_projected_deg_3d"] = math.copysign(unsigned, projected)
    return output


def _annotate(
    color: np.ndarray,
    markers: dict[str, MarkerMeasurement],
    angles: dict[str, float | None],
    diagnostics: dict[str, Any],
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    output = color.copy()
    palette = {"metacarpal": (0, 255, 0), "proximal": (255, 0, 255)}
    for marker in markers.values():
        x, y, w, h = marker.bbox_xywh
        marker_color = palette[marker.name]
        cv2.rectangle(output, (x, y), (x + w, y + h), marker_color, 2)
        center = np.asarray(marker.centroid_uv)
        axis = np.asarray(marker.axis2_uv)
        half = 0.65 * max(w, h)
        p0 = tuple(np.round(center - half * axis).astype(int))
        p1 = tuple(np.round(center + half * axis).astype(int))
        cv2.line(output, p0, p1, marker_color, 3)
        cv2.putText(
            output,
            f"{marker.name} {marker.heading2_deg:+.2f} deg",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            marker_color,
            2,
            cv2.LINE_AA,
        )
    endpoint = tuple(
        np.round(diagnostics["palm_endpoint_uv"]).astype(int)
    )
    cv2.circle(output, endpoint, 6, (0, 255, 255), -1)
    cv2.putText(
        output,
        f"MCP={angles['mcp_projected_deg_2d']:+.2f} deg",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    x0, y0, x1, y1 = roi
    cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 255), 1)
    return output


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import pyrealsense2 as rs

    context = rs.context()
    devices = list(context.query_devices())
    matches = [
        device
        for device in devices
        if device.get_info(rs.camera_info.serial_number) == args.serial
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one D435 serial {args.serial}, found {len(matches)}"
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    config.enable_stream(
        rs.stream.depth,
        args.width,
        args.height,
        rs.format.z16,
        args.fps,
    )
    profile = pipeline.start(config)
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
            rs.option.enable_auto_exposure
        ),
        "exposure": color_sensor.get_option(rs.option.exposure),
        "gain": color_sensor.get_option(rs.option.gain),
        "enable_auto_white_balance": color_sensor.get_option(
            rs.option.enable_auto_white_balance
        ),
        "white_balance": color_sensor.get_option(rs.option.white_balance),
    }
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    intrinsics = {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(value) for value in intr.coeffs],
    }
    for _ in range(args.warmup_frames):
        pipeline.wait_for_frames(5000)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    last_color: np.ndarray | None = None
    last_depth: np.ndarray | None = None
    last_markers: dict[str, MarkerMeasurement] | None = None
    last_angles: dict[str, float | None] | None = None
    last_diagnostics: dict[str, Any] | None = None
    started = time.time()
    try:
        for frame_index in range(args.frames):
            frames = align.process(pipeline.wait_for_frames(5000))
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                failures.append(
                    {"frame": frame_index, "error": "incomplete frameset"}
                )
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            try:
                markers, _, diagnostics = measure_mcp_pitch_frame(
                    color=color,
                    depth=depth,
                    intrinsics=intrinsics,
                    depth_scale=depth_scale,
                    roi=args.roi,
                    hsv_lower=tuple(args.hsv_lower),
                    hsv_upper=tuple(args.hsv_upper),
                    component_min_area=args.component_min_area,
                    palm_min_area=args.palm_min_area,
                    proximal_min_area=args.proximal_min_area,
                    proximal_along_min_px=args.proximal_along_min_px,
                    proximal_along_max_px=args.proximal_along_max_px,
                    proximal_radius_max_px=args.proximal_radius_max_px,
                    segmentation_depth_min_m=args.segmentation_depth_min_m,
                    segmentation_depth_max_m=args.segmentation_depth_max_m,
                )
                angles = mcp_pitch_angles(markers)
            except ValueError as exc:
                failures.append({"frame": frame_index, "error": str(exc)})
                continue
            row: dict[str, Any] = {
                "frame": frame_index,
                "timestamp_s": time.time(),
                **angles,
                "selected_proximal_component_count": diagnostics[
                    "selected_proximal_component_count"
                ],
                "proximal_along_from_palm_endpoint_px": diagnostics[
                    "proximal_along_from_palm_endpoint_px"
                ],
                "proximal_radius_from_palm_endpoint_px": diagnostics[
                    "proximal_radius_from_palm_endpoint_px"
                ],
            }
            for name, marker in markers.items():
                row[f"{name}_heading_deg_2d"] = marker.heading2_deg
                row[f"{name}_area_px"] = marker.area_px
                row[f"{name}_elongation_2d"] = marker.elongation2
                row[f"{name}_depth_valid_fraction"] = (
                    marker.depth_valid_fraction
                )
                row[f"{name}_depth_median_m"] = marker.depth_median_m
            rows.append(row)
            last_color, last_depth = color, depth
            last_markers, last_angles = markers, angles
            last_diagnostics = diagnostics
    finally:
        pipeline.stop()

    if (
        not rows
        or last_color is None
        or last_depth is None
        or last_markers is None
        or last_angles is None
        or last_diagnostics is None
    ):
        raise RuntimeError(
            f"no usable MCP pitch measurements; failures={failures[:5]}"
        )
    failure_fraction = len(failures) / args.frames
    if failure_fraction > args.max_failure_fraction:
        raise RuntimeError(
            f"marker detection failure fraction {failure_fraction:.1%} exceeds "
            f"limit {args.max_failure_fraction:.1%}"
        )

    prefix = args.out_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{prefix}_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(Path(f"{prefix}_color.png")), last_color)
    np.save(str(Path(f"{prefix}_depth_raw.npy")), last_depth)
    cv2.imwrite(
        str(Path(f"{prefix}_annotated.png")),
        _annotate(
            last_color,
            last_markers,
            last_angles,
            last_diagnostics,
            args.roi,
        ),
    )

    angle_keys = ("mcp_projected_deg_2d", "mcp_projected_deg_3d")
    summary = {
        "schema_version": 1,
        "camera_only": True,
        "linkerhand_sdk_imported": False,
        "hand_motion_sent": False,
        "camera": {
            "name": matches[0].get_info(rs.camera_info.name),
            "serial": args.serial,
            "firmware": matches[0].get_info(
                rs.camera_info.firmware_version
            ),
            "usb": matches[0].get_info(
                rs.camera_info.usb_type_descriptor
            ),
            "profile": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
            },
            "intrinsics": intrinsics,
            "depth_scale_m_per_unit": depth_scale,
            "rgb_controls": rgb_controls,
        },
        "detector": {
            "mode": "index_mcp_palm_endpoint_proximal_region",
            "roi_xyxy": list(args.roi),
            "hsv_lower": list(args.hsv_lower),
            "hsv_upper": list(args.hsv_upper),
            "component_min_area_px": args.component_min_area,
            "palm_min_area_px": args.palm_min_area,
            "proximal_min_area_px": args.proximal_min_area,
            "proximal_along_min_px": args.proximal_along_min_px,
            "proximal_along_max_px": args.proximal_along_max_px,
            "proximal_radius_max_px": args.proximal_radius_max_px,
            "segmentation_depth_min_m": args.segmentation_depth_min_m,
            "segmentation_depth_max_m": args.segmentation_depth_max_m,
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
            key: _summary(
                [float(row[key]) for row in rows if row[key] is not None]
            )
            for key in angle_keys
            if any(row[key] is not None for row in rows)
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
                "axis3_xyz": (
                    None
                    if marker.axis3_xyz is None
                    else list(marker.axis3_xyz)
                ),
                "elongation3": marker.elongation3,
            }
            for name, marker in last_markers.items()
        },
        "last_diagnostics": last_diagnostics,
        "failures": failures,
        "interpretation": (
            "Physical index MCP pitch is the angle change from the separately "
            "recorded raw-255 reference; fixed tape offsets are not absolute "
            "joint coordinates."
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
    parser.add_argument(
        "--roi", type=_parse_roi, default=(500, 185, 1100, 500)
    )
    parser.add_argument(
        "--hsv-lower", type=int, nargs=3, default=(75, 40, 20)
    )
    parser.add_argument(
        "--hsv-upper", type=int, nargs=3, default=(165, 255, 255)
    )
    parser.add_argument("--component-min-area", type=int, default=15)
    parser.add_argument("--palm-min-area", type=int, default=1000)
    parser.add_argument("--proximal-min-area", type=int, default=100)
    parser.add_argument("--proximal-along-min-px", type=float, default=5.0)
    parser.add_argument("--proximal-along-max-px", type=float, default=125.0)
    parser.add_argument("--proximal-radius-max-px", type=float, default=125.0)
    parser.add_argument(
        "--segmentation-depth-min-m", type=float, default=0.330
    )
    parser.add_argument(
        "--segmentation-depth-max-m", type=float, default=0.390
    )
    parser.add_argument("--max-failure-fraction", type=float, default=0.1)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.frames <= 0 or args.warmup_frames < 0:
        parser.error("--frames must be positive and --warmup-frames nonnegative")
    if not 0.0 <= args.max_failure_fraction < 1.0:
        parser.error("--max-failure-fraction must be in [0,1)")
    if not (
        0.0
        < args.segmentation_depth_min_m
        < args.segmentation_depth_max_m
    ):
        parser.error("segmentation depth bounds must satisfy 0 < min < max")
    if not (
        0.0
        <= args.proximal_along_min_px
        < args.proximal_along_max_px
    ):
        parser.error("proximal along bounds are invalid")
    if args.proximal_radius_max_px <= 0:
        parser.error("--proximal-radius-max-px must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = capture(parse_args(argv))
    print(
        json.dumps(
            {
                "camera_only": summary["camera_only"],
                "hand_motion_sent": summary["hand_motion_sent"],
                "capture": summary["capture"],
                "angle_summary_deg": summary["angle_summary_deg"],
                "outputs": summary["outputs"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
