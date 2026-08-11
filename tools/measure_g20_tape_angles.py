#!/usr/bin/env python3
"""Measure blue-tape link orientations from a fixed RealSense D435.

The tool opens only the camera.  It does not import the LinkerHand SDK, open
CAN, or send hand commands.  Four differently sized blue markers in the
selected ROI are associated anatomically: the large palm marker is metacarpal,
the smallest link marker is distal, and the remaining proximal/middle markers
follow the fixed kinematic chain.  Their PCA long axes provide projected
MCP/PIP/DIP angles; aligned depth additionally provides a 3D axis-angle estimate.

The tape axes may have fixed installation offsets.  For joint calibration use
angle changes relative to a separately recorded physical reference pose; do
not treat one frame's raw tape-to-tape angle as an absolute URDF coordinate.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import numpy as np


MARKER_NAMES = ("metacarpal", "proximal", "middle", "distal")
ANGLE_PAIRS = {
    "mcp_projected": ("metacarpal", "proximal"),
    "pip_projected": ("proximal", "middle"),
    "dip_projected": ("middle", "distal"),
}


@dataclass(frozen=True)
class MarkerMeasurement:
    name: str
    area_px: int
    bbox_xywh: tuple[int, int, int, int]
    centroid_uv: tuple[float, float]
    axis2_uv: tuple[float, float]
    heading2_deg: float
    elongation2: float
    depth_valid_fraction: float
    depth_median_m: float | None
    axis3_xyz: tuple[float, float, float] | None
    elongation3: float | None


def _normalize_axis_heading(degrees: float) -> float:
    """Normalize an unoriented line heading to [-90, 90)."""

    return (degrees + 90.0) % 180.0 - 90.0


def _line_difference(parent_heading: float, child_heading: float) -> float:
    """Signed projected flexion; positive for the present side-view geometry."""

    return _normalize_axis_heading(parent_heading - child_heading)


def _unwrap_near_reference(value_deg: float, reference_deg: float) -> float:
    """Choose the equivalent unoriented-line angle nearest a prior point."""

    return value_deg + 180.0 * round((reference_deg - value_deg) / 180.0)


def _parse_roi(text: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1") from exc
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1 with positive area")
    return values


def _pca_axis(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 4:
        raise ValueError("at least four points are required for PCA")
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    major = eigenvectors[:, order[-1]].astype(np.float64)
    if major[0] < 0.0:
        major = -major
    minor_value = max(float(eigenvalues[order[-2]]), 1.0e-12)
    elongation = float(eigenvalues[order[-1]]) / minor_value
    return major, elongation


def _deproject_mask_points(
    pixels_uv: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_scale: float,
) -> tuple[np.ndarray, float, float | None]:
    u = pixels_uv[:, 0].astype(np.int64)
    v = pixels_uv[:, 1].astype(np.int64)
    raw = depth[v, u]
    valid = raw > 0
    valid_fraction = float(np.mean(valid))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), valid_fraction, None
    z = raw[valid].astype(np.float64) * depth_scale
    median = float(np.median(z))
    # Remove background leakage around tape edges while retaining the link face.
    near = np.abs(z - median) <= 0.012
    u_valid = u[valid][near].astype(np.float64)
    v_valid = v[valid][near].astype(np.float64)
    z = z[near]
    x = (u_valid - intrinsics["ppx"]) / intrinsics["fx"] * z
    y = (v_valid - intrinsics["ppy"]) / intrinsics["fy"] * z
    return np.column_stack((x, y, z)), valid_fraction, median


def _order_candidates_anatomically(
    candidates: Sequence[tuple[np.ndarray, int, float, float]],
) -> list[tuple[np.ndarray, int, float, float]]:
    """Associate fixed-camera tape blobs without assuming monotonic image x."""

    if len(candidates) != 4:
        raise ValueError(f"candidate count must be four, got {len(candidates)}")
    metacarpal = max(candidates, key=lambda item: item[1])
    link_markers = [item for item in candidates if item is not metacarpal]
    distal = min(link_markers, key=lambda item: item[1])
    proximal_middle = [item for item in link_markers if item is not distal]
    proximal = min(proximal_middle, key=lambda item: item[2])
    middle = next(item for item in proximal_middle if item is not proximal)

    if metacarpal[1] < 3 * max(item[1] for item in link_markers):
        raise ValueError("metacarpal area identity is ambiguous")
    if distal[1] >= 0.75 * min(proximal[1], middle[1]):
        raise ValueError("distal area identity is ambiguous")
    if proximal[2] >= middle[2]:
        raise ValueError("proximal/middle chain order is ambiguous")
    return [metacarpal, proximal, middle, distal]


def _discard_small_extra_components(
    candidates: list[tuple[np.ndarray, int, float, float]],
) -> list[tuple[np.ndarray, int, float, float]]:
    # Keep four dominant markers when glare fragments create extra blobs.
    # Fail closed if an extra is close enough in area to be a real marker.
    if len(candidates) <= 4:
        return candidates
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    kept = ranked[:4]
    rejected = ranked[4:]
    smallest_kept = min(item[1] for item in kept)
    largest_rejected = max(item[1] for item in rejected)
    if largest_rejected >= 0.75 * smallest_kept:
        detail = [
            (area, round(cx, 1), round(cy, 1))
            for _, area, cx, cy in ranked
        ]
        raise ValueError(
            "extra blue component is too large to discard safely: "
            f"{detail}"
        )
    return kept


def _order_candidates_by_chain_distance(
    candidates: list[tuple[np.ndarray, int, float, float]],
) -> list[tuple[np.ndarray, int, float, float]]:
    if len(candidates) != 4:
        raise ValueError(f"chain-distance association needs four markers, got {len(candidates)}")
    metacarpal = max(candidates, key=lambda item: item[1])
    links = [item for item in candidates if item is not metacarpal]
    if metacarpal[1] < 3 * max(item[1] for item in links):
        raise ValueError("metacarpal area identity is ambiguous")

    def distance(left, right):
        return math.hypot(left[2] - right[2], left[3] - right[3])

    palm_ranked = sorted(links, key=lambda item: distance(metacarpal, item))
    if distance(metacarpal, palm_ranked[0]) >= 0.95 * distance(metacarpal, palm_ranked[1]):
        raise ValueError("proximal chain-distance identity is ambiguous")
    proximal = palm_ranked[0]
    remaining = [item for item in links if item is not proximal]
    next_ranked = sorted(remaining, key=lambda item: distance(proximal, item))
    if distance(proximal, next_ranked[0]) >= 0.95 * distance(proximal, next_ranked[1]):
        raise ValueError("middle/distal chain-distance identity is ambiguous")
    middle, distal = next_ranked
    return [metacarpal, proximal, middle, distal]


def _order_candidates_by_chain_geometry(
    candidates: Sequence[tuple[np.ndarray, int, float, float]],
    expected_pip_deg: float | None = None,
    expected_dip_deg: float | None = None,
    expected_proximal_heading_deg: float | None = None,
) -> list[tuple[np.ndarray, int, float, float]]:
    """Select the fixed-camera tape chain despite fragmented thin markers."""

    if len(candidates) < 4:
        raise ValueError(
            f"chain-geometry association needs at least four components, got {len(candidates)}"
        )
    metacarpal = max(candidates, key=lambda item: item[1])
    links = [item for item in candidates if item is not metacarpal]
    if metacarpal[1] < 3 * max(item[1] for item in links):
        raise ValueError("metacarpal area identity is ambiguous")

    elongation = {id(item): _pca_axis(item[0])[1] for item in links}

    def distance(left, right):
        return math.hypot(left[2] - right[2], left[3] - right[3])

    minimum_primary_elongation = (
        15.0
        if expected_pip_deg is not None
        and expected_dip_deg is not None
        and expected_proximal_heading_deg is not None
        else 30.0
    )
    ranked = []
    for proximal, middle, distal in itertools.permutations(links, 3):
        if (
            elongation[id(proximal)] < minimum_primary_elongation
            or elongation[id(middle)] < minimum_primary_elongation
        ):
            continue
        if elongation[id(distal)] < 8.0:
            continue
        palm_proximal = distance(metacarpal, proximal)
        palm_middle = distance(metacarpal, middle)
        palm_distal = distance(metacarpal, distal)
        proximal_middle = distance(proximal, middle)
        middle_distal = distance(middle, distal)
        high_flexion = middle[3] >= proximal[3] + 15.0
        layout_valid = (
            distal[3] >= middle[3] + 30.0
            if high_flexion
            else palm_proximal < palm_middle < palm_distal
        )
        if not (
            150.0 <= palm_proximal <= 215.0
            and 45.0 <= proximal_middle <= 105.0
            and 35.0 <= middle_distal <= 100.0
            and layout_valid
        ):
            continue
        geometry_score = (
            ((palm_proximal - 183.0) / 22.0) ** 2
            + ((proximal_middle - 72.0) / 22.0) ** 2
            + ((middle_distal - 70.0) / 22.0) ** 2
            - 0.02
            * sum(
                math.log(min(elongation[id(item)], 150.0))
                for item in (proximal, middle, distal)
            )
        )
        headings = []
        for item in (proximal, middle, distal):
            axis, _ = _pca_axis(item[0])
            headings.append(
                _normalize_axis_heading(math.degrees(math.atan2(axis[1], axis[0])))
            )
        pip_deg = _line_difference(headings[0], headings[1])
        dip_deg = _line_difference(headings[1], headings[2])
        prior_score = 0.0
        if expected_proximal_heading_deg is not None:
            proximal_heading = _unwrap_near_reference(
                headings[0], expected_proximal_heading_deg
            )
            proximal_error = proximal_heading - expected_proximal_heading_deg
            if abs(proximal_error) > 3.0:
                continue
            prior_score += (proximal_error / 0.5) ** 2
        for measured, expected in (
            (pip_deg, expected_pip_deg),
            (dip_deg, expected_dip_deg),
        ):
            if expected is None:
                continue
            measured = _unwrap_near_reference(measured, expected)
            error = measured - expected
            if abs(error) > 12.0:
                break
            prior_score += (error / 2.0) ** 2
        else:
            ranked.append((geometry_score + prior_score, proximal, middle, distal))
    if not ranked:
        detail = [
            (area, round(cx, 1), round(cy, 1), round(elongation.get(id(item), 0.0), 1))
            for item in candidates
            for _, area, cx, cy in (item,)
        ]
        raise ValueError(f"no valid fixed-camera marker chain: {detail}")
    ranked.sort(key=lambda item: item[0])
    if (
        expected_pip_deg is None
        and expected_dip_deg is None
        and len(ranked) > 1
        and ranked[1][0] - ranked[0][0] < 0.20
    ):
        raise ValueError("fixed-camera marker chain is ambiguous")
    _, proximal, middle, distal = ranked[0]
    return [metacarpal, proximal, middle, distal]


def measure_frame(
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float],
    depth_scale: float,
    roi: tuple[int, int, int, int],
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    min_area: int,
    marker_association: str = "area",
    morph_open_kernel: int = 3,
    split_merged_markers: bool = True,
    pip_angle_prior_deg: float | None = None,
    dip_angle_prior_deg: float | None = None,
    proximal_heading_prior_deg: float | None = None,
    segmentation_depth_min_m: float | None = None,
    segmentation_depth_max_m: float | None = None,
) -> tuple[dict[str, MarkerMeasurement], np.ndarray]:
    height, width = color.shape[:2]
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"ROI {roi} is outside image {width}x{height}")

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower, np.uint8), np.array(hsv_upper, np.uint8))
    if segmentation_depth_min_m is not None and segmentation_depth_max_m is not None:
        depth_m = depth.astype(np.float64) * depth_scale
        depth_foreground = (
            (depth > 0)
            & (depth_m >= segmentation_depth_min_m)
            & (depth_m <= segmentation_depth_max_m)
        )
        mask = cv2.bitwise_and(
            mask, np.where(depth_foreground, 255, 0).astype(np.uint8)
        )
    roi_mask = np.zeros_like(mask)
    roi_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    if morph_open_kernel > 1:
        roi_mask = cv2.morphologyEx(
            roi_mask,
            cv2.MORPH_OPEN,
            np.ones((morph_open_kernel, morph_open_kernel), dtype=np.uint8),
        )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask)
    candidates: list[tuple[np.ndarray, int, float, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            pixels = np.column_stack(np.where(labels == label))[:, ::-1].astype(np.float64)
            candidates.append((pixels, area, float(centroids[label, 0]), float(centroids[label, 1])))
    # Auto exposure or high flexion can create a one-pixel bridge between
    # neighboring link markers.  With three components, the palm is the largest;
    # among the two non-palm components the larger one is the only split candidate.
    # Split it at the weakest vertical column between its lobes and fail closed if
    # either recovered marker is below the configured area threshold.
    if len(candidates) == 3 and split_merged_markers:
        metacarpal_index = max(
            range(len(candidates)), key=lambda index: candidates[index][1]
        )
        merged_index = max(
            (index for index in range(len(candidates)) if index != metacarpal_index),
            key=lambda index: candidates[index][1],
        )
        pixels, _, _, _ = candidates[merged_index]
        x_min = int(np.min(pixels[:, 0]))
        x_max = int(np.max(pixels[:, 0]))
        width_px = x_max - x_min + 1
        # Ring pitch raw32 produced a valid compact two-lobe component only 32 px wide.
        # Both recovered lobes must still independently exceed min_area.
        if width_px >= 28:
            columns = np.bincount(
                pixels[:, 0].astype(np.int64) - x_min,
                minlength=width_px,
            ).astype(np.float64)
            smooth = np.convolve(columns, np.ones(5, dtype=np.float64), mode="same")
            if width_px < 80:
                # Compact merges can put the proximal lobe on either side of
                # the column valley.  A supplied heading prior authorizes the
                # full split search; otherwise retain the fail-closed late-neck
                # behavior validated for the PIP sweep.
                search_lo = (
                    1
                    if proximal_heading_prior_deg is not None
                    else max(1, int(round(0.65 * width_px)))
                )
                search_hi = min(width_px - 1, int(round(0.90 * width_px)))
            else:
                search_lo = max(1, int(round(0.45 * width_px)))
                search_hi = min(width_px - 1, int(round(0.80 * width_px)))
            if width_px < 80:
                valid_splits = []
                for candidate_local in range(search_lo, search_hi):
                    candidate_x = x_min + candidate_local
                    candidate_left = pixels[pixels[:, 0] < candidate_x]
                    candidate_right = pixels[pixels[:, 0] >= candidate_x]
                    if (
                        len(candidate_left) >= min_area
                        and len(candidate_right) >= min_area
                    ):
                        left_axis, left_elongation = _pca_axis(candidate_left)
                        right_elongation = _pca_axis(candidate_right)[1]
                        if proximal_heading_prior_deg is None:
                            split_score = left_elongation * right_elongation
                        else:
                            left_heading = _normalize_axis_heading(
                                math.degrees(math.atan2(left_axis[1], left_axis[0]))
                            )
                            left_heading = _unwrap_near_reference(
                                left_heading, proximal_heading_prior_deg
                            )
                            split_score = -abs(
                                left_heading - proximal_heading_prior_deg
                            )
                        valid_splits.append((split_score, candidate_local))
                split_local = (
                    max(valid_splits)[1] if valid_splits else search_lo
                )
            else:
                split_local = search_lo + int(
                    np.argmin(smooth[search_lo:search_hi])
                )
            split_x = x_min + split_local
            left = pixels[pixels[:, 0] < split_x]
            right = pixels[pixels[:, 0] >= split_x]
            if len(left) >= min_area and len(right) >= min_area:
                replacement = []
                for part in (left, right):
                    center = np.mean(part, axis=0)
                    replacement.append(
                        (part, len(part), float(center[0]), float(center[1]))
                    )
                candidates[merged_index:merged_index + 1] = replacement
    if marker_association in ("chain-geometry", "chain-angle-prior"):
        candidates = _order_candidates_by_chain_geometry(
            candidates,
            pip_angle_prior_deg if marker_association == "chain-angle-prior" else None,
            dip_angle_prior_deg if marker_association == "chain-angle-prior" else None,
            proximal_heading_prior_deg if marker_association == "chain-angle-prior" else None,
        )
    else:
        candidates = _discard_small_extra_components(candidates)
    # At high flexion the distal centroid crosses the middle centroid in image
    # x, so global left-to-right sorting is invalid.  Use the intentionally
    # different marker areas plus the proximal-to-middle chain order instead.
    if len(candidates) != 4:
        detail = [(area, round(cx, 1), round(cy, 1)) for _, area, cx, cy in candidates]
        raise ValueError(f"expected exactly four blue markers in ROI, found {len(candidates)}: {detail}")
    if marker_association == "area":
        candidates = _order_candidates_anatomically(candidates)
    elif marker_association == "chain-distance":
        candidates = _order_candidates_by_chain_distance(candidates)
    elif marker_association in ("chain-geometry", "chain-angle-prior"):
        pass  # Already ordered before extra-component rejection.
    else:
        raise ValueError(f"unknown marker association mode: {marker_association}")

    result: dict[str, MarkerMeasurement] = {}
    for name, (pixels, area, _, _) in zip(MARKER_NAMES, candidates):
        axis2, elongation2 = _pca_axis(pixels)
        heading = _normalize_axis_heading(math.degrees(math.atan2(axis2[1], axis2[0])))
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
        bbox = (
            int(min_xy[0]), int(min_xy[1]),
            int(max_xy[0] - min_xy[0] + 1), int(max_xy[1] - min_xy[1] + 1),
        )
        center = np.mean(pixels, axis=0)
        result[name] = MarkerMeasurement(
            name=name,
            area_px=area,
            bbox_xywh=bbox,
            centroid_uv=tuple(float(value) for value in center),
            axis2_uv=tuple(float(value) for value in axis2),
            heading2_deg=heading,
            elongation2=elongation2,
            depth_valid_fraction=valid_fraction,
            depth_median_m=depth_median,
            axis3_xyz=axis3,
            elongation3=elongation3,
        )
    return result, roi_mask


def relative_angles(markers: dict[str, MarkerMeasurement]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for label, (parent_name, child_name) in ANGLE_PAIRS.items():
        parent = markers[parent_name]
        child = markers[child_name]
        projected = _line_difference(parent.heading2_deg, child.heading2_deg)
        output[f"{label}_deg_2d"] = projected
        if parent.axis3_xyz is None or child.axis3_xyz is None:
            output[f"{label}_deg_3d"] = None
            continue
        parent3 = np.asarray(parent.axis3_xyz)
        child3 = np.asarray(child.axis3_xyz)
        dot = float(np.clip(abs(np.dot(parent3, child3)), 0.0, 1.0))
        unsigned = math.degrees(math.acos(dot))
        output[f"{label}_deg_3d"] = math.copysign(unsigned, projected)
    return output


def _annotate(
    color: np.ndarray,
    markers: dict[str, MarkerMeasurement],
    angles: dict[str, float | None],
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    output = color.copy()
    palette = {
        "metacarpal": (0, 255, 0),
        "proximal": (255, 0, 255),
        "middle": (0, 0, 255),
        "distal": (0, 255, 255),
    }
    for marker in markers.values():
        x, y, w, h = marker.bbox_xywh
        color_value = palette[marker.name]
        cv2.rectangle(output, (x, y), (x + w, y + h), color_value, 2)
        center = np.asarray(marker.centroid_uv)
        axis = np.asarray(marker.axis2_uv)
        half = 0.65 * max(w, h)
        p0 = tuple(np.round(center - half * axis).astype(int))
        p1 = tuple(np.round(center + half * axis).astype(int))
        cv2.line(output, p0, p1, color_value, 3)
        cv2.putText(
            output,
            f"{marker.name} {marker.heading2_deg:+.2f} deg",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_value,
            2,
            cv2.LINE_AA,
        )
    text = "  ".join(
        f"{name.split('_')[0].upper()}={value:+.2f}deg"
        for name, value in angles.items()
        if name.endswith("_2d") and value is not None
    )
    cv2.putText(output, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    x0, y0, x1, y1 = roi
    cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 255), 1)
    return output


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "peak_to_peak": float(np.ptp(array)),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import pyrealsense2 as rs

    context = rs.context()
    devices = list(context.query_devices())
    matches = [
        device for device in devices
        if device.get_info(rs.camera_info.serial_number) == args.serial
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one D435 serial {args.serial}, found {len(matches)}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    color_sensor = next(
        sensor
        for sensor in profile.get_device().query_sensors()
        if "RGB" in sensor.get_info(rs.camera_info.name)
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
    rgb_controls = {
        "enable_auto_exposure": color_sensor.get_option(rs.option.enable_auto_exposure),
        "exposure": color_sensor.get_option(rs.option.exposure),
        "gain": color_sensor.get_option(rs.option.gain),
        "enable_auto_white_balance": color_sensor.get_option(
            rs.option.enable_auto_white_balance
        ),
        "white_balance": color_sensor.get_option(rs.option.white_balance),
    }
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    intrinsics = {
        "width": int(intr.width), "height": int(intr.height),
        "fx": float(intr.fx), "fy": float(intr.fy),
        "ppx": float(intr.ppx), "ppy": float(intr.ppy),
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
    started = time.time()
    try:
        for frame_index in range(args.frames):
            frames = align.process(pipeline.wait_for_frames(5000))
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                failures.append({"frame": frame_index, "error": "incomplete frameset"})
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            if args.live_preview:
                cv2.imshow("G20 tape-angle calibration live", color)
                cv2.waitKey(1)
            try:
                markers, _ = measure_frame(
                    color, depth, intrinsics, depth_scale, args.roi,
                    tuple(args.hsv_lower), tuple(args.hsv_upper), args.min_area,
                    args.marker_association,
                    args.morph_open_kernel,
                    not args.disable_merged_split,
                    args.unwrap_pip_near_deg,
                    args.unwrap_dip_near_deg,
                    args.proximal_heading_prior_deg,
                    args.segmentation_depth_min_m,
                    args.segmentation_depth_max_m,
                )
                angles = relative_angles(markers)
                for label, reference in (
                    ("pip_projected", args.unwrap_pip_near_deg),
                    ("dip_projected", args.unwrap_dip_near_deg),
                ):
                    if reference is None:
                        continue
                    for dimension in ("2d", "3d"):
                        key = f"{label}_deg_{dimension}"
                        if angles[key] is not None:
                            angles[key] = _unwrap_near_reference(
                                float(angles[key]), reference
                            )
            except ValueError as exc:
                failures.append({"frame": frame_index, "error": str(exc)})
                continue
            row: dict[str, Any] = {
                "frame": frame_index,
                "timestamp_s": time.time(),
                **angles,
            }
            for name, marker in markers.items():
                row[f"{name}_heading_deg_2d"] = marker.heading2_deg
                row[f"{name}_area_px"] = marker.area_px
                row[f"{name}_elongation_2d"] = marker.elongation2
                row[f"{name}_depth_valid_fraction"] = marker.depth_valid_fraction
                row[f"{name}_depth_median_m"] = marker.depth_median_m
                row[f"{name}_elongation_3d"] = marker.elongation3
            rows.append(row)
            last_color, last_depth = color, depth
            last_markers, last_angles = markers, angles
    finally:
        if args.live_preview:
            cv2.destroyWindow("G20 tape-angle calibration live")
            cv2.waitKey(1)
        pipeline.stop()

    if not rows or last_color is None or last_depth is None or last_markers is None or last_angles is None:
        raise RuntimeError(f"no usable tape measurements; failures={failures[:5]}")
    failure_fraction = len(failures) / args.frames
    if failure_fraction > args.max_failure_fraction:
        raise RuntimeError(
            f"marker detection failure fraction {failure_fraction:.1%} exceeds "
            f"limit {args.max_failure_fraction:.1%}"
        )

    prefix = args.out_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{prefix}_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(Path(f"{prefix}_color.png")), last_color)
    np.save(str(Path(f"{prefix}_depth_raw.npy")), last_depth)
    annotated = _annotate(last_color, last_markers, last_angles, args.roi)
    cv2.imwrite(str(Path(f"{prefix}_annotated.png")), annotated)

    angle_keys = [key for key in rows[0] if key.endswith("_deg_2d") or key.endswith("_deg_3d")]
    summary = {
        "schema_version": 1,
        "camera_only": True,
        "linkerhand_sdk_imported": False,
        "hand_motion_sent": False,
        "camera": {
            "name": matches[0].get_info(rs.camera_info.name),
            "serial": args.serial,
            "firmware": matches[0].get_info(rs.camera_info.firmware_version),
            "usb": matches[0].get_info(rs.camera_info.usb_type_descriptor),
            "profile": {"width": args.width, "height": args.height, "fps": args.fps},
            "intrinsics": intrinsics,
            "depth_scale_m_per_unit": depth_scale,
            "rgb_controls": rgb_controls,
        },
        "detector": {
            "roi_xyxy": list(args.roi),
            "hsv_lower": list(args.hsv_lower),
            "hsv_upper": list(args.hsv_upper),
            "min_area_px": args.min_area,
            "marker_order": list(MARKER_NAMES),
            "marker_association": args.marker_association,
            "morph_open_kernel": args.morph_open_kernel,
            "live_preview_during_capture": args.live_preview,
            "split_merged_neighbor_markers_by_column_valley": not args.disable_merged_split,
            "unwrap_pip_near_deg": args.unwrap_pip_near_deg,
            "unwrap_dip_near_deg": args.unwrap_dip_near_deg,
            "proximal_heading_prior_deg": args.proximal_heading_prior_deg,
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
            key: _summary([float(row[key]) for row in rows if row[key] is not None])
            for key in angle_keys
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
                "axis3_xyz": None if marker.axis3_xyz is None else list(marker.axis3_xyz),
                "elongation3": marker.elongation3,
            }
            for name, marker in last_markers.items()
        },
        "failures": failures,
        "interpretation": (
            "Tape-to-link installation offsets are not removed. Use differences "
            "from a recorded physical reference pose for calibrated joint angles."
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
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--rgb-exposure", type=float, default=None)
    parser.add_argument("--rgb-gain", type=float, default=None)
    parser.add_argument("--rgb-white-balance", type=float, default=None)
    parser.add_argument("--roi", type=_parse_roi, default=(430, 185, 1045, 370))
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=(85, 70, 25))
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=(145, 255, 255))
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument(
        "--marker-association",
        choices=("area", "chain-distance", "chain-geometry", "chain-angle-prior"),
        default="area",
    )
    parser.add_argument(
        "--morph-open-kernel",
        type=int,
        choices=(0, 3),
        default=3,
    )
    parser.add_argument("--max-failure-fraction", type=float, default=0.1)
    parser.add_argument(
        "--disable-merged-split", action="store_true",
        help="require four real marker components; never synthesize a fourth by splitting",
    )
    parser.add_argument(
        "--live-preview", action="store_true",
        help="show the same color frames being measured; does not open a second camera",
    )
    parser.add_argument("--unwrap-pip-near-deg", type=float, default=None)
    parser.add_argument("--unwrap-dip-near-deg", type=float, default=None)
    parser.add_argument("--proximal-heading-prior-deg", type=float, default=None)
    parser.add_argument("--segmentation-depth-min-m", type=float, default=None)
    parser.add_argument("--segmentation-depth-max-m", type=float, default=None)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.frames <= 0 or args.warmup_frames < 0:
        parser.error("--frames must be positive and --warmup-frames nonnegative")
    if not 0.0 <= args.max_failure_fraction < 1.0:
        parser.error("--max-failure-fraction must be in [0,1)")
    for value, label in (
        (args.rgb_exposure, "exposure"),
        (args.rgb_gain, "gain"),
        (args.rgb_white_balance, "white balance"),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            parser.error(f"--rgb-{label.replace(chr(32), chr(45))} must be finite and nonnegative")
    if (args.segmentation_depth_min_m is None) != (
        args.segmentation_depth_max_m is None
    ):
        parser.error("both segmentation depth bounds must be provided together")
    if args.segmentation_depth_min_m is not None and not (
        0.0 < args.segmentation_depth_min_m < args.segmentation_depth_max_m
    ):
        parser.error("segmentation depth bounds must satisfy 0 < min < max")
    if args.marker_association == "chain-angle-prior" and (
        args.unwrap_pip_near_deg is None
        or args.unwrap_dip_near_deg is None
        or args.proximal_heading_prior_deg is None
    ):
        parser.error(
            "--marker-association chain-angle-prior requires PIP, DIP, and proximal references"
        )
    for value, label in (
        (args.unwrap_pip_near_deg, "PIP"),
        (args.unwrap_dip_near_deg, "DIP"),
        (args.proximal_heading_prior_deg, "proximal heading"),
    ):
        if value is not None and not math.isfinite(value):
            parser.error(f"--unwrap-{label.lower()}-near-deg must be finite")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = capture(parse_args(argv))
    print(json.dumps({
        "camera_only": summary["camera_only"],
        "hand_motion_sent": summary["hand_motion_sent"],
        "capture": summary["capture"],
        "angle_summary_deg": summary["angle_summary_deg"],
        "outputs": summary["outputs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
