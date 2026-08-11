from __future__ import annotations

import cv2
import numpy as np

from tools.measure_g20_tape_angles import measure_frame, relative_angles


def _draw_marker(image, center, size, angle_deg):
    box = cv2.boxPoints((center, size, angle_deg)).astype(np.int32)
    cv2.fillConvexPoly(image, box, (255, 0, 0))
    return box


def test_wide_blue_tape_axes_and_merged_distal_split():
    color = np.zeros((720, 1280, 3), dtype=np.uint8)
    color[:] = (180, 180, 180)
    _draw_marker(color, (600, 310), (245, 50), -4.0)
    _draw_marker(color, (815, 282), (58, 16), -8.0)
    _draw_marker(color, (905, 265), (76, 28), -13.0)
    _draw_marker(color, (972, 246), (40, 22), -17.0)
    # Simulate a reflected blue bridge that makes the final two markers one
    # connected component after the 3x3 opening.
    cv2.line(color, (941, 255), (955, 250), (255, 0, 0), 5)

    depth = np.full((720, 1280), 318, dtype=np.uint16)
    intrinsics = {
        "fx": 911.4485,
        "fy": 910.0629,
        "ppx": 648.0690,
        "ppy": 370.1065,
    }
    markers, _ = measure_frame(
        color,
        depth,
        intrinsics,
        0.001,
        (430, 185, 1045, 370),
        (85, 70, 25),
        (145, 255, 255),
        80,
        marker_association="chain-distance",
    )
    assert list(markers) == ["metacarpal", "proximal", "middle", "distal"]
    assert all(marker.depth_valid_fraction == 1.0 for marker in markers.values())
    assert markers["metacarpal"].elongation2 > 20.0
    assert markers["proximal"].elongation2 > 8.0

    angles = relative_angles(markers)
    assert 3.0 <= angles["mcp_projected_deg_2d"] <= 5.0
    assert 3.5 <= angles["pip_projected_deg_2d"] <= 6.5
    assert 2.0 <= angles["dip_projected_deg_2d"] <= 6.0
    assert angles["pip_projected_deg_3d"] is not None

def test_chain_angle_prior_splits_three_components_before_association():
    color = np.full((720, 1280, 3), 180, dtype=np.uint8)
    _draw_marker(color, (600, 310), (245, 50), -4.0)
    _draw_marker(color, (812, 282), (58, 8), -8.0)
    _draw_marker(color, (895, 265), (60, 10), -13.0)
    _draw_marker(color, (965, 246), (40, 8), -17.0)
    # Join middle/distal into one component while preserving a low-density
    # vertical valley that the production splitter can recover.
    cv2.line(color, (923, 258), (946, 251), (255, 0, 0), 7)

    depth = np.full((720, 1280), 318, dtype=np.uint16)
    intrinsics = {
        "fx": 911.4485,
        "fy": 910.0629,
        "ppx": 648.0690,
        "ppy": 370.1065,
    }
    markers, _ = measure_frame(
        color,
        depth,
        intrinsics,
        0.001,
        (430, 185, 1045, 370),
        (85, 70, 25),
        (145, 255, 255),
        20,
        marker_association="chain-angle-prior",
        pip_angle_prior_deg=5.0,
        dip_angle_prior_deg=4.0,
        proximal_heading_prior_deg=-8.0,
    )
    assert list(markers) == ["metacarpal", "proximal", "middle", "distal"]
    angles = relative_angles(markers)
    assert 3.5 <= angles["pip_projected_deg_2d"] <= 6.5
    assert 2.0 <= angles["dip_projected_deg_2d"] <= 6.0

