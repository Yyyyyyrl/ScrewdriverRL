from __future__ import annotations

import cv2
import numpy as np

from tools.measure_g20_index_mcp_pitch import (
    mcp_pitch_angles,
    measure_mcp_pitch_frame,
)


def _draw_marker(image, center, size, angle_deg):
    box = cv2.boxPoints((center, size, angle_deg)).astype(np.int32)
    cv2.fillConvexPoly(image, box, (255, 0, 0))


def test_mcp_detector_merges_fragmented_proximal_and_rejects_other_tapes():
    color = np.full((720, 1280, 3), 180, dtype=np.uint8)
    _draw_marker(color, (685, 300), (225, 40), -2.0)
    # One physical proximal tape interrupted by a dark crease.
    _draw_marker(color, (860, 284), (23, 7), -7.0)
    _draw_marker(color, (890, 280), (25, 7), -7.0)
    # PIP/DIP markers and an identically colored marker on another finger.
    _draw_marker(color, (965, 270), (60, 10), -12.0)
    _draw_marker(color, (1030, 258), (35, 8), -16.0)
    _draw_marker(color, (775, 230), (34, 8), 5.0)

    depth = np.full((720, 1280), 352, dtype=np.uint16)
    depth[275:325, 565:805] = 341
    intrinsics = {
        "fx": 911.4485,
        "fy": 910.0629,
        "ppx": 648.0690,
        "ppy": 370.1065,
    }
    markers, _, diagnostics = measure_mcp_pitch_frame(
        color,
        depth,
        intrinsics,
        0.001,
        (500, 185, 1100, 500),
    )

    assert list(markers) == ["metacarpal", "proximal"]
    assert diagnostics["selected_proximal_component_count"] == 2
    assert 250 <= markers["proximal"].area_px <= 450
    assert markers["proximal"].elongation2 > 20.0
    assert markers["metacarpal"].depth_median_m == 0.341
    assert markers["proximal"].depth_median_m == 0.352
    angles = mcp_pitch_angles(markers)
    assert 3.0 <= angles["mcp_projected_deg_2d"] <= 7.0
    assert angles["mcp_projected_deg_3d"] is not None


def test_mcp_detector_fails_when_palm_is_removed_by_bad_depth_gate():
    color = np.full((720, 1280, 3), 180, dtype=np.uint8)
    _draw_marker(color, (685, 300), (225, 40), -2.0)
    _draw_marker(color, (875, 282), (55, 8), -7.0)
    depth = np.full((720, 1280), 352, dtype=np.uint16)
    depth[275:325, 565:805] = 341
    intrinsics = {
        "fx": 911.4485,
        "fy": 910.0629,
        "ppx": 648.0690,
        "ppy": 370.1065,
    }

    try:
        measure_mcp_pitch_frame(
            color,
            depth,
            intrinsics,
            0.001,
            (500, 185, 1100, 500),
            segmentation_depth_min_m=0.350,
            segmentation_depth_max_m=0.370,
        )
    except ValueError as exc:
        assert "palm" in str(exc) or "found" in str(exc)
    else:
        raise AssertionError("bad palm-excluding depth gate must fail closed")
