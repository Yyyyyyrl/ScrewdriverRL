"""Compatibility re-export for the pure-Python diameter posture table.

Canonical data lives under ``screwdriver_rl.utils`` so CPU/XML tools can import
it without loading the Isaac task package.
"""

from __future__ import annotations

from screwdriver_rl.utils.linker_topdown_diameter_postures import (
    TOPDOWN_HANDLE_DIAMETERS_M,
    TOPDOWN_HANDLE_LENGTH_M,
    TOPDOWN_HANDLE_RADII_M,
    TOPDOWN_NOMINAL_BUCKET_INDEX,
    TOPDOWN_NOMINAL_DIAMETER_M,
    TOPDOWN_NOMINAL_RADIUS_M,
    TOPDOWN_PREGRASP_POSITIONS_BUCKETS,
    TOPDOWN_RESET_POSITIONS_BUCKETS,
    TOPDOWN_ROOT_POS_OFFSETS_BUCKETS,
    TOPDOWN_ROOT_POS_W,
    TOPDOWN_ROOT_QUAT_WXYZ,
    TOPDOWN_ROOT_QUATS_WXYZ_BUCKETS,
    bucket_for_diameter_mm,
)


__all__ = [name for name in globals() if name.startswith("TOPDOWN_")] + [
    "bucket_for_diameter_mm"
]
