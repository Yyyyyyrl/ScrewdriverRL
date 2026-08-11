from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run_g20_hardcoded_screwdriver_turn.py"
SPEC = importlib.util.spec_from_file_location("g20_hardcoded_turn", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_full_scale_waypoints_are_16d_and_bounded() -> None:
    rows = MODULE.waypoints(1.0)
    assert len(rows) == 4
    assert all(len(row) == 16 for row in rows)
    # Keep the deployed sequence inside the explicit CEM search envelope.
    assert max(
        abs(value - home)
        for row in rows
        for value, home in zip(row, MODULE.HOME)
    ) <= 0.140001


def test_waypoints_clip_to_exact_urdf_limits() -> None:
    anchor = [upper for _lower, upper in MODULE.ISAAC_TARGET_LIMITS]
    rows = MODULE.waypoints(1.0, anchor=anchor)
    for row in rows:
        for value, (lower, upper) in zip(row, MODULE.ISAAC_TARGET_LIMITS):
            assert lower <= value <= upper


def test_amplitude_scales_exactly_about_home() -> None:
    full = MODULE.waypoints(1.0)
    low = MODULE.waypoints(0.35)
    for full_row, low_row in zip(full, low):
        for home, full_value, low_value in zip(MODULE.HOME, full_row, low_row):
            assert abs((low_value - home) - 0.35 * (full_value - home)) < 1.0e-12


def test_turn_plan_is_closed_and_ends_at_home() -> None:
    rows = MODULE.build_phases(
        start=list(MODULE.HOME),
        mode="turn",
        amplitude=0.35,
        cycles=1,
        rate_hz=20.0,
        preposition_s=6.0,
        leg_s=1.2,
        hold_s=0.15,
        return_to_anchor=True,
    )
    assert rows
    assert rows[-1]["phase"] == "hold_anchor_final"
    assert rows[-1]["semantic"] == list(MODULE.HOME)
    assert any(row["phase"] == "cycle1_close_to_wp1" for row in rows)


def test_turn_plan_is_relative_to_measured_anchor() -> None:
    anchor = [value + 0.01 for value in MODULE.HOME]
    rows = MODULE.build_phases(
        start=anchor,
        mode="turn",
        amplitude=0.35,
        cycles=1,
        rate_hz=20.0,
        preposition_s=6.0,
        leg_s=1.2,
        hold_s=0.15,
        return_to_anchor=True,
    )
    assert rows[-1]["semantic"] == anchor


def test_raw_preposition_handles_lut_outside_start_without_jump() -> None:
    start = [255, 67, 62, 47, 32, 64, 14, 16, 159, 254, 132, 0, 0, 0, 0, 95, 186, 241, 214, 198]
    target = [191, 154, 112, 128, 124, 165, 193, 160, 140, 145, 63, 0, 0, 0, 0, 213, 138, 76, 176, 120]
    rows = MODULE._raw_linear_frames(
        start, target, duration_s=6.0, rate_hz=10.0, max_raw_step=1
    )
    assert rows[-1] == target
    previous = start
    for row in rows:
        assert max(abs(a - b) for a, b in zip(row, previous)) <= 1
        previous = row


def test_preposition_plan_only_moves_to_home() -> None:
    start = [value - 0.01 for value in MODULE.HOME]
    rows = MODULE.build_phases(
        start=start,
        mode="preposition",
        amplitude=0.35,
        cycles=1,
        rate_hz=20.0,
        preposition_s=6.0,
        leg_s=1.2,
        hold_s=0.15,
    )
    assert rows[-1]["semantic"] == list(MODULE.HOME)
    assert all("cycle" not in row["phase"] for row in rows)
