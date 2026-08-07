"""The fingertip contact check must price penetration against EVERY handle part.

Regression for a validator bug that passed a posture with 4.86 mm of
interpenetration.  The check scores each fingertip against whichever handle part
it is nearest, which is the right way to decide what the finger *bears on*.  It
also used to apply the penetration limit only to that part.  Once a fingertip
intersects both the body and the cap their surface distances are both exactly
zero, so "nearest" stops discriminating: the tie selected the cap, where the
ring finger was 0.50 mm deep, and the 4.86 mm it was simultaneously buried in
the body went unmeasured.  The posture was reported as passing all gates.

The fixture is that exact posture, kept as a file rather than reconstructed so
the test cannot drift away from the case it is guarding.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/topdown/posture_deep_body_penetration_shallow_cap.json"


@pytest.fixture(scope="module")
def validator():
    spec = importlib.util.spec_from_file_location(
        "topdown_validator",
        ROOT / "tools/validate_linker_l20_screwdriver_topdown.py",
    )
    module = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["validator"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


@pytest.fixture(scope="module")
def contact_rows(validator):
    posture = json.loads(FIXTURE.read_text())
    hand = validator.UrdfGeometry(validator.HAND_URDF).collision_meshes_world(
        posture["joint_positions_independent"],
        posture["root_pos_w"],
        posture["root_quat_wxyz"],
    )
    screwdriver = validator.UrdfGeometry(
        validator.TOPDOWN_SCREWDRIVER_URDF
    ).collision_meshes_world(
        {}, validator.SCREWDRIVER_ROOT_POS_W, (1.0, 0.0, 0.0, 0.0)
    )
    return validator._contact_checks(hand, screwdriver)


def test_deep_penetration_in_a_non_nearest_part_is_rejected(contact_rows, validator):
    ring = contact_rows["fingers"]["ring"]
    # The finger genuinely bears on the cap: that part of the report is correct.
    assert ring["target"] == "screwdriver_cap"
    assert ring["penetration_by_target_m"]["screwdriver_cap"] < 0.001
    # And it is simultaneously buried in the body, which is what must be caught.
    assert ring["penetration_by_target_m"]["screwdriver_body"] > 0.004
    assert ring["deepest_penetration_m"] == pytest.approx(
        ring["penetration_by_target_m"]["screwdriver_body"], rel=1e-6
    )
    assert not ring["pass"]
    assert not contact_rows["pass"]


def test_the_other_fingers_are_not_collateral_damage(contact_rows):
    # The fix must reject the ring only; scoring every part must not start
    # failing fingers whose deepest penetration is legitimately small.
    for finger in ("index", "middle", "pinky", "thumb"):
        row = contact_rows["fingers"][finger]
        assert row["deepest_penetration_m"] <= 0.001, finger
        assert row["pass"], finger
