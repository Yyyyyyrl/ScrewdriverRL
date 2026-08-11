from __future__ import annotations

import json

import pytest

from screwdriver_rl.deploy import linker_sdk_map as sdkmap
from tools import plan_g20_physical_joint_calibration as planner


def _inputs():
    config, _ = planner._load_bundle(
        planner.REPO_ROOT
        / "deliverables/linker_g20_topdown_d64_action008_stage2_commissioning_20260727/runtime/deploy.pth"
    )
    overlay = sdkmap.load_calibration_file(
        str(planner.REPO_ROOT / "linker_calib_thumbfit.json")
    )
    table = tuple(sdkmap.build_joint_table(overlay))
    lower = tuple(float(value) for value in config["finger_lower"])
    upper = tuple(float(value) for value in config["finger_upper"])
    baseline = tuple(float(value) for value in config["startup_reset_targets"])
    return lower, upper, baseline, table


def test_main_measurement_plan_has_independent_holdouts_and_pip_gate():
    lower, upper, baseline, table = _inputs()
    rows = planner._build_rows(lower, upper, baseline, table, sdkmap)

    assert len(rows) == 203
    measured = [row for row in rows if row["requires_physical_measurement"] == "YES"]
    assert len(measured) == 85
    assert sum(
        row["joint"] == "index_pip"
        and row["phase"] == "repeatability"
        and row["requires_physical_measurement"] == "YES"
        for row in rows
    ) == 5

    for row in rows:
        raw20 = json.loads(row["raw20_json"])
        assert len(raw20) == 20
        assert all(isinstance(value, int) and 0 <= value <= 255 for value in raw20)
        assert all(raw20[slot] == 0 for slot in planner.RESERVED_SLOTS)
        if row["joint"].endswith("_pip"):
            assert row["execution_gate"] == "REGENERATE_AFTER_PHYSICAL_PIP_RANGE"

    for joint in planner.JOINT_ORDER:
        ordinary = [
            row for row in measured
            if row["joint"] == joint and row["phase"] in ("fit", "holdout")
        ]
        assert [float(row["q_fraction"]) for row in ordinary if row["phase"] == "fit"] == list(planner.FIT_POINTS)
        assert [float(row["q_fraction"]) for row in ordinary if row["phase"] == "holdout"] == list(planner.HOLDOUT_POINTS)


def test_pip_range_discovery_uses_direct_raw_without_choosing_rad_limit():
    lower, upper, baseline, table = _inputs()
    rows = planner._build_pip_range_rows(lower, upper, baseline, table, sdkmap)

    assert len(rows) == 4 * (len(planner.PIP_RANGE_RAW_DESCENDING) + 12)
    assert {row["joint"] for row in rows} == {
        "index_pip", "middle_pip", "ring_pip", "pinky_pip"
    }
    assert all("planned_sim_q_rad" not in row for row in rows)
    assert all(row["candidate_only"] == "YES" for row in rows)
    assert all(row["requires_operator_release"] == "YES" for row in rows)

    for joint in ("index_pip", "middle_pip", "ring_pip", "pinky_pip"):
        joint_rows = [row for row in rows if row["joint"] == joint]
        descending = [
            int(row["direct_command_raw"])
            for row in joint_rows
            if row["phase"] == "coarse_to_fine_upper_range_discovery"
        ]
        assert descending == list(planner.PIP_RANGE_RAW_DESCENDING)
        target_slot = int(joint_rows[0]["sdk_slot"])
        reference = json.loads(joint_rows[0]["raw20_json"])
        for row in joint_rows:
            raw20 = json.loads(row["raw20_json"])
            assert all(raw20[slot] == 0 for slot in planner.RESERVED_SLOTS)
            assert raw20[target_slot] == int(row["direct_command_raw"])
            assert all(
                raw20[slot] == reference[slot]
                for slot in range(20)
                if slot != target_slot
            )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "2026-08-04: the runtime URDF was reverted to the vendor (OG) 1.57 for "
        "training, so it no longer carries the per-finger measured endpoints "
        "asserted below; the measurement is archived at "
        "assets/linker_hand_l20_calibfit/. What is unconfirmed is not the numbers "
        "but their datum — whether the projected 2D angle equals URDF local joint "
        "q. Practically this changes nothing for the current top-down task: the "
        "+-0.35 rad home box binds before the PIP limit on all four fingers (max "
        "commanded 1.129 rad), so 1.08 / 1.57 / 1.75 are indistinguishable there. "
        "Flip this marker off once the datum is confirmed and the asset adopted."
    ),
)
def test_pip_software_limits_agree_on_the_measured_endpoints():
    """The 1.08-vs-1.57 conflict this test used to record is resolved.

    Both numbers were guesses: 1.08 was the SDK tip arc and 1.57 the untouched
    vendor URDF.  Phase A measured each finger separately, so URDF and semantic
    schema must now carry the same per-finger physical endpoint, and the four
    fingers must *not* share one value.
    """

    measured = {
        "index_pip": 1.5387503110826093,
        "middle_pip": 1.7493646502326758,
        "ring_pip": 1.696652275710845,
        "pinky_pip": 1.7332971062074665,
    }
    urdf, _ = planner._load_urdf(
        planner.REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
    )
    schema = planner._schema_limits(
        planner.REPO_ROOT / "assets/calibrations/linker_g20_left_semantic_schema_v1.json"
    )
    for name, upper in measured.items():
        assert abs(float(urdf[name]["upper"]) - upper) < 1.0e-12
        assert abs(schema[name][1] - upper) < 1.0e-12
        assert float(urdf[name]["lower"]) == 0.0
    assert len(set(measured.values())) == 4
