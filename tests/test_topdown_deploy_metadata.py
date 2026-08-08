"""Pure-Python tests for diameter-specific top-down deploy metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from screwdriver_rl.utils.linker_topdown_diameter_postures import (
    TOPDOWN_PREGRASP_POSITIONS_BUCKETS,
    TOPDOWN_RESET_POSITIONS_BUCKETS,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/repackage_linker_topdown_deploy.py"
SPEC = importlib.util.spec_from_file_location("topdown_deploy_repackage", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _urdf_upper_limits() -> dict[str, float]:
    import xml.etree.ElementTree as ET

    root = ET.parse(TOOL.HAND_URDF).getroot()
    return {
        str(joint.get("name")): float(joint.find("limit").get("upper"))
        for joint in root.findall("joint")
        if joint.find("limit") is not None
    }

def test_train_bundle_builder_wires_nominal_geometry_row():
    source = (ROOT / "train.py").read_text()
    assert "deployment_variant = nominal_geometry_row_index" in source
    assert "t[deployment_row].detach().cpu().tolist()" in source
    assert '"startup_reset_targets": startup_reset_targets' in source
    assert '"deployment_geometry_bucket": deployment_bucket' in source
    assert '"deployment_geometry_scale": deployment_scale' in source
    assert '"observation_semantics_version": str(' in source



def _source_config() -> dict:
    return {
        "task": TOOL.TASK_ID,
        "n_finger": 16,
        "prop_hist_len": 30,
        "home_targets": [0.0] * 16,
        "finger_lower": [-1.0] * 16,
        "finger_upper": [1.0] * 16,
        "proprio_codec": {},
        "observation_semantics_version": "linker-l20-force-free-reset-dr-v1",
    }


def test_64mm_vectors_match_nominal_bucket_and_simulator_limit_formula():
    bucket, home, lower, upper = TOOL.deployment_vectors(64)
    expected = [
        float(value)
        for finger in TOOL.FINGERS
        for value in TOPDOWN_PREGRASP_POSITIONS_BUCKETS[1][finger]
    ]

    assert bucket == 1
    assert home == pytest.approx(expected, abs=1e-12)
    assert all(lo <= target <= hi for lo, target, hi in zip(lower, home, upper))
    # Before the 21/21 calibration all four PIP uppers were pinned at the
    # guessed URDF limit minus margin (1.08 - 0.02 = 1.06).  The measured
    # endpoints are 1.539..1.749, so the limit no longer binds and the envelope
    # is set by the intended motion range around home instead.  Assert that
    # regime explicitly, and that the measured limit is still a live backstop.
    urdf_upper = _urdf_upper_limits()
    for index, name in ((2, "index_pip"), (8, "ring_pip"), (11, "pinky_pip")):
        assert upper[index] == pytest.approx(
            expected[index] + TOOL.JOINT_MOTION_RANGE_RAD, abs=1e-12
        ), name
        assert upper[index] < urdf_upper[name] - TOOL.JOINT_TARGET_MARGIN_RAD

    # middle_pip is the exception, and it is forced rather than chosen.  Its home
    # sits at 1.372 rad, so the intended +0.35 motion would reach 1.722 against a
    # 1.55 backstop: the envelope is clipped by 0.17 rad and the finger keeps
    # only half its upward action range on hardware.
    #
    # This cannot be dialled out by flexing the middle less.  Sweeping its PIP
    # down from the fitted value drives the fingertip into the handle almost
    # at once -- 5.6 mm of penetration at -0.10 rad, 10.2 mm at -0.20 -- so the
    # 0.17 rad needed to clear the backstop would cost roughly 8 mm of
    # interpenetration.  The clip is a geometric consequence of where this hand
    # has to put its middle finger to reach a 64 mm handle at all, so it is
    # asserted as a bounded, known cost rather than silently dropped.
    middle_home = expected[5]
    middle_backstop = urdf_upper["middle_pip"] - TOOL.JOINT_TARGET_MARGIN_RAD
    assert middle_home + TOOL.JOINT_MOTION_RANGE_RAD > middle_backstop
    assert upper[5] == pytest.approx(middle_backstop, abs=1e-12)
    assert middle_home + TOOL.JOINT_MOTION_RANGE_RAD - middle_backstop < 0.20

    expected_reset = [
        float(value)
        for finger in TOOL.FINGERS
        for value in TOPDOWN_RESET_POSITIONS_BUCKETS[1][finger]
    ]
    assert TOOL.startup_reset_vector(64) == pytest.approx(
        expected_reset, abs=1e-12
    )


def test_corrected_config_is_copied_and_marks_scale_one_geometry():
    source = _source_config()
    corrected = TOOL.corrected_config(source, 64)

    assert source["home_targets"] == [0.0] * 16
    assert corrected["deployment_env_index"] == 1
    assert corrected["deployment_geometry_bucket"] == 1
    assert corrected["deployment_geometry_scale"] == pytest.approx([1.0, 1.0])
    assert corrected["deployment_handle_diameter_mm"] == 64.0
    assert corrected["startup_reset_targets"] == pytest.approx(
        TOOL.startup_reset_vector(64), abs=1e-12
    )
    assert corrected["observation_semantics_version"] == (
        "linker-l20-force-free-reset-dr-v1"
    )
    assert corrected["proprio_codec"]["codec_id"] == "linker-g20-mounted-proprio-v1"


def test_corrected_config_rejects_another_task_family():
    source = _source_config()
    source["task"] = "wrong"

    with pytest.raises(ValueError, match="expected task"):
        TOOL.corrected_config(source, 64)

