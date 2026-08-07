"""Architecture V2.1 M0 identity, calibration, and mapping gates."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from screwdriver_rl.deploy import linker_sdk_map as legacy_map
from screwdriver_rl.deploy.linker_calibration import (
    LinkerMapper,
    canonical_json_digest,
    load_linker_calibration,
    load_semantic_schema,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SCHEMA = ASSETS / "calibrations" / "linker_g20_left_semantic_schema_v1.json"
CALIBRATION = ASSETS / "calibrations" / "linker_g20_left_lht20_010_415_v1.json"
MODEL_DIR = ASSETS / "linker_hand_l20"
MODEL_MANIFEST = MODEL_DIR / "model_manifest_v1.json"
GOLDEN = ROOT / "tests" / "fixtures" / "architecture" / "linker_mapping_golden_v1.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_schema_and_calibration_are_content_addressed_and_frozen():
    schema_raw = _json(SCHEMA)
    calibration_raw = _json(CALIBRATION)
    assert canonical_json_digest(schema_raw, "digest") == schema_raw["digest"]
    assert (
        canonical_json_digest(calibration_raw, "artifact_digest")
        == calibration_raw["artifact_digest"]
    )
    schema = load_semantic_schema(SCHEMA)
    calibration = load_linker_calibration(CALIBRATION, SCHEMA)
    assert schema.digest == calibration.semantic_schema_digest
    assert calibration.hand_joint == "G20"
    assert calibration.hand_side == "left"
    assert calibration.serial_number == "LHT20-010-415-L-B-1-D"
    assert calibration.calibration_author == "manual same-view registration; user-reviewed"
    with pytest.raises(FrozenInstanceError):
        calibration.hand_side = "right"  # type: ignore[misc]


def test_golden_mapping_and_thumb_slot_regression():
    mapper = LinkerMapper.load(CALIBRATION, SCHEMA)
    fixture = _json(GOLDEN)
    assert fixture["semantic_schema_digest"] == mapper.calibration.semantic_schema_digest
    assert fixture["calibration_digest"] == mapper.calibration.artifact_digest
    for case in fixture["cases"]:
        prepared = mapper.prepare(case["semantic_radians"])
        assert list(prepared.native_range) == case["native_range"], case["name"]
        assert max(abs(error) for error in prepared.round_trip_error) < 0.01
    by_name = {joint.name: joint for joint in mapper.calibration.joints}
    assert by_name["thumb_cmc_yaw"].slot == 10
    assert by_name["thumb_cmc_roll"].slot == 5


def test_immutable_mapper_matches_existing_verified_mapping():
    mapper = LinkerMapper.load(CALIBRATION, SCHEMA)
    legacy_map.apply_calibration(str(ROOT / "linker_calib_deploy.json"))
    try:
        for semantic in (
            legacy_map.PREGRASP_16,
            [joint.lo for joint in legacy_map.DEFAULT_JOINTS],
            [joint.hi for joint in legacy_map.DEFAULT_JOINTS],
        ):
            immutable = list(mapper.prepare(semantic).native_range)
            legacy = legacy_map.joints16_to_sdk_range(semantic)
            assert max(abs(a - b) for a, b in zip(immutable, legacy)) <= 1
    finally:
        legacy_map.reset_calibration()


def test_canonical_model_manifest_and_urdf_identity():
    manifest = _json(MODEL_MANIFEST)
    assert canonical_json_digest(manifest, "model_digest") == manifest["model_digest"]
    urdf_path = MODEL_DIR / manifest["urdf"]["path"]
    assert _sha256(urdf_path) == manifest["urdf"]["sha256"]

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_names = {element.attrib["name"] for element in root.findall("link")}
    joint_elements = {element.attrib["name"]: element for element in root.findall("joint")}
    assert set(manifest["fingertip_links"]).issubset(link_names)
    assert set(manifest["actuated_joints"]).issubset(joint_elements)
    for relation in manifest["mimic_joints"]:
        mimic = joint_elements[relation["joint"]].find("mimic")
        assert mimic is not None
        assert mimic.attrib["joint"] == relation["source"]
        assert float(mimic.attrib["multiplier"]) == pytest.approx(relation["multiplier"])
        assert float(mimic.attrib["offset"]) == pytest.approx(relation["offset"])

    for filename, digest in manifest["meshes"].items():
        assert _sha256(MODEL_DIR / "meshes" / filename) == digest


def test_schema_covers_exactly_the_manifest_semantic_order():
    schema = load_semantic_schema(SCHEMA)
    manifest = _json(MODEL_MANIFEST)
    assert list(schema.joint_names) == manifest["actuated_joints"]
    assert schema.digest == manifest["semantic_schema_digest"]
