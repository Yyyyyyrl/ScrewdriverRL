#!/usr/bin/env python3
"""Promote the reviewed G20 OG-local-q LUT into training/deploy assets.

The deployment overlay retains the complete reviewed physical LUT, while its
commandable semantic limits and the training URDF use the intersection of the
reviewed range with the unchanged OG geometric limits.  Mechanical follower
mimic relationships are copied from the OG runtime asset and are never fitted
or changed by this promotion.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "assets/calibrations/linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805.json"
PROVENANCE = CANDIDATE.with_name(CANDIDATE.stem + "_provenance.json")
OG_URDF = ROOT / "assets/linker_hand_l20_OG/linkerhand_l20_left.urdf"
TRAINING_URDF = ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
DEPLOY_OVERLAY = ROOT / "linker_calib_deploy.json"
SCHEMA = ROOT / "assets/calibrations/linker_g20_left_semantic_schema_v1.json"
CALIBRATION = ROOT / "assets/calibrations/linker_g20_left_lht20_010_415_v1.json"
MODEL_MANIFEST = ROOT / "assets/linker_hand_l20/model_manifest_v1.json"
PROMOTION_ROOT = ROOT / "records/g20_og_local_q_asset_promotion_20260805"
ROLLBACK = PROMOTION_ROOT / "rollback"
PROMOTION_MANIFEST = PROMOTION_ROOT / "promotion_manifest.json"
EXPECTED_OG_SHA256 = "697fe08490c957e4c9fa595ac0750cc80512b1f28dd3f53382db0d24ed202b2f"
FOLLOWERS = ("index_dip", "middle_dip", "ring_dip", "pinky_dip", "thumb_ip")
DYNAMIC_VIDEOS = (
    ROOT / "records/g20_og_local_q_candidate_dynamic_validation_20260805/videos/01_flex_extend_AB.mp4",
    ROOT / "records/g20_og_local_q_candidate_dynamic_validation_20260805/videos/02_mcp_roll_fan_AB.mp4",
    ROOT / "records/g20_og_local_q_candidate_dynamic_validation_20260805/videos/03_thumb_opposition_AB.mp4",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Mapping[str, Any], excluded: str) -> str:
    body = dict(value)
    body.pop(excluded, None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def joint_nodes(path: Path) -> dict[str, ET.Element]:
    return {node.attrib["name"]: node for node in ET.parse(path).getroot().findall("joint")}


def limits(path: Path, names: list[str]) -> dict[str, tuple[float, float]]:
    nodes = joint_nodes(path)
    result = {}
    for name in names:
        node = nodes[name].find("limit")
        if node is None:
            raise ValueError(f"{path}: {name} has no limit")
        result[name] = (float(node.attrib["lower"]), float(node.attrib["upper"]))
    return result


def mimic_rows(path: Path) -> list[dict[str, Any]]:
    nodes = joint_nodes(path)
    rows = []
    for name in FOLLOWERS:
        mimic = nodes[name].find("mimic")
        if mimic is None:
            raise ValueError(f"{path}: {name} has no mimic")
        rows.append({
            "joint": name,
            "source": mimic.attrib["joint"],
            "multiplier": float(mimic.attrib["multiplier"]),
            "offset": float(mimic.attrib.get("offset", 0.0)),
        })
    return rows


def replace_urdf_limits(text: str, safe: Mapping[str, tuple[float, float]]) -> str:
    for name, (lower, upper) in safe.items():
        pattern = re.compile(
            rf'(<joint\s+name="{re.escape(name)}".*?<limit\s+lower=")[^"]+("\s+upper=")[^"]+(")',
            re.DOTALL,
        )
        text, count = pattern.subn(
            rf'\g<1>{lower!r}\g<2>{upper!r}\g<3>', text, count=1
        )
        if count != 1:
            raise ValueError(f"could not uniquely replace {name} limits")
    return text


def backup(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT)
    target = ROLLBACK / relative
    if path.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return {
            "path": str(relative),
            "sha256": sha256(path),
            "rollback_path": str(target.relative_to(ROOT)),
        }
    return {"path": str(relative), "missing_before_promotion": True}


def build_model_manifest(
    schema: Mapping[str, Any], active_names: list[str], mimic: list[dict[str, Any]]
) -> dict[str, Any]:
    mesh_dir = TRAINING_URDF.parent / "meshes"
    meshes = {path.name: sha256(path) for path in sorted(mesh_dir.glob("*.STL"))}
    manifest: dict[str, Any] = {
        "model_id": "linkerhand-g20-left-og-local-q-v1",
        "model_version": 1,
        "urdf": {"path": TRAINING_URDF.name, "sha256": sha256(TRAINING_URDF)},
        "semantic_schema_id": schema["schema_id"],
        "semantic_schema_digest": schema["digest"],
        "coordinate_conventions": {
            "length_units": "meter",
            "angle_units": "radian",
            "joint_coordinates": "linker_hand_l20_OG local revolute q",
            "base_frame": "hand_base_link",
        },
        "actuated_joints": active_names,
        "mimic_joints": mimic,
        "fingertip_links": ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
        "meshes": meshes,
        "digest_algorithm": "sha256-canonical-json-excluding-model-digest",
        "model_digest": "pending",
    }
    manifest["model_digest"] = canonical_digest(manifest, "model_digest")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-by", required=True)
    args = parser.parse_args()

    candidate = read_json(CANDIDATE)
    provenance = read_json(PROVENANCE)
    if sha256(OG_URDF) != EXPECTED_OG_SHA256:
        raise ValueError("production OG URDF digest drift")
    if provenance.get("status") != "candidate_only_not_promoted":
        raise ValueError("unexpected candidate provenance status")
    if provenance.get("candidate_overlay_sha256") != sha256(CANDIDATE):
        raise ValueError("candidate/provenance digest mismatch")
    if any(not path.is_file() or path.stat().st_size == 0 for path in DYNAMIC_VIDEOS):
        raise ValueError("dynamic A/B evidence is incomplete")

    active_names = list(candidate["joints"])
    og_limits = limits(OG_URDF, active_names)
    safe: dict[str, tuple[float, float]] = {}
    for name, entry in candidate["joints"].items():
        lower = max(float(entry["lo"]), og_limits[name][0])
        upper = min(float(entry["hi"]), og_limits[name][1])
        if upper <= lower:
            raise ValueError(f"{name}: reviewed/OG intersection is empty")
        safe[name] = (lower, upper)

    PROMOTION_ROOT.mkdir(parents=True, exist_ok=True)
    before = [backup(path) for path in (DEPLOY_OVERLAY, TRAINING_URDF, SCHEMA, CALIBRATION, MODEL_MANIFEST)]
    original_mimic = mimic_rows(TRAINING_URDF)

    promoted_overlay = deepcopy(candidate)
    promoted_overlay["version"] = 5
    promoted_overlay["note"] = (
        "PRODUCTION 2026-08-05, user-approved after static 15-pose and three-trajectory "
        "D435/Isaac A/B review. LHT20-010-415-L-B-1-D. Complete physical raw to OG-local-q "
        "LUT; command limits are the reviewed-range intersection with unchanged OG geometry. "
        "Follower mimic remains OG and is not represented in this 16-actuator overlay."
    )
    for name, (lower, upper) in safe.items():
        promoted_overlay["joints"][name]["lo"] = lower
        promoted_overlay["joints"][name]["hi"] = upper
    write_json(DEPLOY_OVERLAY, promoted_overlay)

    training_text = TRAINING_URDF.read_text(encoding="utf-8")
    TRAINING_URDF.write_text(replace_urdf_limits(training_text, safe), encoding="utf-8")
    if mimic_rows(TRAINING_URDF) != original_mimic:
        raise ValueError("promotion changed a follower mimic relationship")

    old_schema = read_json(SCHEMA)
    schema = deepcopy(old_schema)
    schema["schema_version"] = max(1, int(schema.get("schema_version", 1)))
    by_name = {row["name"]: row for row in schema["ordered_joints"]}
    for name, bound in safe.items():
        by_name[name]["position_limit"] = list(bound)
    schema["mimic_relationships"] = original_mimic
    schema["digest"] = canonical_digest(schema, "digest")
    write_json(SCHEMA, schema)

    old_calibration = read_json(CALIBRATION)
    calibration = deepcopy(old_calibration)
    calibration.update({
        "artifact_version": 2,
        "semantic_schema_id": schema["schema_id"],
        "semantic_schema_digest": schema["digest"],
        "mapping_mode": "piecewise-linear-raw-to-semantic",
        "calibration_procedure_version": "og-local-q-same-view-registration-v1",
        "calibration_author": "manual same-view registration; user-reviewed",
        "calibration_date": "2026-08-05",
        "physical_verification_evidence": [
            "docs/g20-og-urdf-local-q-visual-calibration-20260805.md",
            "records/g20_og_local_q_candidate_validation_20260805/report/report.html",
            *[str(path.relative_to(ROOT)) for path in DYNAMIC_VIDEOS],
        ],
        "experimental_calibrations_excluded": [
            "five follower affine fits; production retains OG mimic",
            "diagnostic joint-limit expansions outside OG geometry",
        ],
    })
    calibration["semantic_to_native"] = [
        {
            "name": name,
            "slot": int(promoted_overlay["joints"][name]["slot"]),
            "soft_limit": list(safe[name]),
            "flip": False,
            "offset": 0.0,
            "physical_lut": deepcopy(promoted_overlay["joints"][name]["physical_lut"]),
        }
        for name in active_names
    ]
    calibration["artifact_digest"] = canonical_digest(calibration, "artifact_digest")
    write_json(CALIBRATION, calibration)

    model_manifest = build_model_manifest(schema, active_names, original_mimic)
    write_json(MODEL_MANIFEST, model_manifest)

    outputs = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for path in (DEPLOY_OVERLAY, TRAINING_URDF, SCHEMA, CALIBRATION, MODEL_MANIFEST)
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "production_promoted_user_approved",
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
        "approved_by": args.approved_by,
        "hand_serial": "LHT20-010-415-L-B-1-D",
        "decisions": {
            "active_lut": "promoted",
            "training_limits": "reviewed_active_range_intersection_with_unchanged_og_limits",
            "og_reference_urdf": "unchanged",
            "follower_mimic": "unchanged_og_not_promoted",
            "diagnostic_limit_expansion": "not_promoted",
        },
        "inputs": {
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "candidate_sha256": sha256(CANDIDATE),
            "provenance": str(PROVENANCE.relative_to(ROOT)),
            "provenance_sha256": sha256(PROVENANCE),
            "og_urdf": str(OG_URDF.relative_to(ROOT)),
            "og_urdf_sha256": sha256(OG_URDF),
            "dynamic_ab_videos": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in DYNAMIC_VIDEOS
            ],
        },
        "safe_training_limits_rad": {name: list(bound) for name, bound in safe.items()},
        "before": before,
        "outputs": outputs,
        "rollback_root": str(ROLLBACK.relative_to(ROOT)),
        "manifest_digest_algorithm": "sha256-canonical-json-excluding-manifest-digest",
        "manifest_digest": "pending",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, "manifest_digest")
    write_json(PROMOTION_MANIFEST, manifest)
    print(f"promoted 16 active LUTs; manifest={PROMOTION_MANIFEST.relative_to(ROOT)}")
    print(f"deployment overlay sha256={sha256(DEPLOY_OVERLAY)}")
    print(f"training URDF sha256={sha256(TRAINING_URDF)}")
    print(f"schema digest={schema['digest']}")
    print(f"calibration digest={calibration['artifact_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
