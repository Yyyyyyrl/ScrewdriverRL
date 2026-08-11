from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from screwdriver_rl.utils.inhand_grasp_cache_manifest import (
    GraspCacheManifestError,
    load_and_validate_manifest,
    load_manifest_for_update,
    model_identity,
    new_manifest,
    record_cache_entry,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_model_identity_binds_current_urdf_and_semantic_schema() -> None:
    identity = model_identity(ROOT)
    source = json.loads(
        (ROOT / "assets/linker_hand_l20/model_manifest_v1.json").read_text()
    )
    assert identity["model_id"] == source["model_id"]
    assert identity["model_digest"] == source["model_digest"]
    assert identity["semantic_schema_id"] == source["semantic_schema_id"]
    assert identity["urdf_sha256"] == source["urdf"]["sha256"]


def test_manifest_rejects_file_tamper_and_identity_drift(tmp_path: Path) -> None:
    cache = tmp_path / "bank_grasp_cub0_s1.npy"
    cache.write_bytes(b"cache-v1")
    manifest = new_manifest(
        cache_name="bank",
        orientation="top-down-tilted-45deg",
        object_scales=(0.9375, 1.0, 1.0625),
        cube_edge_m=0.064,
        repo_root=ROOT,
        certification={
            "schema": "dex-forge-inhand-cache-replay-cert-v1",
            "replays_per_row": 4,
        },
    )
    record_cache_entry(
        manifest,
        cache_path=cache,
        scale=1.0,
        shape="cuboid",
        prototype=0,
        rows=12500,
    )
    path = tmp_path / "bank_manifest.json"
    write_manifest(path, manifest)
    expected = deepcopy(manifest)
    expected.pop("entries")
    loaded = load_and_validate_manifest(
        path=path,
        expected=expected,
        required_files=[cache],
    )
    assert loaded["entries"][cache.name]["rows"] == 12500
    assert loaded["schema_version"] == 2
    assert loaded["certification"]["replays_per_row"] == 4

    cache.write_bytes(b"cache-v2")
    with pytest.raises(GraspCacheManifestError, match="digest mismatch"):
        load_and_validate_manifest(
            path=path,
            expected=expected,
            required_files=[cache],
        )

    incompatible = deepcopy(manifest)
    incompatible["orientation"] = "bottom-up-tilted-45deg"
    with pytest.raises(GraspCacheManifestError, match="incompatible"):
        load_manifest_for_update(path, incompatible)


def test_pipeline_tools_encode_closed_loop_and_no_palm_gates() -> None:
    acceptance = (ROOT / "tools/evaluate_inhand_stage2_acceptance.py").read_text()
    for token in (
        "adapter_fall <= 0.40",
        "adapter_fall <= oracle_fall + 0.05",
        "adapter_turns >= 0.70 * oracle_turns",
        "adapter_turns >= 1.5",
        "fraction <= 0.01",
        "--fixed_inhand_cube_mm",
    ):
        assert token in acceptance

    exporter = (ROOT / "tools/export_inhand_policy_package.py").read_text()
    for token in (
        "acceptance.get(\"promotion_status\") != \"sim-qualified\"",
        "deployment_object_size_mm",
        "operator-loaded 64 mm PLA cube",
        "no palm support; finger-side contact permitted",
        "load_linker_calibration",
    ):
        assert token in exporter

    env = (
        ROOT
        / "screwdriver_rl/tasks/linker_l20/inhand_rotation_env.py"
    ).read_text()
    for token in (
        'PALM_BODY_NAMES = ("hand_base_link",)',
        '"eval_palm_support_force"',
        '"eval_nontip_surface_force"',
        "self._read_named_nontip_object_forces(self.PALM_BODY_NAMES)",
    ):
        assert token in env

    certifier = (ROOT / "tools/certify_inhand_grasp_cache.py").read_text()
    assert 'base.extras.get("eval_nontip_surface_force")' in certifier

    train = (ROOT / "train.py").read_text()
    assert "free-object Stage-2 cannot assemble a deployable bundle" in train
