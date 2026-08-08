from __future__ import annotations

import hashlib
import json

from safetensors.torch import load_file

from screwdriver_rl.deploy.policy_package import (
    canonical_json,
    export_policy_package,
    package_content_digest,
)
from test_deploy_policy_bundle import make_bundle


def package_metadata() -> dict:
    return {
        "display_name": "Synthetic mounted rotation policy",
        "task_id": "test-bundle",
        "task_version": "1.0",
        "hand": {
            "model": "LinkerHand G20",
            "side": "left",
            "semantic_schema_id": "linker-g20-left-semantic-16-v1",
            "semantic_schema_digest": "ce53ccafeb70a7bd9ba203576f7e54f330e106c0676ef90f8262d2a9ffa34ba7",
        },
        "calibration_compatibility": [
            {
                "calibration_id": "linker-g20-left-lht20-010-415-v1",
                "artifact_digest": "1e20d989a14aa9fe127e78680decb9bb29679858e223c41ad28ae67a598d51df",
            }
        ],
        "state_requirements": {
            "fields": ["semantic_position", "last_effective_target"],
            "acknowledgement_level": "sent-to-bus",
            "maximum_state_age_ns": 100_000_000,
            "maximum_effective_target_age_ns": 100_000_000,
        },
        "task_frame": {
            "task_frame_id": "fixture-axis",
            "wrist_frame_id": "hand-base",
            "desired_task_from_wrist": [0, 0, 0, 1, 0, 0, 0],
            "position_envelope_m": [0.01, 0.01, 0.01],
            "orientation_envelope_rad": 0.1,
            "maximum_wrist_twist_rad_s": 0.05,
            "gravity_relative_orientation": "fixture-axis-parallel-gravity",
            "object_fixture_assumptions": ["fixture-secured"],
            "contact_target_gap_conditions": ["operator-confirmed"],
        },
        "provenance": {
            "training_commit": "0123456789abcdef0123456789abcdef01234567",
            "training_dirty": False,
            "resolved_training_config_digest": "a" * 64,
            "urdf_digest": "b" * 64,
            "asset_digests": {"screwdriver": "c" * 64},
        },
        "evaluation": {
            "results": {"success_rate": 0.9, "episodes": 100},
            "promotion_status": "commissioning",
        },
        "supported_runtime_api": {"min": "1.0", "max": "1.0"},
        "readiness_provider_ids": [
            "operator-confirmation-v1",
            "hand-state-freshness-v1",
            "gateway-health-v1",
        ],
    }


def test_export_is_canonical_content_addressed_and_separates_tensors(tmp_path) -> None:
    output = export_policy_package(make_bundle(), package_metadata(), tmp_path / "policy")
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest_text == canonical_json(manifest) + "\n"
    assert manifest["package_digest"] == package_content_digest(manifest)
    assert manifest["package_id"] == f"sha256:{manifest['package_digest']}"
    assert manifest["package_format_version"] == 2
    assert manifest["observation_contract"] == {
        "semantics_version": "test-force-free-v1",
        "history_obs_dim": 32,
        "privileged_obs_dim": 19,
    }
    assert manifest["trust"] == {"mode": "unsigned-local", "signature": None}
    assert manifest["action_transform"]["initial_effective_target_rad"]
    assert manifest["startup_sequence"] == {
        "kind": "collision-safe-reset-then-contact-home",
        "approach_target_rad": manifest["action_transform"][
            "initial_effective_target_rad"
        ],
        "contact_target_rad": manifest["action_transform"][
            "initial_effective_target_rad"
        ],
        "durations_are_runtime_configurable": True,
        "hardware_lower_rad": None,
        "hardware_upper_rad": None,
    }
    for name in ("actor", "adapter"):
        entry = manifest["weights"][name]
        path = output / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert load_file(str(path))


def test_export_requires_non_inferable_metadata_and_new_output(tmp_path) -> None:
    metadata = package_metadata()
    del metadata["task_frame"]
    try:
        export_policy_package(make_bundle(), metadata, tmp_path / "bad")
    except ValueError as exc:
        assert "task_frame" in str(exc)
    else:
        raise AssertionError("missing task-frame metadata was accepted")

    output = tmp_path / "immutable"
    export_policy_package(make_bundle(), package_metadata(), output)
    try:
        export_policy_package(make_bundle(), package_metadata(), output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing immutable package output was overwritten")


def test_staged_startup_uses_hardware_envelope_not_policy_window(tmp_path) -> None:
    bundle = make_bundle()
    config = bundle["config"]
    joint_count = int(config["n_finger"])
    reset = list(config["home_targets"])
    reset[0] = float(config["finger_lower"][0]) - 0.1
    config["startup_reset_targets"] = reset
    config["startup_reset_hardware_lower"] = [-3.14] * joint_count
    config["startup_reset_hardware_upper"] = [3.14] * joint_count

    output = export_policy_package(
        bundle, package_metadata(), tmp_path / "staged"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["startup_sequence"]["approach_target_rad"] == reset
    assert manifest["startup_sequence"]["hardware_lower_rad"] == [-3.14] * joint_count

    config["startup_reset_targets"][0] = -4.0
    try:
        export_policy_package(
            bundle, package_metadata(), tmp_path / "unsafe-staged"
        )
    except ValueError as exc:
        assert "outside hardware limits" in str(exc)
    else:
        raise AssertionError("unsafe staged startup reset was accepted")
