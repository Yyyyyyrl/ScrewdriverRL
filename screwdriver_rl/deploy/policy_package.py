"""Export immutable, content-addressed runtime policy packages.

The output is a canonical JSON manifest plus separate Safetensors actor and
adapter files.  This exporter intentionally requires deployment metadata that
cannot be reconstructed safely from a training checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from safetensors.torch import save_file
import torch

from .codecs import COLLECT_FRESH_HISTORY, ProprioCodecSpec


PACKAGE_FORMAT = "dex-policy-package"
PACKAGE_FORMAT_VERSION = 2
PROTOCOL_VERSION = "1.0"
ACTOR_FILENAME = "actor.safetensors"
ADAPTER_FILENAME = "adapter.safetensors"
MANIFEST_FILENAME = "manifest.json"

_METADATA_FIELDS = {
    "display_name",
    "task_id",
    "task_version",
    "hand",
    "calibration_compatibility",
    "state_requirements",
    "task_frame",
    "provenance",
    "evaluation",
    "supported_runtime_api",
    "readiness_provider_ids",
}


class PolicyPackageExportError(ValueError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_content_digest(manifest: Mapping[str, object]) -> str:
    content = deepcopy(dict(manifest))
    content.pop("package_id", None)
    content.pop("package_digest", None)
    return sha256_bytes(canonical_json(content).encode("utf-8"))


def _require_exact_fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise PolicyPackageExportError(f"invalid {label} fields; missing={missing}, extra={extra}")


def _tensor_state(value: object, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise PolicyPackageExportError(f"{label} state must be a non-empty tensor mapping")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise PolicyPackageExportError(f"{label} state contains a non-tensor entry")
        if not bool(torch.isfinite(tensor).all()):
            raise PolicyPackageExportError(f"{label} tensor {key!r} contains non-finite values")
        result[key] = tensor.detach().cpu().contiguous()
    return result


def _validate_metadata(metadata: Mapping[str, object], task_id: str) -> None:
    _require_exact_fields(metadata, _METADATA_FIELDS, "package metadata")
    if metadata["task_id"] != task_id:
        raise PolicyPackageExportError("metadata task_id does not match the deploy bundle")
    for name in ("display_name", "task_id", "task_version"):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise PolicyPackageExportError(f"metadata {name} must be a non-empty string")

    hand = metadata["hand"]
    if not isinstance(hand, Mapping):
        raise PolicyPackageExportError("hand metadata must be an object")
    _require_exact_fields(
        hand,
        {"model", "side", "semantic_schema_id", "semantic_schema_digest"},
        "hand metadata",
    )
    if hand["side"] not in ("left", "right") or any(
        not isinstance(hand[name], str) or not hand[name]
        for name in ("model", "semantic_schema_id", "semantic_schema_digest")
    ):
        raise PolicyPackageExportError("hand identity metadata is incomplete")

    compatibility = metadata["calibration_compatibility"]
    if not isinstance(compatibility, list) or not compatibility:
        raise PolicyPackageExportError("at least one calibration compatibility entry is required")
    for entry in compatibility:
        if not isinstance(entry, Mapping):
            raise PolicyPackageExportError("calibration compatibility entries must be objects")
        _require_exact_fields(entry, {"calibration_id", "artifact_digest"}, "calibration entry")
        if not all(isinstance(entry[name], str) and entry[name] for name in entry):
            raise PolicyPackageExportError("calibration compatibility identity is incomplete")

    for name in (
        "state_requirements",
        "task_frame",
        "provenance",
        "evaluation",
        "supported_runtime_api",
    ):
        if not isinstance(metadata[name], Mapping) or not metadata[name]:
            raise PolicyPackageExportError(f"metadata {name} must be a non-empty object")
    if not isinstance(metadata["readiness_provider_ids"], list):
        raise PolicyPackageExportError("readiness_provider_ids must be a list")
    state = metadata["state_requirements"]
    _require_exact_fields(
        state,
        {"fields", "acknowledgement_level", "maximum_state_age_ns", "maximum_effective_target_age_ns"},
        "state requirements",
    )
    if (
        not isinstance(state["fields"], list)
        or "semantic_position" not in state["fields"]
        or "last_effective_target" not in state["fields"]
        or not isinstance(state["acknowledgement_level"], str)
        or not state["acknowledgement_level"]
        or int(state["maximum_state_age_ns"]) <= 0
        or int(state["maximum_effective_target_age_ns"]) <= 0
    ):
        raise PolicyPackageExportError("state requirements are incomplete")

    task_frame = metadata["task_frame"]
    _require_exact_fields(
        task_frame,
        {
            "task_frame_id", "wrist_frame_id", "desired_task_from_wrist",
            "position_envelope_m", "orientation_envelope_rad",
            "maximum_wrist_twist_rad_s", "gravity_relative_orientation",
            "object_fixture_assumptions", "contact_target_gap_conditions",
        },
        "task frame",
    )
    if not all(isinstance(task_frame[name], str) and task_frame[name] for name in ("task_frame_id", "wrist_frame_id")):
        raise PolicyPackageExportError("task and wrist frame IDs are required")

    provenance = metadata["provenance"]
    _require_exact_fields(
        provenance,
        {"training_commit", "training_dirty", "resolved_training_config_digest", "urdf_digest", "asset_digests"},
        "provenance",
    )
    if (
        not isinstance(provenance["training_dirty"], bool)
        or not isinstance(provenance["asset_digests"], Mapping)
        or any(not isinstance(provenance[name], str) or not provenance[name] for name in ("training_commit", "resolved_training_config_digest", "urdf_digest"))
    ):
        raise PolicyPackageExportError("training provenance is incomplete")

    evaluation = metadata["evaluation"]
    _require_exact_fields(evaluation, {"results", "promotion_status"}, "evaluation")
    if not isinstance(evaluation["results"], Mapping) or not isinstance(evaluation["promotion_status"], str) or not evaluation["promotion_status"]:
        raise PolicyPackageExportError("evaluation metadata is incomplete")
    runtime_api = metadata["supported_runtime_api"]
    _require_exact_fields(runtime_api, {"min", "max"}, "supported runtime API")
    if not all(isinstance(runtime_api[name], str) and runtime_api[name] for name in ("min", "max")):
        raise PolicyPackageExportError("supported runtime API range is incomplete")
    if any(not isinstance(item, str) or not item for item in metadata["readiness_provider_ids"]):
        raise PolicyPackageExportError("readiness provider IDs must be non-empty strings")


def _load_bundle(bundle: str | Path | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(bundle, (str, Path)):
        loaded = torch.load(str(bundle), map_location="cpu", weights_only=False)
        if not isinstance(loaded, Mapping):
            raise PolicyPackageExportError("deploy bundle root must be an object")
        return loaded
    return bundle


def export_policy_package(
    bundle: str | Path | Mapping[str, object],
    metadata: Mapping[str, object],
    output_directory: str | Path,
) -> Path:
    """Export one immutable directory; refuses an existing output path."""

    bundle_value = _load_bundle(bundle)
    for key in ("actor", "actor_arch", "adapter", "net_dims", "config"):
        if key not in bundle_value:
            raise PolicyPackageExportError(f"deploy bundle is missing {key!r}")
    config = bundle_value["config"]
    actor_arch = bundle_value["actor_arch"]
    net_dims = bundle_value["net_dims"]
    if not isinstance(config, Mapping) or not isinstance(actor_arch, Mapping) or not isinstance(net_dims, Mapping):
        raise PolicyPackageExportError("bundle config and network declarations must be objects")

    required_config = {
        "task",
        "n_finger",
        "action_delta_scale",
        "finger_lower",
        "finger_upper",
        "home_targets",
        "prop_hist_len",
        "history_obs_dim",
        "observation_semantics_version",
        "proprio_codec",
    }
    missing_config = sorted(required_config - set(config))
    if missing_config:
        raise PolicyPackageExportError(f"bundle config is missing {missing_config}")
    task_id = str(config["task"])
    fixed_wrist_task = (
        "topdown" in task_id.lower()
        or "inhand-rotation" in task_id.lower()
    )
    if (
        fixed_wrist_task
        and config.get("startup_reset_targets") is None
    ):
        raise PolicyPackageExportError(
            "fixed-wrist bundle is missing collision-safe startup_reset_targets"
        )
    observation_semantics_version = config["observation_semantics_version"]
    if (
        not isinstance(observation_semantics_version, str)
        or not observation_semantics_version
    ):
        raise PolicyPackageExportError("observation semantics version is required")
    _validate_metadata(metadata, task_id)
    codec_value = config["proprio_codec"]
    if not isinstance(codec_value, Mapping):
        raise PolicyPackageExportError("bundle ProprioCodec must be an object")
    codec = ProprioCodecSpec.from_dict(codec_value)

    joint_count = int(config["n_finger"])
    history_length = int(net_dims["hist_len"])
    frame_dim = int(net_dims["frame_dim"])
    actor_width = int(actor_arch["proprio_dim"])
    if codec.joint_count != joint_count:
        raise PolicyPackageExportError("codec joint count does not match action width")
    if codec.history_length != history_length or codec.frame_dim != frame_dim:
        raise PolicyPackageExportError("codec history shape does not match adapter")
    if codec.actor_frame_count * codec.frame_dim != actor_width:
        raise PolicyPackageExportError("codec actor assembly does not match actor width")
    if codec.history_reset_semantics != COLLECT_FRESH_HISTORY:
        raise PolicyPackageExportError("package must require fresh effective-target history")

    lower = [float(value) for value in config["finger_lower"]]
    upper = [float(value) for value in config["finger_upper"]]
    home = [float(value) for value in config["home_targets"]]
    startup_reset = [
        float(value)
        for value in config.get("startup_reset_targets", home)
    ]
    if any(
        len(values) != joint_count
        for values in (lower, upper, home, startup_reset)
    ):
        raise PolicyPackageExportError(
            "action limits, home, or startup reset width does not match joint count"
        )
    if any(high <= low for low, high in zip(lower, upper)):
        raise PolicyPackageExportError("action upper limits must exceed lower limits")
    if any(
        value < low - 1.0e-6 or value > high + 1.0e-6
        for value, low, high in zip(home, lower, upper)
    ):
        raise PolicyPackageExportError(
            "home target lies outside the exported action limits"
        )
    startup_hardware_lower = config.get("startup_reset_hardware_lower")
    startup_hardware_upper = config.get("startup_reset_hardware_upper")
    if startup_reset != home:
        if startup_hardware_lower is None or startup_hardware_upper is None:
            raise PolicyPackageExportError(
                "staged startup reset requires explicit hardware limits"
            )
        startup_hardware_lower = [
            float(value) for value in startup_hardware_lower
        ]
        startup_hardware_upper = [
            float(value) for value in startup_hardware_upper
        ]
        if any(
            len(values) != joint_count
            for values in (startup_hardware_lower, startup_hardware_upper)
        ):
            raise PolicyPackageExportError(
                "startup hardware limits do not match joint count"
            )
        if any(
            value < low - 1.0e-6 or value > high + 1.0e-6
            for value, low, high in zip(
                startup_reset, startup_hardware_lower, startup_hardware_upper
            )
        ):
            raise PolicyPackageExportError(
                "startup reset target lies outside hardware limits"
            )
    action_delta_scale = float(config["action_delta_scale"])
    if action_delta_scale <= 0:
        raise PolicyPackageExportError("action delta scale must be positive")

    actor_state = _tensor_state(bundle_value["actor"], "actor")
    adapter_state = _tensor_state(bundle_value["adapter"], "adapter")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"immutable package output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        actor_path = temporary / ACTOR_FILENAME
        adapter_path = temporary / ADAPTER_FILENAME
        save_file(actor_state, str(actor_path))
        save_file(adapter_state, str(adapter_path))

        manifest: dict[str, object] = {
            "package_format": PACKAGE_FORMAT,
            "package_format_version": PACKAGE_FORMAT_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "package_id": "pending",
            "package_digest": "pending",
            "display_name": metadata["display_name"],
            "task": {"id": task_id, "version": metadata["task_version"]},
            "hand": deepcopy(metadata["hand"]),
            "calibration_compatibility": deepcopy(metadata["calibration_compatibility"]),
            "control_period_ns": codec.control_period_ns,
            "weights": {
                "actor": {
                    "path": ACTOR_FILENAME,
                    "format": "safetensors",
                    "sha256": sha256_file(actor_path),
                },
                "adapter": {
                    "path": ADAPTER_FILENAME,
                    "format": "safetensors",
                    "sha256": sha256_file(adapter_path),
                },
            },
            "network": {
                "actor": deepcopy(dict(actor_arch)),
                "adapter": {
                    "architecture_id": "proprio-adapt-tconv-v1",
                    "frame_dim": frame_dim,
                    "history_length": history_length,
                    "output_dim": int(net_dims["out_dim"]),
                    "frame_encoder_units": [32, 32],
                    "temporal_convolutions": [
                        {"channels": 32, "kernel": 9, "stride": 2},
                        {"channels": 32, "kernel": 5, "stride": 1},
                        {"channels": 32, "kernel": 5, "stride": 1},
                    ],
                    "activation": "elu",
                },
            },
            "proprio_codec": codec.as_dict(),
            "actor_input_assembler": {
                "kind": "latest-frames-flatten",
                "frame_count": codec.actor_frame_count,
                "output_width": codec.actor_frame_count * codec.frame_dim,
            },
            "action_transform": {
                "kind": "bounded-delta-position",
                "action_clip": [-1.0, 1.0],
                "delta_scale_rad": action_delta_scale,
                "position_lower_rad": lower,
                "position_upper_rad": upper,
                "initial_effective_target_rad": home,
                "integration_semantics": "acknowledged-effective-target-plus-delta",
            },
            "startup_sequence": {
                "kind": "collision-safe-reset-then-contact-home",
                "approach_target_rad": startup_reset,
                "contact_target_rad": home,
                "durations_are_runtime_configurable": True,
                "hardware_lower_rad": startup_hardware_lower,
                "hardware_upper_rad": startup_hardware_upper,
            },
            "history": {
                "length": history_length,
                "reset_semantics": COLLECT_FRESH_HISTORY,
                "activation_requires_full_history": True,
            },
            "observation_contract": {
                "semantics_version": observation_semantics_version,
                "history_obs_dim": int(config["history_obs_dim"]),
                "privileged_obs_dim": int(config.get("privileged_obs_dim", 0)),
            },
            "state_requirements": deepcopy(metadata["state_requirements"]),
            "task_frame": deepcopy(metadata["task_frame"]),
            "provenance": deepcopy(metadata["provenance"]),
            "evaluation": deepcopy(metadata["evaluation"]),
            "readiness_provider_ids": deepcopy(metadata["readiness_provider_ids"]),
            "supported_runtime_api": deepcopy(metadata["supported_runtime_api"]),
            "trust": {"mode": "unsigned-local", "signature": None},
        }
        if config.get("deployment_object_kind") is not None:
            manifest["deployment_object"] = {
                "kind": config["deployment_object_kind"],
                "size_mm": deepcopy(config.get("deployment_object_size_mm")),
                "training_mass_range_g": deepcopy(
                    config.get("deployment_object_mass_range_g")
                ),
                "grasp_orientation": config.get(
                    "deployment_grasp_orientation"
                ),
                "hand_root_position_m": deepcopy(
                    config.get("deployment_hand_root_position_m")
                ),
                "hand_root_quaternion_wxyz": deepcopy(
                    config.get("deployment_hand_root_quaternion_wxyz")
                ),
                "object_seed_position_m": deepcopy(
                    config.get("deployment_object_seed_position_m")
                ),
                "grasp_cache": deepcopy(config.get("deployment_grasp_cache")),
                "grasp_manifest_sha256": config.get(
                    "deployment_grasp_manifest_sha256"
                ),
                "operator_loaded": True,
                "palm_support_allowed": False,
            }
        digest = package_content_digest(manifest)
        manifest["package_id"] = f"sha256:{digest}"
        manifest["package_digest"] = digest
        (temporary / MANIFEST_FILENAME).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        os.rename(temporary, output)
        return output
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
