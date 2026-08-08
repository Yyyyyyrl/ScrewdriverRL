#!/usr/bin/env python3
"""Repackage a geometry-DR top-down Stage-2 bundle for one real handle diameter.

The Stage-2 run launched before nominal-row bundle selection was fixed exported
env row 0 (60 mm). The adapter and actor weights are valid; only deployment
home/clamp metadata must be rebuilt for the physical 64 mm handle. This tool
does that without changing the source bundle and emits an auditable manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from screwdriver_rl.deploy.codecs import mounted_linker_g20_codec_spec
from screwdriver_rl.deploy.policy import DeployPolicy
from screwdriver_rl.utils.linker_topdown_diameter_postures import (
    TOPDOWN_HANDLE_DIAMETERS_M,
    TOPDOWN_NOMINAL_DIAMETER_M,
    TOPDOWN_PREGRASP_POSITIONS_BUCKETS,
    TOPDOWN_RESET_POSITIONS_BUCKETS,
    bucket_for_diameter_mm,
)


TASK_ID = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
HAND_URDF = ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
FINGERS = ("index", "middle", "ring", "pinky", "thumb")
ACTIVE_JOINTS = (
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
    "thumb_cmc_yaw",
    "thumb_cmc_roll",
    "thumb_cmc_pitch",
    "thumb_mcp",
)
JOINT_MOTION_RANGE_RAD = 0.35
JOINT_TARGET_MARGIN_RAD = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_vectors(
    diameter_mm: int, urdf_path: Path = HAND_URDF
) -> tuple[int, list[float], list[float], list[float]]:
    """Return bucket, home, lower and upper vectors in policy joint order."""
    bucket = bucket_for_diameter_mm(diameter_mm)
    posture = TOPDOWN_PREGRASP_POSITIONS_BUCKETS[bucket]
    home = [float(value) for finger in FINGERS for value in posture[finger]]
    if len(home) != len(ACTIVE_JOINTS):
        raise ValueError("top-down posture width does not match active joints")

    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None:
            limits[str(joint.get("name"))] = (
                float(limit.get("lower")),
                float(limit.get("upper")),
            )
    missing = [name for name in ACTIVE_JOINTS if name not in limits]
    if missing:
        raise ValueError(f"URDF is missing active joint limits: {missing}")

    lower = [
        max(
            limits[name][0] + JOINT_TARGET_MARGIN_RAD,
            value - JOINT_MOTION_RANGE_RAD,
        )
        for name, value in zip(ACTIVE_JOINTS, home)
    ]
    upper = [
        min(
            limits[name][1] - JOINT_TARGET_MARGIN_RAD,
            value + JOINT_MOTION_RANGE_RAD,
        )
        for name, value in zip(ACTIVE_JOINTS, home)
    ]
    if any(not lo <= value <= hi for lo, value, hi in zip(lower, home, upper)):
        raise ValueError("deployment home target falls outside reconstructed bounds")
    return bucket, home, lower, upper


def hardware_vectors(
    urdf_path: Path = HAND_URDF,
) -> tuple[list[float], list[float]]:
    """Return physical URDF limits in policy joint order."""
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None:
            limits[str(joint.get("name"))] = (
                float(limit.get("lower")),
                float(limit.get("upper")),
            )
    missing = [name for name in ACTIVE_JOINTS if name not in limits]
    if missing:
        raise ValueError(f"URDF is missing active joint limits: {missing}")
    return (
        [limits[name][0] for name in ACTIVE_JOINTS],
        [limits[name][1] for name in ACTIVE_JOINTS],
    )


def startup_reset_vector(diameter_mm: int) -> list[float]:
    """Return the collision-safe approach posture in policy joint order."""
    bucket = bucket_for_diameter_mm(diameter_mm)
    posture = TOPDOWN_RESET_POSITIONS_BUCKETS[bucket]
    values = [
        float(value) for finger in FINGERS for value in posture[finger]
    ]
    if len(values) != len(ACTIVE_JOINTS):
        raise ValueError("top-down reset posture width does not match active joints")
    return values


def corrected_config(source: dict, diameter_mm: int) -> dict:
    """Return a copied deploy config with diameter-specific home and bounds."""
    if source.get("task") != TASK_ID:
        raise ValueError(
            f"expected task {TASK_ID!r}, got {source.get('task')!r}"
        )
    if int(source.get("n_finger", -1)) != len(ACTIVE_JOINTS):
        raise ValueError("bundle does not contain the 16-joint Linker policy")
    bucket, home, lower, upper = deployment_vectors(diameter_mm)
    hardware_lower, hardware_upper = hardware_vectors()
    startup_reset = startup_reset_vector(diameter_mm)
    if any(
        not lo <= value <= hi
        for value, lo, hi in zip(
            startup_reset, hardware_lower, hardware_upper
        )
    ):
        raise ValueError("startup reset target falls outside URDF hardware limits")
    history_length = int(source["prop_hist_len"])

    result = dict(source)
    result.update(
        {
            "home_targets": home,
            "startup_reset_targets": startup_reset,
            "startup_reset_hardware_lower": hardware_lower,
            "startup_reset_hardware_upper": hardware_upper,
            "finger_lower": lower,
            "finger_upper": upper,
            "proprio_codec": mounted_linker_g20_codec_spec(
                history_length
            ).as_dict(),
            "deployment_env_index": bucket,
            "deployment_geometry_bucket": bucket,
            "deployment_geometry_scale": [
                (float(diameter_mm) / 1000.0) / TOPDOWN_NOMINAL_DIAMETER_M,
                1.0,
            ],
            "deployment_handle_diameter_mm": float(diameter_mm),
        }
    )
    return result


def repackage(source_path: Path, output_path: Path, diameter_mm: int) -> dict:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    if source_path == output_path:
        raise ValueError("source and output paths must differ")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path} or its manifest")

    bundle = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("config"), dict):
        raise ValueError("source is not a Stage-2 deploy bundle")
    corrected = dict(bundle)
    corrected["config"] = corrected_config(bundle["config"], diameter_mm)
    corrected["deployment_metadata_repackage"] = {
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "reason": "select physical diameter row from geometry-DR bundle",
        "diameter_mm": int(diameter_mm),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(corrected, temporary)
    try:
        policy = DeployPolicy(str(temporary), device="cpu")
        expected_home = torch.tensor(
            corrected["config"]["home_targets"], dtype=torch.float32
        )
        if not torch.allclose(
            policy.home_targets.cpu(), expected_home, atol=1e-7
        ):
            raise RuntimeError("repackaged DeployPolicy home target mismatch")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, output_path)

    manifest = {
        "artifact_type": "topdown-deploy-bundle-repackage",
        "task": TASK_ID,
        "source": str(source_path),
        "source_sha256": corrected["deployment_metadata_repackage"][
            "source_sha256"
        ],
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "iter": corrected.get("iter"),
        "loss": corrected.get("loss"),
        "diameter_mm": int(diameter_mm),
        "geometry_bucket": corrected["config"]["deployment_geometry_bucket"],
        "geometry_scale": corrected["config"]["deployment_geometry_scale"],
        "home_targets": corrected["config"]["home_targets"],
        "startup_reset_targets": corrected["config"][
            "startup_reset_targets"
        ],
        "finger_lower": corrected["config"]["finger_lower"],
        "finger_upper": corrected["config"]["finger_upper"],
        "deploy_policy_load": "pass",
    }
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diameter-mm",
        type=int,
        choices=[round(value * 1000) for value in TOPDOWN_HANDLE_DIAMETERS_M],
        default=64,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = repackage(args.source, args.output, args.diameter_mm)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

