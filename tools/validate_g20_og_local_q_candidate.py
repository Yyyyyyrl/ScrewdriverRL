#!/usr/bin/env python3
"""Static and numerical validation for the G20 OG-local-q candidate overlay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from screwdriver_rl.deploy import linker_sdk_map as sdkmap  # noqa: E402


DEFAULT_CANDIDATE = (
    ROOT
    / "assets"
    / "calibrations"
    / "linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805.json"
)
DEFAULT_PROVENANCE = DEFAULT_CANDIDATE.with_name(
    DEFAULT_CANDIDATE.stem + "_provenance.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "records"
    / "g20_og_local_q_candidate_validation_20260805"
    / "static_validation.json"
)
PRODUCTION_URDF = ROOT / "assets/linker_hand_l20_OG/linkerhand_l20_left.urdf"
EXPECTED_URDF_SHA256 = (
    "697fe08490c957e4c9fa595ac0750cc80512b1f28dd3f53382db0d24ed202b2f"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    default_before = sdkmap.active_joints()
    if default_before != sdkmap.DEFAULT_JOINTS:
        raise RuntimeError("candidate was already active at process start")
    overlay = read_json(args.candidate)
    provenance = read_json(args.provenance)
    candidate_digest = sha256(args.candidate)
    if provenance["candidate_overlay_sha256"] != candidate_digest:
        raise ValueError("candidate digest differs from provenance")
    if sha256(PRODUCTION_URDF) != EXPECTED_URDF_SHA256:
        raise ValueError("production OG URDF digest drift")

    specs = tuple(sdkmap.build_joint_table(overlay))
    if sdkmap.active_joints() != sdkmap.DEFAULT_JOINTS:
        raise RuntimeError("build_joint_table unexpectedly activated the candidate")
    slots = [joint.slot for joint in specs]
    if len(set(slots)) != 16 or any(slot in (11, 12, 13, 14) for slot in slots):
        raise ValueError("candidate slot routing is invalid")

    sdkmap.apply_calibration(overlay)
    rows: list[dict[str, Any]] = []
    global_q_error = 0.0
    global_raw_error = 0
    for joint_index, joint in enumerate(specs):
        if joint.physical_raw_knots is None or joint.physical_rad_knots is None:
            raise ValueError(f"{joint.name}: candidate LUT missing")
        raw_lower = math.ceil(joint.physical_raw_knots[0])
        raw_upper = math.floor(joint.physical_raw_knots[-1])
        max_raw_error = 0
        worst_raw = None
        for raw in range(raw_lower, raw_upper + 1):
            sample = [0.0] * 20
            sample[joint.slot] = raw
            q = sdkmap.sdk_range_to_joints16(sample)[joint_index]
            values = [spec.lo for spec in specs]
            values[joint_index] = q
            recovered = sdkmap.joints16_to_sdk_range(values)[joint.slot]
            error = abs(recovered - raw)
            if error > max_raw_error:
                max_raw_error = error
                worst_raw = raw

        max_q_error = 0.0
        worst_q = None
        for sample_index in range(1001):
            q = joint.lo + (joint.hi - joint.lo) * sample_index / 1000.0
            values = [spec.lo for spec in specs]
            values[joint_index] = q
            raw20 = sdkmap.joints16_to_sdk_range(values)
            recovered = sdkmap.sdk_range_to_joints16(raw20)[joint_index]
            error = abs(recovered - q)
            if error > max_q_error:
                max_q_error = error
                worst_q = q

        direction = (
            "increasing"
            if joint.physical_rad_knots[0] < joint.physical_rad_knots[-1]
            else "decreasing"
        )
        rows.append(
            {
                "joint": joint.name,
                "slot": joint.slot,
                "q_direction_with_increasing_raw": direction,
                "runtime_knot_count": len(joint.physical_raw_knots),
                "raw_domain": [joint.physical_raw_knots[0], joint.physical_raw_knots[-1]],
                "q_domain_rad": [joint.lo, joint.hi],
                "integer_raw_roundtrip": {
                    "tested_range_inclusive": [raw_lower, raw_upper],
                    "max_abs_raw_error": max_raw_error,
                    "worst_input_raw": worst_raw,
                },
                "q_roundtrip_after_uint8_command": {
                    "sample_count": 1001,
                    "max_abs_q_error_rad": max_q_error,
                    "worst_input_q_rad": worst_q,
                },
            }
        )
        global_q_error = max(global_q_error, max_q_error)
        global_raw_error = max(global_raw_error, max_raw_error)

    sdkmap.reset_calibration()
    if sdkmap.active_joints() != sdkmap.DEFAULT_JOINTS:
        raise RuntimeError("failed to restore default mapper")

    follower_rows = []
    og_multipliers = {
        "index_dip": 0.8917,
        "middle_dip": 0.8917,
        "ring_dip": 0.8917,
        "pinky_dip": 0.8917,
        "thumb_ip": 1.1619,
    }
    for fit in provenance["follower_mimic_candidates"]:
        joint = fit["joint"]
        archive = provenance["joint_archives"][joint]
        summary = read_json(ROOT / archive["summary"])
        follower_lut = summary["physical_lut_urdf_local_q"]
        parent_summary = read_json(
            ROOT / provenance["joint_archives"][fit["source"]]["summary"]
        )
        parent_lut = parent_summary["physical_lut_urdf_local_q"]

        def interpolate(raw: float) -> float:
            xs = parent_lut["raw"]
            ys = parent_lut["q_urdf_local_rad"]
            if raw <= xs[0]:
                return float(ys[0])
            if raw >= xs[-1]:
                return float(ys[-1])
            for index, (left, right) in enumerate(zip(xs, xs[1:])):
                if left <= raw <= right:
                    fraction = (raw - left) / (right - left)
                    return float(ys[index] + fraction * (ys[index + 1] - ys[index]))
            raise AssertionError("unreachable")

        parent_q = [interpolate(raw) for raw in follower_lut["raw"]]
        follower_q = follower_lut["q_urdf_local_rad"]

        def rmse(multiplier: float, offset: float) -> float:
            residuals = [
                y - (multiplier * x + offset)
                for x, y in zip(parent_q, follower_q)
            ]
            return math.sqrt(sum(value * value for value in residuals) / len(residuals))

        candidate = fit["affine_fit"]
        candidate_rmse = rmse(candidate["multiplier"], candidate["offset"])
        og_rmse = rmse(og_multipliers[joint], 0.0)
        follower_rows.append(
            {
                "joint": joint,
                "source": fit["source"],
                "candidate_affine_rmse_rad": candidate_rmse,
                "og_mimic_multiplier": og_multipliers[joint],
                "og_mimic_rmse_rad": og_rmse,
                "rmse_reduction_fraction": (og_rmse - candidate_rmse) / og_rmse,
            }
        )

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "static_validation_pass_hardware_visual_ab_pending",
        "candidate": str(args.candidate.relative_to(ROOT)),
        "candidate_sha256": candidate_digest,
        "provenance": str(args.provenance.relative_to(ROOT)),
        "provenance_sha256": sha256(args.provenance),
        "production_og_urdf_sha256": sha256(PRODUCTION_URDF),
        "checks": {
            "candidate_not_active_at_process_start": True,
            "candidate_build_did_not_mutate_active_table": True,
            "active_joint_count": len(specs),
            "unique_non_reserved_slot_count": len(set(slots)),
            "all_active_joints_have_lut": True,
            "default_mapper_restored": True,
            "max_integer_raw_roundtrip_error": global_raw_error,
            "max_q_roundtrip_error_after_uint8_rad": global_q_error,
        },
        "active_joints": rows,
        "follower_mimic_diagnostics": follower_rows,
        "remaining_gates": [
            "same-view discrete-pose Isaac versus real-hand A/B",
            "short continuous-motion video A/B",
            "manual review of follower mimic candidate before sim URDF change",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["checks"], indent=2))
    print(f"wrote {args.output.relative_to(ROOT)} sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
