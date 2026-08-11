#!/usr/bin/env python3
"""Prepare formal G20 photos and an explicit URDF-local-q matching grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEMANTIC_ORDER = (
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
RAW_RE = re.compile(r"raw[_-]?(\d{1,3})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", required=True, choices=SEMANTIC_ORDER)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base-pose", type=Path, required=True)
    parser.add_argument("--render-urdf", type=Path, required=True)
    parser.add_argument("--production-urdf", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--q-lower", type=float, required=True)
    parser.add_argument("--q-upper", type=float, required=True)
    parser.add_argument("--q-step", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_single_pose(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if len(payload) != 1:
            raise ValueError("--base-pose must contain exactly one named pose")
        values = next(iter(payload.values()))
    else:
        values = payload
    if not isinstance(values, list) or len(values) != len(SEMANTIC_ORDER):
        raise ValueError("base pose must resolve to 16 semantic joint values")
    return [float(value) for value in values]


def joint_limit(urdf: Path, joint_name: str) -> tuple[float, float]:
    root = ET.parse(urdf).getroot()
    joint = next(
        (item for item in root.findall("joint") if item.get("name") == joint_name),
        None,
    )
    if joint is None:
        raise ValueError(f"{joint_name} not found in {urdf}")
    limit = joint.find("limit")
    if limit is None:
        raise ValueError(f"{joint_name} has no URDF limit")
    return float(limit.get("lower")), float(limit.get("upper"))


def q_grid(lower: float, upper: float, step: float) -> list[float]:
    if upper <= lower:
        raise ValueError("--q-upper must exceed --q-lower")
    if step <= 0.0:
        raise ValueError("--q-step must be positive")
    count = int(math.floor((upper - lower) / step + 1.0e-9))
    values = [round(lower + index * step, 10) for index in range(count + 1)]
    if values[-1] < upper - 1.0e-9:
        values.append(float(upper))
    else:
        values[-1] = float(upper)
    return values


def color_from_summary(summary_path: Path) -> Path:
    suffix = "_summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"unexpected summary filename: {summary_path}")
    return summary_path.with_name(
        summary_path.name.removesuffix(suffix) + "_color.png"
    )


def sample_row(
    *,
    direction: str,
    command_raw: int,
    stable_raw: int,
    summary_path: Path,
    old_measurement_rad: float | None,
    suffix: str = "",
) -> dict[str, Any]:
    sample_id = (
        f"{direction}_cmd{command_raw:03d}_read{stable_raw:03d}{suffix}"
    )
    return {
        "sample_id": sample_id,
        "direction": direction,
        "command_raw": command_raw,
        "stable_readback_raw": stable_raw,
        "real_color": str(color_from_summary(summary_path).resolve()),
        "camera_summary": str(summary_path.resolve()),
        "old_visual_measurement_rad_context_only": old_measurement_rad,
        "match_status": "unreviewed",
    }


def _camera_summary_for_anchor(session: Path, anchor: dict[str, Any]) -> Path:
    source = anchor.get("camera_source")
    if source and source != ".":
        return session / str(source)
    record = Path(str(anchor["record"]))
    return session / record.parent / "camera_fixed_exp_180f_summary.json"


def _raw_from_token(token: str) -> int | None:
    match = RAW_RE.search(token)
    if match is None:
        return None
    raw = int(match.group(1))
    return raw if 0 <= raw <= 255 else None


def build_anchor_samples(
    session: Path, summary: dict[str, Any]
) -> list[dict[str, Any]]:
    anchors = summary.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("summary does not contain a non-empty anchors list")

    samples: list[dict[str, Any]] = []
    zero_runs = summary.get("zero_runs", [])
    for anchor in anchors:
        command = int(round(float(anchor["command_raw"])))
        stable = int(round(float(anchor["stable_readback_raw"])))
        old_rad = (
            float(anchor["physical_rad"])
            if anchor.get("physical_rad") is not None
            else None
        )
        if command == 255 and zero_runs:
            for index, run in enumerate(zero_runs):
                prefix = str(run["prefix"])
                summary_path = session / f"{prefix}_summary.json"
                samples.append(
                    sample_row(
                        direction="forward_reference",
                        command_raw=command,
                        stable_raw=stable,
                        summary_path=summary_path,
                        old_measurement_rad=old_rad,
                        suffix=f"_{chr(ord('a') + index)}",
                    )
                )
            continue
        samples.append(
            sample_row(
                direction="forward",
                command_raw=command,
                stable_raw=stable,
                summary_path=_camera_summary_for_anchor(session, anchor),
                old_measurement_rad=old_rad,
            )
        )

    for key, value in summary.items():
        if not key.startswith("return_") or not isinstance(value, dict):
            continue
        prefix = value.get("prefix")
        raw = _raw_from_token(key)
        if prefix is None or raw is None:
            continue
        samples.append(
            sample_row(
                direction="reverse",
                command_raw=raw,
                stable_raw=raw,
                summary_path=session / f"{prefix}_summary.json",
                old_measurement_rad=None,
            )
        )

    missing = [
        path
        for sample in samples
        for path in (
            Path(sample["real_color"]),
            Path(sample["camera_summary"]),
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing formal evidence:\n" + "\n".join(map(str, missing))
        )
    return sorted(
        samples,
        key=lambda item: (
            -int(item["stable_readback_raw"]),
            str(item["direction"]),
            str(item["sample_id"]),
        ),
    )


def build_point_samples(
    session: Path, summary: dict[str, Any], joint: str
) -> list[dict[str, Any]]:
    points = summary.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("summary does not contain a non-empty points list")

    stable_by_command: dict[int, int] = {}
    hardware_return = summary.get("hardware_return", {})
    for record in hardware_return.get("records", []):
        if "correct_exposure_down" not in str(record.get("record", "")):
            continue
        stable_by_command[int(record["target_raw"])] = int(
            round(float(record["stable_readback_raw"]))
        )
    if hardware_return.get("final_stable_readback_raw") is not None:
        stable_by_command[int(hardware_return["final_target_raw"])] = int(
            round(float(hardware_return["final_stable_readback_raw"]))
        )

    if joint == "thumb_cmc_yaw":
        role = "yaw"
        measurement_keys = (
            f"{joint}_rad",
            "yaw_rad",
            "physical_rad",
        )
    elif joint.endswith("_mcp_roll"):
        role = "roll"
        measurement_keys = (
            f"{joint}_rad",
            "abduction_rad",
            "physical_rad",
        )
    else:
        role = "pitch" if joint.endswith("_mcp_pitch") else "pip"
        measurement_keys = (
            f"{joint}_rad",
            "pip_physical_rad" if joint.endswith("_pip") else "physical_rad",
        )
    stable_keys = (
        f"{role}_stable_readback_raw",
        "stable_readback_raw",
    )
    camera_keys = (
        f"{role}_camera_source",
        "camera_summary",
        "source",
    )

    def first_value(point: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if point.get(key) is not None:
                return point[key]
        raise KeyError(f"none of {keys} is present in calibration point")

    samples: list[dict[str, Any]] = []
    for point in points:
        command = int(point["command_raw"])
        measurement_value = next(
            (
                point[key] for key in measurement_keys
                if point.get(key) is not None
            ),
            None,
        )
        if measurement_value is None:
            print(
                f"[prepare] skip command_raw={command}: target measurement is "
                f"null for {joint} (quality={point.get(f'{role}_camera_quality')})",
                flush=True,
            )
            continue
        stable_source = next(
            (
                point[key] for key in stable_keys
                if point.get(key) is not None
            ),
            stable_by_command.get(command, command),
        )
        stable = int(round(float(stable_source)))
        old_measurement = float(measurement_value)
        camera_sources = [
            token.strip()
            for token in str(first_value(point, camera_keys)).split(",")
            if token.strip()
        ]
        camera_summary_paths: list[Path] = []
        for camera_source in camera_sources:
            source_path = session / camera_source
            if source_path.is_dir():
                expanded = sorted(
                    source_path.glob("camera*fixed_exp_180f_summary.json")
                )
                if not expanded:
                    raise FileNotFoundError(
                        f"no formal camera summaries in {source_path}"
                    )
                camera_summary_paths.extend(expanded)
            else:
                camera_summary_paths.append(source_path)
        for source_index, camera_summary_path in enumerate(camera_summary_paths):
            suffix = f"_{point.get('role', 'sample').replace('-', '_')}"
            if len(camera_summary_paths) > 1:
                suffix += f"_{chr(ord('a') + source_index)}"
            item = sample_row(
                direction=(
                    "forward_reference"
                    if len(camera_summary_paths) > 1
                    else "forward"
                ),
                command_raw=command,
                stable_raw=stable,
                summary_path=camera_summary_path,
                old_measurement_rad=old_measurement,
                suffix=suffix,
            )
            item["evidence_role"] = point.get(
                "role", "formal-full-sweep"
            )
            item["stable_readback_raw_source"] = float(stable_source)
            samples.append(item)

    limit_points = summary.get(
        "physical_limit_characterization", {}
    ).get("points", [])
    for point in limit_points:
        stable = int(point["readback_raw"])
        command = 0 if point.get("fault_endpoint") else stable
        item = sample_row(
            direction="forward_limit_characterization",
            command_raw=command,
            stable_raw=stable,
            summary_path=session / str(point["camera_summary"]),
            old_measurement_rad=float(first_value(point, measurement_keys)),
            suffix="_fault_endpoint" if point.get("fault_endpoint") else "",
        )
        item["evidence_role"] = "physical-limit-characterization"
        item["fault_endpoint"] = bool(point.get("fault_endpoint"))
        samples.append(item)

    missing = [
        path
        for sample in samples
        for path in (
            Path(sample["real_color"]),
            Path(sample["camera_summary"]),
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing formal evidence:\n" + "\n".join(map(str, missing))
        )
    return sorted(
        samples,
        key=lambda item: (
            -int(item["stable_readback_raw"]),
            str(item["direction"]),
            str(item["sample_id"]),
        ),
    )


def normalize_thumb_combined_summary(
    session: Path, summary: dict[str, Any], joint: str
) -> dict[str, Any]:
    """Expose one joint from the archived combined thumb sweep as points."""
    if joint not in ("thumb_cmc_pitch", "thumb_mcp"):
        return summary
    joint_summary = summary.get(joint)
    if not isinstance(joint_summary, dict):
        return summary
    points = joint_summary.get("points")
    if not isinstance(points, list) or not points:
        return summary

    zero_sources = summary.get("formal_zero_reference", {}).get("sources", [])
    if not zero_sources:
        raise ValueError(
            "combined thumb summary has no formal zero-reference source"
        )
    zero_samples = Path(str(zero_sources[0]))
    zero_summary = zero_samples.with_name(
        zero_samples.name.replace("_samples.csv", "_summary.json")
    )
    prefix = "pitch" if joint == "thumb_cmc_pitch" else "mcp"
    normalized_points: list[dict[str, Any]] = []
    for source_point in points:
        point = dict(source_point)
        raw = int(point["command_raw"])
        camera_summary = (
            zero_summary
            if raw == 255
            else session
            / f"{prefix}_raw{raw:03d}"
            / "camera_fixed_exp_180f_summary.json"
        )
        point["camera_summary"] = str(camera_summary)
        point["physical_rad"] = float(point[f"{joint}_rad"])
        point["role"] = "formal-full-sweep"
        normalized_points.append(point)
    return {
        "joint": joint,
        "points": normalized_points,
        "combined_summary_source": summary,
    }


def main() -> int:
    args = parse_args()
    for path in (
        args.summary,
        args.base_pose,
        args.render_urdf,
        args.production_urdf,
        args.camera,
        args.intrinsics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.session.is_dir():
        raise NotADirectoryError(args.session)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary = normalize_thumb_combined_summary(
        args.session, summary, args.joint
    )
    if summary.get("joint") not in (None, args.joint):
        raise ValueError(
            f"summary joint {summary.get('joint')!r} != {args.joint!r}"
        )
    if isinstance(summary.get("anchors"), list):
        samples = build_anchor_samples(args.session, summary)
    elif isinstance(summary.get("points"), list):
        samples = build_point_samples(args.session, summary, args.joint)
    else:
        raise ValueError("unsupported summary schema: need anchors or points")

    production_lower, production_upper = joint_limit(
        args.production_urdf, args.joint
    )
    render_lower, render_upper = joint_limit(args.render_urdf, args.joint)
    tolerance = 1.0e-9
    if args.q_lower < render_lower - tolerance:
        raise ValueError("q grid lower bound is outside render URDF limit")
    if args.q_upper > render_upper + tolerance:
        raise ValueError("q grid upper bound is outside render URDF limit")

    values = q_grid(args.q_lower, args.q_upper, args.q_step)
    base = load_single_pose(args.base_pose)
    target_index = SEMANTIC_ORDER.index(args.joint)
    poses: dict[str, list[float]] = {}
    candidates: list[dict[str, Any]] = []
    for index, q in enumerate(values):
        sign = "p" if q >= 0.0 else "m"
        token = f"{abs(q):.4f}".replace(".", "p")
        name = f"{args.joint}_q_{index:03d}_{sign}{token}"
        pose = list(base)
        pose[target_index] = q
        poses[name] = pose
        candidates.append(
            {
                "index": index,
                "name": name,
                "q_urdf_rad": q,
                "q_urdf_deg": math.degrees(q),
                "outside_production_og_limit": (
                    q < production_lower - tolerance
                    or q > production_upper + tolerance
                ),
            }
        )

    poses_path = args.out_dir / "poses_urdf_local_q.json"
    dataset_path = args.out_dir / "manual_match_dataset.json"
    poses_path.write_text(
        json.dumps(poses, indent=2) + "\n", encoding="utf-8"
    )
    dataset = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "formal_photo_to_og_urdf_local_q_manual_registration",
        "status": "unreviewed_candidate_only",
        "joint": args.joint,
        "coordinate_contract": {
            "ground_truth_coordinate": "OG URDF local joint q in radians",
            "raw_coordinate": "physical stable SDK readback [0,255]",
            "old_visual_measurement_role": (
                "context only; hidden from the matcher to avoid anchoring bias"
            ),
            "selection_rule": (
                "human chooses the same-view OG-URDF render that best matches "
                "each formal D435 photo"
            ),
        },
        "production_urdf": str(args.production_urdf.resolve()),
        "production_urdf_sha256": sha256(args.production_urdf),
        "production_urdf_limit_rad": [production_lower, production_upper],
        "urdf_limit_rad": [production_lower, production_upper],
        "render_urdf": str(args.render_urdf.resolve()),
        "render_urdf_sha256": sha256(args.render_urdf),
        "urdf": str(args.render_urdf.resolve()),
        "urdf_sha256": sha256(args.render_urdf),
        "render_urdf_limit_rad": [render_lower, render_upper],
        "diagnostic_extended_limit_only": (
            render_lower != production_lower or render_upper != production_upper
        ),
        "camera": str(args.camera.resolve()),
        "camera_sha256": sha256(args.camera),
        "intrinsics": str(args.intrinsics.resolve()),
        "intrinsics_sha256": sha256(args.intrinsics),
        "base_pose": str(args.base_pose.resolve()),
        "base_pose_sha256": sha256(args.base_pose),
        "poses_json": str(poses_path.resolve()),
        "semantic_order": list(SEMANTIC_ORDER),
        "target_semantic_index": target_index,
        "q_grid_rad": {
            "lower": args.q_lower,
            "upper": args.q_upper,
            "step": args.q_step,
            "margin_below_production_lower": production_lower - args.q_lower,
            "margin_above_production_upper": args.q_upper - production_upper,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "sample_count": len(samples),
        "samples": samples,
        "source_summary": str(args.summary.resolve()),
        "source_summary_sha256": sha256(args.summary),
    }
    dataset_path.write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "poses": str(poses_path),
                "samples": len(samples),
                "candidates": len(candidates),
                "q_grid_rad": [args.q_lower, args.q_upper, args.q_step],
                "production_limit_rad": [production_lower, production_upper],
                "margin_rad": [
                    production_lower - args.q_lower,
                    args.q_upper - production_upper,
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
