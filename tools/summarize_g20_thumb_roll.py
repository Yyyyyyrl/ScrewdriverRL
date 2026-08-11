#!/usr/bin/env python3
"""Summarize the accepted G20 thumb-CMC-roll sweep into a candidate LUT.

The camera measures the azimuth of the thumb marker pair.  In that image
coordinate convention the azimuth increases with raw.  The Linker mapper requires physical radians to decrease with raw. The camera-relative curve is ``raw58_azimuth - measured_azimuth``; it is then shifted so the highest accepted stable readback is 0 rad, matching the non-negative URDF and policy contract.

All motion points are keyed by stable encoder readback, not command raw.  The
LUT is the readback-domain midline of the increasing and decreasing passes
where both exist, with the single available pass retained at either end.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np


ANGLE_KEY = "thumb_cmc_roll_deg_3d"
PALM_KEY = "palm_heading_deg_2d"
ROLL_SLOT = 5
ZERO_REPEAT_GATE_DEG = 0.05
ZERO_RETURN_GATE_DEG = 1.5
HYSTERESIS_GATE_DEG = 3.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_value(path: Path, key: str = ANGLE_KEY) -> float:
    return float(read_json(path)["angle_summary_deg"][key]["median"])


def two_capture_mean(directory: Path, pattern: str) -> tuple[float, float, list[Path]]:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 2:
        raise ValueError(f"expected two captures matching {directory / pattern}, got {len(paths)}")
    values = [camera_value(path) for path in paths]
    return fmean(values), abs(values[0] - values[1]), paths


def point_from_files(
    root: Path,
    motion_path: Path,
    camera_path: Path,
    zero_azimuth_deg: float,
) -> dict[str, Any]:
    motion = read_json(motion_path)
    camera = read_json(camera_path)
    readback = int(round(motion["result"]["settled_state20_median"][ROLL_SLOT]))
    command = int(motion["approved_scope"]["target_raw"])
    measured = float(camera["angle_summary_deg"][ANGLE_KEY]["median"])
    blocks = [float(v) for v in camera["block_medians_deg"][ANGLE_KEY]]
    palm = float(camera["angle_summary_deg"][PALM_KEY]["median"])
    return {
        "command_raw": command,
        "stable_readback_raw": readback,
        "undershoot_raw": readback - command,
        "measured_azimuth_deg": measured,
        "physical_deg": zero_azimuth_deg - measured,
        "physical_rad": math.radians(zero_azimuth_deg - measured),
        "camera_usable_frames": int(camera["capture"]["usable_frames"]),
        "camera_failure_fraction": float(camera["capture"]["failure_fraction"]),
        "block_median_range_deg": max(blocks) - min(blocks),
        "palm_heading_deg": palm,
        "motion_source": str(motion_path.relative_to(root)),
        "camera_source": str(camera_path.relative_to(root)),
    }


def collect_dirs(
    root: Path,
    patterns: Iterable[str],
    zero_azimuth_deg: float,
    inputs: list[Path],
) -> dict[int, dict[str, Any]]:
    points: dict[int, dict[str, Any]] = {}
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in root.glob(pattern) if path.is_dir())
    for directory in sorted(directories):
        motions = sorted(directory.glob("motion_*.json"))
        cameras = sorted(directory.glob("*summary.json"))
        if len(motions) != 1 or len(cameras) != 1:
            raise ValueError(
                f"{directory}: expected exactly one motion and one camera summary, "
                f"got {len(motions)} and {len(cameras)}"
            )
        point = point_from_files(root, motions[0], cameras[0], zero_azimuth_deg)
        readback = point["stable_readback_raw"]
        if readback in points:
            raise ValueError(f"duplicate stable readback raw {readback} in accepted pass")
        points[readback] = point
        inputs.extend((motions[0], cameras[0]))
    return points


def interpolate(points: dict[int, dict[str, Any]], raw: int) -> float | None:
    keys = sorted(points)
    if not keys or raw < keys[0] or raw > keys[-1]:
        return None
    return float(np.interp(raw, keys, [points[key]["physical_deg"] for key in keys]))


def heldout_max_abs_error_rad(knots: list[tuple[int, float]]) -> float:
    worst = 0.0
    for index in range(1, len(knots) - 1):
        x0, y0 = knots[index - 1]
        x1, y1 = knots[index]
        x2, y2 = knots[index + 1]
        predicted = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0)
        worst = max(worst, abs(predicted - y1))
    return worst


def validate_pass(points: dict[int, dict[str, Any]], name: str) -> None:
    keys = sorted(points)
    if len(keys) < 3:
        raise ValueError(f"{name}: too few points")
    values = [points[key]["physical_deg"] for key in keys]
    if any(values[index] <= values[index + 1] for index in range(len(values) - 1)):
        raise ValueError(f"{name}: physical angle is not strictly decreasing with raw")
    for point in points.values():
        if point["camera_usable_frames"] < 162:
            raise ValueError(f"{name}: fewer than 90% usable camera frames")
        if point["camera_failure_fraction"] > 0.10:
            raise ValueError(f"{name}: camera failure gate exceeded")
        if point["block_median_range_deg"] > 0.60:
            raise ValueError(f"{name}: block stability gate exceeded")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    inputs: list[Path] = []

    reference_dir = root
    zero, zero_repeat, zero_paths = two_capture_mean(
        reference_dir, "reference_raw058_camera_*_fixed_exp_180f_summary.json"
    )
    inputs.extend(zero_paths)
    palm_zero = fmean(camera_value(path, PALM_KEY) for path in zero_paths)

    direction_motion = root / "direction_motion_raw058_to_raw072.json"
    direction_camera = root / "direction_raw071_camera_fixed_exp_180f_summary.json"
    forward = {
        58: {
            "command_raw": 58,
            "stable_readback_raw": 58,
            "undershoot_raw": 0,
            "measured_azimuth_deg": zero,
            "physical_deg": 0.0,
            "physical_rad": 0.0,
            "camera_usable_frames": 360,
            "camera_failure_fraction": 0.0,
            "block_median_range_deg": max(
                max(read_json(path)["block_medians_deg"][ANGLE_KEY])
                - min(read_json(path)["block_medians_deg"][ANGLE_KEY])
                for path in zero_paths
            ),
            "palm_heading_deg": palm_zero,
            "motion_source": "reference only",
            "camera_source": str(reference_dir.relative_to(root)),
        }
    }
    direction = point_from_files(root, direction_motion, direction_camera, zero)
    forward[direction["stable_readback_raw"]] = direction
    inputs.extend((direction_motion, direction_camera))
    forward.update(
        collect_dirs(
            root,
            (
                "forward_up_cmd*",
                "forward_up_resume_cmd*",
                "forward_up_tail_cmd*",
            ),
            zero,
            inputs,
        )
    )
    backward = collect_dirs(root, ("reverse_down_cmd*",), zero, inputs)
    validate_pass(forward, "forward")
    validate_pass(backward, "reverse")

    # At each readback knot use the mean of interpolated forward/reverse curves
    # where both are defined.  End knots retain the only pass that reached them.
    rows: list[dict[str, Any]] = []
    for raw in sorted(set(forward) | set(backward)):
        fwd = interpolate(forward, raw)
        rev = interpolate(backward, raw)
        if fwd is not None and rev is not None:
            physical_deg = fmean((fwd, rev))
            source = "readback_domain_midline"
        elif fwd is not None:
            physical_deg = fwd
            source = "forward_only_endpoint"
        elif rev is not None:
            physical_deg = rev
            source = "reverse_only_endpoint"
        else:  # pragma: no cover
            raise AssertionError(raw)
        rows.append(
            {
                "stable_readback_raw": raw,
                "physical_deg": physical_deg,
                "physical_rad": math.radians(physical_deg),
                "lut_source": source,
            }
        )
    # raw58 is a camera-relative anchor. The URDF/policy contract is
    # non-negative, so the highest accepted stable readback is semantic 0 rad.
    relative_at_semantic_zero_deg = rows[-1]["physical_deg"]
    for row in rows:
        row["relative_deg_from_raw58"] = row["physical_deg"]
        row["physical_deg"] -= relative_at_semantic_zero_deg
        row["physical_rad"] = math.radians(row["physical_deg"])
    if any(
        rows[index]["physical_rad"] <= rows[index + 1]["physical_rad"]
        for index in range(len(rows) - 1)
    ):
        raise ValueError("midline LUT is not strictly decreasing")

    overlap = sorted(raw for raw in forward if min(backward) <= raw <= max(backward))
    differences = {
        str(raw): forward[raw]["physical_deg"] - float(interpolate(backward, raw))
        for raw in overlap
    }
    hysteresis_max = max(abs(value) for value in differences.values())

    final_dir = root / "final_zero_raw058"
    final_zero, final_repeat, final_paths = two_capture_mean(
        final_dir, "final_zero_*_fixed_exp_180f_summary.json"
    )
    inputs.extend(final_paths)
    final_palm = fmean(camera_value(path, PALM_KEY) for path in final_paths)
    final_snapshot_path = root / "final_readonly/sdk_snapshot_raw058.json"
    final_snapshot = read_json(final_snapshot_path)
    inputs.append(final_snapshot_path)
    final_states = [sample["state20"] for sample in final_snapshot["samples"]]
    final_faults = [sample["faults20"] for sample in final_snapshot["samples"]]
    final_health_ok = (
        len(final_states) == 20
        and all(state is not None and int(round(state[ROLL_SLOT])) == 58 for state in final_states)
        and all(faults is not None and not any(faults) for faults in final_faults)
    )
    if not final_health_ok:
        raise ValueError("final read-only health snapshot did not pass")

    motion_paths = sorted(root.glob("**/motion_*.json"))
    motion_records = [read_json(path) for path in motion_paths]
    inputs.extend(motion_paths)
    motion_fault_free = all(
        record.get("motion_sent") and record.get("result", {}).get("fault_free")
        for record in motion_records
    )
    max_temperature_c = max(
        value
        for record in motion_records
        for sample in record.get("post_command_samples", [])
        for value in sample.get("temperature20", [])
    )
    if not motion_fault_free:
        raise ValueError("one or more white-marker motion records are not fault-free")

    knot_pairs = [(row["stable_readback_raw"], row["physical_rad"]) for row in rows]
    heldout = heldout_max_abs_error_rad(knot_pairs)
    result = {
        "schema_version": 1,
        "status": "complete_candidate_only",
        "joint": "thumb_cmc_roll",
        "raw20_slot": ROLL_SLOT,
        "identity": {
            "hand_serial": "LHT20-010-415-L-B-1-D",
            "hand": "left G20",
            "sdk_version": "3.1.0",
            "embedded_version": "1.0.7",
            "camera": "RealSense D435 143322073091",
        },
        "measurement_contract": {
            "primary_value": "3D azimuth of the vector between the two blue thumb markers",
            "fixed_reference": "white horizontal palm marker; heading is a drift witness",
            "relative_sign": "relative_deg = raw58_reference_azimuth_deg - measured_azimuth_deg",
            "camera_reference_readback_raw": 58,
            "semantic_zero_readback_raw": rows[-1]["stable_readback_raw"],
            "semantic_zero_rule": "highest accepted stable readback is 0 rad",
            "semantic_range_contract": "thumb_cmc_roll is non-negative in the URDF and policy",
            "points_keyed_by_stable_readback_not_command": True,
            "holds_raw": {"thumb_cmc_yaw": 125, "thumb_cmc_pitch": 247, "thumb_mcp": 254},
            "settle_tolerance_raw": 4,
            "settle_tolerance_reason": "observed fault-free short-side undershoot reached -4 raw twice",
        },
        "camera_reference_raw058": {
            "initial_azimuth_deg": zero,
            "initial_repeat_abs_deg": zero_repeat,
            "initial_repeat_gate_deg": ZERO_REPEAT_GATE_DEG,
            "initial_repeat_within_gate": zero_repeat < ZERO_REPEAT_GATE_DEG,
            "final_azimuth_deg": final_zero,
            "final_repeat_abs_deg": final_repeat,
            "return_drift_deg": final_zero - zero,
            "return_drift_gate_deg": ZERO_RETURN_GATE_DEG,
            "return_drift_within_gate": abs(final_zero - zero) <= ZERO_RETURN_GATE_DEG,
            "palm_heading_drift_deg": final_palm - palm_zero,
        },
        "range": {
            "min_stable_readback_raw": rows[0]["stable_readback_raw"],
            "max_stable_readback_raw": rows[-1]["stable_readback_raw"],
            "physical_at_min_raw_deg": rows[0]["physical_deg"],
            "physical_at_max_raw_deg": rows[-1]["physical_deg"],
            "full_span_deg": rows[0]["physical_deg"] - rows[-1]["physical_deg"],
            "full_span_rad": rows[0]["physical_rad"] - rows[-1]["physical_rad"],
            "raw_0_and_255_not_claimed": True,
            "end_reason": "remaining command distance was below the measured 8-raw deadband floor",
        },
        "direction_probe": {
            "stable_readback_raw": direction["stable_readback_raw"],
            "measured_azimuth_delta_deg": direction["measured_azimuth_deg"] - zero,
            "relative_physical_delta_deg": direction["physical_deg"],
        },
        "hysteresis_readback_domain": {
            "max_abs_deg": hysteresis_max,
            "max_abs_rad": math.radians(hysteresis_max),
            "gate_deg": HYSTERESIS_GATE_DEG,
            "within_gate": hysteresis_max <= HYSTERESIS_GATE_DEG,
            "forward_minus_interpolated_reverse_deg": differences,
        },
        "heldout_piecewise_linear_max_abs_error_rad": heldout,
        "heldout_piecewise_linear_max_abs_error_deg": math.degrees(heldout),
        "final_readonly": {
            "path": str(final_snapshot_path.relative_to(root)),
            "samples": len(final_states),
            "all_raw58": all(int(round(state[ROLL_SLOT])) == 58 for state in final_states),
            "all_fault_free": all(not any(faults) for faults in final_faults),
        },
        "motion_health": {
            "records": len(motion_records),
            "sent": sum(bool(record.get("motion_sent")) for record in motion_records),
            "fault_free": sum(
                bool(record.get("result", {}).get("fault_free"))
                for record in motion_records
            ),
            "max_observed_temperature_c": max_temperature_c,
            "settle_tolerance_evidence": {
                str(path.relative_to(root)): record.get("result", {}).get("target_error_raw")
                for path, record in zip(motion_paths, motion_records)
                if not record.get("result", {}).get("settled")
            },
        },
        "rejected_evidence": [
            "All measurements made before the loose horizontal blue palm marker was replaced are rejected.",
            "The old-marker zero repeat differed by about 0.457 deg; no old-marker point enters this LUT.",
        ],
        "forward_points": [forward[key] for key in sorted(forward)],
        "reverse_points": [backward[key] for key in sorted(backward)],
        "lut_points": rows,
        "input_sha256": {
            str(path.relative_to(root)): sha256(path) for path in sorted(set(inputs))
        },
    }
    write_json(Path(f"{args.out_prefix}_summary.json"), result)
    write_json(
        Path(f"{args.out_prefix}_lut.json"),
        {
            "thumb_cmc_roll": {
                "flip": False,
                "lo": rows[-1]["physical_rad"],
                "hi": rows[0]["physical_rad"],
                "physical_lut": {
                    "raw": [row["stable_readback_raw"] for row in rows],
                    "rad": [row["physical_rad"] for row in rows],
                },
            }
        },
    )
    csv_path = Path(f"{args.out_prefix}_points.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "knots": len(rows),
                "range_deg": result["range"]["full_span_deg"],
                "hysteresis_max_deg": hysteresis_max,
                "heldout_max_deg": math.degrees(heldout),
                "zero_return_drift_deg": final_zero - zero,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
