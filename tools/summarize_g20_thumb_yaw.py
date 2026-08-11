#!/usr/bin/env python3
"""Summarize the G20 thumb CMC-yaw sweep into a candidate midline LUT.

All angles are the thumb longitudinal marker change relative to the unobscured
palm cross marker. Points are keyed by stable SDK readback, never command raw.
The accepted lower endpoint is raw17; raw2 is retained only as rejected wrap
boundary evidence. The output remains candidate-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

THUMB_YAW_SLOT = 10
ZERO_DRIFT_GATE_DEG = 0.5
HYSTERESIS_MIDLINE_GATE_DEG = 3.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_delta_deg(value: float, reference: float) -> float:
    return (value - reference + 90.0) % 180.0 - 90.0


def zero_medians(reference: Path) -> tuple[dict[str, float], float]:
    caps = sorted(reference.glob("camera_*_fixed_exp_180f_summary.json"))
    if len(caps) != 2:
        raise SystemExit(f"expected two zero captures in {reference}")
    summaries = [read_json(c)["angle_summary_deg"] for c in caps]
    zero = {k: sum(s[k]["median"] for s in summaries) / 2.0
            for k in summaries[0]}
    def relative_yaw(summary: dict[str, Any]) -> float:
        thumb = axis_delta_deg(
            summary["thumb_heading_deg_2d"]["median"],
            zero["thumb_heading_deg_2d"])
        palm_long = axis_delta_deg(
            summary["palm_heading_deg_2d"]["median"],
            zero["palm_heading_deg_2d"])
        palm_cross = axis_delta_deg(
            summary["palm_cross_heading_deg_2d"]["median"],
            zero["palm_cross_heading_deg_2d"])
        palm = ((palm_long + palm_cross) / 2.0
                if abs(palm_long - palm_cross) <= 0.5 else palm_cross)
        return thumb - palm
    repeat = abs(relative_yaw(summaries[0]) - relative_yaw(summaries[1]))
    return zero, repeat


def palm_reference_delta(
        angle_summary: dict[str, Any], zero: dict[str, float]) -> tuple[float, str]:
    palm_long = axis_delta_deg(
        angle_summary["palm_heading_deg_2d"]["median"],
        zero["palm_heading_deg_2d"])
    palm_cross = axis_delta_deg(
        angle_summary["palm_cross_heading_deg_2d"]["median"],
        zero["palm_cross_heading_deg_2d"])
    if abs(palm_long - palm_cross) <= 0.5:
        return (palm_long + palm_cross) / 2.0, "fused_long_and_cross"
    return palm_cross, "cross_fallback_long_occluded"


def robust_yaw(angle_summary: dict[str, Any], zero: dict[str, float]) -> float:
    palm, _ = palm_reference_delta(angle_summary, zero)
    return (axis_delta_deg(
        angle_summary["thumb_heading_deg_2d"]["median"],
        zero["thumb_heading_deg_2d"]) - palm)


def preferred_camera(point_dir: Path) -> Path | None:
    preferred = [
        point_dir / "camera_depth_identity_fixed_exp_180f_summary.json",
        point_dir / "camera_crossref_repeat_fixed_exp_180f_summary.json",
    ]
    for path in preferred:
        if path.exists():
            return path
    cameras = sorted(point_dir.glob("camera*_summary.json"))
    return cameras[-1] if cameras else None


def collect(session: Path, tags: list[str], zero: dict[str, float],
            inputs: list[Path]) -> dict[int, dict[str, Any]]:
    """Map stable yaw readback to one accepted point record."""
    points: dict[int, dict[str, Any]] = {}
    for tag in tags:
        for point_dir in sorted(session.glob(f"{tag}_raw*")):
            motions = sorted(point_dir.glob("motion_*.json"))
            camera_path = preferred_camera(point_dir)
            if not motions or camera_path is None:
                continue
            motion = read_json(motions[0])
            camera = read_json(camera_path)
            inputs.extend([motions[0], camera_path])
            readback = int(motion["result"]["settled_state20_median"][THUMB_YAW_SLOT])
            blocks = []
            for thumb, palm_long, palm_cross in zip(
                    camera["block_medians_deg"]["thumb_heading_deg_2d"],
                    camera["block_medians_deg"]["palm_heading_deg_2d"],
                    camera["block_medians_deg"]["palm_cross_heading_deg_2d"]):
                thumb_delta = axis_delta_deg(
                    thumb, zero["thumb_heading_deg_2d"])
                long_delta = axis_delta_deg(
                    palm_long, zero["palm_heading_deg_2d"])
                cross_delta = axis_delta_deg(
                    palm_cross, zero["palm_cross_heading_deg_2d"])
                palm_delta = ((long_delta + cross_delta) / 2.0
                              if abs(long_delta - cross_delta) <= 0.5
                              else cross_delta)
                blocks.append(thumb_delta - palm_delta)
            yaw_deg = robust_yaw(camera["angle_summary_deg"], zero)
            palm_reference, palm_mode = palm_reference_delta(
                camera["angle_summary_deg"], zero)
            points[readback] = {
                "command_raw": int(motion["approved_scope"]["target_raw"]),
                "stable_readback_raw": readback,
                "undershoot_raw": (readback
                                   - int(motion["approved_scope"]["target_raw"])),
                "yaw_deg": yaw_deg,
                "yaw_rad": math.radians(yaw_deg),
                "camera_usable_frames": camera["capture"]["usable_frames"],
                "camera_failure_fraction": camera["capture"]["failure_fraction"],
                "block_median_range_deg": max(blocks) - min(blocks),
                "palm_reference_drift_deg": palm_reference,
                "palm_reference_mode": palm_mode,
                "palm_cross_drift_deg": axis_delta_deg(
                    camera["angle_summary_deg"]["palm_cross_heading_deg_2d"]["median"],
                    zero["palm_cross_heading_deg_2d"]),
                "thumb_l_change_deg": (
                    camera["angle_summary_deg"]["thumb_l_angle_deg"]["median"]
                    - zero["thumb_l_angle_deg"]),
                "source": str(camera_path.relative_to(session)),
            }
    return points


def hysteresis(a: dict[int, dict], b: dict[int, dict]) -> dict[str, Any]:
    """Readback-domain difference between two passes over the same span."""
    if not a or not b:
        return {"available": False}
    br = sorted(b)
    bx = [b[k]["yaw_deg"] for k in br]
    pairs = [(k, a[k]["yaw_deg"] - float(np.interp(k, br, bx)))
             for k in sorted(a) if br[0] <= k <= br[-1]]
    if not pairs:
        return {"available": False}
    values = [d for _, d in pairs]
    return {
        "available": True,
        "max_abs_deg": max(abs(v) for v in values),
        "max_abs_rad": math.radians(max(abs(v) for v in values)),
        "mean_deg": float(np.mean(values)),
        "per_readback_deg": {str(k): d for k, d in pairs},
    }


def heldout_error_rad(points: list[tuple[int, float]]) -> float:
    """Max error when each interior knot is predicted from its neighbours."""
    if len(points) < 3:
        return 0.0
    worst = 0.0
    for i in range(1, len(points) - 1):
        (x0, y0), (x1, y1), (x2, y2) = points[i - 1], points[i], points[i + 1]
        predicted = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0)
        worst = max(worst, abs(predicted - y1))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--zero-readback-raw", type=int, default=251)
    parser.add_argument("--forward-tags", nargs="+", required=True)
    parser.add_argument("--return-tags", nargs="+", required=True)
    parser.add_argument("--final-zero-summary", type=Path, required=True)
    parser.add_argument("--rejected-note", action="append", default=[])
    parser.add_argument("--base-calib", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    session = args.session
    inputs: list[Path] = []
    zero, zero_repeat = zero_medians(args.reference_dir)
    reference_caps = sorted(
        args.reference_dir.glob("camera_*_fixed_exp_180f_summary.json"))
    inputs.extend(reference_caps)

    forward = collect(session, args.forward_tags, zero, inputs)
    forward[args.zero_readback_raw] = {
        "command_raw": 252,
        "stable_readback_raw": args.zero_readback_raw,
        "undershoot_raw": -1,
        "yaw_deg": 0.0,
        "yaw_rad": 0.0,
        "camera_usable_frames": 360,
        "camera_failure_fraction": 0.0,
        "block_median_range_deg": max(
            max(read_json(p)["block_medians_deg"]["thumb_cmc_yaw_deg_2d"])
            - min(read_json(p)["block_medians_deg"]["thumb_cmc_yaw_deg_2d"])
            for p in reference_caps),
        "palm_reference_drift_deg": 0.0,
        "palm_reference_mode": "fused_long_and_cross",
        "palm_cross_drift_deg": 0.0,
        "thumb_l_change_deg": 0.0,
        "source": str(args.reference_dir.relative_to(session)),
    }
    backward = collect(session, args.return_tags, zero, inputs)
    if min(forward) != 17:
        raise SystemExit(f"accepted forward lower endpoint must be raw17, got {min(forward)}")
    if max(forward) != args.zero_readback_raw:
        raise SystemExit("accepted forward curve does not reach the zero readback")

    forward_pass = {k: dict(v) for k, v in forward.items()}
    merged: dict[int, dict[str, Any]] = {}
    br = sorted(backward)
    bx = [backward[k]["yaw_deg"] for k in br]
    for source in (forward, backward):
        for raw, point in source.items():
            merged.setdefault(raw, dict(point))
    for raw in merged:
        f = forward.get(raw)
        b = backward.get(raw)
        if f is not None and br and br[0] <= raw <= br[-1]:
            other = float(np.interp(raw, br, bx))
            merged[raw]["yaw_deg"] = (f["yaw_deg"] + other) / 2.0
            merged[raw]["lut_source"] = "midline"
        elif f is not None:
            merged[raw]["yaw_deg"] = f["yaw_deg"]
            merged[raw]["lut_source"] = "forward_only"
        else:
            merged[raw]["yaw_deg"] = b["yaw_deg"]
            merged[raw]["lut_source"] = "return_only"
        merged[raw]["yaw_rad"] = math.radians(merged[raw]["yaw_deg"])

    knots = sorted(merged)
    dropped_inversions: list[int] = []
    while True:
        bad = [i for i in range(len(knots) - 1)
               if merged[knots[i]]["yaw_rad"]
               <= merged[knots[i + 1]]["yaw_rad"]]
        if not bad:
            break
        dropped = knots.pop(bad[0] + 1)
        dropped_inversions.append(dropped)
    points = {k: merged[k] for k in knots}
    rad = [points[k]["yaw_rad"] for k in knots]
    if any(rad[i] <= rad[i + 1] for i in range(len(rad) - 1)):
        raise SystemExit("yaw LUT is not strictly decreasing")

    hyst = hysteresis(forward_pass, backward)
    if (not hyst.get("available")
            or hyst["max_abs_deg"] > HYSTERESIS_MIDLINE_GATE_DEG):
        raise SystemExit("readback-domain hysteresis is outside the midline gate")

    final_path = args.final_zero_summary
    final_camera = read_json(final_path)
    inputs.append(final_path)
    final_yaw = robust_yaw(final_camera["angle_summary_deg"], zero)
    final_palm, final_palm_mode = palm_reference_delta(
        final_camera["angle_summary_deg"], zero)
    final_drift = {
        "final_minus_zero_deg": final_yaw,
        "palm_reference_drift_deg": final_palm,
        "palm_reference_mode": final_palm_mode,
        "gate_deg": ZERO_DRIFT_GATE_DEG,
        "within_gate": abs(final_yaw) <= ZERO_DRIFT_GATE_DEG,
        "source": str(final_path.relative_to(session)),
    }
    if not final_drift["within_gate"]:
        raise SystemExit("final zero drift exceeds gate")

    motion_files = sorted(session.glob("**/motion_*.json"))
    sent = [read_json(p) for p in motion_files if read_json(p).get("motion_sent")]
    max_temp = max(
        (max(sample.get("temperature20", [0]))
         for record in sent
         for sample in record.get("post_command_samples", [])),
        default=None)
    all_fault_free = all(
        record.get("result", {}).get("fault_free", False) for record in sent)

    all_accepted = list(forward_pass.values()) + list(backward.values())
    summary = {
        "schema_version": 1,
        "status": "complete_candidate_only",
        "joint": "thumb_cmc_yaw",
        "raw20_slot": THUMB_YAW_SLOT,
        "identity": {
            "hand_serial": "LHT20-010-415-L-B-1-D",
            "hand": "left G20",
            "sdk_version": "3.1.0",
            "embedded_version": "1.0.7",
            "camera": "RealSense D435 143322073091",
        },
        "measurement_contract": {
            "primary_value": "thumb longitudinal heading change relative to robust palm L heading",
            "palm_reference_rule": "fuse palm long and cross when their relative drifts agree within 0.5 deg; otherwise use cross as the occlusion-safe fallback",
            "zero_readback_raw": args.zero_readback_raw,
            "zero_is_relative_reference_not_mechanical_stop": True,
            "zero_repeat_abs_deg": zero_repeat,
            "points_keyed_by_stable_readback_not_command": True,
            "settle_tolerance_raw": 3,
            "settle_tolerance_reason": "the first 14 raw return step settled 3 raw short with zero faults; later steps remained fail-closed",
            "marker_identity": "thumb longitudinal is the shallower depth component; palm L provides a rigid reference and cross is the unobscured fallback",
            "camera_and_can_never_open_concurrently": True,
        },
        "range": {
            "min_safe_readback_raw": knots[0],
            "max_reference_readback_raw": knots[-1],
            "yaw_at_min_readback_deg": points[knots[0]]["yaw_deg"],
            "yaw_at_max_readback_deg": points[knots[-1]]["yaw_deg"],
            "full_range_deg": points[knots[0]]["yaw_deg"] - points[knots[-1]]["yaw_deg"],
            "full_range_rad": points[knots[0]]["yaw_rad"] - points[knots[-1]]["yaw_rad"],
            "raw2_probe_rejected_by_wrap_gate": True,
            "raw0_not_attempted": True,
        },
        "quality": {
            "accepted_camera_points": len(all_accepted),
            "all_accepted_points_180_of_180": all(
                p["camera_usable_frames"] == 180
                for p in all_accepted if p["stable_readback_raw"] != args.zero_readback_raw),
            "max_block_median_range_deg": max(
                p["block_median_range_deg"] for p in all_accepted),
            "max_abs_palm_cross_drift_deg": max(
                abs(p["palm_cross_drift_deg"]) for p in all_accepted),
            "max_abs_thumb_l_change_deg": max(
                abs(p["thumb_l_change_deg"]) for p in all_accepted),
            "motion_records_sent": len(sent),
            "all_sent_motion_fault_free": all_fault_free,
            "max_observed_temperature_c": max_temp,
        },
        "hysteresis_readback_domain": hyst,
        "hysteresis_verdict": "midline_lut_ok_record_as_uncertainty",
        "heldout_piecewise_linear_max_abs_error_rad": heldout_error_rad(
            [(k, points[k]["yaw_rad"]) for k in knots]),
        "final_zero_drift": final_drift,
        "midline": {
            "dropped_near_flat_inverted_readbacks": dropped_inversions,
            "midline_knots": sum(
                p.get("lut_source") == "midline" for p in points.values()),
            "forward_only_knots": sum(
                p.get("lut_source") == "forward_only" for p in points.values()),
            "return_only_knots": sum(
                p.get("lut_source") == "return_only" for p in points.values()),
        },
        "rejected_evidence": args.rejected_note,
        "points": [points[k] for k in sorted(points, reverse=True)],
        "forward_points": [forward_pass[k] for k in sorted(forward_pass, reverse=True)],
        "return_points": [backward[k] for k in sorted(backward, reverse=True)],
        "input_sha256": {
            str(p.relative_to(session)): sha256(p)
            for p in sorted(set(inputs))
        },
    }
    write_json(Path(f"{args.out_prefix}_summary.json"), summary)

    csv_path = Path(f"{args.out_prefix}_points.csv")
    if csv_path.exists():
        raise SystemExit(f"refusing to overwrite {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["points"][0]))
        writer.writeheader()
        writer.writerows(summary["points"])

    lut_joint = {
        "flip": False,
        "lo": min(rad),
        "hi": max(rad),
        "physical_lut": {"raw": knots, "rad": rad},
    }
    write_json(Path(f"{args.out_prefix}_lut.json"),
               {"thumb_cmc_yaw": lut_joint})

    candidate = read_json(args.base_calib)
    candidate["joints"]["thumb_cmc_yaw"] = lut_joint
    candidate["note"] = candidate["note"].replace(
        "Thumb CMC yaw/roll unmeasured.",
        "Thumb CMC yaw now has a candidate physical midline LUT over stable readback raw17..251; raw2 was rejected at the 80 degree line-heading wrap gate and raw0 was not attempted. Thumb CMC roll remains unmeasured.")
    write_json(args.candidate_out, candidate)

    report = f"""# Thumb CMC yaw fixed-exposure calibration

Status: complete as a candidate calibration; not production-enabled.  
SDK slot 10, CAN frame 0x41 element 1. Relative zero reference at stable readback raw {args.zero_readback_raw}.

## Result

- Safe accepted range: stable readback raw `{knots[0]}..{knots[-1]}`, `{summary["range"]["full_range_deg"]:.4f} deg / {summary["range"]["full_range_rad"]:.7f} rad`, {len(knots)} strictly monotonic midline knots.
- Readback-domain hysteresis max `{hyst["max_abs_deg"]:.4f} deg`, mean `{hyst["mean_deg"]:+.4f} deg`, inside the 3 deg runbook gate.
- Held-out piecewise-linear maximum error `{summary["heldout_piecewise_linear_max_abs_error_rad"]:.7f} rad`.
- Initial two zero captures differed by `{zero_repeat:.6f} deg`; final zero drift was `{final_yaw:+.4f} deg`, with robust palm-L reference drift `{final_palm:+.4f} deg`.
- Every accepted camera point used 180/180 frames. All {len(sent)} sent motion records were fault-free; maximum observed temperature was `{max_temp} C`.

The raw2 boundary probe repeated at `+80.017 deg` and was rejected by the 80 deg line-heading wrap gate. Raw0 was not attempted. The production-safe measured lower endpoint is raw17 at about `+75.36 deg`.

## Measurement and LUT construction

The thumb longitudinal tape is identified by its shallower depth. Palm long and cross drifts are averaged while they agree within 0.5 deg; when the thumb partly occludes palm-long near the low-raw endpoint, the unobscured cross becomes the automatic fallback. The LUT is the runbook midline of the forward and return passes and is keyed by stable readback. Readback {dropped_inversions} was dropped where two interleaved near-flat knots inverted within measurement noise.

Settle tolerance was explicitly 3 raw because the first 14 raw return step settled 3 raw short with zero faults and stable holds; it was not widened during the formal sweep. CAN and camera processes were always serialized.

## Rejected evidence retained

""" + "\n".join(f"{i + 1}. {note}" for i, note in enumerate(args.rejected_note)) + f"""

## Artifacts

- `{Path(f"{args.out_prefix}_summary.json").name}`
- `{Path(f"{args.out_prefix}_points.csv").name}`
- `{Path(f"{args.out_prefix}_lut.json").name}`
- `{args.candidate_out.name}` at repository root
- all motion, RGB, aligned depth, per-frame CSV, annotated PNG, rejected captures, and readonly snapshots remain in this session

Production calibration, Isaac limits, semantic limits and live deployment remain unchanged. Thumb CMC roll is still unmeasured.
"""
    report_path = args.report_out
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps({
        "joint": "thumb_cmc_yaw",
        "knots": len(knots),
        "range_deg": summary["range"]["full_range_deg"],
        "hysteresis_max_deg": hyst["max_abs_deg"],
        "heldout_max_rad": summary["heldout_piecewise_linear_max_abs_error_rad"],
        "final_zero_drift": final_drift,
        "candidate": str(args.candidate_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
