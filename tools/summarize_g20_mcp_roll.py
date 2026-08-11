#!/usr/bin/env python3
"""Summarize one finger's MCP-roll sweep into a candidate LUT and report.

Roll differs structurally from every Phase A joint: it is bidirectional, its
zero is a mid-range reference rather than a mechanical stop, and both raw ends
are the 0/255 command bounds rather than mechanical limits. The LUT therefore
carries positive physical angle below the zero readback and negative above it,
while still satisfying the runbook section 10 rule that raw increases as
physical angle decreases.

Points are keyed by stable readback raw, because this actuator stops one to
three raw short of every command and a follow-up step small enough to close
that gap lands inside the deadband.
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

ROLL_SLOT = {"index": 6, "middle": 7, "ring": 8, "pinky": 9}
ZERO_DRIFT_GATE_DEG = 0.5
HYSTERESIS_MIDLINE_GATE_DEG = 3.0


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


def zero_mean(reference: Path, key: str) -> tuple[float, float]:
    caps = sorted(reference.glob("camera_*_fixed_exp_180f_summary.json"))
    if len(caps) != 2:
        raise SystemExit(f"expected two zero captures in {reference}")
    vals = [read_json(c)["angle_summary_deg"][key]["median"] for c in caps]
    return sum(vals) / 2.0, abs(vals[0] - vals[1])


def collect(session: Path, tags: list[str], finger: str, zero: float,
            inputs: list[Path]) -> dict[int, dict[str, Any]]:
    """Map stable readback raw -> point record, for the given directory tags."""
    slot = ROLL_SLOT[finger]
    key = f"{finger}_abduction_deg_2d"
    points: dict[int, dict[str, Any]] = {}
    for tag in tags:
        for point_dir in sorted(session.glob(f"{tag}_raw*")):
            motions = sorted(point_dir.glob("motion_*.json"))
            cameras = sorted(point_dir.glob("camera*_summary.json"))
            if not motions or not cameras:
                continue
            motion = read_json(motions[0])
            camera = read_json(cameras[-1])
            inputs.extend([motions[0], cameras[-1]])
            readback = motion["result"]["settled_state20_median"][slot]
            blocks = camera["block_medians_deg"][key]
            points[readback] = {
                "command_raw": motion["approved_scope"]["target_raw"],
                "stable_readback_raw": readback,
                "undershoot_raw": readback - motion["approved_scope"]["target_raw"],
                "abduction_deg": camera["angle_summary_deg"][key]["median"] - zero,
                "abduction_rad": math.radians(
                    camera["angle_summary_deg"][key]["median"] - zero),
                "camera_usable_frames": camera["capture"]["usable_frames"],
                "block_median_range_deg": max(blocks) - min(blocks),
                "source": str(cameras[-1].relative_to(session)),
            }
    return points


def hysteresis(a: dict[int, dict], b: dict[int, dict]) -> dict[str, Any]:
    """Readback-domain difference between two passes over the same span."""
    if not a or not b:
        return {"available": False}
    br = sorted(b)
    bx = [b[k]["abduction_deg"] for k in br]
    pairs = [(k, a[k]["abduction_deg"] - float(np.interp(k, br, bx)))
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
    parser.add_argument("--finger", required=True, choices=tuple(ROLL_SLOT))
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--zero-readback-raw", type=int, required=True)
    parser.add_argument("--forward-tags", nargs="+", required=True,
                        help="dir tags forming the accepted forward curve")
    parser.add_argument("--return-tags", nargs="+", default=[],
                        help="dir tags forming the accepted return curve")
    parser.add_argument("--final-zero-dir", type=Path, default=None)
    parser.add_argument("--rejected-note", action="append", default=[])
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    finger, session = args.finger, args.session
    key = f"{finger}_abduction_deg_2d"
    inputs: list[Path] = []
    zero, zero_repeat = zero_mean(args.reference_dir, key)
    palm_zero, _ = zero_mean(args.reference_dir, "palm_reference_heading_deg_2d")

    forward = collect(session, args.forward_tags, finger, zero, inputs)
    forward[args.zero_readback_raw] = {
        "command_raw": args.zero_readback_raw,
        "stable_readback_raw": args.zero_readback_raw,
        "undershoot_raw": 0,
        "abduction_deg": 0.0, "abduction_rad": 0.0,
        "camera_usable_frames": 360,
        "block_median_range_deg": 0.0,
        "source": str(args.reference_dir.relative_to(session)),
    }
    backward = collect(session, args.return_tags, finger, zero, inputs)

    # Runbook 8.3: with readback-domain hysteresis inside the 3 deg gate the
    # LUT is the midline of the two directions. Neither direction alone spans
    # the whole range here - the adduction end is only reached descending and
    # the abduction end only ascending - so the midline is also what makes the
    # LUT cover both ends.
    # Keep the un-merged forward pass: hysteresis must stay forward-vs-return.
    # Measuring the midline against the return would halve it by construction.
    forward_pass = {k: dict(v) for k, v in forward.items()}
    merged: dict[int, dict[str, Any]] = {}
    br = sorted(backward)
    bx = [backward[k]["abduction_deg"] for k in br]
    for source in (forward, backward):
        for raw, point in source.items():
            merged.setdefault(raw, dict(point))
    for raw in merged:
        f = forward.get(raw)
        b = backward.get(raw)
        if f is not None and br and br[0] <= raw <= br[-1]:
            other = float(np.interp(raw, br, bx))
            merged[raw]["abduction_deg"] = (f["abduction_deg"] + other) / 2.0
            merged[raw]["lut_source"] = "midline"
        elif f is not None:
            merged[raw]["abduction_deg"] = f["abduction_deg"]
            merged[raw]["lut_source"] = "forward_only"
        else:
            merged[raw]["abduction_deg"] = b["abduction_deg"]
            merged[raw]["lut_source"] = "return_only"
        merged[raw]["abduction_rad"] = math.radians(merged[raw]["abduction_deg"])
    # Averaging two noisy passes can invert a pair that is within noise of flat;
    # drop the shallower of any inverted neighbours rather than reorder them.
    knots = sorted(merged)
    while True:
        bad = [i for i in range(len(knots) - 1)
               if merged[knots[i]]["abduction_rad"]
               <= merged[knots[i + 1]]["abduction_rad"]]
        if not bad:
            break
        knots.pop(bad[0] + 1)
    forward = {k: merged[k] for k in knots}
    rad = [forward[k]["abduction_rad"] for k in knots]
    if any(rad[i] <= rad[i + 1] for i in range(len(rad) - 1)):
        raise SystemExit("LUT is not strictly decreasing in physical angle")

    final_drift = None
    if args.final_zero_dir is not None:
        final, _ = zero_mean(args.final_zero_dir, key)
        palm_final, _ = zero_mean(args.final_zero_dir,
                                  "palm_reference_heading_deg_2d")
        final_drift = {
            "final_minus_zero_deg": final - zero,
            "palm_reference_drift_deg": palm_final - palm_zero,
            "gate_deg": ZERO_DRIFT_GATE_DEG,
            "within_gate": abs(final - zero) <= ZERO_DRIFT_GATE_DEG,
        }

    hyst = hysteresis(forward_pass, backward)
    summary = {
        "schema_version": 1,
        "status": "complete_candidate_only",
        "joint": f"{finger}_mcp_roll",
        "raw20_slot": ROLL_SLOT[finger],
        "identity": {
            "hand_serial": "LHT20-010-415-L-B-1-D",
            "hand": "left G20", "sdk_version": "3.1.0",
            "embedded_version": "1.0.7",
            "camera": "RealSense D435 143322073091",
        },
        "measurement_contract": {
            "primary_value": "2D in-plane abduction, palm tape as reference",
            "three_d_axes": "diagnostic only; palm tape carries too little "
                            "depth in this pose to fit a palm plane",
            "zero_is_mid_range_not_a_mechanical_stop": True,
            "zero_readback_raw": args.zero_readback_raw,
            "zero_repeat_abs_deg": zero_repeat,
            "points_keyed_by_stable_readback_not_command": True,
            "isolation": "non-target fingers pitched away; only the target "
                         "finger's proximal tape is required to be visible",
        },
        "range": {
            "min_readback_raw": knots[0],
            "max_readback_raw": knots[-1],
            "abduction_at_min_readback_deg": forward[knots[0]]["abduction_deg"],
            "abduction_at_max_readback_deg": forward[knots[-1]]["abduction_deg"],
            "full_range_deg": (forward[knots[0]]["abduction_deg"]
                               - forward[knots[-1]]["abduction_deg"]),
            "full_range_rad": math.radians(
                forward[knots[0]]["abduction_deg"]
                - forward[knots[-1]]["abduction_deg"]),
            "both_ends_are_raw_command_bounds_not_mechanical_limits": True,
        },
        "hysteresis_readback_domain": hyst,
        "hysteresis_verdict": (
            "midline_lut_ok_record_as_uncertainty"
            if hyst.get("available")
            and hyst["max_abs_deg"] <= HYSTERESIS_MIDLINE_GATE_DEG
            else "review"),
        "heldout_piecewise_linear_max_abs_error_rad": heldout_error_rad(
            [(k, forward[k]["abduction_rad"]) for k in knots]),
        "final_zero_drift": final_drift,
        "rejected_evidence": args.rejected_note,
        "points": [forward[k] for k in sorted(forward, reverse=True)],
        "return_points": [backward[k] for k in sorted(backward, reverse=True)],
        "input_sha256": {
            str(p.relative_to(session)): sha256(p) for p in sorted(set(inputs))
        },
    }
    write_json(Path(f"{args.out_prefix}_summary.json"), summary)

    csv_path = Path(f"{args.out_prefix}_points.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["points"][0]))
        writer.writeheader()
        writer.writerows(summary["points"])

    lut = {"raw": knots, "rad": rad}
    write_json(Path(f"{args.out_prefix}_lut.json"),
               {f"{finger}_mcp_roll": {"flip": False,
                                       "lo": min(rad), "hi": max(rad),
                                       "zero_readback_raw": args.zero_readback_raw,
                                       "physical_lut": lut}})
    print(json.dumps({
        "joint": f"{finger}_mcp_roll",
        "knots": len(knots),
        "range_deg": summary["range"]["full_range_deg"],
        "hysteresis_max_deg": hyst.get("max_abs_deg"),
        "heldout_max_rad": summary["heldout_piecewise_linear_max_abs_error_rad"],
        "final_zero_drift": final_drift,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
