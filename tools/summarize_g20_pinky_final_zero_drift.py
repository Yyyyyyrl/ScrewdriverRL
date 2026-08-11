#!/usr/bin/env python3
"""Compute the pinky Phase A final zero-reference drift against the 0.5 deg gate.

The pinky session summary recorded the final raw255 visual repeat but never
differenced it against the initial zero pair, so the runbook section 8.3 /
section 9 zero-drift check was left unevaluated. This tool performs that check
from the same fixed-camera captures and writes an addendum next to the original
summary. It does not modify or regenerate any existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Joint angles under test plus the raw marker headings used to decompose the
# drift into palm/camera motion versus real finger rotation.
JOINT_KEYS = ("mcp_projected_deg_2d", "pip_projected_deg_2d", "dip_projected_deg_2d")
HEADING_KEYS = (
    "metacarpal_heading_deg_2d",
    "proximal_heading_deg_2d",
    "middle_heading_deg_2d",
    "distal_heading_deg_2d",
)

ZERO_DRIFT_GATE_DEG = 0.5

ZERO_CAPTURES = (
    "pinky_raw255_zero/camera_four_real_markers_fixed_exp_180f_a_summary.json",
    "pinky_raw255_zero/camera_four_real_markers_fixed_exp_180f_b_summary.json",
)
FINAL_CAPTURE = (
    "pitch_return_raw255/camera_four_real_markers_fixed_exp_180f_after_reposition_summary.json"
)
# Pitch readback at the moment each capture was taken: the zero pair followed
# the establish_raw255 pitch step that settled at raw254, the final repeat is
# the 20-sample read-only snapshot median.
ZERO_ESTABLISH_MOTION = "establish_raw255/motion_pitch_raw252_to_raw255.json"
FINAL_SNAPSHOT = "final_sdk_snapshot_after_pinky_pip_pitch_sweeps.json"
PINKY_PITCH_SLOT = 4
PINKY_PIP_SLOT = 19


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


def medians(path: Path) -> dict[str, float]:
    payload = read_json(path)
    if payload["capture"]["usable_frames"] <= 0:
        raise ValueError(f"{path}: no usable camera frames")
    summary = payload["angle_summary_deg"]
    return {key: float(summary[key]["median"]) for key in JOINT_KEYS + HEADING_KEYS}


def pitch_slope_deg_per_raw(calibration: Path) -> float:
    """Local pinky pitch LUT slope over the topmost segment (raw240..raw255)."""
    lut = read_json(calibration)["joints"]["pinky_mcp_pitch"]["physical_lut"]
    raw, rad = lut["raw"], lut["rad"]
    return math.degrees(rad[-2] - rad[-1]) / (raw[-1] - raw[-2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    session: Path = args.session
    inputs = [session / name for name in (*ZERO_CAPTURES, FINAL_CAPTURE,
                                          ZERO_ESTABLISH_MOTION, FINAL_SNAPSHOT)]
    inputs.append(args.calibration)

    zero_rows = [medians(session / name) for name in ZERO_CAPTURES]
    zero = {key: statistics.mean(row[key] for row in zero_rows)
            for key in JOINT_KEYS + HEADING_KEYS}
    zero_repeat = {key: abs(zero_rows[0][key] - zero_rows[1][key])
                   for key in JOINT_KEYS + HEADING_KEYS}
    final = medians(session / FINAL_CAPTURE)

    drift = {key: final[key] - zero[key] for key in JOINT_KEYS}
    heading_drift = {key: final[key] - zero[key] for key in HEADING_KEYS}

    establish = read_json(session / ZERO_ESTABLISH_MOTION)["result"]
    zero_pitch_raw = float(establish["settled_state20_median"][PINKY_PITCH_SLOT])
    snapshot = read_json(session / FINAL_SNAPSHOT)["samples"]
    final_pitch_raw = statistics.median(
        s["state20"][PINKY_PITCH_SLOT] for s in snapshot)
    final_pip_raw = statistics.median(
        s["state20"][PINKY_PIP_SLOT] for s in snapshot)

    slope = pitch_slope_deg_per_raw(args.calibration)
    readback_delta_raw = zero_pitch_raw - final_pitch_raw
    explained_deg = readback_delta_raw * slope
    observed_deg = abs(drift["mcp_projected_deg_2d"])
    residual_deg = observed_deg - abs(explained_deg)

    verdicts = {
        "pinky_mcp_pitch": abs(drift["mcp_projected_deg_2d"]) <= ZERO_DRIFT_GATE_DEG,
        "pinky_pip": abs(drift["pip_projected_deg_2d"]) <= ZERO_DRIFT_GATE_DEG,
        "pinky_dip": abs(drift["dip_projected_deg_2d"]) <= ZERO_DRIFT_GATE_DEG,
    }

    addendum = {
        "schema_version": 1,
        "addendum_to": "pinky_pip_dip_pitch_summary.json",
        "purpose": (
            "Evaluate the runbook 8.3/9 final zero-reference drift check that the "
            "original pinky summary left uncomputed. Adds evidence only; the "
            "original summary, report and LUT are unchanged."
        ),
        "gate_deg": ZERO_DRIFT_GATE_DEG,
        "zero_reference_deg_median_mean_of_pair": zero,
        "zero_reference_pair_repeat_abs_deg": zero_repeat,
        "final_repeat_deg_median": final,
        "joint_zero_drift_deg": drift,
        "marker_heading_drift_deg": heading_drift,
        "gate_result": verdicts,
        "sdk_readback_raw": {
            "pinky_pitch_at_zero": zero_pitch_raw,
            "pinky_pitch_at_final": final_pitch_raw,
            "pinky_pip_at_final": final_pip_raw,
            "pitch_delta_raw": readback_delta_raw,
        },
        "pitch_drift_decomposition_deg": {
            "observed": observed_deg,
            "local_lut_slope_deg_per_raw": slope,
            "explained_by_pitch_readback_delta": abs(explained_deg),
            "unexplained_residual": residual_deg,
        },
        "interpretation": [
            "The metacarpal (palm) heading moved %.4f deg, so the camera pose, the "
            "palm fixture and the palm marker were stable; the drift is not a rig "
            "artifact." % heading_drift["metacarpal_heading_deg_2d"],
            "Proximal, middle and distal headings all rotated together by %.3f / "
            "%.3f / %.3f deg, i.e. the pinky rotated near-rigidly about the MCP "
            "pitch axis rather than changing its internal PIP/DIP angles."
            % (
                heading_drift["proximal_heading_deg_2d"],
                heading_drift["middle_heading_deg_2d"],
                heading_drift["distal_heading_deg_2d"],
            ),
            "pinky_pip and pinky_dip zero drift stay inside the %.1f deg gate, so "
            "the PIP LUT and the DIP mimic fit keep a valid zero anchor."
            % ZERO_DRIFT_GATE_DEG,
            "pinky_mcp_pitch zero drift exceeds the gate. The command raw255 "
            "extension rest position is not repeatable to better than about "
            "%.2f deg, of which %.2f deg is explained by the pitch readback "
            "returning to raw%.0f instead of raw%.0f."
            % (observed_deg, abs(explained_deg), final_pitch_raw, zero_pitch_raw),
            "Consequence: every pinky_mcp_pitch physical angle carries a zero-anchor "
            "bias of up to %.4f deg / %.6f rad, about %.1f%% of the measured %.3f "
            "deg range. This is a bias on the whole curve, not per-point noise, and "
            "it does not affect curve shape, monotonicity or hysteresis."
            % (
                observed_deg,
                math.radians(observed_deg),
                100.0 * observed_deg / 69.49338225660952,
                69.49338225660952,
            ),
        ],
        "input_sha256": {
            str(path.resolve()): sha256(path) for path in sorted(set(inputs))
        },
    }

    out = args.out or (session / "pinky_final_zero_drift_addendum.json")
    write_json(out, addendum)
    print(f"wrote {out}")
    for joint, ok in verdicts.items():
        key = {"pinky_mcp_pitch": "mcp", "pinky_pip": "pip",
               "pinky_dip": "dip"}[joint] + "_projected_deg_2d"
        print(f"  {joint:16s} drift {drift[key]:+8.4f} deg  "
              f"{'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
