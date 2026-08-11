#!/usr/bin/env python3
"""Split the recorded raw128 hysteresis into command-domain and readback-domain parts.

Each finger summary reports one hysteresis number: the physical angle difference
between the forward and return captures taken at the same *command* raw. Those
two captures do not sit at the same *readback* raw, because the joint stops
short of the command in whichever direction it is travelling (middle PIP settles
at raw130 coming from extension and raw127 coming from flexion for the same
command raw128).

That single number is correct for command-domain behaviour, but the runbook
section 10 contract uses one LUT for both command inversion *and* readback
interpretation. The readback-domain question - at the same readback, how far
apart are the two directions - is a different and larger quantity that no
session computed. This tool derives it from the accepted curves and writes a
combined addendum. It does not modify any existing artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Approach(NamedTuple):
    """Stable readback raw reached at the shared command raw, per direction."""

    forward_raw: int
    return_raw: int
    source: str


# Readback raws at the shared command raw128. Ring and pinky record these in
# their own summaries; middle records only the PIP pair, so the pitch pair is
# recovered here from the motion records named in `source`.
APPROACHES: dict[tuple[str, str], Approach] = {
    ("middle", "pip"): Approach(130, 127, "pip_hysteresis_new_camera/{forward_motion_raw144_to_raw128,return_motion_actual114_to_command128}.json"),
    ("middle", "pitch"): Approach(128, 127, "middle_pitch_new_camera/{flex_raw128/motion_raw144_to_raw128,return_to_raw255/motion_raw112_to_raw128}.json"),
    ("ring", "pip"): Approach(129, 128, "ring_pip_dip_pitch_summary.json"),
    ("ring", "pitch"): Approach(128, 127, "ring_pip_dip_pitch_summary.json"),
    ("pinky", "pip"): Approach(129, 128, "pinky_pip_dip_pitch_summary.json"),
    ("pinky", "pitch"): Approach(128, 127, "pinky_pip_dip_pitch_summary.json"),
}

# Fingers whose PIP/DIP LUT was captured in a different camera pose than their
# PIP hysteresis pair, so the local slope carries a cross-pose projection caveat.
CROSS_POSE_PIP = {"middle"}

SESSIONS = {
    "middle": "20260802T_middle_pip_dip_pitch_thin_tape_full_raw_sweep",
    "ring": "20260802T_ring_pip_dip_pitch_thin_tape_full_raw_sweep",
    "pinky": "20260802T_pinky_pip_dip_pitch_thin_tape_full_raw_sweep",
}

HYSTERESIS_GATE_DEG = (3.0, 6.0)


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


def curve(rows: list[dict[str, str]], readback_col: str, rad_col: str) -> list[tuple[float, float]]:
    """Accepted curve as (stable readback raw, physical deg), ascending by raw."""
    points = [
        (float(row[readback_col]), math.degrees(float(row[rad_col])))
        for row in rows
        if row[readback_col] and row[rad_col]
    ]
    return sorted(set(points))


def interpolate(points: list[tuple[float, float]], raw: float) -> float:
    if raw <= points[0][0] or raw >= points[-1][0]:
        raise ValueError(f"readback raw {raw} outside accepted curve")
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= raw <= x1:
            return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)
    raise AssertionError("unreachable")


def verdict(deg: float) -> str:
    low, high = HYSTERESIS_GATE_DEG
    if abs(deg) <= low:
        return "midline_lut_ok_record_as_uncertainty"
    if abs(deg) <= high:
        return "consider_direction_aware_mapping"
    return "blocks_deployment"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    root: Path = args.record_root
    inputs: list[Path] = []
    results: dict[str, dict[str, Any]] = {}

    for finger, session_name in SESSIONS.items():
        session = root / "sessions" / session_name
        points_path = session / f"{finger}_pip_dip_pitch_calibration_points.csv"
        summary_path = session / f"{finger}_pip_dip_pitch_summary.json"
        inputs.extend([points_path, summary_path])

        with points_path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        recorded = read_json(summary_path)["hysteresis_raw128"]

        finger_result: dict[str, Any] = {}
        # DIP is passive, so it is indexed by the PIP readback like its driver.
        for joint, readback_col, rad_col, approach_key in (
            ("pip", "pip_stable_readback_raw", f"{finger}_pip_rad", "pip"),
            ("dip", "pip_stable_readback_raw", f"{finger}_dip_rad", "pip"),
            ("pitch", "pitch_stable_readback_raw", f"{finger}_mcp_pitch_rad", "pitch"),
        ):
            approach = APPROACHES[(finger, approach_key)]
            points = curve(rows, readback_col, rad_col)
            expected = (interpolate(points, approach.forward_raw)
                        - interpolate(points, approach.return_raw))
            command_domain = float(
                recorded[f"{finger}_{'mcp_pitch' if joint == 'pitch' else joint}_deg"])
            readback_domain = command_domain - expected

            finger_result[joint] = {
                "forward_readback_raw": approach.forward_raw,
                "return_readback_raw": approach.return_raw,
                "readback_raw_offset": approach.forward_raw - approach.return_raw,
                "approach_source": approach.source,
                "command_domain_deg": command_domain,
                "command_domain_rad": math.radians(command_domain),
                "expected_from_readback_offset_deg": expected,
                "readback_domain_deg": readback_domain,
                "readback_domain_rad": math.radians(readback_domain),
                "command_domain_verdict": verdict(command_domain),
                "readback_domain_verdict": verdict(readback_domain),
                "cross_pose_slope_caveat": finger in CROSS_POSE_PIP and joint in ("pip", "dip"),
            }
        results[finger] = finger_result

    addendum = {
        "schema_version": 1,
        "purpose": (
            "Separate the recorded raw128 hysteresis into the command-domain value "
            "the sessions already report and the readback-domain value none of them "
            "computed. Evidence only; no existing artifact is modified."
        ),
        "definitions": {
            "command_domain_deg": (
                "physical(forward @ command raw128) - physical(return @ command "
                "raw128). What a controller sees when it commands the same raw from "
                "either direction. Already reported per session."
            ),
            "expected_from_readback_offset_deg": (
                "Angle difference the accepted curve predicts purely from the two "
                "stable readbacks being different raws."
            ),
            "readback_domain_deg": (
                "command_domain minus expected. Lost motion between encoder and "
                "link: how far apart the two directions sit at the *same* readback. "
                "This is the figure that matters when the same LUT is used to "
                "interpret SDK readback, per runbook section 10."
            ),
        },
        "gate_deg": {"midline_lut": HYSTERESIS_GATE_DEG[0],
                     "direction_aware": HYSTERESIS_GATE_DEG[1]},
        "fingers": results,
        "input_sha256": {str(p.resolve()): sha256(p) for p in sorted(set(inputs))},
    }

    prefix = args.out_prefix or (root / "hysteresis_readback_domain_addendum")
    write_json(prefix.with_suffix(".json"), addendum)
    print(f"wrote {prefix.with_suffix('.json')}")
    print(f"{'joint':16s} {'rb raws':>9s} {'command':>9s} {'expected':>9s} {'readback':>9s}")
    for finger, joints in results.items():
        for joint, r in joints.items():
            print("%-16s %4d/%-4d %+9.3f %+9.3f %+9.3f%s" % (
                f"{finger}_{joint}", r["forward_readback_raw"], r["return_readback_raw"],
                r["command_domain_deg"], r["expected_from_readback_offset_deg"],
                r["readback_domain_deg"],
                "  (cross-pose)" if r["cross_pose_slope_caveat"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
