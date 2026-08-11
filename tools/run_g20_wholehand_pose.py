#!/usr/bin/env python3
"""Drive the G20 to whole-hand semantic poses and record what actually came back.

This is the hardware half of the sim->SDK->real visual A/B check.  Unlike the
Phase A/B/C calibration runners, which moved exactly one joint and *verified*
every other slot, this one commands all sixteen active joints at once -- that is
the whole point, since single-joint calibration cannot catch a slot swap or a
sign error that only shows up when several joints move together.

Fail-closed behaviour kept from the calibration runners:

* a preflight record with ``motion_sent=false`` is written *before* any frame is
  sent, so an aborted run still leaves evidence of what was about to happen;
* the target is rejected unless every semantic value is inside the joint's signed
  limits and every resulting raw is in 0..255 with the reserved slots at 0;
* motion is ramped, never stepped, and each pose is followed by a settle and a
  median over several read-only samples;
* faults and temperatures are snapshotted before and after every pose.

It reports the readback in *semantic* units by inverting the same mapper used to
command, so the numbers are directly comparable to what the simulator was told.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SERIAL = "LHT20-010-415-L-B-1-D"
RESERVED_SLOTS = (11, 12, 13, 14)
MAX_TEMPERATURE_C = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", default="/home/user/linkerhand-ros-sdk")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--calib", required=True, help="physical-LUT overlay to command through")
    parser.add_argument("--poses", required=True, help="JSON: {name: [16 semantic rad]}")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--speed", type=int, default=60, help="SDK speed for all five groups")
    parser.add_argument("--ramp-steps", type=int, default=25)
    parser.add_argument("--ramp-hz", type=float, default=20.0)
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument(
        "--capture-cmd",
        help=(
            "optional shell command run after each pose settles, with {out} replaced "
            "by an image prefix; used to trigger the RealSense capture, which lives in "
            "a different interpreter than the SDK"
        ),
    )
    return parser.parse_args()


def _median20(rows: list[list[int]]) -> list[int]:
    return [int(round(statistics.median(col))) for col in zip(*rows)]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses: dict[str, list[float]] = json.loads(Path(args.poses).read_text())

    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from screwdriver_rl.deploy import hw_utils
    from screwdriver_rl.deploy import linker_sdk_map as sdkmap
    from tools.run_g20_index_pip_candidate_step import (
        _fault_snapshot,
        _read_state20,
        _temperature_snapshot,
        _write_json,
    )

    sdkmap.apply_calibration(args.calib)
    specs = sdkmap.active_joints()
    names = [js.name for js in specs]

    # Reject anything outside the signed limits before the SDK is even opened.
    for name, semantic in poses.items():
        if len(semantic) != 16:
            raise ValueError(f"pose {name!r} has {len(semantic)} values, expected 16")
        for js, value in zip(specs, semantic):
            if not (js.lo - 1e-9 <= value <= js.hi + 1e-9):
                raise ValueError(
                    f"pose {name!r}: {js.name}={value:.6f} outside signed "
                    f"[{js.lo:.6f}, {js.hi:.6f}]"
                )
        raw = sdkmap.joints16_to_sdk_range(list(semantic))
        if len(raw) != 20 or any(not (0 <= v <= 255) for v in raw):
            raise ValueError(f"pose {name!r} produced an out-of-range raw20: {raw}")
        if any(raw[slot] != 0 for slot in RESERVED_SLOTS):
            raise ValueError(f"pose {name!r} wrote a reserved slot: {raw}")

    hw_utils.bootstrap_sdk(str(args.sdk_root))
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    api = LinkerHandApi(hand_joint="G20", hand_type="left", can=args.can)
    serial = str(api.get_serial_number()).strip().strip("\x00")
    if serial != EXPECTED_SERIAL:
        raise RuntimeError(f"serial mismatch: expected {EXPECTED_SERIAL}, got {serial!r}")

    api.set_speed(speed=[args.speed] * 5)
    time.sleep(0.1)

    results: dict[str, Any] = {}
    for name, semantic in poses.items():
        target_raw = sdkmap.joints16_to_sdk_range(list(semantic))
        start_raw = _read_state20(api, hw_utils)
        fault_groups, fault_flat = _fault_snapshot(api)
        temps = _temperature_snapshot(api)

        preflight = {
            "pose": name,
            "motion_sent": False,
            "serial": serial,
            "calib": args.calib,
            "target_semantic": list(map(float, semantic)),
            "target_raw20": target_raw,
            "start_raw20": start_raw,
            "faults_before": fault_groups,
            "faults20_before": fault_flat,
            "temperature20_before": temps,
        }
        _write_json(out_dir / f"{name}_preflight.json", preflight)

        if any(fault_flat):
            raise RuntimeError(f"pose {name!r}: pre-existing faults {fault_flat}")
        if max(temps) > MAX_TEMPERATURE_C:
            raise RuntimeError(f"pose {name!r}: temperature {max(temps)}C over limit")

        start_semantic = sdkmap.sdk_range_to_joints16(start_raw)
        frames = hw_utils.ramp_frames(
            list(start_semantic), list(semantic),
            duration_s=args.ramp_steps / args.ramp_hz, rate_hz=args.ramp_hz,
        )
        for frame in frames:
            api.finger_move(pose=sdkmap.joints16_to_sdk_range(frame))
            time.sleep(1.0 / args.ramp_hz)
        api.finger_move(pose=target_raw)
        time.sleep(args.settle_s)

        rows = []
        for _ in range(args.samples):
            rows.append(_read_state20(api, hw_utils))
            time.sleep(0.05)
        settled_raw = _median20(rows)
        settled_semantic = sdkmap.sdk_range_to_joints16(settled_raw)
        fault_groups_after, fault_flat_after = _fault_snapshot(api)
        temps_after = _temperature_snapshot(api)

        image_prefix = None
        if args.capture_cmd:
            import subprocess

            image_prefix = str(out_dir / f"{name}_real")
            subprocess.run(
                args.capture_cmd.format(out=image_prefix), shell=True, check=True
            )

        record = dict(preflight)
        record.update({
            "motion_sent": True,
            "ramp_frames": len(frames),
            "settled_raw20": settled_raw,
            "settled_semantic": [float(v) for v in settled_semantic],
            "raw_samples": rows,
            "faults_after": fault_groups_after,
            "faults20_after": fault_flat_after,
            "temperature20_after": temps_after,
            "image_prefix": image_prefix,
            "per_joint": [
                {
                    "name": js.name,
                    "slot": js.slot,
                    "target_rad": float(t),
                    "settled_rad": float(s),
                    "error_rad": float(s - t),
                    "target_raw": target_raw[js.slot],
                    "settled_raw": settled_raw[js.slot],
                    "raw_error": settled_raw[js.slot] - target_raw[js.slot],
                }
                for js, t, s in zip(specs, semantic, settled_semantic)
            ],
        })
        _write_json(out_dir / f"{name}_result.json", record)
        results[name] = record

        worst = max(record["per_joint"], key=lambda row: abs(row["error_rad"]))
        print(
            f"[pose] {name}: worst {worst['name']} {worst['error_rad']:+.4f} rad "
            f"(raw {worst['raw_error']:+d}), faults={any(fault_flat_after)}, "
            f"Tmax={max(temps_after)}C",
            flush=True,
        )
        if any(fault_flat_after):
            raise RuntimeError(f"pose {name!r} raised faults {fault_flat_after}")

    _write_json(out_dir / "summary.json", {"joint_order16": names, "poses": results})
    print(f"[pose] wrote {out_dir/'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
