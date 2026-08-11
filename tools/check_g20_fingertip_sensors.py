#!/usr/bin/env python3
"""Interactively verify all five G20 fingertip tactile sensors.

This tool is read-only.  It only calls identity and ``get_matrix_touch`` query
methods; it never sends a joint command, changes speed/torque, or clears faults.

For each fingertip, follow the terminal prompt:

1. leave the hand untouched and press Enter;
2. press and hold the requested fingertip, then press Enter;
3. release it, then press Enter.

The script compares the pressed sensor against its own baseline, the other four
sensors (cross-talk), and its released value.  A complete JSON record is saved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any


EXPECTED_SERIAL = "LHT20-010-415-L-B-1-D"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
DISPLAY_NAMES = {
    "thumb": "大拇指 thumb",
    "index": "食指 index",
    "middle": "中指 middle",
    "ring": "无名指 ring",
    "pinky": "小指 pinky",
}


def _read_masses(api: Any) -> list[float]:
    matrices = api.get_matrix_touch()
    time.sleep(0.02)
    if not isinstance(matrices, (list, tuple)) or len(matrices) != 5:
        raise RuntimeError(
            f"get_matrix_touch returned {type(matrices).__name__}, expected 5 matrices"
        )

    masses: list[float] = []
    for sensor_index, matrix in enumerate(matrices):
        try:
            values = [float(value) for row in matrix for value in row]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"sensor {sensor_index} contains non-numeric data"
            ) from exc
        if len(values) != 72:
            raise RuntimeError(
                f"sensor {sensor_index} has {len(values)} cells, expected 72"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise RuntimeError(f"sensor {sensor_index} has invalid cell values")
        masses.append(sum(values))
    return masses


def _sample(api: Any, *, count: int, hz: float) -> list[list[float]]:
    rows: list[list[float]] = []
    period = 1.0 / hz
    for _ in range(count):
        started = time.monotonic()
        rows.append(_read_masses(api))
        remaining = period - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)
    return rows


def _median5(rows: list[list[float]]) -> list[float]:
    return [float(statistics.median(column)) for column in zip(*rows)]


def _max5(rows: list[list[float]]) -> list[float]:
    return [float(max(column)) for column in zip(*rows)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", default="/home/user/linkerhand-ros-sdk")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=20.0,
        help="minimum target-sensor mass increase required to pass",
    )
    parser.add_argument(
        "--crosstalk-ratio",
        type=float,
        default=1.5,
        help="target delta must exceed the largest other-sensor delta by this ratio",
    )
    parser.add_argument(
        "--release-fraction",
        type=float,
        default=0.25,
        help="released residual must be at most this fraction of the pressed delta",
    )
    parser.add_argument(
        "--expected-serial",
        default=EXPECTED_SERIAL,
        help="exact hand serial; pass an empty string to disable the identity gate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "records/g20_fingertip_sensor_check_20260806/sensor_check.json"
        ),
    )
    args = parser.parse_args()

    if args.samples < 3 or args.hz <= 0.0:
        parser.error("--samples must be >= 3 and --hz must be positive")
    if args.min_delta <= 0.0 or args.crosstalk_ratio <= 1.0:
        parser.error("--min-delta must be positive and --crosstalk-ratio must be > 1")
    if not 0.0 < args.release_fraction < 1.0:
        parser.error("--release-fraction must be in (0, 1)")

    from screwdriver_rl.deploy import hw_utils

    hw_utils.bootstrap_sdk(args.sdk_root)
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    api = LinkerHandApi(hand_joint="G20", hand_type="left", can=args.can)
    try:
        serial = str(api.get_serial_number()).strip().strip("\x00")
        if args.expected_serial and serial != args.expected_serial:
            raise RuntimeError(
                f"serial mismatch: expected {args.expected_serial!r}, got {serial!r}"
            )

        print(f"\n已连接 G20: {serial}")
        print("这个脚本只读取触觉，不会发送任何运动命令。")
        print("SDK sensor 顺序按 thumb/index/middle/ring/pinky 检查。\n")

        input("请不要触碰任何指尖，按 Enter 采集公共基线...")
        common_baseline_rows = _sample(api, count=args.samples, hz=args.hz)
        common_baseline = _median5(common_baseline_rows)
        print(
            "公共基线: "
            + "  ".join(
                f"{name}={value:.1f}"
                for name, value in zip(FINGERS, common_baseline)
            )
        )

        checks: list[dict[str, Any]] = []
        for target_index, finger in enumerate(FINGERS):
            print(f"\n[{target_index + 1}/5] {DISPLAY_NAMES[finger]}")
            input("保持所有指尖松开，按 Enter 采集该轮基线...")
            baseline_rows = _sample(api, count=args.samples, hz=args.hz)
            baseline = _median5(baseline_rows)

            input(
                f"现在按住 {DISPLAY_NAMES[finger]} 的触觉区域，保持压力并按 Enter..."
            )
            pressed_rows = _sample(api, count=args.samples, hz=args.hz)
            pressed_peak = _max5(pressed_rows)

            input("现在完全松开该指尖，按 Enter 采集释放值...")
            released_rows = _sample(api, count=args.samples, hz=args.hz)
            released = _median5(released_rows)

            deltas = [
                max(0.0, peak - base)
                for peak, base in zip(pressed_peak, baseline)
            ]
            target_delta = deltas[target_index]
            other_delta = max(
                delta for index, delta in enumerate(deltas) if index != target_index
            )
            released_residual = max(
                0.0, released[target_index] - baseline[target_index]
            )

            response_pass = target_delta >= args.min_delta
            crosstalk_pass = target_delta >= args.crosstalk_ratio * max(
                other_delta, 1.0e-9
            )
            release_pass = released_residual <= max(
                0.25 * args.min_delta,
                args.release_fraction * target_delta,
            )
            passed = response_pass and crosstalk_pass and release_pass

            row = {
                "finger": finger,
                "sensor_index": target_index,
                "baseline_median5": baseline,
                "pressed_peak5": pressed_peak,
                "released_median5": released,
                "delta5": deltas,
                "target_delta": target_delta,
                "largest_other_delta": other_delta,
                "released_residual": released_residual,
                "response_pass": response_pass,
                "crosstalk_pass": crosstalk_pass,
                "release_pass": release_pass,
                "pass": passed,
                "baseline_samples": baseline_rows,
                "pressed_samples": pressed_rows,
                "released_samples": released_rows,
            }
            checks.append(row)
            print(
                f"结果: {'PASS' if passed else 'FAIL'}  "
                f"目标增量={target_delta:.1f}  "
                f"最大串扰={other_delta:.1f}  "
                f"释放残留={released_residual:.1f}"
            )

        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "read_only": True,
            "serial": serial,
            "can": args.can,
            "sensor_order": list(FINGERS),
            "thresholds": {
                "min_delta": args.min_delta,
                "crosstalk_ratio": args.crosstalk_ratio,
                "release_fraction": args.release_fraction,
            },
            "common_baseline_median5": common_baseline,
            "common_baseline_samples": common_baseline_rows,
            "checks": checks,
            "all_pass": all(row["pass"] for row in checks),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print("\n汇总:")
        for row in checks:
            print(
                f"  {row['finger']:<6} "
                f"{'PASS' if row['pass'] else 'FAIL'} "
                f"delta={row['target_delta']:.1f}"
            )
        print(f"总体: {'PASS' if payload['all_pass'] else 'FAIL'}")
        print(f"记录: {args.out}")
        return 0 if payload["all_pass"] else 2
    finally:
        close = getattr(api.hand, "close_can_interface", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
