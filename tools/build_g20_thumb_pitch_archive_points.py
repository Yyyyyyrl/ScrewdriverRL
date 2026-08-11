#!/usr/bin/env python3
"""Build camera-source point tables for archived thumb MCP/CMC-pitch sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--joint", choices=("thumb_mcp", "thumb_cmc_pitch"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(
        (args.session / "thumb_mcp_ip_pitch_summary.json").read_text()
    )
    if args.joint == "thumb_mcp":
        points = summary["thumb_mcp"]["points"]
        q_key = "thumb_mcp_rad"
        prefix = "mcp"
    else:
        points = summary["thumb_cmc_pitch"]["points"]
        q_key = "thumb_cmc_pitch_rad"
        prefix = "pitch"

    zero_samples = Path(summary["formal_zero_reference"]["sources"][0])
    zero_summary = zero_samples.with_name(
        zero_samples.name.replace("_samples.csv", "_summary.json")
    )
    rows = []
    for point in points:
        raw = int(point["command_raw"])
        source = (
            zero_summary
            if raw == 255
            else args.session
            / f"{prefix}_raw{raw:03d}"
            / "camera_fixed_exp_180f_summary.json"
        )
        rows.append(
            {
                "command_raw": raw,
                "stable_readback_raw": point["stable_readback_raw"],
                "physical_rad": point[q_key],
                "camera_summary": str(source),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
