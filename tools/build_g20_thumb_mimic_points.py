#!/usr/bin/env python3
"""Extract paired thumb MCP/IP measurements from the archived summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    rows = [
        {
            "command_raw": point["command_raw"],
            "stable_readback_raw": point["stable_readback_raw"],
            "thumb_mcp_rad": point["thumb_mcp_rad"],
            "thumb_ip_rad": point["thumb_ip_rad"],
        }
        for point in payload["thumb_mcp"]["points"]
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
