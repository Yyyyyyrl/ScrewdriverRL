#!/usr/bin/env python3
"""Record explicit user approval of a generated G20 visual view gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if payload.get("status") not in (
        "pending_user_view_approval", "user_view_approved"
    ):
        raise ValueError(f"unexpected view gate status: {payload.get('status')}")
    payload["status"] = "user_view_approved"
    payload["user_review"] = {
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "result": "pass",
        "scope": "camera view, scale, target visibility, and focus rendering",
    }
    args.manifest.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(args.manifest.resolve()),
        "joint": payload.get("joint"),
        "status": payload["status"],
        "focus_finger": payload.get("focus_finger"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
