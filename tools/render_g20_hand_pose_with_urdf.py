#!/usr/bin/env python3
"""Select a LinkerHand URDF, then delegate to render_g20_hand_pose."""

from __future__ import annotations

import sys
from pathlib import Path

from tools import render_g20_hand_pose


def main() -> int:
    try:
        index = sys.argv.index("--urdf")
        urdf = Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("--urdf PATH is required") from exc
    if not urdf.is_file():
        raise SystemExit(f"URDF does not exist: {urdf}")
    del sys.argv[index : index + 2]
    render_g20_hand_pose.HAND_URDF = urdf
    return render_g20_hand_pose.main()


if __name__ == "__main__":
    raise SystemExit(main())
