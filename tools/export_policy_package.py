#!/usr/bin/env python3
"""Export a Stage-2 deploy bundle to the adopted immutable runtime format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from screwdriver_rl.deploy.policy_package import export_policy_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Stage-2 deploy.pth")
    parser.add_argument(
        "--metadata",
        required=True,
        help="JSON containing all non-inferable deployment metadata",
    )
    parser.add_argument("--output", required=True, help="new immutable package directory")
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    output = export_policy_package(args.bundle, metadata, args.output)
    print(output)


if __name__ == "__main__":
    main()
