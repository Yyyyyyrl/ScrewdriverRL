#!/usr/bin/env python3
"""Compose per-finger Isaac search results into a deterministic joint candidate bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FINGER_JOINTS = {
    "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
    "ring": ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
    "pinky": ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
    "thumb": ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
}


def _search_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data.get("top_candidates")
    if not rows:
        raise ValueError(f"{path} has no top_candidates")
    return rows


def _candidate_by_index(path: Path, index: int) -> dict:
    for row in _search_rows(path):
        if int(row["candidate_index"]) == index:
            return row
    raise ValueError(f"candidate {index} is not retained in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base_index", type=int, required=True)
    for finger in FINGER_JOINTS:
        parser.add_argument(f"--{finger}", type=Path, required=True)
    parser.add_argument("--bank_top", type=int, default=32)
    parser.add_argument("--num_candidates", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.bank_top < 1 or args.num_candidates < 1:
        raise ValueError("bank_top and num_candidates must be positive")
    base = _candidate_by_index(args.base, args.base_index)
    paths = {finger: getattr(args, finger) for finger in FINGER_JOINTS}
    banks = {finger: _search_rows(path)[: args.bank_top] for finger, path in paths.items()}
    if any(len(rows) < args.bank_top for rows in banks.values()):
        raise ValueError("one or more source banks contain fewer rows than --bank_top")

    # Candidate zero preserves the already verified index+pinky seed.  Candidate
    # one combines every bank's best row; the remaining unique tuples uniformly
    # cover the retained Cartesian product with a deterministic RNG.
    selections: list[tuple[int, int, int, int] | None] = [None]
    if args.num_candidates > 1:
        selections.append((0, 0, 0, 0))
    seen = {(0, 0, 0, 0)}
    rng = np.random.default_rng(args.seed)
    while len(selections) < args.num_candidates:
        selection = tuple(int(v) for v in rng.integers(0, args.bank_top, size=4))
        if selection in seen:
            continue
        seen.add(selection)
        selections.append(selection)

    fingers = tuple(FINGER_JOINTS)
    candidates: list[dict] = []
    for candidate_index, selection in enumerate(selections):
        q = dict(base["joint_positions_independent"])
        sources: dict[str, dict] = {}
        if selection is not None:
            for finger, rank in zip(fingers, selection):
                row = banks[finger][rank]
                for name in FINGER_JOINTS[finger]:
                    q[name] = float(row["joint_positions_independent"][name])
                sources[finger] = {
                    "rank": rank,
                    "candidate_index": int(row["candidate_index"]),
                }
        candidates.append(
            {
                "candidate_index": candidate_index,
                "label": "base_seed" if selection is None else "combined_finger_banks",
                "root_pos_w": list(base["root_pos_w"]),
                "root_quat_wxyz": list(base["root_quat_wxyz"]),
                "joint_positions_independent": q,
                "finger_sources": sources,
            }
        )

    output = {
        "base": {"path": str(args.base), "candidate_index": args.base_index},
        "finger_banks": {finger: str(path) for finger, path in paths.items()},
        "bank_top": args.bank_top,
        "seed": args.seed,
        "num_candidates": len(candidates),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
