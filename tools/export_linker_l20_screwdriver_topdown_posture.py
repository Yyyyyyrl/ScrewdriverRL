#!/usr/bin/env python3
"""Export one ranked Isaac candidate as the flat posture consumed by validators."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    PALM_DOWN_BASE_QUAT_WXYZ,
    PALM_NORMAL_LOCAL,
    UrdfGeometry,
    joint_limit_margins,
    quat_multiply_wxyz,
    rotate_vector_wxyz,
)


def _yaw_from_topdown_quaternion(quat: list[float]) -> float:
    # q = q_yaw * q_base, so q_yaw = q * conjugate(q_base).
    base = PALM_DOWN_BASE_QUAT_WXYZ
    relative = quat_multiply_wxyz(quat, (base[0], -base[1], -base[2], -base[3]))
    return float(2.0 * math.atan2(relative[3], relative[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--candidate_index", type=int, required=True)
    parser.add_argument(
        "--settled",
        action="store_true",
        help="Export the actual settled hand and screwdriver state instead of the command target.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.search.read_text())
    row = next(
        candidate for candidate in data["top_candidates"]
        if int(candidate["candidate_index"]) == args.candidate_index
    )
    hand = UrdfGeometry(REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf")
    joint_field = (
        "settled_joint_positions_independent"
        if args.settled
        else "joint_positions_independent"
    )
    joints = {name: float(value) for name, value in row[joint_field].items()}
    screwdriver_joints = (
        {name: float(value) for name, value in row.get(
            "settled_screwdriver_joint_positions", {}
        ).items()}
        if args.settled
        else {}
    )
    quat = [float(value) for value in row["root_quat_wxyz"]]
    margins = joint_limit_margins(hand, joints)
    output = {
        "posture_source": str(args.search),
        "posture_state": "settled" if args.settled else "controller_target",
        "physics_candidate_index": args.candidate_index,
        "physics_search_metrics": {
            key: row[key]
            for key in (
                "cost",
                "physics_contact_gate",
                "role_contact_fraction",
                "role_force_max_n",
                "role_force_mean_n",
                "terminated_or_truncated",
                "tilt_max_rad",
                "wrong_surface_force_max_n",
            )
        },
        "root_pos_w": [float(value) for value in row["root_pos_w"]],
        "root_quat_wxyz": quat,
        "yaw_rad": _yaw_from_topdown_quaternion(quat),
        "palm_normal_w": [float(value) for value in rotate_vector_wxyz(quat, PALM_NORMAL_LOCAL)],
        "joint_positions_independent": joints,
        "joint_positions_expanded": hand.expanded_positions(joints),
        "screwdriver_joint_positions": screwdriver_joints,
        "joint_limit_margins_rad": margins,
        "minimum_joint_limit_margin_rad": float(min(margins.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {output['posture_state']} candidate {args.candidate_index} to {args.output}"
    )


if __name__ == "__main__":
    main()
