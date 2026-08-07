#!/usr/bin/env python3
"""Refine top-down contacts into explicit cap/body height roles.

The dense contact refiner establishes shallow contact but does not by itself
distinguish the top edge of the body from the cap.  This deterministic second
pass keeps every distal mesh in its shallow contact band while placing index on
the cap and staggering middle/ring/pinky/thumb down the handle body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    SCREWDRIVER_ROOT_POS_W,
    TIP_MARKER_LINKS,
    UrdfGeometry,
    palm_down_quaternion_wxyz,
)
from tools.refine_linker_l20_screwdriver_topdown import (  # noqa: E402
    DenseRefinement,
    HAND_URDF,
    JOINT_ORDER,
    _joint_dict,
    _safe_joint_bounds,
)


# Index remains on the 1 mm cap at z=1.405..1.406.  Drive fingers are deliberately
# separated from that boundary and distributed along the 100 mm body.
ROLE_Z_TARGETS_M = np.asarray((1.4055, 1.3820, 1.3600, 1.3400, 1.3600), dtype=np.float64)


class RoleRefinement:
    def __init__(self, dense: DenseRefinement) -> None:
        self.dense = dense

    def __call__(self, x: np.ndarray) -> np.ndarray:
        root, yaw, q = self.dense.unpack(x)
        fk = self.dense.model.forward_kinematics(
            _joint_dict(q), root, palm_down_quaternion_wxyz(yaw)
        )
        marker_z = np.asarray([fk[name][2, 3] for name in TIP_MARKER_LINKS])
        role_height = (marker_z - ROLE_Z_TARGETS_M) / 0.0040

        # Retain the mechanically opposed wrap: middle/ring/pinky on one side,
        # thumb on the other.  Distal-mesh contact itself comes from dense(x).
        marker_xy = np.vstack([fk[name][:2, 3] for name in TIP_MARKER_LINKS])
        drive = marker_xy[1:] - np.asarray(SCREWDRIVER_ROOT_POS_W[:2])[None, :]
        drive_radius = np.linalg.norm(drive, axis=1)
        drive_unit = drive / np.maximum(drive_radius[:, None], 1.0e-9)
        non_thumb = np.sum(drive_unit[:3], axis=0)
        non_thumb /= max(float(np.linalg.norm(non_thumb)), 1.0e-9)
        opposed = np.asarray(((float(np.dot(drive_unit[3], non_thumb)) + 0.90) / 0.12,))
        return np.concatenate((self.dense(x), role_height, opposed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_nfev", type=int, default=1500)
    args = parser.parse_args()

    initial = json.loads(args.input.read_text())
    model = UrdfGeometry(HAND_URDF)
    dense = DenseRefinement(model, initial)
    objective = RoleRefinement(dense)
    q_lo, q_hi = _safe_joint_bounds(model)
    lower = np.concatenate(((-0.080, 0.100, 1.420, -0.70), q_lo))
    upper = np.concatenate(((0.060, 0.260, 1.580, 0.70), q_hi))
    q0 = np.asarray(
        [initial["joint_positions_independent"][name] for name in JOINT_ORDER],
        dtype=np.float64,
    )
    x0 = np.concatenate(
        (
            np.asarray(initial["root_pos_w"], dtype=np.float64),
            (float(initial["yaw_rad"]),),
            np.clip(q0, q_lo + 1.0e-8, q_hi - 1.0e-8),
        )
    )
    result = least_squares(
        objective,
        np.clip(x0, lower + 1.0e-8, upper - 1.0e-8),
        bounds=(lower, upper),
        max_nfev=args.max_nfev,
        loss="soft_l1",
        f_scale=1.0,
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        verbose=1,
    )
    summary = dense.summary(result.x, result)
    actual = np.asarray(
        [summary["tip_marker_positions_w"][finger][2] for finger in ("index", "middle", "ring", "pinky", "thumb")]
    )
    summary.update(
        {
            "role_refinement_cost": float(np.sum(objective(result.x) ** 2)),
            "role_height_targets_m": {
                finger: float(value)
                for finger, value in zip(("index", "middle", "ring", "pinky", "thumb"), ROLE_Z_TARGETS_M)
            },
            "role_height_errors_m": {
                finger: float(value)
                for finger, value in zip(
                    ("index", "middle", "ring", "pinky", "thumb"), actual - ROLE_Z_TARGETS_M
                )
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

