#!/usr/bin/env python3
"""Refine a fitted Linker L20 top-down grasp against every distal mesh vertex.

The coarse multi-start fitter deliberately uses sparse vertices for speed.  This
second deterministic solve starts from its best candidate, welds only exact STL
vertex duplicates (no face reduction), and enforces a shallow contact band on
all five real distal collision meshes.  The strict full-mesh/convex validator is
still the authority after this refinement.
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
    FINGERTIP_LINKS,
    NON_DISTAL_LINKS,
    PALM_NORMAL_LOCAL,
    SCREWDRIVER_ROOT_POS_W,
    TIP_MARKER_LINKS,
    TOPDOWN_CAP_THICKNESS_M,
    TOPDOWN_HANDLE_LENGTH_M,
    TOPDOWN_HANDLE_RADIUS_M,
    UrdfGeometry,
    cylindrical_surface_clearance,
    joint_limit_margins,
    palm_down_quaternion_wxyz,
    rotate_vector_wxyz,
)


HAND_URDF = REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
JOINT_ORDER = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
)
FINGERS = ("index", "middle", "ring", "pinky", "thumb")
MIN_JOINT_MARGIN_RAD = 0.11
TARGET_PENETRATION_M = 0.00055
MIN_NON_DISTAL_CLEARANCE_M = 0.0020


def _safe_joint_bounds(model: UrdfGeometry) -> tuple[np.ndarray, np.ndarray]:
    limits = model.joint_limits(MIN_JOINT_MARGIN_RAD)
    lo = np.asarray([limits[name][0] for name in JOINT_ORDER], dtype=np.float64)
    hi = np.asarray([limits[name][1] for name in JOINT_ORDER], dtype=np.float64)
    index = {name: i for i, name in enumerate(JOINT_ORDER)}
    for follower in model.joints.values():
        if follower.mimic is None or follower.mimic.source not in index:
            continue
        if follower.lower is None or follower.upper is None:
            continue
        source = follower.mimic.source
        mult = follower.mimic.multiplier
        offset = follower.mimic.offset
        values = (
            (follower.lower + MIN_JOINT_MARGIN_RAD - offset) / mult,
            (follower.upper - MIN_JOINT_MARGIN_RAD - offset) / mult,
        )
        i = index[source]
        lo[i] = max(lo[i], min(values))
        hi[i] = min(hi[i], max(values))
    return lo, hi


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(JOINT_ORDER, values)}


class DenseRefinement:
    def __init__(self, model: UrdfGeometry, initial: dict) -> None:
        self.model = model
        self.initial = initial
        self.root0 = np.asarray(initial["root_pos_w"], dtype=np.float64)
        self.yaw0 = float(initial["yaw_rad"])
        self.q0 = np.asarray(
            [initial["joint_positions_independent"][name] for name in JOINT_ORDER],
            dtype=np.float64,
        )
        self.body_base_z = float(SCREWDRIVER_ROOT_POS_W[2] + 0.100)
        self.body_top_z = self.body_base_z + TOPDOWN_HANDLE_LENGTH_M
        self.cap_top_z = self.body_top_z + TOPDOWN_CAP_THICKNESS_M
        # Every unique vertex of every distal collision mesh is used.  The
        # geometry loader preserves all triangles and only welds exact duplicate
        # coordinates from the STL encoding.
        self.distal_vertices = {
            name: np.asarray(model.collision_mesh_local(name).vertices, dtype=np.float64)
            for name in FINGERTIP_LINKS
        }
        # Non-distal clearance already passed the strict validator.  A stable
        # deterministic subset prevents refinement from eroding that margin;
        # the final validator rechecks every vertex and convex hull.
        self.non_distal_vertices = {}
        for name in NON_DISTAL_LINKS:
            vertices = np.asarray(model.collision_mesh_local(name).vertices, dtype=np.float64)
            take = np.linspace(0, len(vertices) - 1, min(768, len(vertices)), dtype=np.int64)
            self.non_distal_vertices[name] = vertices[take]

    @staticmethod
    def _world(local: np.ndarray, transform: np.ndarray) -> np.ndarray:
        return local @ transform[:3, :3].T + transform[:3, 3]

    def _body(self, points: np.ndarray) -> np.ndarray:
        return cylindrical_surface_clearance(
            points,
            SCREWDRIVER_ROOT_POS_W[:2],
            TOPDOWN_HANDLE_RADIUS_M,
            self.body_base_z,
            self.body_top_z,
        )

    def _cap(self, points: np.ndarray) -> np.ndarray:
        return cylindrical_surface_clearance(
            points,
            SCREWDRIVER_ROOT_POS_W[:2],
            TOPDOWN_HANDLE_RADIUS_M,
            self.body_top_z,
            self.cap_top_z,
        )

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return x[:3], float(x[3]), x[4:]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        root, yaw, q = self.unpack(x)
        joints = _joint_dict(q)
        fk = self.model.forward_kinematics(joints, root, palm_down_quaternion_wxyz(yaw))
        residuals: list[float] = []

        for i, name in enumerate(FINGERTIP_LINKS):
            points = self._world(self.distal_vertices[name], fk[name])
            clearance = self._cap(points) if i == 0 else self._body(points)
            penetration = max(0.0, -float(np.min(clearance)))
            closest = float(np.min(np.abs(clearance)))
            residuals.append((penetration - TARGET_PENETRATION_M) / 0.00018)
            residuals.append(closest / 0.00030)

        for name in NON_DISTAL_LINKS:
            points = self._world(self.non_distal_vertices[name], fk[name])
            clearance = np.minimum(self._body(points), self._cap(points))
            residuals.append(
                max(0.0, MIN_NON_DISTAL_CLEARANCE_M - float(np.min(clearance))) / 0.0010
            )

        # Preserve the already good contact distribution and self-clearance,
        # while allowing sub-millimetre surface corrections.
        residuals.extend(0.035 * ((root - self.root0) / 0.004))
        residuals.append(0.025 * ((yaw - self.yaw0) / 0.05))
        residuals.extend(0.030 * ((q - self.q0) / 0.08))
        return np.asarray(residuals, dtype=np.float64)

    def summary(self, x: np.ndarray, result) -> dict:
        root, yaw, q = self.unpack(x)
        joints = _joint_dict(q)
        quat = palm_down_quaternion_wxyz(yaw)
        fk = self.model.forward_kinematics(joints, root, quat)
        expanded = self.model.expanded_positions(joints)
        margins = joint_limit_margins(self.model, joints)
        contacts: dict[str, dict[str, float]] = {}
        for i, (finger, name) in enumerate(zip(FINGERS, FINGERTIP_LINKS)):
            points = self._world(self.distal_vertices[name], fk[name])
            clearance = self._cap(points) if i == 0 else self._body(points)
            contacts[finger] = {
                "minimum_clearance_m": float(np.min(clearance)),
                "maximum_penetration_m": max(0.0, -float(np.min(clearance))),
                "closest_vertex_to_surface_m": float(np.min(np.abs(clearance))),
                "unique_vertices_checked": int(len(points)),
            }
        tips = {
            finger: [float(v) for v in fk[name][:3, 3]]
            for finger, name in zip(FINGERS, TIP_MARKER_LINKS)
        }
        output = dict(self.initial)
        output.update(
            {
                "refinement_source": self.initial.get("posture_json", "coarse multi-start fit"),
                "refinement_nfev": int(result.nfev),
                "refinement_status": int(result.status),
                "refinement_cost": float(np.sum(self(x) ** 2)),
                "minimum_required_joint_margin_rad": MIN_JOINT_MARGIN_RAD,
                "root_pos_w": [float(v) for v in root],
                "root_quat_wxyz": [float(v) for v in quat],
                "yaw_rad": yaw,
                "palm_normal_w": [
                    float(v) for v in rotate_vector_wxyz(quat, PALM_NORMAL_LOCAL)
                ],
                "joint_positions_independent": joints,
                "joint_positions_expanded": expanded,
                "joint_limit_margins_rad": margins,
                "minimum_joint_limit_margin_rad": float(min(margins.values())),
                "tip_marker_positions_w": tips,
                "dense_distal_contact_diagnostics": contacts,
            }
        )
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_nfev", type=int, default=900)
    args = parser.parse_args()

    initial = json.loads(args.input.read_text())
    model = UrdfGeometry(HAND_URDF)
    objective = DenseRefinement(model, initial)
    q_lo, q_hi = _safe_joint_bounds(model)
    lower = np.concatenate(((-0.080, 0.100, 1.420, -0.70), q_lo))
    upper = np.concatenate(((0.060, 0.260, 1.580, 0.70), q_hi))
    x0 = np.concatenate(
        (
            objective.root0,
            (objective.yaw0,),
            np.clip(objective.q0, q_lo + 1.0e-8, q_hi - 1.0e-8),
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
    summary = objective.summary(result.x, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
