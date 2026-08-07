#!/usr/bin/env python3
"""Fit a shallow, collision-checkable reset pose at a fixed physics-tested wrist."""

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

from screwdriver_rl.tasks.linker_l20.screwdriver_rotation_topdown_posture import (  # noqa: E402
    TOPDOWN_PREGRASP_POSITIONS,
)
from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    FINGERTIP_LINKS,
    NON_DISTAL_LINKS,
    PALM_NORMAL_LOCAL,
    SCREWDRIVER_ROOT_POS_W,
    TOPDOWN_CAP_THICKNESS_M,
    TOPDOWN_HANDLE_LENGTH_M,
    TOPDOWN_HANDLE_RADIUS_M,
    UrdfGeometry,
    cylindrical_surface_clearance,
    joint_limit_margins,
    rotate_vector_wxyz,
)
from tools.refine_linker_l20_screwdriver_topdown import (  # noqa: E402
    HAND_URDF,
    JOINT_ORDER,
    _joint_dict,
    _safe_joint_bounds,
)


TARGET_PENETRATION_M = 0.00050
MIN_NON_DISTAL_CLEARANCE_M = 0.0030


class FixedWristObjective:
    def __init__(
        self,
        model: UrdfGeometry,
        wrist: dict,
        *,
        handle_radius_m: float,
        target_penetration_m: float,
        target_surface_gap_m: float,
        use_convex_hulls: bool = False,
    ) -> None:
        self.model = model
        self.handle_radius_m = float(handle_radius_m)
        self.target_penetration_m = float(target_penetration_m)
        self.target_surface_gap_m = float(target_surface_gap_m)
        self.use_convex_hulls = bool(use_convex_hulls)
        self.root = np.asarray(wrist["root_pos_w"], dtype=np.float64)
        self.quat = np.asarray(wrist["root_quat_wxyz"], dtype=np.float64)
        source_joints = wrist.get("joint_positions_independent")
        if source_joints is None:
            values = [
                value
                for finger in ("index", "middle", "ring", "pinky", "thumb")
                for value in TOPDOWN_PREGRASP_POSITIONS[finger]
            ]
        else:
            values = [source_joints[name] for name in JOINT_ORDER]
        self.q0 = np.asarray(values, dtype=np.float64)
        self.body_base_z = float(SCREWDRIVER_ROOT_POS_W[2] + 0.100)
        self.body_top_z = self.body_base_z + TOPDOWN_HANDLE_LENGTH_M
        self.cap_top_z = self.body_top_z + TOPDOWN_CAP_THICKNESS_M
        self.distal_vertices = {}
        for name in FINGERTIP_LINKS:
            mesh = model.collision_mesh_local(name)
            if self.use_convex_hulls:
                mesh = mesh.convex_hull
            self.distal_vertices[name] = np.asarray(mesh.vertices, dtype=np.float64)
        self.non_distal_vertices = {}
        for name in NON_DISTAL_LINKS:
            mesh = model.collision_mesh_local(name)
            if self.use_convex_hulls:
                mesh = mesh.convex_hull
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            take = np.linspace(0, len(vertices) - 1, min(1024, len(vertices)), dtype=np.int64)
            self.non_distal_vertices[name] = vertices[take]

    @staticmethod
    def _world(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
        return points @ tf[:3, :3].T + tf[:3, 3]

    def _body(self, points: np.ndarray) -> np.ndarray:
        return cylindrical_surface_clearance(
            points,
            SCREWDRIVER_ROOT_POS_W[:2],
            self.handle_radius_m,
            self.body_base_z,
            self.body_top_z,
        )

    def _cap(self, points: np.ndarray) -> np.ndarray:
        return cylindrical_surface_clearance(
            points,
            SCREWDRIVER_ROOT_POS_W[:2],
            self.handle_radius_m,
            self.body_top_z,
            self.cap_top_z,
        )

    def diagnostics(self, q: np.ndarray) -> tuple[dict, dict]:
        joints = _joint_dict(q)
        fk = self.model.forward_kinematics(joints, self.root, self.quat)
        contacts: dict[str, dict] = {}
        for i, (finger, link) in enumerate(
            zip(("index", "middle", "ring", "pinky", "thumb"), FINGERTIP_LINKS)
        ):
            points = self._world(self.distal_vertices[link], fk[link])
            clearance = self._cap(points) if i == 0 else self._body(points)
            contacts[finger] = {
                "minimum_clearance_m": float(np.min(clearance)),
                "maximum_penetration_m": max(0.0, -float(np.min(clearance))),
                "closest_vertex_to_surface_m": float(np.min(np.abs(clearance))),
                "unique_vertices_checked": int(len(points)),
            }
        non_distal: dict[str, float] = {}
        for link, local in self.non_distal_vertices.items():
            points = self._world(local, fk[link])
            non_distal[link] = float(np.min(np.minimum(self._body(points), self._cap(points))))
        return contacts, non_distal

    def __call__(self, q: np.ndarray) -> np.ndarray:
        contacts, non_distal = self.diagnostics(q)
        residuals: list[float] = []
        desired_clearance = (
            self.target_surface_gap_m - self.target_penetration_m
        )
        for finger in ("index", "middle", "ring", "pinky", "thumb"):
            row = contacts[finger]
            residuals.append(
                (row["minimum_clearance_m"] - desired_clearance) / 0.00018
            )
            residuals.append(
                (row["closest_vertex_to_surface_m"] - abs(desired_clearance))
                / 0.00030
            )
        residuals.extend(
            max(0.0, MIN_NON_DISTAL_CLEARANCE_M - clearance) / 0.00025
            for clearance in non_distal.values()
        )
        residuals.extend(0.005 * ((q - self.q0) / 0.08))
        return np.asarray(residuals, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrist_posture", type=Path, required=True)
    parser.add_argument(
        "--handle_radius_m",
        type=float,
        default=TOPDOWN_HANDLE_RADIUS_M,
        help="Handle radius used by the selected diameter bucket.",
    )
    parser.add_argument(
        "--target_penetration_m", type=float, default=TARGET_PENETRATION_M
    )
    parser.add_argument("--target_surface_gap_m", type=float, default=0.0)
    parser.add_argument("--use_convex_hulls", action="store_true")
    parser.add_argument("--diff_step", type=float, default=None)
    parser.add_argument("--max_nfev", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target_penetration_m < 0.0 or args.target_surface_gap_m < 0.0:
        raise ValueError("reset penetration and surface gap targets must be non-negative")

    wrist = json.loads(args.wrist_posture.read_text())
    model = UrdfGeometry(HAND_URDF)
    objective = FixedWristObjective(
        model,
        wrist,
        handle_radius_m=args.handle_radius_m,
        target_penetration_m=args.target_penetration_m,
        target_surface_gap_m=args.target_surface_gap_m,
        use_convex_hulls=args.use_convex_hulls,
    )
    lo, hi = _safe_joint_bounds(model)
    result = least_squares(
        objective,
        np.clip(objective.q0, lo + 1.0e-8, hi - 1.0e-8),
        bounds=(lo, hi),
        max_nfev=args.max_nfev,
        diff_step=args.diff_step,
        loss="soft_l1",
        f_scale=1.0,
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        verbose=1,
    )
    joints = _joint_dict(result.x)
    margins = joint_limit_margins(model, joints)
    contacts, non_distal = objective.diagnostics(result.x)
    output = {
        "posture_source": str(args.wrist_posture),
        "purpose": "collision-safe reset state before compliant contact settling",
        "handle_radius_m": objective.handle_radius_m,
        "target_penetration_m": objective.target_penetration_m,
        "target_surface_gap_m": objective.target_surface_gap_m,
        "use_convex_hulls": objective.use_convex_hulls,
        "root_pos_w": [float(value) for value in objective.root],
        "root_quat_wxyz": [float(value) for value in objective.quat],
        "yaw_rad": float(wrist.get("yaw_rad", 0.0)),
        "palm_normal_w": [
            float(value) for value in rotate_vector_wxyz(objective.quat, PALM_NORMAL_LOCAL)
        ],
        "joint_positions_independent": joints,
        "joint_positions_expanded": model.expanded_positions(joints),
        "joint_limit_margins_rad": margins,
        "minimum_joint_limit_margin_rad": float(min(margins.values())),
        "dense_distal_contact_diagnostics": contacts,
        "sampled_non_distal_min_clearance_m": non_distal,
        "optimizer": {
            "cost": float(np.sum(objective(result.x) ** 2)),
            "nfev": int(result.nfev),
            "diff_step": args.diff_step,
            "status": int(result.status),
            "message": str(result.message),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
