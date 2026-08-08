#!/usr/bin/env python3
"""Global per-finger workspace solve for top-down cap/body contact roles.

Each finger is independent kinematically once the fixed palm transform is chosen.
This tool uses differential evolution over that finger's guarded URDF range,
checks every unique distal collision vertex, and finds shallow contact at an
explicit handle height.  It escapes the local minimum near the cap that a single
all-hand least-squares refinement can retain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    FINGERTIP_LINKS,
    SCREWDRIVER_ROOT_POS_W,
    TIP_MARKER_LINKS,
    TOPDOWN_CAP_THICKNESS_M,
    TOPDOWN_HANDLE_LENGTH_M,
    TOPDOWN_HANDLE_RADIUS_M,
    UrdfGeometry,
    cylindrical_surface_clearance,
    palm_down_quaternion_wxyz,
)
from tools.refine_linker_l20_screwdriver_topdown import (  # noqa: E402
    DenseRefinement,
    HAND_URDF,
    JOINT_ORDER,
    _safe_joint_bounds,
)


FINGERS = ("index", "middle", "ring", "pinky", "thumb")
FINGER_JOINTS = {
    "index": ("index_mcp_roll", "index_mcp_pitch", "index_pip"),
    "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
    "ring": ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
    "pinky": ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
    "thumb": ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
}
FINGER_NON_DISTAL_LINKS = {
    "index": ("index_metacarpals", "index_proximal", "index_middle"),
    "middle": ("middle_metacarpals", "middle_proximal", "middle_middle"),
    "ring": ("ring_metacarpals", "ring_proximal", "ring_middle"),
    "pinky": ("pinky_metacarpals", "pinky_proximal", "pinky_middle"),
    "thumb": (
        "thumb_metacarpals_base2", "thumb_metacarpals_base1",
        "thumb_metacarpals", "thumb_proximal",
    ),
}
MIN_NON_DISTAL_CLEARANCE_M = 0.0050
MIN_WRONG_DISTAL_CLEARANCE_M = 0.0030
BASELINE_TARGET_SIGNED_CLEARANCE_M = {
    "index": -0.0004973257535692,
    "middle": -0.0016753144531690446,
    "ring": -0.0003177543183882116,
    "pinky": -0.0006097035834885384,
    "thumb": -0.003016124369946374,
}
ROLE_Z_TARGETS_M = {
    "index": 1.4055,
    "middle": 1.3850,
    "ring": 1.3650,
    "pinky": 1.3450,
    "thumb": 1.3600,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--popsize", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--root_x_offset", type=float, default=0.0)
    parser.add_argument("--root_y_offset", type=float, default=0.0)
    parser.add_argument(
        "--root_z_offset", type=float, default=0.0,
        help="World-z wrist offset applied before the per-finger global solve.",
    )
    parser.add_argument("--yaw_offset", type=float, default=0.0)
    parser.add_argument(
        "--baseline_preload_profile",
        action="store_true",
        help="Match each finger's signed target clearance from the baseline task.",
    )
    parser.add_argument(
        "--target_surface_clearance_m",
        type=float,
        default=None,
        help="Fit the minimum signed distal/surface distance instead of 0.5 mm penetration.",
    )
    args = parser.parse_args()

    initial = json.loads(args.input.read_text())
    model = UrdfGeometry(HAND_URDF)
    q_lo, q_hi = _safe_joint_bounds(model)
    safe = {
        name: (float(q_lo[i]), float(q_hi[i])) for i, name in enumerate(JOINT_ORDER)
    }
    root = np.asarray(initial["root_pos_w"], dtype=np.float64).copy()
    root[0] += args.root_x_offset
    root[1] += args.root_y_offset
    root[2] += args.root_z_offset
    yaw = float(initial["yaw_rad"]) + args.yaw_offset
    quat = palm_down_quaternion_wxyz(yaw)
    joints = dict(initial["joint_positions_independent"])

    body_base_z = float(SCREWDRIVER_ROOT_POS_W[2] + 0.100)
    body_top_z = body_base_z + TOPDOWN_HANDLE_LENGTH_M
    cap_top_z = body_top_z + TOPDOWN_CAP_THICKNESS_M
    axis_xy = np.asarray(SCREWDRIVER_ROOT_POS_W[:2], dtype=np.float64)
    diagnostics: dict[str, dict] = {}

    for finger_index, finger in enumerate(FINGERS):
        names = FINGER_JOINTS[finger]
        distal = FINGERTIP_LINKS[finger_index]
        marker = TIP_MARKER_LINKS[finger_index]
        local = np.asarray(model.collision_mesh_local(distal).vertices, dtype=np.float64)
        non_distal_local = {}
        for link in FINGER_NON_DISTAL_LINKS[finger]:
            hull = model.collision_mesh_local(link).convex_hull
            non_distal_local[link] = np.vstack(
                (np.asarray(hull.vertices), np.asarray(hull.triangles_center))
            ).astype(np.float64, copy=False)
        q_initial = np.asarray([joints[name] for name in names], dtype=np.float64)
        fk_initial = model.forward_kinematics(joints, root, quat)
        xy_initial = fk_initial[marker][:2, 3].copy()

        def clearance(points: np.ndarray) -> np.ndarray:
            if finger == "index":
                return cylindrical_surface_clearance(
                    points, axis_xy, TOPDOWN_HANDLE_RADIUS_M, body_top_z, cap_top_z
                )
            return cylindrical_surface_clearance(
                points, axis_xy, TOPDOWN_HANDLE_RADIUS_M, body_base_z, body_top_z
            )

        def objective(values: np.ndarray) -> float:
            candidate = dict(joints)
            candidate.update({name: float(value) for name, value in zip(names, values)})
            fk = model.forward_kinematics(candidate, root, quat)
            transform = fk[distal]
            points = local @ transform[:3, :3].T + transform[:3, 3]
            distance = clearance(points)
            penetration = max(0.0, -float(np.min(distance)))
            closest = float(np.min(np.abs(distance)))
            wrong_distal_penalty = 0.0
            if finger != "index":
                wrong_cap_clearance = cylindrical_surface_clearance(
                    points, axis_xy, TOPDOWN_HANDLE_RADIUS_M, body_top_z, cap_top_z
                )
                wrong_distal_penalty = max(
                    0.0,
                    MIN_WRONG_DISTAL_CLEARANCE_M
                    - float(np.min(wrong_cap_clearance)),
                ) / 0.0005
            if args.target_surface_clearance_m is not None:
                contact_fit = (
                    (float(np.min(distance)) - args.target_surface_clearance_m) / 0.00018
                ) ** 2
            elif args.baseline_preload_profile:
                contact_fit = (
                    (
                        float(np.min(distance))
                        - BASELINE_TARGET_SIGNED_CLEARANCE_M[finger]
                    )
                    / 0.00018
                ) ** 2
            else:
                contact_fit = (
                    ((penetration - 0.00050) / 0.00018) ** 2
                    + (closest / 0.00030) ** 2
                )
            marker_pos = fk[marker][:3, 3]
            height = (float(marker_pos[2]) - ROLE_Z_TARGETS_M[finger]) / 0.0035
            xy_drift = np.linalg.norm(marker_pos[:2] - xy_initial) / 0.018
            q_drift = np.linalg.norm((values - q_initial) / 0.35)
            non_distal_penalty = 0.0
            for link, link_local in non_distal_local.items():
                link_tf = fk[link]
                link_points = link_local @ link_tf[:3, :3].T + link_tf[:3, 3]
                link_clearance = np.minimum(
                    cylindrical_surface_clearance(
                        link_points, axis_xy, TOPDOWN_HANDLE_RADIUS_M, body_base_z, body_top_z
                    ),
                    cylindrical_surface_clearance(
                        link_points, axis_xy, TOPDOWN_HANDLE_RADIUS_M, body_top_z, cap_top_z
                    ),
                )
                violation = max(
                    0.0, MIN_NON_DISTAL_CLEARANCE_M - float(np.min(link_clearance))
                ) / 0.0005
                non_distal_penalty += violation**2
            return float(
                contact_fit
                + height**2
                + 0.12 * xy_drift**2
                + 0.025 * q_drift**2
                + 4.0 * non_distal_penalty
                + 4.0 * wrong_distal_penalty**2
            )

        result = differential_evolution(
            objective,
            bounds=[safe[name] for name in names],
            seed=args.seed + finger_index,
            maxiter=args.maxiter,
            popsize=args.popsize,
            tol=1.0e-8,
            polish=True,
            workers=1,
            updating="immediate",
        )
        joints.update({name: float(value) for name, value in zip(names, result.x)})
        fk = model.forward_kinematics(joints, root, quat)
        transform = fk[distal]
        points = local @ transform[:3, :3].T + transform[:3, 3]
        distance = clearance(points)
        diagnostics[finger] = {
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "joint_positions": {name: joints[name] for name in names},
            "marker_position_w": [float(v) for v in fk[marker][:3, 3]],
            "target_marker_z_m": ROLE_Z_TARGETS_M[finger],
            "maximum_penetration_m": max(0.0, -float(np.min(distance))),
            "closest_vertex_to_surface_m": float(np.min(np.abs(distance))),
        }
        print(f"{finger}: {json.dumps(diagnostics[finger], sort_keys=True)}", flush=True)

    # Reuse the dense refiner's audited output schema without another solve.
    dense_input = dict(initial)
    dense_input["joint_positions_independent"] = joints
    dense = DenseRefinement(model, dense_input)
    x = np.concatenate(
        (
            root,
            (yaw,),
            np.asarray([joints[name] for name in JOINT_ORDER], dtype=np.float64),
        )
    )

    class Result:
        nfev = sum(row["evaluations"] for row in diagnostics.values())
        status = 1

    summary = dense.summary(x, Result())
    summary["fingerwise_global_solve"] = diagnostics
    summary["fingerwise_fit_seed"] = args.seed
    summary["target_surface_clearance_m"] = args.target_surface_clearance_m
    summary["baseline_preload_profile"] = args.baseline_preload_profile
    summary["baseline_target_signed_clearance_m"] = (
        BASELINE_TARGET_SIGNED_CLEARANCE_M if args.baseline_preload_profile else None
    )
    summary["root_offset_m"] = [
        args.root_x_offset, args.root_y_offset, args.root_z_offset
    ]
    summary["yaw_offset_rad"] = args.yaw_offset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

