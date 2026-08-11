#!/usr/bin/env python3
"""Refine a G20 camera pose with bounded projective CAD-to-depth alignment.

Unlike unconstrained nearest-neighbour ICP, this solver keeps the update close
to the supplied camera pose and compares each visible hand-base CAD sample with
the D435 depth at the same image location.  A caller-supplied polygon isolates
the rigid hand base from the robot flange and articulated fingers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation

from calibrate_g20_camera_extrinsic import (
    base_to_camera,
    camera_json,
    camera_pose,
    draw_projection,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True, help="aligned uint16 .npy")
    parser.add_argument("--depth-scale", type=float, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--initial-camera", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument(
        "--polygon",
        nargs="+",
        type=int,
        required=True,
        metavar="X_Y",
        help="flat x y vertex sequence, at least three vertices",
    )
    parser.add_argument("--depth-range", nargs=2, type=float, default=(0.32, 0.58))
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--max-translation-mm", type=float, default=35.0)
    parser.add_argument("--max-rotation-deg", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--de-popsize", type=int, default=12)
    parser.add_argument("--de-iterations", type=int, default=45)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def compose_delta(initial: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Apply a camera-frame SE(3) correction to the initial base transform."""
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(parameters[3:]).as_matrix()
    delta[:3, 3] = parameters[:3]
    return delta @ initial


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image.astype(np.float32),
        u.astype(np.float32).reshape(-1, 1),
        v.astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(-1).astype(np.float64)


def robust_depth_cost(residual: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() < 300:
        return 1.0
    absolute = np.abs(residual[valid])
    clipped = np.minimum(absolute, 0.030)
    cutoff = np.quantile(clipped, 0.78)
    trimmed = clipped[clipped <= cutoff]
    # A small coverage term prevents the optimizer from moving CAD samples out
    # of the mask merely to discard difficult correspondences.
    coverage_penalty = 0.015 * (1.0 - float(valid.mean()))
    return float(np.mean(trimmed * trimmed) + coverage_penalty * coverage_penalty)


def main() -> int:
    args = parse_args()
    if len(args.polygon) < 6 or len(args.polygon) % 2:
        raise ValueError("--polygon requires x y pairs for at least three vertices")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    color = cv2.imread(str(args.color), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f"failed to read {args.color}")
    raw_depth = np.load(args.depth)
    depth = raw_depth.astype(np.float64) * args.depth_scale
    intr = json.loads(args.intrinsics.read_text())
    polygon = np.asarray(args.polygon, dtype=np.int32).reshape(-1, 2)
    roi_mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon], 255)
    lo, hi = args.depth_range
    target_valid = (roi_mask > 0) & (depth >= lo) & (depth <= hi)
    target_depth = np.where(target_valid, depth, 0.0)

    mesh = trimesh.load_mesh(args.mesh, process=False)
    rng_state = np.random.get_state()
    np.random.seed(args.seed)
    sampled, face_ids = trimesh.sample.sample_surface(mesh, args.samples)
    np.random.set_state(rng_state)
    normals = np.asarray(mesh.face_normals)[face_ids]

    view_data = json.loads(args.initial_camera.read_text())
    view = view_data[0] if isinstance(view_data, list) else view_data
    position, rotation_base_from_camera = camera_pose(view)
    initial = base_to_camera(position, rotation_base_from_camera)

    sampled_initial = transform_points(sampled, initial)
    normal_initial = normals @ initial[:3, :3].T
    ray_to_camera = -sampled_initial / np.linalg.norm(sampled_initial, axis=1, keepdims=True)
    front_facing = np.einsum("ij,ij->i", normal_initial, ray_to_camera) > 0.12
    sampled = sampled[front_facing]
    normals = normals[front_facing]
    # Deterministic thinning keeps global optimization quick.
    if len(sampled) > 18_000:
        choose = np.random.default_rng(args.seed).choice(len(sampled), 18_000, replace=False)
        sampled = sampled[choose]
        normals = normals[choose]

    h, w = depth.shape

    def evaluate(parameters: np.ndarray) -> float:
        transform = compose_delta(initial, parameters)
        points = transform_points(sampled, transform)
        z = points[:, 2]
        u = intr["fx"] * points[:, 0] / np.maximum(z, 1.0e-6) + intr["ppx"]
        v = intr["fy"] * points[:, 1] / np.maximum(z, 1.0e-6) + intr["ppy"]
        observed = bilinear_sample(target_depth, u, v)
        mask_value = bilinear_sample(roi_mask, u, v)
        valid = (
            (z > 0.0)
            & (u >= 1.0) & (u < w - 2.0)
            & (v >= 1.0) & (v < h - 2.0)
            & (mask_value > 200.0)
            & (observed >= lo) & (observed <= hi)
        )
        return robust_depth_cost(z - observed, valid)

    max_t = args.max_translation_mm / 1000.0
    max_r = np.radians(args.max_rotation_deg)
    bounds = [(-max_t, max_t)] * 3 + [(-max_r, max_r)] * 3
    initial_cost = evaluate(np.zeros(6))
    global_result = differential_evolution(
        evaluate,
        bounds,
        seed=args.seed,
        popsize=args.de_popsize,
        maxiter=args.de_iterations,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1.0e-4,
    )
    local_result = minimize(
        evaluate,
        global_result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 500, "xtol": 2.0e-5, "ftol": 1.0e-7},
    )
    parameters = local_result.x
    refined = compose_delta(initial, parameters)
    refined_cost = evaluate(parameters)

    overlay = color.copy()
    cv2.polylines(overlay, [polygon], True, (255, 180, 0), 2, cv2.LINE_AA)
    draw_projection(overlay, transform_points(sampled, initial), intr, (0, 0, 255))
    draw_projection(overlay, transform_points(sampled, refined), intr, (0, 255, 0))
    cv2.putText(
        overlay,
        "cyan=rigid ROI red=initial green=bounded fit",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "cyan=rigid ROI red=initial green=bounded fit",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "projection_overlay.png"), overlay)

    camera = camera_json(refined)
    (args.out_dir / "camera_projective.json").write_text(json.dumps([camera], indent=2) + "\n")
    payload = {
        "schema_version": 1,
        "method": "bounded_projective_rigid_hand_base_depth",
        "inputs": {
            "color": str(args.color),
            "depth": str(args.depth),
            "mesh": str(args.mesh),
            "intrinsics": intr,
            "polygon_xy": polygon.tolist(),
            "depth_range_m": list(args.depth_range),
            "seed": args.seed,
            "sample_count_requested": args.samples,
            "sample_count_visible": int(len(sampled)),
            "max_translation_mm": args.max_translation_mm,
            "max_rotation_deg": args.max_rotation_deg,
        },
        "initial_camera": view,
        "refined_camera": camera,
        "delta_camera_frame": {
            "translation_m": parameters[:3].tolist(),
            "rotation_vector_rad": parameters[3:].tolist(),
            "translation_norm_mm": float(np.linalg.norm(parameters[:3]) * 1000.0),
            "rotation_angle_deg": float(np.degrees(np.linalg.norm(parameters[3:]))),
        },
        "optimization": {
            "initial_cost": initial_cost,
            "global_cost": float(global_result.fun),
            "final_cost": refined_cost,
            "relative_improvement": float((initial_cost - refined_cost) / initial_cost),
            "global_success": bool(global_result.success),
            "global_message": str(global_result.message),
            "local_success": bool(local_result.success),
            "local_message": str(local_result.message),
        },
    }
    (args.out_dir / "registration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["delta_camera_frame"], indent=2))
    print(json.dumps(payload["optimization"], indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
