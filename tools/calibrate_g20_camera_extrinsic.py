#!/usr/bin/env python3
"""Register the rigid G20 hand-base CAD to an aligned D435 depth snapshot.

The solver uses only a caller-selected rigid palm ROI.  Articulated fingers are
excluded so joint-mapping errors cannot leak into the camera extrinsic.  The
input camera JSON is an initialization, not the result: a deterministic,
multi-scale point-to-plane ICP refines the hand-base-to-camera transform and
writes auditable clouds, overlays, and residual metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True, help="aligned uint16 .npy")
    parser.add_argument("--depth-scale", type=float, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--initial-camera", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"), required=True)
    parser.add_argument("--depth-range", nargs=2, type=float, default=(0.30, 0.58))
    parser.add_argument("--samples", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def look_at_rotation(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def camera_pose(view: dict) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(view["pos"], dtype=float)
    if "quat" in view:
        w, x, y, z = map(float, view["quat"])
        rotation_base_from_camera = Rotation.from_quat([x, y, z, w]).as_matrix()
    else:
        rotation_base_from_camera = look_at_rotation(
            position,
            np.asarray(view["look_at"], dtype=float),
            np.asarray(view.get("up", [-1.0, 0.0, 0.0]), dtype=float),
        )
    return position, rotation_base_from_camera


def base_to_camera(position: np.ndarray, rotation_base_from_camera: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation_base_from_camera.T
    transform[:3, 3] = -rotation_base_from_camera.T @ position
    return transform


def camera_json(transform_camera_from_base: np.ndarray) -> dict:
    rotation_camera_from_base = transform_camera_from_base[:3, :3]
    translation_camera_from_base = transform_camera_from_base[:3, 3]
    rotation_base_from_camera = rotation_camera_from_base.T
    position_base = -rotation_base_from_camera @ translation_camera_from_base
    x, y, z, w = Rotation.from_matrix(rotation_base_from_camera).as_quat()
    return {
        "name": "d435_icp",
        "pos": position_base.tolist(),
        "quat": [float(w), float(x), float(y), float(z)],
    }


def cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    return result


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def project(points_camera: np.ndarray, intr: dict) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    valid = z > 1.0e-5
    pixels = np.empty((len(points_camera), 2), dtype=float)
    pixels[:, 0] = intr["fx"] * points_camera[:, 0] / np.maximum(z, 1.0e-5) + intr["ppx"]
    pixels[:, 1] = intr["fy"] * points_camera[:, 1] / np.maximum(z, 1.0e-5) + intr["ppy"]
    return pixels, valid


def draw_projection(image: np.ndarray, points_camera: np.ndarray, intr: dict, color: tuple[int, int, int]) -> None:
    pixels, valid = project(points_camera, intr)
    h, w = image.shape[:2]
    valid &= (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    rounded = np.rint(pixels[valid]).astype(int)
    for u, v in rounded[::4]:
        cv2.circle(image, (int(u), int(v)), 1, color, -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    color = cv2.imread(str(args.color), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f"failed to read {args.color}")
    depth = np.load(args.depth).astype(np.float64) * args.depth_scale
    intr = json.loads(args.intrinsics.read_text())
    x0, y0, x1, y1 = args.roi
    lo, hi = args.depth_range
    yy, xx = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    mask = (
        (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        & (depth >= lo) & (depth <= hi)
    )
    z = depth[mask]
    u = xx[mask].astype(float)
    v = yy[mask].astype(float)
    target_points = np.column_stack((
        (u - intr["ppx"]) / intr["fx"] * z,
        (v - intr["ppy"]) / intr["fy"] * z,
        z,
    ))
    target = cloud(target_points).voxel_down_sample(0.0015)
    target.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

    mesh = trimesh.load_mesh(args.mesh, process=False)
    np.random.seed(args.seed)
    sampled, face_ids = trimesh.sample.sample_surface(mesh, args.samples)
    normals = np.asarray(mesh.face_normals)[face_ids]

    view_data = json.loads(args.initial_camera.read_text())
    view = view_data[0] if isinstance(view_data, list) else view_data
    position, rotation_base_from_camera = camera_pose(view)
    initial = base_to_camera(position, rotation_base_from_camera)
    sampled_camera = transform_points(sampled, initial)
    normal_camera = normals @ initial[:3, :3].T
    ray_to_camera = -sampled_camera / np.linalg.norm(sampled_camera, axis=1, keepdims=True)
    front_facing = np.einsum("ij,ij->i", normal_camera, ray_to_camera) > 0.05
    pixels, positive = project(sampled_camera, intr)
    visible_roi = (
        positive & front_facing
        & (pixels[:, 0] >= x0) & (pixels[:, 0] < x1)
        & (pixels[:, 1] >= y0) & (pixels[:, 1] < y1)
        & (sampled_camera[:, 2] >= lo - 0.08)
        & (sampled_camera[:, 2] <= hi + 0.08)
    )
    source_points = sampled[visible_roi]
    if len(source_points) < 500:
        raise RuntimeError(f"only {len(source_points)} initialized CAD samples in ROI")
    source = cloud(source_points)

    current = initial.copy()
    stages = []
    for voxel, threshold, iterations in (
        (0.010, 0.040, 80),
        (0.005, 0.022, 80),
        (0.0025, 0.012, 120),
    ):
        source_down = source.voxel_down_sample(voxel)
        target_down = target.voxel_down_sample(voxel)
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4.0, max_nn=50))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4.0, max_nn=50))
        loss = o3d.pipelines.registration.TukeyLoss(k=threshold * 0.5)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            threshold,
            current,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        current = result.transformation
        stages.append({
            "voxel_m": voxel,
            "threshold_m": threshold,
            "iterations": iterations,
            "fitness": float(result.fitness),
            "inlier_rmse_m": float(result.inlier_rmse),
        })

    registered = source.transform(current.copy())
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, 0.012, current
    )
    rotation_delta = current[:3, :3] @ initial[:3, :3].T
    angle_delta_deg = float(np.degrees(Rotation.from_matrix(rotation_delta).magnitude()))
    translation_delta_m = float(np.linalg.norm(current[:3, 3] - initial[:3, 3]))

    o3d.io.write_point_cloud(str(args.out_dir / "target_real_roi.ply"), target)
    initial_cloud = cloud(transform_points(source_points, initial))
    o3d.io.write_point_cloud(str(args.out_dir / "source_initial.ply"), initial_cloud)
    o3d.io.write_point_cloud(str(args.out_dir / "source_registered.ply"), registered)

    overlay = color.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 180, 0), 2)
    draw_projection(overlay, transform_points(source_points, initial), intr, (0, 0, 255))
    draw_projection(overlay, transform_points(source_points, current), intr, (0, 255, 0))
    cv2.putText(overlay, "red=initial green=ICP", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, "red=initial green=ICP", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out_dir / "projection_overlay.png"), overlay)

    camera = camera_json(current)
    (args.out_dir / "camera_icp.json").write_text(json.dumps([camera], indent=2) + "\n")
    payload = {
        "schema_version": 1,
        "method": "rigid_hand_base_depth_icp",
        "inputs": {
            "color": str(args.color),
            "depth": str(args.depth),
            "mesh": str(args.mesh),
            "intrinsics": intr,
            "roi_xyxy": list(args.roi),
            "depth_range_m": list(args.depth_range),
            "seed": args.seed,
            "sample_count": args.samples,
        },
        "counts": {
            "target_points_raw": int(len(target_points)),
            "target_points_downsampled": int(len(target.points)),
            "source_points_visible": int(len(source_points)),
        },
        "initial_camera": view,
        "refined_camera": camera,
        "transform_camera_from_hand_base": current.tolist(),
        "delta_from_initial": {
            "translation_m": translation_delta_m,
            "rotation_deg": angle_delta_deg,
        },
        "stages": stages,
        "final": {
            "fitness_at_12mm": float(evaluation.fitness),
            "inlier_rmse_m_at_12mm": float(evaluation.inlier_rmse),
        },
    }
    (args.out_dir / "registration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["final"], indent=2))
    print(json.dumps(payload["delta_from_initial"], indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
