#!/usr/bin/env python3
"""Fit a strict palm-down Linker L20 posture around the 64 mm screwdriver.

This is an offline, deterministic multi-start solve over the real Linker URDF
kinematics and collision meshes.  It does not import Isaac Lab and never edits a
task config; the selected candidate is written as JSON for review before its
numbers are copied into the task-specific posture module.

The solve enforces the orientation structurally: local palm +X maps exactly to
world -Z for every candidate.  It fits all 16 independent joints plus hand-root
translation/yaw, keeps both independent and mimic joints at least 0.10 rad from
their URDF limits, targets the index distal pad on the cap, targets the four
drive distal pads on the handle body, and penalises every non-distal link that
approaches the 64 mm body/cap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from screwdriver_rl.utils.linker_topdown_geometry import (
    FINGERTIP_LINKS,
    NON_DISTAL_LINKS,
    SCREWDRIVER_ROOT_POS_W,
    TIP_MARKER_LINKS,
    TOPDOWN_CAP_THICKNESS_M,
    TOPDOWN_HANDLE_LENGTH_M,
    TOPDOWN_HANDLE_RADIUS_M,
    UrdfGeometry,
    cylindrical_surface_clearance,
    joint_limit_margins,
    palm_down_quaternion_wxyz,
    posture_frame,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_URDF = REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"

JOINT_ORDER = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
)

FINGERS = ("index", "middle", "ring", "pinky", "thumb")
MIN_JOINT_MARGIN_RAD = 0.10
MIN_NON_DISTAL_CLEARANCE_M = 0.0025


def _safe_joint_bounds(model: UrdfGeometry) -> tuple[np.ndarray, np.ndarray]:
    limits = model.joint_limits(MIN_JOINT_MARGIN_RAD)
    lo = np.asarray([limits[name][0] for name in JOINT_ORDER], dtype=np.float64)
    hi = np.asarray([limits[name][1] for name in JOINT_ORDER], dtype=np.float64)

    # Mimic followers also need the same hard-limit margin.  Convert each
    # follower's guarded range back into a guarded range for its source.
    index = {name: i for i, name in enumerate(JOINT_ORDER)}
    for follower in model.joints.values():
        if follower.mimic is None:
            continue
        source = follower.mimic.source
        if source not in index or follower.lower is None or follower.upper is None:
            continue
        mult = follower.mimic.multiplier
        offset = follower.mimic.offset
        a = (follower.lower + MIN_JOINT_MARGIN_RAD - offset) / mult
        b = (follower.upper - MIN_JOINT_MARGIN_RAD - offset) / mult
        source_lo, source_hi = min(a, b), max(a, b)
        i = index[source]
        lo[i] = max(lo[i], source_lo)
        hi[i] = min(hi[i], source_hi)
    if bool(np.any(lo >= hi)):
        raise ValueError("joint/mimic guard margins leave an empty range")
    return lo, hi


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(JOINT_ORDER, values)}


def _tip_targets() -> np.ndarray:
    """Five task-role targets in world coordinates.

    The four drive targets span the near (+Y) side of the handle and the thumb
    opposes from -Y.  The solver is also constrained by the actual distal meshes,
    so these marker targets establish roles/distribution rather than pretending a
    zero-volume point is the final contact geometry.
    """

    axis_x, axis_y = SCREWDRIVER_ROOT_POS_W[:2]
    body_base_z = SCREWDRIVER_ROOT_POS_W[2] + 0.100
    cap_top_z = body_base_z + TOPDOWN_HANDLE_LENGTH_M + TOPDOWN_CAP_THICKNESS_M
    return np.asarray(
        (
            (axis_x - 0.018, axis_y + 0.002, cap_top_z),
            (axis_x - 0.021, axis_y + 0.024, body_base_z + 0.080),
            (axis_x + 0.000, axis_y + 0.032, body_base_z + 0.060),
            (axis_x + 0.021, axis_y + 0.024, body_base_z + 0.040),
            (axis_x - 0.004, axis_y - 0.032, body_base_z + 0.055),
        ),
        dtype=np.float64,
    )


class PostureObjective:
    def __init__(self, model: UrdfGeometry, joint_mid: np.ndarray, joint_span: np.ndarray) -> None:
        self.model = model
        self.joint_mid = joint_mid
        self.joint_span = joint_span
        self.targets = _tip_targets()
        self.local_vertices_full = {
            name: np.asarray(model.collision_mesh_local(name).vertices, dtype=np.float64)
            for name in set(FINGERTIP_LINKS).union(NON_DISTAL_LINKS)
        }
        self.local_vertices_fit = {
            name: vertices[
                np.linspace(0, len(vertices) - 1, min(len(vertices), 256), dtype=np.int64)
            ]
            for name, vertices in self.local_vertices_full.items()
        }
        self.body_base_z = SCREWDRIVER_ROOT_POS_W[2] + 0.100
        self.body_top_z = self.body_base_z + TOPDOWN_HANDLE_LENGTH_M
        self.cap_top_z = self.body_top_z + TOPDOWN_CAP_THICKNESS_M

    @staticmethod
    def unpack(x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return x[:3], float(x[3]), x[4:]

    @staticmethod
    def _world_vertices(local: np.ndarray, link_tf: np.ndarray) -> np.ndarray:
        return local @ link_tf[:3, :3].T + link_tf[:3, 3]

    def _body_clearance(self, points: np.ndarray) -> np.ndarray:
        return cylindrical_surface_clearance(
            points,
            SCREWDRIVER_ROOT_POS_W[:2],
            TOPDOWN_HANDLE_RADIUS_M,
            self.body_base_z,
            self.body_top_z,
        )

    def _cap_clearance(self, points: np.ndarray) -> np.ndarray:
        return cylindrical_surface_clearance(
            points,
            SCREWDRIVER_ROOT_POS_W[:2],
            TOPDOWN_HANDLE_RADIUS_M,
            self.body_top_z,
            self.cap_top_z,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        root_pos, yaw, q_values = self.unpack(x)
        joint_pos = _joint_dict(q_values)
        root_quat = palm_down_quaternion_wxyz(yaw)
        fk = self.model.forward_kinematics(joint_pos, root_pos, root_quat)
        tips = np.vstack([fk[name][:3, 3] for name in TIP_MARKER_LINKS])

        axis_xy = SCREWDRIVER_ROOT_POS_W[:2]
        drive_xy = tips[1:, :2] - axis_xy[None, :]
        drive_radius = np.linalg.norm(drive_xy, axis=1)
        drive_unit = drive_xy / np.maximum(drive_radius[:, None], 1.0e-9)
        residuals: list[float] = list((drive_radius - TOPDOWN_HANDLE_RADIUS_M) / 0.004)
        residuals.extend((tips[1:, 2] - self.targets[1:, 2]) / 0.020)

        index_xy = tips[0, :2] - axis_xy
        index_radius = float(np.linalg.norm(index_xy))
        residuals.append((tips[0, 2] - self.targets[0, 2]) / 0.003)
        residuals.append((index_radius - 0.018) / 0.008)

        # Middle/ring/pinky form one side of the wrap while the thumb must be on
        # the opposing side.  This prevents a five-point-looking but mechanically
        # useless same-hemisphere contact solution.
        non_thumb_dir = np.sum(drive_unit[:3], axis=0)
        non_thumb_dir /= max(float(np.linalg.norm(non_thumb_dir)), 1.0e-9)
        residuals.append((float(np.dot(drive_unit[3], non_thumb_dir)) + 0.90) / 0.15)
        residuals.append((float(np.dot(drive_unit[0], drive_unit[1])) - 0.72) / 0.25)
        residuals.append((float(np.dot(drive_unit[1], drive_unit[2])) - 0.72) / 0.25)

        # Real distal meshes must touch their task-role surface without deep
        # penetration.  Index targets body+cap union; drive fingers target body.
        for i, link in enumerate(FINGERTIP_LINKS):
            vertices = self._world_vertices(self.local_vertices_fit[link], fk[link])
            body = self._body_clearance(vertices)
            clearance = np.minimum(body, self._cap_clearance(vertices)) if i == 0 else body
            closest = float(clearance[np.argmin(np.abs(clearance))])
            residuals.append(closest / 0.0015)
            residuals.append(max(0.0, -float(np.min(clearance)) - 0.0005) / 0.0010)

        # Every palm/metacarpal/proximal/middle mesh must remain clear of the
        # body+cap.  A smooth-ish one-residual-per-link hinge is adequate for the
        # least-squares fit; the final validator performs dense full-mesh checks.
        for link in NON_DISTAL_LINKS:
            vertices = self._world_vertices(self.local_vertices_fit[link], fk[link])
            clearance = np.minimum(self._body_clearance(vertices), self._cap_clearance(vertices))
            min_clearance = float(np.min(clearance))
            residuals.append(max(0.0, MIN_NON_DISTAL_CLEARANCE_M - min_clearance) / 0.0015)

        # Gentle regularisation selects a controllable middle-of-range solution
        # when multiple mesh-valid postures satisfy the same role targets.
        residuals.extend(0.03 * ((q_values - self.joint_mid) / self.joint_span))
        residuals.extend(0.02 * ((root_pos - np.asarray((-0.009, 0.175, 1.485))) / 0.05))
        residuals.append(0.01 * yaw)
        return np.asarray(residuals, dtype=np.float64)

    def diagnostics(self, x: np.ndarray) -> dict:
        root_pos, yaw, q_values = self.unpack(x)
        joints = _joint_dict(q_values)
        quat = palm_down_quaternion_wxyz(yaw)
        fk = self.model.forward_kinematics(joints, root_pos, quat)
        frame = posture_frame(self.model, root_pos, quat, joints)
        fingertip_clearance: dict[str, float] = {}
        non_distal_clearance: dict[str, float] = {}
        for i, link in enumerate(FINGERTIP_LINKS):
            vertices = self._world_vertices(self.local_vertices_full[link], fk[link])
            body = self._body_clearance(vertices)
            clearance = np.minimum(body, self._cap_clearance(vertices)) if i == 0 else body
            fingertip_clearance[FINGERS[i]] = float(clearance[np.argmin(np.abs(clearance))])
        for link in NON_DISTAL_LINKS:
            vertices = self._world_vertices(self.local_vertices_full[link], fk[link])
            clearance = np.minimum(self._body_clearance(vertices), self._cap_clearance(vertices))
            non_distal_clearance[link] = float(np.min(clearance))
        margins = joint_limit_margins(self.model, joints)
        return {
            "objective_cost": float(np.sum(self(x) ** 2)),
            "root_pos_w": [float(v) for v in root_pos],
            "root_quat_wxyz": [float(v) for v in quat],
            "yaw_rad": yaw,
            "palm_normal_w": [float(v) for v in frame["palm_normal_w"]],
            "long_axis_w": [float(v) for v in frame["long_axis_w"]],
            "joint_positions_independent": joints,
            "joint_positions_expanded": self.model.expanded_positions(joints),
            "joint_limit_margins_rad": margins,
            "minimum_joint_limit_margin_rad": float(min(margins.values())),
            "tip_marker_positions_w": {
                finger: [float(v) for v in point]
                for finger, point in zip(FINGERS, frame["tip_points_w"])
            },
            "tip_marker_targets_w": {
                finger: [float(v) for v in point]
                for finger, point in zip(FINGERS, self.targets)
            },
            "fingertip_mesh_surface_clearance_m": fingertip_clearance,
            "non_distal_mesh_clearance_m": non_distal_clearance,
            "minimum_non_distal_mesh_clearance_m": float(min(non_distal_clearance.values())),
        }


def _seed(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, index: int) -> np.ndarray:
    root = np.asarray((-0.009, 0.175, 1.485), dtype=np.float64)
    nominal = np.asarray(
        (
            0.00, 0.65, 0.95,
            0.00, 0.65, 0.90,
            0.00, 0.65, 0.90,
            0.00, 0.65, 0.85,
            0.75, 0.60, 0.35, 0.60,
        ),
        dtype=np.float64,
    )
    if index > 0:
        root += rng.normal(0.0, (0.012, 0.018, 0.015))
        nominal += rng.normal(0.0, 0.18, size=len(nominal))
    nominal = np.clip(nominal, lo + 1.0e-6, hi - 1.0e-6)
    yaw = 0.0 if index == 0 else float(rng.normal(0.0, 0.18))
    return np.concatenate((root, (yaw,), nominal))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--max_nfev", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/linker_l20_screwdriver_topdown/posture_fit.json",
    )
    args = parser.parse_args()

    model = UrdfGeometry(HAND_URDF)
    joint_lo, joint_hi = _safe_joint_bounds(model)
    joint_mid = 0.5 * (joint_lo + joint_hi)
    joint_span = np.maximum(joint_hi - joint_lo, 1.0e-6)
    objective = PostureObjective(model, joint_mid, joint_span)

    lower = np.concatenate(((-0.080, 0.100, 1.420, -0.70), joint_lo))
    upper = np.concatenate(((0.060, 0.260, 1.580, 0.70), joint_hi))
    rng = np.random.default_rng(args.seed)
    candidates: list[tuple[float, np.ndarray, int, int]] = []
    for index in range(args.starts):
        x0 = np.clip(_seed(rng, joint_lo, joint_hi, index), lower + 1.0e-8, upper - 1.0e-8)
        result = least_squares(
            objective,
            x0,
            bounds=(lower, upper),
            max_nfev=args.max_nfev,
            loss="soft_l1",
            f_scale=1.0,
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
            verbose=0,
        )
        score = float(np.sum(objective(result.x) ** 2))
        candidates.append((score, result.x.copy(), int(result.nfev), int(result.status)))
        print(f"start={index:02d} score={score:.6f} nfev={result.nfev} status={result.status}", flush=True)

    candidates.sort(key=lambda item: item[0])
    score, best, nfev, status = candidates[0]
    summary = objective.diagnostics(best)
    summary.update(
        {
            "fit_seed": args.seed,
            "fit_starts": args.starts,
            "best_nfev": nfev,
            "best_status": status,
            "minimum_required_joint_margin_rad": MIN_JOINT_MARGIN_RAD,
            "minimum_required_non_distal_clearance_m": MIN_NON_DISTAL_CLEARANCE_M,
            "all_candidate_scores": [float(item[0]) for item in candidates],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
