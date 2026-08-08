#!/usr/bin/env python3
"""Numerically refit fixed-wrist reset/target postures for each handle diameter."""

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

from screwdriver_rl.utils.linker_topdown_diameter_postures import (  # noqa: E402
    TOPDOWN_HANDLE_DIAMETERS_M,
    TOPDOWN_HANDLE_RADII_M,
    TOPDOWN_NOMINAL_BUCKET_INDEX,
    TOPDOWN_PREGRASP_POSITIONS_BUCKETS,
    TOPDOWN_RESET_POSITIONS_BUCKETS,
    TOPDOWN_ROOT_POS_OFFSETS_BUCKETS,
    TOPDOWN_ROOT_POS_W,
    TOPDOWN_ROOT_QUAT_WXYZ,
)
from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    FINGERTIP_LINKS,
    NON_DISTAL_LINKS,
    SCREWDRIVER_ROOT_POS_W,
    TOPDOWN_CAP_THICKNESS_M,
    TOPDOWN_HANDLE_LENGTH_M,
    UrdfGeometry,
    cylindrical_surface_clearance,
    joint_limit_margins,
)


HAND_URDF = REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts/linker_l20_screwdriver_topdown/diameter_bucket_refit.json"
)
FINGERS = ("index", "middle", "ring", "pinky", "thumb")
JOINT_NAMES_BY_FINGER = {
    "index": ("index_mcp_roll", "index_mcp_pitch", "index_pip"),
    "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
    "ring": ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
    "pinky": ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
    "thumb": ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
}
JOINT_ORDER = tuple(
    joint for finger in FINGERS for joint in JOINT_NAMES_BY_FINGER[finger]
)
MIN_JOINT_MARGIN_RAD = 0.105


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(JOINT_ORDER, values)}


def _flatten(posture: dict[str, tuple[float, ...]]) -> np.ndarray:
    return np.asarray(
        [value for finger in FINGERS for value in posture[finger]],
        dtype=np.float64,
    )


def _per_finger(joints: dict[str, float]) -> dict[str, tuple[float, ...]]:
    return {
        finger: tuple(float(joints[name]) for name in names)
        for finger, names in JOINT_NAMES_BY_FINGER.items()
    }


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
        mult = follower.mimic.multiplier
        offset = follower.mimic.offset
        guarded = (
            (follower.lower + MIN_JOINT_MARGIN_RAD - offset) / mult,
            (follower.upper - MIN_JOINT_MARGIN_RAD - offset) / mult,
        )
        i = index[follower.mimic.source]
        lo[i] = max(lo[i], min(guarded))
        hi[i] = min(hi[i], max(guarded))
    if bool(np.any(lo >= hi)):
        raise ValueError("joint and mimic limits leave an empty guarded range")
    return lo, hi


class FixedWristObjective:
    def __init__(
        self,
        model: UrdfGeometry,
        *,
        radius: float,
        root: np.ndarray,
        quat: np.ndarray,
        q0: np.ndarray,
        target_penetration_m: float,
        min_non_distal_clearance_m: float,
    ) -> None:
        self.model = model
        self.radius = float(radius)
        self.root = root
        self.quat = quat
        self.q0 = q0
        self.target_penetration_m = float(target_penetration_m)
        self.min_non_distal_clearance_m = float(min_non_distal_clearance_m)
        self.body_base_z = float(SCREWDRIVER_ROOT_POS_W[2] + 0.100)
        self.body_top_z = self.body_base_z + TOPDOWN_HANDLE_LENGTH_M
        self.cap_top_z = self.body_top_z + TOPDOWN_CAP_THICKNESS_M
        self.distal_vertices = {
            name: np.asarray(model.collision_mesh_local(name).vertices, dtype=np.float64)
            for name in FINGERTIP_LINKS
        }
        self.non_distal_vertices: dict[str, np.ndarray] = {}
        for name in NON_DISTAL_LINKS:
            vertices = np.asarray(model.collision_mesh_local(name).vertices, dtype=np.float64)
            indices = np.linspace(
                0, len(vertices) - 1, min(1200, len(vertices)), dtype=np.int64
            )
            self.non_distal_vertices[name] = vertices[indices]

    @staticmethod
    def _world(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        return points @ transform[:3, :3].T + transform[:3, 3]

    def _body(self, points: np.ndarray) -> np.ndarray:
        return cylindrical_surface_clearance(
            points,
            SCREWDRIVER_ROOT_POS_W[:2],
            self.radius,
            self.body_base_z,
            self.body_top_z,
        )

    def _cap(self, points: np.ndarray) -> np.ndarray:
        return cylindrical_surface_clearance(
            points,
            SCREWDRIVER_ROOT_POS_W[:2],
            self.radius,
            self.body_top_z,
            self.cap_top_z,
        )

    def diagnostics(self, q: np.ndarray) -> tuple[dict, dict]:
        joints = _joint_dict(q)
        fk = self.model.forward_kinematics(joints, self.root, self.quat)
        contacts: dict[str, dict] = {}
        for index, (finger, link) in enumerate(zip(FINGERS, FINGERTIP_LINKS)):
            points = self._world(self.distal_vertices[link], fk[link])
            clearance = self._cap(points) if index == 0 else self._body(points)
            contacts[finger] = {
                "minimum_clearance_m": float(np.min(clearance)),
                "maximum_penetration_m": max(0.0, -float(np.min(clearance))),
                "closest_vertex_to_surface_m": float(np.min(np.abs(clearance))),
            }
        non_distal: dict[str, float] = {}
        for link, local in self.non_distal_vertices.items():
            points = self._world(local, fk[link])
            non_distal[link] = float(
                np.min(np.minimum(self._body(points), self._cap(points)))
            )
        return contacts, non_distal

    def __call__(self, q: np.ndarray) -> np.ndarray:
        contacts, non_distal = self.diagnostics(q)
        residuals: list[float] = []
        for finger in FINGERS:
            row = contacts[finger]
            residuals.append(
                (row["maximum_penetration_m"] - self.target_penetration_m)
                / 0.00018
            )
            residuals.append(row["closest_vertex_to_surface_m"] / 0.00030)
        residuals.extend(
            max(0.0, self.min_non_distal_clearance_m - clearance) / 0.00035
            for clearance in non_distal.values()
        )
        residuals.extend(0.010 * ((q - self.q0) / 0.08))
        return np.asarray(residuals, dtype=np.float64)


def _fit_state(
    model: UrdfGeometry,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    bucket: int,
    state: str,
    initial: dict[str, tuple[float, ...]],
    max_nfev: int,
) -> dict:
    offset = np.asarray(TOPDOWN_ROOT_POS_OFFSETS_BUCKETS[bucket], dtype=np.float64)
    root = np.asarray(TOPDOWN_ROOT_POS_W, dtype=np.float64) + offset
    quat = np.asarray(TOPDOWN_ROOT_QUAT_WXYZ, dtype=np.float64)
    q0 = _flatten(initial)
    initial_within_guarded_limits = bool(
        np.all(q0 >= lo - 1.0e-12) and np.all(q0 <= hi + 1.0e-12)
    )
    if bucket == TOPDOWN_NOMINAL_BUCKET_INDEX and initial_within_guarded_limits:
        q = q0
        result_meta = {"preserved_validated_nominal": True, "nfev": 0, "cost": 0.0}
    else:
        objective = FixedWristObjective(
            model,
            radius=TOPDOWN_HANDLE_RADII_M[bucket],
            root=root,
            quat=quat,
            q0=q0,
            target_penetration_m=0.00035 if state == "reset" else 0.00055,
            min_non_distal_clearance_m=0.0030 if state == "reset" else 0.0020,
        )
        result = least_squares(
            objective,
            np.clip(q0, lo + 1.0e-8, hi - 1.0e-8),
            bounds=(lo, hi),
            max_nfev=max_nfev,
            loss="soft_l1",
            f_scale=1.0,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
            verbose=0,
        )
        q = result.x
        result_meta = {
            "preserved_validated_nominal": False,
            "nfev": int(result.nfev),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(np.sum(objective(result.x) ** 2)),
        }

    diagnostics_objective = FixedWristObjective(
        model,
        radius=TOPDOWN_HANDLE_RADII_M[bucket],
        root=root,
        quat=quat,
        q0=q0,
        target_penetration_m=0.00035 if state == "reset" else 0.00055,
        min_non_distal_clearance_m=0.0030 if state == "reset" else 0.0020,
    )
    joints = _joint_dict(q)
    contacts, non_distal = diagnostics_objective.diagnostics(q)
    margins = joint_limit_margins(model, joints)
    return {
        "root_pos_w": [float(value) for value in root],
        "root_quat_wxyz": [float(value) for value in quat],
        "joint_positions_independent": joints,
        "joint_positions_expanded": model.expanded_positions(joints),
        "per_finger_positions": _per_finger(joints),
        "joint_limit_margins_rad": margins,
        "minimum_joint_limit_margin_rad": float(min(margins.values())),
        "distal_contact_diagnostics": contacts,
        "sampled_non_distal_clearance_m": non_distal,
        "minimum_sampled_non_distal_clearance_m": float(min(non_distal.values())),
        "optimizer": result_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=1600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = UrdfGeometry(HAND_URDF)
    lo, hi = _safe_joint_bounds(model)
    buckets: list[dict] = []
    for bucket, diameter in enumerate(TOPDOWN_HANDLE_DIAMETERS_M):
        print(f"fitting bucket {bucket}: diameter={diameter * 1000:.0f} mm", flush=True)
        reset = _fit_state(
            model,
            lo,
            hi,
            bucket=bucket,
            state="reset",
            initial=TOPDOWN_RESET_POSITIONS_BUCKETS[bucket],
            max_nfev=args.max_nfev,
        )
        target = _fit_state(
            model,
            lo,
            hi,
            bucket=bucket,
            state="target",
            initial=TOPDOWN_PREGRASP_POSITIONS_BUCKETS[bucket],
            max_nfev=args.max_nfev,
        )
        buckets.append(
            {
                "bucket": bucket,
                "diameter_m": diameter,
                "radius_m": TOPDOWN_HANDLE_RADII_M[bucket],
                "reset": reset,
                "target": target,
            }
        )

    output = {
        "minimum_joint_margin_rad": MIN_JOINT_MARGIN_RAD,
        "fixed_wrist": True,
        "buckets": buckets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
