"""Refit the top-down reset posture with a *controlled* tip penetration.

The 2026-08-06 first pass drove the nearest tip vertex to zero clearance and only
penalised penetration beyond 0.5 mm, which let the deepest vertex reach 0.96 mm.
Under a stiff PD that is 9-11 N per finger against an 8 N functional ceiling, and
the reaction drives the two weak chains (thumb, pinky) off the object entirely --
thumb_cmc_pitch settles at its 0.0 rad lower rail, 0.117 rad below command.

This targets the deepest vertex at a small penetration instead, so the reset is a
light valid touch and the grip force comes from the TARGET preload rather than
from reset interpenetration.
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location(
    "fitmod", "tools/fit_linker_l20_screwdriver_topdown.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

MARGIN = 0.115           # RESET must clear 0.109999 with headroom
TARGET_PEN = 0.00010     # 0.1 mm deepest-vertex penetration
ND_MIN = 0.0030
ND_SCALE = 0.0003
STARTS = 96
SEED = 20260806
SCRATCH = ("/tmp/claude-1000/-home-user-dex-forge/"
           "c28006ed-89eb-4321-8443-2f52233f4e0c/scratchpad/")

m.MIN_JOINT_MARGIN_RAD = MARGIN


class Light(m.PostureObjective):
    """Same role/geometry residuals, but explicit penetration and hard non-distal."""

    def __init__(self, model, mid, span):
        super().__init__(model, mid, span)
        # The inherited 256-vertex fit sample misses the deepest-penetrating
        # vertex: the first attempt reported 0.1 mm to the objective while the
        # full mesh was at 1.03 mm.  The penetration residual is the whole point
        # of this fit, so the fingertips get a dense sample of their own.
        self.local_vertices_pen = {}
        for link in m.FINGERTIP_LINKS:
            vertices = self.local_vertices_full[link]
            index = np.linspace(
                0, len(vertices) - 1, min(len(vertices), 3000), dtype=np.int64
            )
            self.local_vertices_pen[link] = vertices[index]

    def __call__(self, x):
        root_pos, yaw, q = self.unpack(x)
        joint_pos = m._joint_dict(q)
        root_quat = m.palm_down_quaternion_wxyz(yaw)
        fk = self.model.forward_kinematics(joint_pos, root_pos, root_quat)
        tips = np.vstack([fk[n][:3, 3] for n in m.TIP_MARKER_LINKS])

        axis_xy = m.SCREWDRIVER_ROOT_POS_W[:2]
        drive_xy = tips[1:, :2] - axis_xy[None, :]
        drive_radius = np.linalg.norm(drive_xy, axis=1)
        drive_unit = drive_xy / np.maximum(drive_radius[:, None], 1.0e-9)
        res = list((drive_radius - m.TOPDOWN_HANDLE_RADIUS_M) / 0.004)
        res.extend((tips[1:, 2] - self.targets[1:, 2]) / 0.020)
        index_xy = tips[0, :2] - axis_xy
        res.append((tips[0, 2] - self.targets[0, 2]) / 0.003)
        res.append((float(np.linalg.norm(index_xy)) - 0.018) / 0.008)

        non_thumb = np.sum(drive_unit[:3], axis=0)
        non_thumb /= max(float(np.linalg.norm(non_thumb)), 1.0e-9)
        res.append((float(np.dot(drive_unit[3], non_thumb)) + 0.90) / 0.15)
        res.append((float(np.dot(drive_unit[0], drive_unit[1])) - 0.72) / 0.25)
        res.append((float(np.dot(drive_unit[1], drive_unit[2])) - 0.72) / 0.25)

        # Deepest vertex at TARGET_PEN, and the nearest vertex on the surface.
        for i, link in enumerate(m.FINGERTIP_LINKS):
            v = self._world_vertices(self.local_vertices_pen[link], fk[link])
            body = self._body_clearance(v)
            cl = np.minimum(body, self._cap_clearance(v)) if i == 0 else body
            penetration = max(0.0, -float(np.min(cl)))
            res.append((penetration - TARGET_PEN) / 0.00018)
            res.append(float(np.min(np.abs(cl))) / 0.00030)

        for link in m.NON_DISTAL_LINKS:
            v = self._world_vertices(self.local_vertices_fit[link], fk[link])
            cl = np.minimum(self._body_clearance(v), self._cap_clearance(v))
            res.append(max(0.0, ND_MIN - float(np.min(cl))) / ND_SCALE)

        res.extend(0.03 * ((q - self.joint_mid) / self.joint_span))
        res.extend(0.02 * ((root_pos - np.asarray((-0.009, 0.175, 1.485))) / 0.05))
        res.append(0.01 * yaw)
        return np.asarray(res, dtype=np.float64)

    def penetrations(self, x):
        root_pos, yaw, q = self.unpack(x)
        fk = self.model.forward_kinematics(
            m._joint_dict(q), root_pos, m.palm_down_quaternion_wxyz(yaw)
        )
        out = {}
        for i, (finger, link) in enumerate(zip(m.FINGERS if hasattr(m, "FINGERS")
                                               else ("index", "middle", "ring",
                                                     "pinky", "thumb"),
                                               m.FINGERTIP_LINKS)):
            v = self._world_vertices(self.local_vertices_full[link], fk[link])
            body = self._body_clearance(v)
            cl = np.minimum(body, self._cap_clearance(v)) if i == 0 else body
            out[finger] = (max(0.0, -float(np.min(cl))), float(np.min(np.abs(cl))))
        return out


def main() -> None:
    model = m.UrdfGeometry(m.HAND_URDF)
    lo, hi = m._safe_joint_bounds(model)
    mid, span = 0.5 * (lo + hi), np.maximum(hi - lo, 1e-6)
    obj = Light(model, mid, span)
    prod = m.PostureObjective(model, mid, span)
    lower = np.concatenate(((-0.080, 0.100, 1.420, -0.70), lo))
    upper = np.concatenate(((0.060, 0.260, 1.580, 0.70), hi))
    rng = np.random.default_rng(SEED)

    feasible = []
    for i in range(STARTS):
        x0 = np.clip(m._seed(rng, lo, hi, i), lower + 1e-8, upper - 1e-8)
        r = least_squares(obj, x0, bounds=(lower, upper), max_nfev=2500,
                          loss="soft_l1", f_scale=1.0,
                          xtol=1e-10, ftol=1e-10, gtol=1e-10)
        d = prod.diagnostics(r.x)
        pen = obj.penetrations(r.x)
        nd = d["minimum_non_distal_mesh_clearance_m"]
        worst_pen = max(p for p, _ in pen.values())
        worst_gap = max(g for _, g in pen.values())
        margin = d["minimum_joint_limit_margin_rad"]
        ok = (nd >= 0.0020 and margin >= 0.110
              and worst_pen <= 0.00030 and worst_gap <= 0.00020)
        score = float(np.sum(obj(r.x) ** 2))
        if ok:
            feasible.append((score, r.x.copy()))
        print(f"start={i:02d} score={score:9.3f} nd={nd*1000:6.3f}mm "
              f"pen<={worst_pen*1000:.3f}mm gap<={worst_gap*1e6:6.1f}um "
              f"margin={margin:.4f} {'FEASIBLE' if ok else ''}", flush=True)

    if not feasible:
        print("NO FEASIBLE CANDIDATE")
        return
    feasible.sort(key=lambda t: t[0])
    score, best = feasible[0]
    summary = prod.diagnostics(best)
    summary.update({
        "fit_seed": SEED, "fit_starts": STARTS,
        "best_nfev": -1, "best_status": -1,
        "minimum_required_joint_margin_rad": MARGIN,
        "minimum_required_non_distal_clearance_m": ND_MIN,
        "target_tip_penetration_m": TARGET_PEN,
        "tip_penetration_m": {k: v[0] for k, v in obj.penetrations(best).items()},
        "all_candidate_scores": [score],
        "feasible_count": len(feasible),
    })
    with open(SCRATCH + "fit_light.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nfeasible={len(feasible)}/{STARTS} selected score={score:.3f}")
    print("penetration mm:", {k: round(v[0]*1000, 4)
                              for k, v in obj.penetrations(best).items()})


if __name__ == "__main__":
    main()
