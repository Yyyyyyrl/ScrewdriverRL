"""Emit a fit-schema JSON for the best *feasible* posture so the production
validator can judge it.  Selection rejects infeasible candidates first, then
takes the lowest production objective score among the survivors -- the opposite
order from the production fitter, which takes lowest score outright.
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

MARGIN = 0.11
m.MIN_JOINT_MARGIN_RAD = MARGIN
ND_MIN = 0.0030          # fit target, above the 0.0015 validator threshold
ND_SCALE = 0.0003        # hard price: 10x the production hinge weight
STARTS = 96
SEED = 20260806
OUT = ("/tmp/claude-1000/-home-user-dex-forge/"
       "c28006ed-89eb-4321-8443-2f52233f4e0c/scratchpad/fit_feasible.json")


class Hard(m.PostureObjective):
    def __call__(self, x):
        res = list(super().__call__(x))
        # Recompute the non-distal block with the hard price.  super() already
        # appended one residual per non-distal link at the soft weight; replace
        # them in place rather than adding a second, double-counted block.
        root_pos, yaw, q = self.unpack(x)
        fk = self.model.forward_kinematics(
            m._joint_dict(q), root_pos, m.palm_down_quaternion_wxyz(yaw)
        )
        n = len(m.NON_DISTAL_LINKS)
        tail = len(q) + 3 + 1  # regularisation block length
        start = len(res) - tail - n
        for i, link in enumerate(m.NON_DISTAL_LINKS):
            v = self._world_vertices(self.local_vertices_fit[link], fk[link])
            cl = np.minimum(self._body_clearance(v), self._cap_clearance(v))
            res[start + i] = max(0.0, ND_MIN - float(np.min(cl))) / ND_SCALE
        return np.asarray(res, dtype=np.float64)


def main() -> None:
    model = m.UrdfGeometry(m.HAND_URDF)
    lo, hi = m._safe_joint_bounds(model)
    mid, span = 0.5 * (lo + hi), np.maximum(hi - lo, 1e-6)
    hard = Hard(model, mid, span)
    prod = m.PostureObjective(model, mid, span)  # production scorer
    lower = np.concatenate(((-0.080, 0.100, 1.420, -0.70), lo))
    upper = np.concatenate(((0.060, 0.260, 1.580, 0.70), hi))
    rng = np.random.default_rng(SEED)

    feasible: list[tuple[float, np.ndarray]] = []
    scores: list[float] = []
    for i in range(STARTS):
        x0 = np.clip(m._seed(rng, lo, hi, i), lower + 1e-8, upper - 1e-8)
        r = least_squares(hard, x0, bounds=(lower, upper), max_nfev=2500,
                          loss="soft_l1", f_scale=1.0,
                          xtol=1e-10, ftol=1e-10, gtol=1e-10)
        d = prod.diagnostics(r.x)
        nd = d["minimum_non_distal_mesh_clearance_m"]
        worst_tip = max(abs(v) for v in d["fingertip_mesh_surface_clearance_m"].values())
        score = float(np.sum(prod(r.x) ** 2))
        scores.append(score)
        ok = (nd >= 0.0020
              and worst_tip <= 0.0008
              and d["minimum_joint_limit_margin_rad"] >= 0.105)
        if ok:
            feasible.append((score, r.x.copy()))
        print(f"start={i:02d} score={score:9.3f} nd={nd*1000:7.3f}mm "
              f"tip={worst_tip*1000:6.3f}mm margin={d['minimum_joint_limit_margin_rad']:.4f} "
              f"{'FEASIBLE' if ok else ''}", flush=True)

    if not feasible:
        print("NO FEASIBLE CANDIDATE")
        return
    feasible.sort(key=lambda t: t[0])
    score, best = feasible[0]
    summary = prod.diagnostics(best)
    summary.update({
        "fit_seed": SEED,
        "fit_starts": STARTS,
        "best_nfev": -1,
        "best_status": -1,
        "minimum_required_joint_margin_rad": MARGIN,
        "minimum_required_non_distal_clearance_m": ND_MIN,
        "all_candidate_scores": sorted(scores),
        "feasible_count": len(feasible),
        "selected_production_score": score,
    })
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nfeasible={len(feasible)}/{STARTS}  selected score={score:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
