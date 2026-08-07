"""Can ANY wrist offset put the middle finger's pad on the handle's side wall?

The middle finger has three behaviours at the current wrist, none acceptable:
back-contact on the cap top face, a near-straight strut carrying 15-33 N, or no
contact.  Before spending hours on a full re-fit from a different wrist, check
cheaply whether a different wrist even admits a solution.

For each wrist offset, sweep the middle finger's own joints and ask whether any
combination lands its contact on the LATERAL wall (not the cap top face) with the
pad facing the handle and penetration under 1 mm.  Other fingers are ignored:
this is a necessary-condition test.  If no offset admits it, the wrist re-search
is not worth running.
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np

spec = importlib.util.spec_from_file_location(
    "fitmod", "tools/fit_linker_l20_screwdriver_topdown.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    SCREWDRIVER_ROOT_POS_W, TOPDOWN_HANDLE_LENGTH_M, TOPDOWN_CAP_THICKNESS_M,
    TOPDOWN_HANDLE_RADIUS_M, cylindrical_surface_clearance,
)
from screwdriver_rl.utils.linker_topdown_grasp_wrench import (  # noqa: E402
    pad_axis, pad_facing,
)

SCRATCH = ("/tmp/claude-1000/-home-user-dex-forge/"
           "c28006ed-89eb-4321-8443-2f52233f4e0c/scratchpad/")
BASE = json.load(open(SCRATCH + "cand_pool/c000.json"))


def main() -> None:
    model = m.UrdfGeometry(m.HAND_URDF)
    joints = {k: float(v) for k, v in BASE["joint_positions_independent"].items()}
    root0 = np.asarray(BASE["root_pos_w"], dtype=np.float64)
    quat = np.asarray(BASE["root_quat_wxyz"], dtype=np.float64)
    axis = np.asarray(SCREWDRIVER_ROOT_POS_W[:2], dtype=np.float64)
    base_z = SCREWDRIVER_ROOT_POS_W[2] + 0.100
    top_z = base_z + TOPDOWN_HANDLE_LENGTH_M
    cap_z = top_z + TOPDOWN_CAP_THICKNESS_M
    limits = model.joint_limits(0.105)
    verts_local = np.asarray(
        model.collision_mesh_local("middle_distal").vertices, dtype=np.float64
    )

    print("wrist offsets in mm; a hit needs lateral-wall contact, pad>0, pen<1mm")
    print(f"{'dx':>5} {'dy':>5} {'dz':>5} {'hits':>6} {'best pad':>9} "
          f"{'at mcp':>8} {'at pip':>8} {'height%':>8}")
    found = []
    for dx in (-8, -4, 0, 4):
        for dy in (-8, -4, 0, 4, 8):
            for dz in (-12, -8, -4, 0, 4):
                root = root0 + np.array([dx, dy, dz]) * 1e-3
                hits, best = 0, None
                for mcp in np.arange(limits["middle_mcp_pitch"][0],
                                     limits["middle_mcp_pitch"][1], 0.08):
                    for pip in np.arange(limits["middle_pip"][0],
                                         limits["middle_pip"][1], 0.06):
                        q = dict(joints)
                        q["middle_mcp_pitch"] = float(mcp)
                        q["middle_pip"] = float(pip)
                        fk = model.forward_kinematics(q, root, quat)
                        w = verts_local @ fk["middle_distal"][:3, :3].T \
                            + fk["middle_distal"][:3, 3]
                        body = cylindrical_surface_clearance(
                            w, axis, TOPDOWN_HANDLE_RADIUS_M, base_z, top_z)
                        cap = cylindrical_surface_clearance(
                            w, axis, TOPDOWN_HANDLE_RADIUS_M, top_z, cap_z)
                        pen = max(0.0, -float(np.minimum(body, cap).min()))
                        if pen > 0.001:
                            continue
                        r = np.linalg.norm(w[:, :2] - axis, axis=1)
                        # distance to the LATERAL wall only (body, not the top face)
                        d = (np.abs(r - TOPDOWN_HANDLE_RADIUS_M)
                             + np.maximum(0.0, base_z - w[:, 2])
                             + np.maximum(0.0, w[:, 2] - top_z))
                        if d.min() > 0.001:
                            continue
                        idx = int(np.argmin(d))
                        q2 = dict(q); q2["middle_pip"] = float(pip) + 0.02
                        fk2 = model.forward_kinematics(q2, root, quat)
                        pa = pad_axis(fk["middle_distal"][:3, 3],
                                      fk["middle_tip"][:3, 3],
                                      fk2["middle_tip"][:3, 3])
                        pc = pad_facing(fk["middle_distal"][:3, 3],
                                        fk["middle_tip"][:3, 3], pa, w[idx])
                        if pc <= 0:
                            continue
                        hits += 1
                        height = 100 * (w[idx, 2] - base_z) / TOPDOWN_HANDLE_LENGTH_M
                        if best is None or pc > best[0]:
                            best = (pc, float(mcp), float(pip), height)
                if hits:
                    found.append((dx, dy, dz, hits, best))
                    print(f"{dx:5d} {dy:5d} {dz:5d} {hits:6d} {best[0]:+9.3f} "
                          f"{best[1]:8.3f} {best[2]:8.3f} {best[3]:8.1f}", flush=True)
    print(f"\nwrist offsets admitting a middle-finger pad contact on the wall: "
          f"{len(found)}")
    json.dump([dict(dx=a, dy=b, dz=c, hits=h,
                    pad=best[0], mcp=best[1], pip=best[2], height=best[3])
               for a, b, c, h, best in found],
              open(SCRATCH + "middle_wrist_probe.json", "w"), indent=2)


if __name__ == "__main__":
    main()
