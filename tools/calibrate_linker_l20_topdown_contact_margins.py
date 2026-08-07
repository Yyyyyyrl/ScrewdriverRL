#!/usr/bin/env python3
"""Fit per-finger distance-contact margins against measured fingertip force.

``compute_surface_clearance`` measures from each distal *body frame*, not the
pad, so every finger carries a fixed frame-to-pad offset that
``contact_d_margin_by_finger`` compensates.  Those offsets are a property of the
grasp posture: when the posture changes, a stale table silently misreports
contact.

That is not a cosmetic error.  On the 2026-08-06 refit the pinky's frame
clearance at the controller target is 0.00936 m against a stale margin of
0.0095 -- 0.14 mm of slack -- so the distance model calls the pinky "in contact"
while physics measures 0.00 N on it.  A false positive there inflates
drive_count and contact_gate and pays turn reward for contact that does not
exist, which is indistinguishable from the object coasting on its own.

This re-runs the original calibration idea: roll the real task out under domain
randomisation, record per-finger surface clearance and per-finger fingertip
force together, and choose the margin that maximises agreement between the
distance predicate (clearance <= margin) and the force predicate
(force > force_threshold).  Reports the achieved agreement per finger so a
regression is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--action_scale", type=float, default=0.35)
parser.add_argument("--force_threshold", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=20260806)
parser.add_argument("--output", type=Path, required=True)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: E402,F401  (registers the tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


FINGERS = ("index", "middle", "ring", "pinky", "thumb")


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # Calibrate against the distribution training actually sees.
    env_cfg.domain_rand.enabled = True
    # The reset guard resamples rows with too few *distance* contacts, which is
    # the very predicate being calibrated; leaving it on would filter the sample
    # through the stale table.
    env_cfg.reset_contact_guard_min_fingers = 0

    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    clearances: list[torch.Tensor] = []
    forces: list[torch.Tensor] = []
    n_actions = base.num_finger_dofs

    for step in range(args.warmup_steps + args.steps):
        action = (
            torch.rand((base.num_envs, n_actions), device=base.device) * 2.0 - 1.0
        ) * args.action_scale
        env.step(action)
        if step < args.warmup_steps:
            continue
        clearance = base.compute_surface_clearance().detach()
        total, body, cap, _wrong = base._read_contact_forces()
        role = total if base.cfg.role_neutral_fingertip_contact else torch.stack(
            (cap[:, 0], body[:, 1], body[:, 2], body[:, 3], body[:, 4]), dim=-1
        )
        clearances.append(clearance.clone())
        forces.append(role.detach().clone())

    clearance = torch.cat(clearances, dim=0).cpu().numpy()
    force = torch.cat(forces, dim=0).cpu().numpy()
    env.close()

    contact = force > args.force_threshold
    result = {
        "task": args.task,
        "num_envs": args.num_envs,
        "samples": int(clearance.shape[0]),
        "force_threshold_n": args.force_threshold,
        "action_scale": args.action_scale,
        "seed": args.seed,
        "fingers": {},
    }
    margins: dict[str, float] = {}
    for i, finger in enumerate(FINGERS[: clearance.shape[1]]):
        c, k = clearance[:, i], contact[:, i]
        rate = float(k.mean())
        # Sweep the margin over the observed clearance support and keep the
        # value with the highest agreement.  Ties resolve to the tighter margin
        # so the predicate errs toward reporting no contact.
        grid = np.unique(np.concatenate([
            np.linspace(float(c.min()) - 0.002, float(c.max()) + 0.002, 4001),
            np.array([0.0]),
        ]))
        agreement = ((c[None, :] <= grid[:, None]) == k[None, :]).mean(axis=1)
        best = int(np.argmax(agreement))
        best_agreement = float(agreement[best])
        ties = np.flatnonzero(agreement >= best_agreement - 1e-12)
        margin = float(grid[ties.min()])
        clamped = max(0.0, margin)
        margins[finger] = clamped
        clamped_agreement = float(((c <= clamped) == k).mean())
        result["fingers"][finger] = {
            "contact_rate_by_force": rate,
            "best_margin_m": margin,
            "best_agreement": best_agreement,
            "clamped_margin_m": clamped,
            "clamped_agreement": clamped_agreement,
            "clearance_p05_m": float(np.percentile(c, 5)),
            "clearance_median_m": float(np.median(c)),
            "clearance_p95_m": float(np.percentile(c, 95)),
            "force_mean_n": float(force[:, i].mean()),
            "force_p95_n": float(np.percentile(force[:, i], 95)),
        }
    result["contact_d_margin_by_finger"] = margins
    result["mean_agreement"] = float(
        np.mean([v["clamped_agreement"] for v in result["fingers"].values()])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
