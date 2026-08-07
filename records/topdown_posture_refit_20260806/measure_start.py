"""Measure the TRAINING START: are all five pads touching, and does it stay up?

Criterion per the operator: contact may be light -- the posture only has to keep
the screwdriver from falling, not grip it firmly.  So contact is judged by
surface clearance rather than by force, and pad-versus-back by orientation
rather than by force direction.  Both matter because ``reset_zero_tension_targets``
snaps the finger targets onto the measured positions at the end of every reset:
contacts persist geometrically while their forces relax to nearly nothing, so a
force-based test reports a perfectly good light contact as no contact at all.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/home/user/dex-forge")

parser = argparse.ArgumentParser()
parser.add_argument("--option", required=True)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--num_envs", type=int, default=64)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora"
SCRATCH = ("/tmp/claude-1000/-home-user-dex-forge/"
           "c28006ed-89eb-4321-8443-2f52233f4e0c/scratchpad/")

cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
cfg.domain_rand.enabled = False
cfg.reset_contact_guard_min_fingers = 0

env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
env.reset(seed=11)

zero = torch.zeros((base.num_envs, base.num_finger_dofs), device=base.device)
z0 = base.screwdriver.data.joint_pos[:, base._screwdriver_z_id].clone()
ori, clr, frc, tilt = [], [], [], []
for k in range(args.steps):
    env.step(zero)
    if k >= args.steps // 3:
        ori.append(base.extras["eval_pad_orientation"].clone())
        clr.append(base.compute_surface_clearance().clone())
        total, body, cap, _ = base._read_contact_forces()
        frc.append((body + cap).clone())
        tilt.append(base.extras["eval_tilt_norm"].clone())

z1 = base.screwdriver.data.joint_pos[:, base._screwdriver_z_id]
drift = float(((z1 - z0).abs().mean() / (args.steps * base.step_dt)).item())
O = torch.stack(ori).mean(0).mean(0)
C = torch.stack(clr).mean(0).mean(0)
F = torch.stack(frc).mean(0).mean(0)
T = float(torch.stack(tilt).mean().item())

print(f"\n=== TRAINING START, option {args.option} (DR off, zero action) ===")
print(f"  tilt={T:.4f} rad   zero-action drift={drift:.5f} rad/s")
print(f"  {'finger':8s} {'clearance mm':>13} {'touch':>7} {'pad orient':>11} "
      f"{'verdict':>8} {'force N':>8}")
n_touch = n_pad = 0
for i, finger in enumerate(base.fingers):
    clearance_mm = float(C[i]) * 1000.0
    orientation = float(O[i])
    force = float(F[i])
    touching = clearance_mm <= 1.0
    verdict = ("PAD" if orientation > 0 else "BACK") if touching else "--"
    if touching:
        n_touch += 1
        if orientation > 0:
            n_pad += 1
    print(f"  {finger:8s} {clearance_mm:13.3f} {str(touching):>7} "
          f"{orientation:+11.3f} {verdict:>8} {force:8.2f}")
print(f"  => {n_touch} touching, {n_pad} of them pad-facing")

json.dump(
    dict(option=args.option, tilt=T, drift=drift, fingers=list(base.fingers),
         clearance_mm=[float(x) * 1000.0 for x in C],
         pad_orientation=[float(x) for x in O],
         force_n=[float(x) for x in F], n_touch=n_touch, n_pad=n_pad),
    open(SCRATCH + f"start_{args.option}.json", "w"), indent=2,
)
env.close()
app.close()
