"""How much pregrasp tension buys five pad contacts without feeding creep.

``reset_zero_tension_targets`` snaps the finger targets onto the settled measured
positions at the end of every reset.  It exists to stop standing target
penetration from feeding the PhysX contact solver, which slowly self-rotates the
mounted handle -- the "it turns by itself" behaviour.  But measured at the actual
training start it releases the grasp so completely that the handle free-spins
under its own load torque at 0.085 rad/s anyway, so the flag trades one source of
self-rotation for another.

The requirement is only that the posture keeps the screwdriver from falling, not
that it grips hard, so the question is quantitative: the least retained tension
that puts all five pads in contact, judged the way the environment itself judges
contact.

Contact uses the calibrated per-finger margins from the config, because
``compute_surface_clearance`` measures from the distal body frame rather than the
pad; a flat 1 mm threshold against that raw clearance called a finger carrying
0.53 N "not touching".  Pad-versus-back uses orientation, because forces are
near zero by construction here and a force direction is undefined.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/home/user/dex-forge")

parser = argparse.ArgumentParser()
parser.add_argument("--retain", type=float, required=True)
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

# One retention level per process.  Isaac Sim supports a single SimulationContext
# per process, so looping gym.make() in here hangs on the SECOND environment --
# it span at 109% CPU for 27 minutes producing nothing.  The caller loops.
rows = []
for retain in [args.retain]:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.domain_rand.enabled = False
    cfg.reset_contact_guard_min_fingers = 0
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    env.reset(seed=11)

    # Blend the zero-tension snap back toward the commanded pregrasp target.
    if retain > 0.0:
        pregrasp = base._default_finger_pos
        base._cur_targets = (
            base._cur_targets * (1.0 - retain) + pregrasp * retain
        )

    margins = base._contact_d_margin[0]  # (n_fingers,) calibrated, metres
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

    detail, n_touch, n_pad, n_back = {}, 0, 0, 0
    for i, finger in enumerate(base.fingers):
        touching = bool(float(C[i]) <= float(margins[i]))
        orientation = float(O[i])
        if touching:
            n_touch += 1
            if orientation > 0:
                n_pad += 1
            else:
                n_back += 1
        detail[finger] = dict(clearance_mm=round(float(C[i]) * 1000, 2),
                              margin_mm=round(float(margins[i]) * 1000, 2),
                              touching=touching,
                              pad=round(orientation, 3),
                              force_n=round(float(F[i]), 2))
    rows.append(dict(retain=retain, tilt=T, drift=drift, n_touch=n_touch,
                     n_pad=n_pad, n_back=n_back, detail=detail))
    print(f"retain={retain:.2f}  touching={n_touch}  pad={n_pad} back={n_back}  "
          f"tilt={T:.4f}  drift={drift:.5f} rad/s  "
          f"forces={[detail[f]['force_n'] for f in base.fingers]}", flush=True)
    env.close()

json.dump(rows, open(SCRATCH + f"tension_{args.retain:.2f}.json", "w"), indent=2)
app.close()
