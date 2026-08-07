"""Measure pad-versus-back contact for a posture INSIDE the simulator."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,'/home/user/dex-forge')
parser=argparse.ArgumentParser()
parser.add_argument('--option',required=True)
parser.add_argument('--steps',type=int,default=120)
parser.add_argument('--num_envs',type=int,default=32)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import gymnasium as gym, torch, numpy as np
import screwdriver_rl.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
TASK='Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora'
cfg=parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
cfg.domain_rand.enabled=False
cfg.reset_contact_guard_min_fingers=0
env=gym.make(TASK,cfg=cfg); base=env.unwrapped
env.reset(seed=7)
zero=torch.zeros((base.num_envs, base.num_finger_dofs), device=base.device)
pads=[]; forces=[]
for k in range(args.steps):
    env.step(zero)
    if k>=args.steps//2:
        pads.append(base.extras['eval_pad_facing'].clone())
        t,b,c,_=base._read_contact_forces(); forces.append((b+c).clone())
P=torch.stack(pads).mean(0).mean(0); F=torch.stack(forces).mean(0).mean(0)
print(f"\n=== OPTION {args.option}: measured IN SIMULATION (DR off, zero action) ===")
print(f"  {'finger':8s} {'pad cos':>9} {'verdict':>9} {'force N':>9}")
for i,f in enumerate(base.fingers):
    pc=float(P[i]); fn=float(F[i])
    v='no load' if fn<0.05 else ('PAD' if pc>0 else 'BACK')
    print(f"  {f:8s} {pc:+9.3f} {v:>9} {fn:9.2f}")
json.dump({'option':args.option,'fingers':list(base.fingers),
           'pad_cos':[float(x) for x in P],'force_n':[float(x) for x in F]},
          open(f'/tmp/claude-1000/-home-user-dex-forge/c28006ed-89eb-4321-8443-2f52233f4e0c/scratchpad/padmeas_{args.option}.json','w'),indent=2)
env.close(); app.close()
