#!/usr/bin/env python3
"""Stage 2: proprioceptive-adaptation student (teacher-student distillation).

Trains only the adaptation module against a frozen stage-1 teacher. The env
runs with the FINAL curriculum phase settings (strict terminations + full DR)
so the student adapts to the deployment distribution.

Usage:
    python scripts/train_student.py --headless --num_envs 8192 \
        --teacher outputs/allegro/<run>/stage1_nn/best.pth \
        --output outputs/allegro
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ScrewdriverRL stage-2 student training")
parser.add_argument("--task", type=str, default="Screwdriver-Continuous-Turning-v0")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--hand", type=str, default="allegro")
parser.add_argument("--teacher", type=str, required=True, help="Stage-1 checkpoint path")
parser.add_argument("--output", type=str, default="outputs/student")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume stage-2 checkpoint")
parser.add_argument("--curriculum_phase", type=str, default="final",
                    help="'final' applies the last curriculum phase env settings; 'none' uses cfg defaults")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cfg", action="append", default=[], help="env cfg override path=value")
parser.add_argument("--train-cfg", dest="train_cfg", action="append", default=[],
                    help="student train cfg override path=value")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: F401, E402
from screwdriver_rl.algo import ProprioAdapt  # noqa: E402
from screwdriver_rl.configs import NetworkCfg, StudentTrainCfg, get_curriculum  # noqa: E402
from screwdriver_rl.tasks.continuous_turning import ContinuousTurningEnvCfg  # noqa: E402
from screwdriver_rl.utils import apply_overrides, dump_config, cli  # noqa: E402


def main():
    device = args.device if args.device is not None else "cuda:0"
    torch.manual_seed(args.seed)

    env_cfg = ContinuousTurningEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.hand_name = args.hand
    env_cfg.seed = args.seed
    if args.curriculum_phase == "final":
        # Match the teacher's final training distribution.
        final_phase = get_curriculum("allegro_default")[-1]
        for path, value in final_phase.env_overrides.items():
            cli.set_dotted(env_cfg, path, value)
    env_overrides = apply_overrides(env_cfg, args.cfg)

    train_cfg = StudentTrainCfg()
    train_overrides = apply_overrides(train_cfg, args.train_cfg)
    net_cfg = NetworkCfg()

    env = gym.make(args.task, cfg=env_cfg).unwrapped

    run_name = args.run_name or args.task.replace("-", "_")
    output_dir = os.path.join(args.output, run_name)
    dump_config(output_dir, env=env_cfg, train=train_cfg, network=net_cfg,
                cli={"env_overrides": env_overrides, "train_overrides": train_overrides,
                     "teacher": args.teacher, "num_envs": args.num_envs,
                     "hand": args.hand, "seed": args.seed})

    print("=" * 70)
    print(f"Stage 2 student | task={args.task} hand={args.hand} envs={env.num_envs}")
    print(f"  teacher: {args.teacher}")
    print(f"  output:  {output_dir}")
    print("=" * 70)

    student = ProprioAdapt(env, output_dir, cfg=train_cfg, net_cfg=net_cfg, device=device)
    student.restore_from_teacher(args.teacher)
    if args.checkpoint:
        print(f"Resuming stage-2 from {args.checkpoint}")
        student.restore(args.checkpoint)

    student.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
