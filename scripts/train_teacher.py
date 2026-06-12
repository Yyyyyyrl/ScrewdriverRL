#!/usr/bin/env python3
"""Stage 1: privileged teacher PPO with curriculum.

Usage:
    python scripts/train_teacher.py --headless --num_envs 8192 \
        --output outputs/allegro --curriculum allegro_default

    # resume
    python scripts/train_teacher.py --headless --num_envs 8192 \
        --output outputs/allegro --checkpoint outputs/allegro/<run>/stage1_nn/last.pth

Any env cfg field can be overridden with repeated --cfg flags, e.g.:
    --cfg reward_turn_weight=500 --cfg dr.randomize_friction=false
Any train cfg field with --train-cfg, e.g. --train-cfg learning_rate=3e-3
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ScrewdriverRL stage-1 teacher training")
parser.add_argument("--task", type=str, default="Screwdriver-Continuous-Turning-v0")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--hand", type=str, default="allegro")
parser.add_argument("--curriculum", type=str, default="allegro_default",
                    help="Curriculum name from screwdriver_rl.configs.curricula (or 'none')")
parser.add_argument("--output", type=str, default="outputs/teacher")
parser.add_argument("--run_name", type=str, default=None, help="Subdirectory name (default: task id)")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from a stage-1 checkpoint")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cfg", action="append", default=[], help="env cfg override path=value")
parser.add_argument("--train-cfg", dest="train_cfg", action="append", default=[],
                    help="PPO train cfg override path=value")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: F401, E402
from screwdriver_rl.algo import PPO  # noqa: E402
from screwdriver_rl.configs import NetworkCfg, PPOTrainCfg, get_curriculum  # noqa: E402
from screwdriver_rl.tasks.continuous_turning import ContinuousTurningEnvCfg  # noqa: E402
from screwdriver_rl.utils import apply_overrides, dump_config  # noqa: E402


def main():
    device = args.device if args.device is not None else "cuda:0"
    torch.manual_seed(args.seed)

    env_cfg = ContinuousTurningEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.hand_name = args.hand
    env_cfg.seed = args.seed
    env_overrides = apply_overrides(env_cfg, args.cfg)

    train_cfg = PPOTrainCfg()
    train_overrides = apply_overrides(train_cfg, args.train_cfg)
    net_cfg = NetworkCfg()
    curriculum = get_curriculum(args.curriculum)

    env = gym.make(args.task, cfg=env_cfg).unwrapped

    run_name = args.run_name or args.task.replace("-", "_")
    output_dir = os.path.join(args.output, run_name)
    dump_config(output_dir, env=env_cfg, train=train_cfg, network=net_cfg,
                cli={"env_overrides": env_overrides, "train_overrides": train_overrides,
                     "curriculum": args.curriculum, "num_envs": args.num_envs,
                     "hand": args.hand, "seed": args.seed})

    print("=" * 70)
    print(f"Stage 1 teacher | task={args.task} hand={args.hand} envs={env.num_envs}")
    print(f"  obs={env.cfg.observation_space.shape} priv={env.cfg.privileged_obs_dim} "
          f"act={env.single_action_space.shape} curriculum={args.curriculum}")
    print(f"  output: {output_dir}")
    print("=" * 70)

    trainer = PPO(env, output_dir, cfg=train_cfg, net_cfg=net_cfg,
                  curriculum=curriculum, device=device)
    if args.checkpoint:
        print(f"Resuming from {args.checkpoint}")
        trainer.restore(args.checkpoint)

    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
