"""Evaluation / visualisation entry point for ScrewdriverRL.

Loads a checkpoint and runs the deterministic policy.  Renders in the
Isaac Sim viewport by default (omit --headless).  Works with both
single-env inspection and many-env statistics collection.

Usage
-----
# Visual playback (16 envs, opens viewport)
python play.py --checkpoint runs/Isaac-Allegro-Screwdriver-Rotation-Direct-v0/nn/allegro_screwdriver_rotation.pth

# Headless stats collection (512 envs, no window)
python play.py --checkpoint <path> --num_envs 512 --headless --num_episodes 20

# Record video to disk
python play.py --checkpoint <path> --video --video_length 300
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a screwdriver rotation policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Allegro-Screwdriver-Rotation-Direct-v0",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the .pth checkpoint to evaluate.",
)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=5, help="Episodes to run per env before exiting.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", help="Record viewport to an mp4.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Number of policy steps to record.",
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = args.video

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import yaml
import gymnasium as gym
import torch

import screwdriver_rl.tasks  # noqa: F401

try:
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rl_games import (
        RlGamesAlgoObserver,
        RlGamesVecEnvWrapper,
        RlGamesGpuEnv,
    )
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.wrappers.rl_games import (
        RlGamesAlgoObserver,
        RlGamesVecEnvWrapper,
    )
    from isaaclab_rl.rl_games import RlGamesGpuEnv

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed

    render_mode = "rgb_array" if args.video else "human"
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)

    if args.video:
        from gymnasium.wrappers import RecordVideo
        video_dir = os.path.join("runs", args.task, "eval_videos")
        env = RecordVideo(
            env,
            video_dir,
            episode_trigger=lambda ep: True,
            video_length=args.video_length,
            disable_logger=True,
        )

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "IsaacLab",
            "env_creator": lambda **_: RlGamesVecEnvWrapper(
                env, env_cfg, args.rl_device, getattr(args, "clip_obs", 5.0)
            ),
        },
    )
    vecenv.register(
        "IsaacLab",
        lambda config_name, num_actors, **_: RlGamesGpuEnv(config_name, num_actors),
    )

    agent_cfg_path = os.path.join(
        os.path.dirname(__file__),
        "screwdriver_rl", "tasks", "allegro", "agents", "rl_games_ppo_cfg.yaml",
    )
    with open(agent_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["device"] = args.rl_device
    agent_cfg["params"]["config"]["device_name"] = args.rl_device
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    # Run deterministic policy; games_num controls how many episodes to play.
    agent_cfg["params"]["config"]["player"]["deterministic"] = True
    agent_cfg["params"]["config"]["player"]["games_num"] = args.num_episodes

    log_dir = os.path.join("runs", args.task)
    agent_cfg["params"]["config"]["train_dir"] = log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(
        f"\n[play] Task        : {args.task}"
        f"\n[play] Checkpoint  : {args.checkpoint}"
        f"\n[play] Num envs    : {env.unwrapped.num_envs}"
        f"\n[play] Num episodes: {args.num_episodes}"
        f"\n[play] Device      : {args.rl_device}\n",
        flush=True,
    )

    runner = Runner(RlGamesAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": False, "play": True, "sigma": None, "checkpoint": args.checkpoint})


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
