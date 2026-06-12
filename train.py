"""Two-stage RMA training entry point for ScrewdriverRL.

Stage 1 — Teacher (asymmetric actor-critic)
  The actor sees policy obs (27-D).  The critic sees privileged obs (17-D:
  exact screwdriver pose, velocity, friction, fingertip distances).  PPO is
  run with RL-Games using a separate central-value network.  The deployment
  policy is the actor alone — it never touches privileged obs.

Stage 2 — Adaptation (proprioceptive history → privileged obs)
  Loads the frozen Stage 1 actor, rolls out the env, and trains a small
  temporal-conv network to predict the 17-D privileged obs from the last 30
  frames of [finger_q, joint_targets] (24-D per frame).  At deployment the
  adaptation network replaces ground-truth state, enabling sim-to-real
  transfer without privileged sensors.

Usage
-----
# Stage 1 (teacher PPO, ~200 M steps recommended)
python train.py --stage 1 --headless

# Resume Stage 1
python train.py --stage 1 --headless \\
    --checkpoint runs/.../stage1_nn/allegro_screwdriver_rotation.pth

# Stage 2 (run after Stage 1 converges)
python train.py --stage 2 --headless \\
    --checkpoint runs/.../stage1_nn/allegro_screwdriver_rotation.pth
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ScrewdriverRL two-stage RMA training.")
parser.add_argument("--task", type=str, default="Isaac-Allegro-Screwdriver-Rotation-Direct-v0")
parser.add_argument(
    "--stage",
    type=int,
    choices=[1, 2],
    default=1,
    help=(
        "1 = teacher PPO with asymmetric critic (privileged obs); "
        "2 = adaptation network training (requires --checkpoint to Stage 1 .pth)"
    ),
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_interval", type=int, default=2000)
# Stage 2 knobs
parser.add_argument("--adapt_iters", type=int, default=500, help="[Stage 2] Training iterations.")
parser.add_argument("--adapt_rollout_steps", type=int, default=512, help="[Stage 2] Rollout steps per iter.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = args.video

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import yaml
import gymnasium as gym
import torch

import screwdriver_rl.tasks  # noqa: F401

try:
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rl_games import (
        RlGamesAlgoObserver, RlGamesVecEnvWrapper, RlGamesGpuEnv,
    )
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.wrappers.rl_games import (
        RlGamesAlgoObserver, RlGamesVecEnvWrapper,
    )
    from isaaclab_rl.rl_games import RlGamesGpuEnv

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def _load_agent_cfg(task: str, num_envs: int, rl_device: str, seed: int) -> dict:
    cfg_path = os.path.join(
        os.path.dirname(__file__),
        "screwdriver_rl", "tasks", "allegro", "agents", "rl_games_ppo_cfg.yaml",
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["params"]["config"]["num_actors"] = num_envs
    cfg["params"]["config"]["device"] = rl_device
    cfg["params"]["config"]["device_name"] = rl_device
    cfg["params"]["seed"] = seed
    cfg["params"]["config"]["train_dir"] = os.path.join("runs", task)
    return cfg


def _register_rl_games(env, env_cfg) -> None:
    env_configurations.register("rlgpu", {
        "vecenv_type": "IsaacLab",
        "env_creator": lambda **_: RlGamesVecEnvWrapper(
            env, env_cfg, args.rl_device, getattr(args, "clip_obs", 5.0)
        ),
    })
    vecenv.register("IsaacLab", lambda name, actors, **_: RlGamesGpuEnv(name, actors))


def run_stage1(env_cfg, log_dir: str) -> None:
    # Enable asymmetric observations: actor sees policy obs (27-D),
    # critic sees privileged obs (17-D) via RL-Games central_value_config.
    env_cfg.asymmetric_obs = True
    env_cfg.state_space = env_cfg.privileged_obs_dim  # 17

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(
            env,
            os.path.join(log_dir, "videos"),
            episode_trigger=lambda ep: ep % args.video_interval == 0,
            disable_logger=True,
        )

    _register_rl_games(env, env_cfg)
    agent_cfg = _load_agent_cfg(args.task, env.unwrapped.num_envs, args.rl_device, args.seed)

    if args.checkpoint:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args.checkpoint

    print(
        f"\n[Stage 1] Task        : {args.task}"
        f"\n[Stage 1] Num envs    : {env.unwrapped.num_envs}"
        f"\n[Stage 1] Log dir     : {log_dir}"
        f"\n[Stage 1] Actor obs   : 27-D (policy)   Critic obs: 17-D (privileged)"
        + (f"\n[Stage 1] Resume from : {args.checkpoint}" if args.checkpoint else "")
        + "\n",
        flush=True,
    )

    runner = Runner(RlGamesAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False, "sigma": None})


def run_stage2(env_cfg, log_dir: str) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint pointing to the Stage 1 .pth is required for Stage 2.")

    env_cfg.asymmetric_obs = True
    env_cfg.state_space = env_cfg.privileged_obs_dim

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    _register_rl_games(env, env_cfg)

    # Build the frozen Stage 1 actor via RL-Games player.
    agent_cfg = _load_agent_cfg(args.task, env.unwrapped.num_envs, args.rl_device, args.seed)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint

    runner = Runner(RlGamesAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args.checkpoint)
    player.init_rnn()

    def frozen_actor(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, _ = player.get_action(obs, is_deterministic=True)
        return action

    from screwdriver_rl.algos.proprio_adapt import ProprioAdaptTrainer, AdaptTrainCfg
    adapt_cfg = AdaptTrainCfg(
        rollout_steps=args.adapt_rollout_steps,
        num_iters=args.adapt_iters,
    )
    stage2_dir = os.path.join(log_dir, "stage2_nn")

    print(
        f"\n[Stage 2] Task             : {args.task}"
        f"\n[Stage 2] Stage 1 ckpt     : {args.checkpoint}"
        f"\n[Stage 2] Adaptation iters : {adapt_cfg.num_iters}"
        f"\n[Stage 2] Rollout steps/it : {adapt_cfg.rollout_steps}"
        f"\n[Stage 2] Output dir       : {stage2_dir}\n",
        flush=True,
    )

    trainer = ProprioAdaptTrainer(
        env=env,
        stage1_actor_fn=frozen_actor,
        cfg=adapt_cfg,
        out_dir=stage2_dir,
        device=args.rl_device,
        priv_obs_dim=env_cfg.privileged_obs_dim,
        frame_dim=env_cfg.history_obs_dim,
        hist_len=env_cfg.prop_hist_len,
    )
    trainer.train()


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    log_dir = os.path.join("runs", args.task)
    os.makedirs(log_dir, exist_ok=True)

    if args.stage == 1:
        run_stage1(env_cfg, log_dir)
    else:
        run_stage2(env_cfg, log_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
