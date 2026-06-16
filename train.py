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
parser.add_argument(
    "--max_epochs",
    type=int,
    default=None,
    help="[Stage 1] Override rl_games max_epochs (useful for smoke tests / short runs).",
)
parser.add_argument(
    "--save_interval_steps",
    type=int,
    default=2_000_000,
    help=(
        "[Stage 1] Target env-step interval between checkpoints. Converted to "
        "rl_games epoch counts based on num_envs so the cadence is predictable "
        "regardless of env count (best-saving starts at half this interval)."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument(
    "--init_global_steps",
    type=int,
    default=0,
    help=(
        "[Stage 1] Seed the curriculum step counter so a resumed run starts in a "
        "later phase instead of Phase 0.  The counter is process state, not saved "
        "in the checkpoint, so a plain --checkpoint resume restarts the curriculum "
        "at Phase 0.  To fine-tune a final-phase policy (e.g. for the anti-wobble "
        "tweak), pass a value >= the last phase's step_start so it stays in the "
        "final phase.  0 (default) = start from Phase 0."
    ),
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help=(
        "Directory for checkpoints, tensorboard logs and videos. "
        "Defaults to runs/<task>. Stage 1 writes <output>/<run-name>/nn/*.pth; "
        "Stage 2 writes <output>/stage2_nn/proprio_adapt.pth."
    ),
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_interval", type=int, default=2000)
# Stage 2 knobs
parser.add_argument("--adapt_iters", type=int, default=500, help="[Stage 2] Training iterations.")
parser.add_argument("--adapt_rollout_steps", type=int, default=512, help="[Stage 2] Rollout steps per iter.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = args.video
args.rl_device = getattr(args, "rl_device", None) or args.device or "cuda:0"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import yaml
import gymnasium as gym
import torch

import screwdriver_rl.tasks  # noqa: F401

# Import paths differ across Isaac Lab releases.  Newest first, with fallbacks
# to the pre-rename (``isaaclab_tasks.utils.wrappers``) and the legacy
# ``omni.isaac.lab_tasks`` layouts.
try:
    # Current Isaac Lab: the RL-Games wrappers live in ``isaaclab_rl``.
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
except ImportError:
    try:
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tasks.utils.wrappers.rl_games import (
            RlGamesGpuEnv, RlGamesVecEnvWrapper,
        )
    except ImportError:  # legacy omni.isaac namespace
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        from omni.isaac.lab_tasks.utils.wrappers.rl_games import (
            RlGamesGpuEnv, RlGamesVecEnvWrapper,
        )

# The RL-Games algo observer was renamed ``RlGamesAlgoObserver`` ->
# ``IsaacAlgoObserver`` and moved into rl_games itself.
try:
    from rl_games.common.algo_observer import IsaacAlgoObserver as RlGamesAlgoObserver
except ImportError:  # very old Isaac Lab
    from isaaclab_tasks.utils.wrappers.rl_games import RlGamesAlgoObserver

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def _resolve_agent_cfg_path(task: str) -> str:
    """Resolve the rl_games YAML for ``task`` from its gym registration, so each
    hand uses its own ``agents/`` config instead of a hardcoded path."""
    import importlib

    entry = gym.spec(task).kwargs["rl_games_cfg_entry_point"]
    module_name, _, file_name = entry.partition(":")
    module = importlib.import_module(module_name)
    return os.path.join(os.path.dirname(module.__file__), file_name)


def _load_agent_cfg(num_envs: int, rl_device: str, seed: int, train_dir: str) -> dict:
    cfg_path = _resolve_agent_cfg_path(args.task)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["params"]["config"]["num_actors"] = num_envs
    cfg["params"]["config"]["device"] = rl_device
    cfg["params"]["config"]["device_name"] = rl_device
    cfg["params"]["seed"] = seed
    cfg["params"]["config"]["train_dir"] = train_dir
    if args.max_epochs is not None:
        cfg["params"]["config"]["max_epochs"] = args.max_epochs

    # RL-Games requires the per-epoch batch (num_actors * horizon_length) to be
    # an exact multiple of minibatch_size.  The shipped config targets the
    # production env count (2048); with smaller --num_envs (e.g. smoke tests)
    # the configured minibatch can exceed the batch, which makes RL-Games stall
    # on zero minibatches.  Shrink the minibatch to the largest divisor of the
    # batch that does not exceed the configured value.
    horizon = int(cfg["params"]["config"]["horizon_length"])
    batch = num_envs * horizon

    def _fit_minibatch(configured: int) -> int:
        mb = min(int(configured), batch)
        while mb > 1 and batch % mb != 0:
            mb -= 1
        return max(mb, 1)

    for path in (cfg["params"]["config"], cfg["params"].get("central_value_config")):
        if not path or "minibatch_size" not in path:
            continue
        fitted = _fit_minibatch(path["minibatch_size"])
        if fitted != path["minibatch_size"]:
            print(
                f"[train] Adjusting minibatch_size {path['minibatch_size']} -> {fitted} "
                f"to divide batch (num_envs={num_envs} * horizon={horizon} = {batch}).",
                flush=True,
            )
            path["minibatch_size"] = fitted

    # Checkpoint cadence.  RL-Games counts save_frequency / save_best_after in
    # *epochs*, and one epoch is num_envs * horizon_length steps — so with large
    # --num_envs the shipped 100/200-epoch gates map to tens of millions of steps
    # before the first checkpoint.  Translate the desired env-step interval into
    # epochs so the cadence is predictable regardless of num_envs.
    epoch_steps = batch  # num_envs * horizon_length
    save_freq = max(1, round(args.save_interval_steps / epoch_steps))
    save_best_after = max(1, round(0.5 * args.save_interval_steps / epoch_steps))
    cfg["params"]["config"]["save_frequency"] = save_freq
    cfg["params"]["config"]["save_best_after"] = save_best_after
    print(
        f"[train] Checkpoint cadence: periodic every {save_freq} epochs "
        f"(~{save_freq * epoch_steps:,} steps), best-saving after {save_best_after} epochs "
        f"(~{save_best_after * epoch_steps:,} steps).",
        flush=True,
    )
    return cfg


def _register_rl_games(env, agent_cfg: dict) -> None:
    # The current RlGamesVecEnvWrapper signature is
    #   (env, rl_device, clip_obs, clip_actions, obs_groups=None, concate_obs_group=True)
    # Asymmetric actor-critic is resolved automatically: when the env exposes a
    # "critic" observation group (i.e. env_cfg.state_space > 0), the wrapper maps
    # it to RL-Games "states" while the actor sees "policy".
    env_section = agent_cfg["params"].get("env", {})
    clip_obs = float(env_section.get("clip_observations", 5.0))
    clip_actions = float(env_section.get("clip_actions", 1.0))
    obs_groups = env_section.get("obs_groups")
    concate_obs_group = env_section.get("concate_obs_groups", True)

    wrapped = RlGamesVecEnvWrapper(
        env, args.rl_device, clip_obs, clip_actions, obs_groups, concate_obs_group
    )
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_: wrapped}
    )


def run_stage1(env_cfg, log_dir: str) -> None:
    # Enable asymmetric observations: actor sees policy obs (27-D),
    # critic sees privileged obs (17-D) via RL-Games central_value_config.
    env_cfg.asymmetric_obs = True
    env_cfg.state_space = env_cfg.privileged_obs_dim  # 17

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.init_global_steps > 0:
        # Seed the curriculum step counter (process state, not in the checkpoint)
        # so a resumed run continues in a later phase instead of restarting at
        # Phase 0.  ``_update_curriculum`` selects the matching phase on the first
        # step and prints the transition banner.
        env.unwrapped._global_steps = int(args.init_global_steps)
        print(
            f"[Stage 1] Curriculum step counter seeded to {args.init_global_steps:,} "
            f"— resumes in a later phase, not Phase 0.",
            flush=True,
        )
    if args.video:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(
            env,
            os.path.join(log_dir, "videos"),
            episode_trigger=lambda ep: ep % args.video_interval == 0,
            disable_logger=True,
        )

    agent_cfg = _load_agent_cfg(env.unwrapped.num_envs, args.rl_device, args.seed, log_dir)
    _register_rl_games(env, agent_cfg)

    if args.checkpoint:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args.checkpoint

    print(
        f"\n[Stage 1] Task        : {args.task}"
        f"\n[Stage 1] Num envs    : {env.unwrapped.num_envs}"
        f"\n[Stage 1] Log dir     : {log_dir}"
        f"\n[Stage 1] Actor obs   : {env_cfg.observation_space.shape[0]}-D (policy)   "
        f"Critic obs: {env_cfg.privileged_obs_dim}-D (privileged)"
        + (f"\n[Stage 1] Resume from : {args.checkpoint}" if args.checkpoint else "")
        + "\n",
        flush=True,
    )

    from screwdriver_rl.utils.rl_games_observer import PhaseCheckpointObserver
    observer = PhaseCheckpointObserver(env.unwrapped)
    runner = Runner(observer)
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False, "sigma": None})

    # Save the final-phase checkpoint: the last curriculum phase has no
    # transition to trigger on, so capture it now that training has ended.
    observer.save_final_phase()

    # Close the env before the app shuts down.  Skipping this leaves the
    # timeline "playing", and Isaac Sim's stop handler then renders a frame
    # during simulation_app.close(), which can deadlock on teardown.
    env.close()


def run_stage2(env_cfg, log_dir: str) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint pointing to the Stage 1 .pth is required for Stage 2.")

    env_cfg.asymmetric_obs = True
    env_cfg.state_space = env_cfg.privileged_obs_dim

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)

    # Build the frozen Stage 1 actor via RL-Games player.
    agent_cfg = _load_agent_cfg(env.unwrapped.num_envs, args.rl_device, args.seed, log_dir)
    _register_rl_games(env, agent_cfg)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint

    runner = Runner(RlGamesAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args.checkpoint)
    player.init_rnn()
    # We call player.get_action() directly on batched observations instead of
    # going through player.run() (which is what normally sets this flag).
    # Without it the player treats the (num_envs, obs_dim) batch as a single
    # unbatched observation and flattens it, breaking the network input.
    player.has_batch_dimension = True

    def frozen_actor(obs: torch.Tensor) -> torch.Tensor:
        # PpoPlayerContinuous.get_action returns the action tensor directly
        # (not a tuple); unpacking it would corrupt the batch dimension.
        with torch.no_grad():
            return player.get_action(obs, is_deterministic=True)

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

    # See run_stage1: close the env before app teardown to avoid a render
    # deadlock during simulation_app.close().
    env.close()


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    log_dir = args.output or os.path.join("runs", args.task)
    os.makedirs(log_dir, exist_ok=True)

    if args.stage == 1:
        run_stage1(env_cfg, log_dir)
    else:
        run_stage2(env_cfg, log_dir)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        # Print the traceback *before* closing the simulator.  Isaac Sim's
        # teardown can hang (render -> cuda.set_device) and would otherwise
        # swallow the real error.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
