"""Evaluation / visualisation entry point for ScrewdriverRL.

Loads a checkpoint and runs the deterministic policy.  Renders in the
Isaac Sim viewport by default (omit --headless).  Works with both
single-env inspection and many-env statistics collection.

Usage
-----
# Visual playback (16 envs, opens viewport)
python play.py --checkpoint runs/Isaac-Allegro-Screwdriver-Rotation-Direct-v0/nn/allegro_screwdriver_rotation.pth

# Linker Hand L20 (pass its task id)
python play.py --task Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0 \
    --checkpoint runs/Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0/nn/linker_l20_screwdriver_rotation.pth

# Headless stats collection (512 envs, no window)
python play.py --checkpoint <path> --num_envs 512 --headless --num_episodes 20

# Record video to disk
python play.py --checkpoint <path> --video --video_length 300

# Clean eval that matches the trained regime (final curriculum phase, no
# randomisation, fixed start) — use this to compare against training metrics
python play.py --checkpoint <path> --no_domain_rand --fixed_start

By default the curriculum phase is pinned to the final trained phase
(``--eval_phase final``) so the printed reward breakdown and the termination
thresholds match training.  Pass ``--eval_phase none`` to reproduce the old
behaviour (phase 0).  Note: the curriculum, domain randomisation, and start
angle change the environment and the reward *display* only — the loaded policy
network is identical regardless of these flags.
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
    help="Path to the Stage-1 rl_games .pth checkpoint (the actor) to visualise.",
)
parser.add_argument(
    "--adapter_checkpoint",
    type=str,
    default=None,
    help="[deploy view] Path to a Stage-2 deploy.pth / proprio_adapt.pth. When "
    "given, render the DEPLOYED policy: the actor is driven by the adapter's "
    "proprioceptively-predicted latent (no privileged obs) instead of the true "
    "env_mlp latent — the HORA vis_s2 analogue. Use the Stage-1 --checkpoint the "
    "adapter was trained against.",
)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=5, help="Episodes to run per env before exiting.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--eval_phase",
    type=str,
    default="final",
    help=(
        "Curriculum phase to evaluate under.  'final' (default) pins the last "
        "phase so the reward display and termination/episode thresholds match "
        "the trained regime; 'none' leaves the env default (phase 0, lenient "
        "termination, phase-0 reward weights); an integer selects a specific "
        "phase index.  This only affects logging and termination — NOT the "
        "policy network, which is identical across phases."
    ),
)
parser.add_argument(
    "--no_domain_rand",
    action="store_true",
    help=(
        "Disable per-reset domain randomisation (screwdriver damping/mass, "
        "finger gains) and per-step observation noise, so the policy's nominal "
        "behaviour is shown instead of worst-case randomised dynamics."
    ),
)
parser.add_argument(
    "--fixed_start",
    action="store_true",
    help=(
        "Start the screwdriver at its fixed reset angle instead of a random Z "
        "orientation, for repeatable side-by-side inspection."
    ),
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Directory for player logs and recorded videos. Defaults to runs/<task>.",
)
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
args.rl_device = getattr(args, "rl_device", None) or args.device or "cuda:0"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import yaml
import gymnasium as gym
import torch

import screwdriver_rl.tasks  # noqa: F401

# Import paths differ across Isaac Lab releases.  Newest first, with fallbacks
# to the pre-rename (``isaaclab_tasks.utils.wrappers``) and the legacy
# ``omni.isaac.lab_tasks`` layouts.
try:
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

try:
    from rl_games.common.algo_observer import IsaacAlgoObserver as RlGamesAlgoObserver
except ImportError:  # very old Isaac Lab
    from isaaclab_tasks.utils.wrappers.rl_games import RlGamesAlgoObserver

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from screwdriver_rl.algos.latent_network import LATENT_NETWORK_NAME, register_latent_network


def _obs_tensor(obses):
    """Extract the actor obs tensor from the rl_games wrapper output (dict or tensor)."""
    if isinstance(obses, dict):
        return obses.get("obs", obses.get("policy"))
    return obses


def _load_adapter(path: str, device: str):
    """Load a ProprioAdaptNet from a deploy.pth (``adapter`` key) or proprio_adapt.pth
    (``net`` key).  Returns ``(net.eval(), euler_dim)``."""
    from screwdriver_rl.algos.proprio_adapt import ProprioAdaptNet

    state = torch.load(path, map_location=device, weights_only=False)
    nd = state["net_dims"]
    sd = state.get("adapter", state.get("net"))
    net = ProprioAdaptNet(
        frame_dim=int(nd["frame_dim"]), hist_len=int(nd["hist_len"]), out_dim=int(nd["out_dim"])
    ).to(device).eval()
    net.load_state_dict(sd)
    euler_dim = int(state.get("config", {}).get("euler_dim", 3))
    return net, euler_dim


def _apply_eval_phase(env, phase_spec: str) -> None:
    """Pin the curriculum phase for evaluation.

    ``play.py`` starts a fresh process, so the env's ``_global_steps`` counter
    — which drives curriculum selection — restarts at 0.  Left alone, the env
    runs under Phase-0 reward weights and the lenient Phase-0 termination
    threshold, not the regime the checkpoint was trained in.  Pinning the phase
    makes the printed reward breakdown and the termination/episode thresholds
    match the trained phase.  It does NOT change the policy network.
    """
    cfg = getattr(env, "cfg", None)
    phases = getattr(cfg, "curriculum_phases", None)
    if not phases:
        return
    spec = phase_spec.strip().lower()
    if spec == "none":
        return
    if spec == "final":
        idx = len(phases) - 1
    else:
        try:
            idx = int(spec)
        except ValueError:
            print(f"[play] Unrecognised --eval_phase '{phase_spec}'; leaving env default.", flush=True)
            return
        if not -len(phases) <= idx < len(phases):
            print(f"[play] --eval_phase index {idx} out of range; leaving env default.", flush=True)
            return

    target = phases[idx]
    # Pre-set both the active phase and the step counter so the env's own
    # ``_update_curriculum`` keeps this phase (and skips the transition banner).
    env._curriculum_phase = target
    env._global_steps = int(target.step_start)
    cfg.episode_length_s = target.episode_length_s
    print(
        f"[play] Curriculum phase pinned to @{target.step_start:,} "
        f"(turn_weight={target.reward_turn_weight}, "
        f"term_threshold={target.upright_termination_threshold} rad, "
        f"episode={target.episode_length_s}s)",
        flush=True,
    )


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed

    # ---- Evaluation-time config overrides (see CLI flags) ----
    if args.no_domain_rand and hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand.enabled = False
        print("[play] Domain randomisation + observation noise: DISABLED", flush=True)
    if args.fixed_start and hasattr(env_cfg, "randomize_obj_start"):
        env_cfg.randomize_obj_start = False
        print("[play] Screwdriver start angle: FIXED (no randomisation)", flush=True)

    if args.adapter_checkpoint:
        # Deploy view needs the proprio-history buffer the adapter reads (only
        # populated when asymmetric_obs is on).  The actor obs is unchanged.
        env_cfg.asymmetric_obs = True
        env_cfg.state_space = env_cfg.privileged_obs_dim

    log_dir = args.output or os.path.join("runs", args.task)

    render_mode = "rgb_array" if args.video else "human"
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    _apply_eval_phase(env.unwrapped, args.eval_phase)

    if args.video:
        from gymnasium.wrappers import RecordVideo
        video_dir = os.path.join(log_dir, "eval_videos")
        env = RecordVideo(
            env,
            video_dir,
            episode_trigger=lambda ep: True,
            video_length=args.video_length,
            disable_logger=True,
        )

    import importlib

    _entry = gym.spec(args.task).kwargs["rl_games_cfg_entry_point"]
    _module_name, _, _file_name = _entry.partition(":")
    _agent_module = importlib.import_module(_module_name)
    agent_cfg_path = os.path.join(os.path.dirname(_agent_module.__file__), _file_name)
    with open(agent_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    # Register the HORA-faithful latent-conditioned network if the config selects
    # it (no-op for legacy ``actor_critic`` configs). Must run before
    # ``Runner.load`` builds the model/player.
    if agent_cfg["params"].get("network", {}).get("name") == LATENT_NETWORK_NAME:
        register_latent_network()

    # Current RlGamesVecEnvWrapper signature:
    #   (env, rl_device, clip_obs, clip_actions, obs_groups=None, concate_obs_group=True)
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

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["device"] = args.rl_device
    agent_cfg["params"]["config"]["device_name"] = args.rl_device
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    # Run deterministic policy; games_num controls how many episodes to play.
    agent_cfg["params"]["config"]["player"]["deterministic"] = True
    agent_cfg["params"]["config"]["player"]["games_num"] = args.num_episodes

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

    if args.adapter_checkpoint:
        _play_deploy(runner, env, agent_cfg)
    else:
        runner.run({"train": False, "play": True, "sigma": None, "checkpoint": args.checkpoint})

    # Close the env before the app shuts down to avoid a render deadlock during
    # simulation_app.close() (Isaac Sim renders a frame in its stop handler).
    env.close()


def _play_deploy(runner, env, agent_cfg) -> None:
    """Render the DEPLOYED policy: drive the actor with the adapter's predicted
    latent (no privileged obs) — the HORA ``vis_s2`` analogue.  Mirrors
    ``eval.py --deploy_eval`` but in the play/render loop.
    """
    base_env = env.unwrapped
    player = runner.create_player()
    player.restore(args.checkpoint)

    adapter, euler_dim = _load_adapter(args.adapter_checkpoint, args.rl_device)
    latent_mode = bool(getattr(base_env.cfg, "latent_conditioned", False))
    a2c = player.model.a2c_network
    proprio_dim = int(getattr(a2c, "proprio_dim", getattr(base_env.cfg, "history_obs_dim", 0)))
    print(
        f"[play] DEPLOY VIEW: actor on adapter-predicted "
        f"{'latent (dim ' + str(int(adapter.head.out_features)) + ')' if latent_mode else 'euler'}"
        f"  |  adapter: {args.adapter_checkpoint}",
        flush=True,
    )

    obses = player.env_reset(player.env)
    player.get_batch_size(obses, 1)
    max_ep = int(base_env.max_episode_length)
    for _ in range(args.num_episodes * max_ep):
        obs_t = _obs_tensor(obses)
        hist = getattr(base_env, "_prop_hist_buf", None)
        with torch.no_grad():
            if latent_mode:
                xn = player.model.norm_obs(obs_t)
                latent = adapter(hist)
                merged = torch.cat([xn[:, :proprio_dim], latent], dim=-1)
                mu = a2c.mu_act(a2c.mu(a2c.actor_mlp(merged)))
                actions = torch.clamp(mu, -1.0, 1.0)
            else:
                if hist is not None and obs_t is not None:
                    obs_t[:, -euler_dim:] = adapter(hist)[:, :euler_dim]
                actions = player.get_action(obs_t, is_deterministic=True)
        obses, _, _, _ = player.env_step(player.env, actions)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        # Surface the real error before the simulator teardown (which can hang
        # on render -> cuda.set_device and would otherwise swallow it).
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
