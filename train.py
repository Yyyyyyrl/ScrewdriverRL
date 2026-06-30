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
        "Stage 2 writes <output>/stage2_nn/proprio_adapt.pth (plus periodic "
        "proprio_adapt_iter_*.pth and a rolling proprio_adapt_last.pth) and the "
        "self-contained deployable bundle <output>/stage2_nn/deploy.pth."
    ),
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_interval", type=int, default=2000)
# Stage 2 knobs
parser.add_argument("--adapt_iters", type=int, default=500, help="[Stage 2] Training iterations.")
parser.add_argument("--adapt_rollout_steps", type=int, default=512, help="[Stage 2] Rollout steps per iter.")
parser.add_argument(
    "--adapt_save_interval",
    type=int,
    default=50,
    help="[Stage 2] Write an intermediate checkpoint every N iters (0 disables).",
)
parser.add_argument(
    "--adapt_onpolicy",
    action="store_true",
    help="[Stage 2] Enable on-policy latent refinement (drive the frozen actor "
    "with the adapter's predicted latent, ramped in). OFF by default: it "
    "destabilises the upright screwdriver task (rollout collapse + rising "
    "AdaptLoss as the mix ramps up). Only enable with a gentle schedule.",
)
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

from screwdriver_rl.algos.latent_network import LATENT_NETWORK_NAME, register_latent_network


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

    # Register the HORA-faithful latent-conditioned network if the config selects
    # it (no-op for the legacy ``actor_critic`` network).  Must run before
    # ``Runner.load`` builds the model.
    if agent_cfg["params"].get("network", {}).get("name") == LATENT_NETWORK_NAME:
        register_latent_network()

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


def _build_deploy_meta(player, agent_cfg: dict, env_cfg, base_env) -> dict | None:
    """Capture the actor + obs normaliser + deployment config for the deploy bundle.

    Returns a dict consumed by ``ProprioAdaptTrainer`` to write ``deploy.pth``,
    or ``None`` if the actor model could not be read (in which case Stage 2 still
    saves the adapter-only checkpoint).  Reuses ``DeployPolicy``'s actor-state
    canonicaliser so the bundle loads into the env-free deploy actor 1:1.
    """
    try:
        from screwdriver_rl.deploy.policy import canonicalize_actor_state

        model = getattr(player, "model", None)
        if model is None:
            return None

        net = agent_cfg["params"]["network"]
        cfg_section = agent_cfg["params"]["config"]
        env_section = agent_cfg["params"].get("env", {})

        obs_dim = int(env_cfg.observation_space.shape[0])
        action_dim = int(env_cfg.action_space.shape[0])
        # HORA-faithful latent design: the actor consumes [proprio(proprio_dim),
        # latent(latent_dim)]; the deploy actor normalises only the proprio block,
        # so the rl_games obs normaliser is sliced to it.  Legacy mode
        # (latent_dim==0): the actor consumes the full obs incl. the raw euler.
        latent_dim = int(net.get("latent_dim", 0))
        proprio_dim = int(net.get("proprio_dim", obs_dim)) if latent_dim > 0 else obs_dim

        actor_state = canonicalize_actor_state(
            model.state_dict(), proprio_dim=proprio_dim if latent_dim > 0 else None
        )

        actor_arch = {
            "mlp_units": list(net["mlp"]["units"]),
            "activation": net["mlp"].get("activation", "elu"),
            "obs_dim": obs_dim,
            "proprio_dim": proprio_dim,
            "latent_dim": latent_dim,
            "action_dim": action_dim,
            "normalize_input": bool(cfg_section.get("normalize_input", True)),
            "clip_obs": float(env_section.get("clip_observations", 5.0)),
        }

        def _row(attr):
            t = getattr(base_env, attr, None)
            return None if t is None else t[0].detach().cpu().tolist()

        home = _row("_home_targets")
        if home is None:
            home = _row("_default_finger_pos") or _row("_cur_targets")

        config = {
            "task": args.task,
            "n_finger": action_dim,
            "action_delta_scale": float(getattr(env_cfg, "action_delta_scale", 0.05)),
            "finger_lower": _row("_finger_lower"),
            "finger_upper": _row("_finger_upper"),
            "home_targets": home,
            "prop_hist_len": int(env_cfg.prop_hist_len),
            "history_obs_dim": int(env_cfg.history_obs_dim),
            "privileged_obs_dim": int(env_cfg.privileged_obs_dim),
        }
        if latent_dim == 0:  # legacy euler-bridge bundle
            config["euler_dim"] = max(0, obs_dim - 2 * action_dim) or 3
        return {"actor": actor_state, "actor_arch": actor_arch, "config": config}
    except Exception as exc:  # never let bundling abort Stage-2 training
        print(f"[Stage 2] WARNING: could not assemble deploy bundle ({exc}); "
              f"saving adapter-only checkpoint.", flush=True)
        return None


def run_stage1(env_cfg, log_dir: str) -> None:
    # Enable asymmetric observations: the actor sees the policy obs, the critic
    # additionally sees the privileged obs via RL-Games central_value_config.
    # (Dims are hand-specific: Allegro 27/17, LinkerL20 35/19.)
    env_cfg.asymmetric_obs = True
    env_cfg.state_space = env_cfg.privileged_obs_dim

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

    _policy_dim = int(env_cfg.observation_space.shape[0])
    _priv_dim = int(env_cfg.privileged_obs_dim)
    print(
        f"\n[Stage 1] Task        : {args.task}"
        f"\n[Stage 1] Num envs    : {env.unwrapped.num_envs}"
        f"\n[Stage 1] Log dir     : {log_dir}"
        f"\n[Stage 1] Actor obs   : {_policy_dim}-D (policy)   Critic obs: {_priv_dim}-D (privileged)"
        + (f"\n[Stage 1] Resume from : {args.checkpoint}" if args.checkpoint else "")
        + "\n",
        flush=True,
    )

    from screwdriver_rl.utils.rl_games_observer import PhaseCheckpointObserver
    observer = PhaseCheckpointObserver(env.unwrapped)
    runner = Runner(observer)
    runner.load(agent_cfg)
    runner.reset()
    # NOTE: this rl_games version restores weights ONLY from the "checkpoint" key
    # in the run() args dict (see rl_games.torch_runner._restore); the
    # params["load_checkpoint"]/["load_path"] set above are ignored on this path.
    # Pass the checkpoint here so --checkpoint actually resumes instead of
    # silently training from scratch.
    runner.run(
        {
            "train": True,
            "play": False,
            "sigma": None,
            "checkpoint": args.checkpoint,
        }
    )

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
        save_interval=args.adapt_save_interval,
        onpolicy_latent=args.adapt_onpolicy,
    )
    stage2_dir = os.path.join(log_dir, "stage2_nn")

    # Assemble a self-contained deployable bundle (actor + obs normaliser + config)
    # so Stage 2 writes a HORA-style deploy.pth, not just the adapter.  All the
    # pieces already live in the restored player / env; see docs/3-deployment.md.
    deploy_meta = _build_deploy_meta(player, agent_cfg, env_cfg, env.unwrapped)

    # HORA-faithful latent mode: the adapter regresses the teacher latent
    # ``tanh(env_mlp(normalize(priv)))`` the Stage-1 actor consumed, and (for
    # on-policy refinement) the frozen actor can be driven by a supplied latent.
    # Both closures reuse the live restored model so no rebuild is needed.
    latent_dim = int(agent_cfg["params"]["network"].get("latent_dim", 0))
    teacher_latent_fn = None
    actor_with_latent_fn = None
    if latent_dim > 0:
        player.model.eval()
        _model = player.model
        _a2c = _model.a2c_network
        _proprio_dim = int(_a2c.proprio_dim)

        def teacher_latent_fn(policy_obs, _m=_model, _a=_a2c, _p=_proprio_dim):
            with torch.no_grad():
                x = _m.norm_obs(policy_obs)
                return torch.tanh(_a.env_mlp(x[:, _p:]))

        def actor_with_latent_fn(policy_obs, latent, _m=_model, _a=_a2c, _p=_proprio_dim):
            with torch.no_grad():
                x = _m.norm_obs(policy_obs)
                merged = torch.cat([x[:, :_p], latent], dim=-1)
                mu = _a.mu_act(_a.mu(_a.actor_mlp(merged)))
                return torch.clamp(mu, -1.0, 1.0)

    print(
        f"\n[Stage 2] Task             : {args.task}"
        f"\n[Stage 2] Stage 1 ckpt     : {args.checkpoint}"
        f"\n[Stage 2] Mode             : "
        f"{'HORA latent (dim ' + str(latent_dim) + ')' if latent_dim > 0 else 'legacy (priv-vector)'}"
        f"\n[Stage 2] Adaptation iters : {adapt_cfg.num_iters}"
        f"\n[Stage 2] Rollout steps/it : {adapt_cfg.rollout_steps}"
        f"\n[Stage 2] Save interval    : every {adapt_cfg.save_interval} iters"
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
        deploy_meta=deploy_meta,
        latent_dim=latent_dim or None,
        teacher_latent_fn=teacher_latent_fn,
        actor_with_latent_fn=actor_with_latent_fn,
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
