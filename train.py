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
import math
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
    "--ppo_learning_rate",
    type=float,
    default=None,
    help="[Stage 1] Optional actor and central-value learning-rate override.",
)
parser.add_argument(
    "--ppo_entropy_coef",
    type=float,
    default=None,
    help="[Stage 1] Optional non-negative actor entropy coefficient.",
)
parser.add_argument(
    "--ppo_score_to_win",
    type=float,
    default=None,
    help=(
        "[Stage 1] Optional RL-Games early-stop score override. Use a value "
        "above the attainable episode reward for fixed-sample runs."
    ),
)
parser.add_argument(
    "--ppo_lr_schedule",
    choices=("adaptive", "identity", "linear"),
    default=None,
    help="[Stage 1] Optional rl_games learning-rate schedule override.",
)
parser.add_argument(
    "--ppo_mini_epochs",
    type=int,
    default=None,
    help="[Stage 1] Optional actor and central-value mini-epoch override.",
)
parser.add_argument(
    "--ppo_sigma_init",
    type=float,
    default=None,
    help=(
        "[Stage 1] Optional initial log-standard-deviation for the continuous "
        "actor. Useful for contact-critical tasks where the default std=1 "
        "destroys the reset grip before a sustained contact gate can open."
    ),
)
parser.add_argument(
    "--ppo_sigma_override",
    type=float,
    default=None,
    help=(
        "[Stage 1] Optional log-standard-deviation applied after checkpoint "
        "restore, for explicit exploration annealing on resumed runs."
    ),
)
parser.add_argument(
    "--ppo_zero_mu_init",
    action="store_true",
    help=(
        "[Stage 1] Initialize the continuous actor mean output layer to zero. "
        "For delta-action tasks this starts from the validated zero-increment "
        "grasp instead of PyTorch's random Linear initialization."
    ),
)
parser.add_argument(
    "--phase0_contact_authority_weight",
    type=float,
    default=None,
    help="[Stage 1] Optional Phase-0 sustained contact-authority reward override.",
)
parser.add_argument(
    "--phase0_load_scale",
    type=float,
    default=None,
    help=(
        "[Stage 1] Optional Phase-0 screwdriver load-scale override. Intended "
        "for hard-contact-gated curriculum experiments; final-phase load is "
        "unchanged."
    ),
)
parser.add_argument(
    "--phase0_action_scale",
    type=float,
    default=None,
    help=(
        "[Stage 1] Optional Phase-0 multiplier on accumulated delta actions. "
        "Final-phase/deployment action scale is unchanged."
    ),
)
parser.add_argument(
    "--phase0_turn_weight",
    type=float,
    default=None,
    help="[Stage 1] Optional Phase-0 forward shaft-spin reward-weight override.",
)
parser.add_argument(
    "--phase_turn_weights",
    type=float,
    nargs=3,
    metavar=("PHASE0", "PHASE1", "PHASE2"),
    default=None,
    help=(
        "[Stage 1] Optional forward shaft-spin reward weights for all three "
        "curriculum phases. Useful for auditable target-free progress A/B runs."
    ),
)
parser.add_argument(
    "--phase_load_scales",
    type=float,
    nargs=3,
    metavar=("PHASE0", "PHASE1", "PHASE2"),
    default=None,
    help=(
        "[Stage 1] Optional screwdriver-load multipliers for all three "
        "curriculum phases. Intended for assisted gait discovery; production "
        "continuations and final evaluation must restore Phase 2 to 1.0."
    ),
)
parser.add_argument(
    "--phase_drive_weights",
    type=float,
    nargs=3,
    metavar=("PHASE0", "PHASE1", "PHASE2"),
    default=None,
    help=(
        "[Stage 1] Optional per-phase fingertip tangential-motion reward "
        "weights. This shapes cyclic contact motion without prescribing a "
        "screwdriver speed target."
    ),
)
parser.add_argument(
    "--target_bound_weight",
    type=float,
    default=None,
    help=(
        "[Stage 1] Optional penalty on accumulated joint targets in the outer "
        "20 percent of their action band. No screwdriver speed is prescribed."
    ),
)
parser.add_argument(
    "--reverse_to_turn_ratio",
    type=float,
    default=None,
    help=(
        "[Stage 1] Reverse-progress cost divided by the active linear forward "
        "weight. Does not prescribe a target speed."
    ),
)
parser.add_argument(
    "--turn_reward_power",
    type=float,
    default=None,
    help=(
        "[Stage 1] Target-free power applied symmetrically to forward and "
        "reverse angular speed; 1 is the legacy linear objective."
    ),
)
parser.add_argument(
    "--joint_motion_range",
    type=float,
    default=None,
    help=(
        "[Stage 1] Optional symmetric motion half-width around the validated "
        "home target. URDF hardware limits remain the final clamp."
    ),
)
parser.add_argument(
    "--action_delta_scale",
    type=float,
    default=None,
    help="[Stage 1] Per-step accumulated joint-target increment in radians.",
)
parser.add_argument(
    "--absolute_action_targets",
    action="store_true",
    help=(
        "[Stage 1] Map normalized actions directly to home-relative targets "
        "instead of integrating delta targets."
    ),
)
parser.add_argument(
    "--topdown_posture_search",
    type=str,
    default=None,
    help="[Stage 1] Audited top-down posture-search JSON for an isolated experiment.",
)
parser.add_argument(
    "--topdown_posture_candidate_index",
    type=int,
    default=None,
    help="[Stage 1] Candidate index selected from --topdown_posture_search.",
)
parser.add_argument(
    "--fixed_geometry_diameter_mm",
    type=int,
    choices=(64,),
    default=None,
    help="[Stage 1] Pin the experimental task to the physical 64 mm asset.",
)
parser.add_argument(
    "--phase_excess_weights",
    type=float,
    nargs=3,
    metavar=("PHASE0", "PHASE1", "PHASE2"),
    default=None,
    help="[Stage 1] Optional per-phase excessive fingertip-force penalties.",
)
parser.add_argument(
    "--phase_wrong_weights",
    type=float,
    nargs=3,
    metavar=("PHASE0", "PHASE1", "PHASE2"),
    default=None,
    help="[Stage 1] Optional per-phase non-fingertip contact-force penalties.",
)
parser.add_argument(
    "--phase0_fall_weight",
    type=float,
    default=None,
    help="[Stage 1] Optional Phase-0 one-shot fall-penalty override.",
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
        "[Stage 1 only] Seed the curriculum step counter so a resumed run starts in "
        "a later phase instead of Phase 0.  The counter is process state, not saved "
        "in the checkpoint, so a plain --checkpoint resume restarts the curriculum "
        "at Phase 0.  To fine-tune a final-phase policy (e.g. for the anti-wobble "
        "tweak), pass a value >= the last phase's step_start so it stays in the "
        "final phase.  0 (default) = start from Phase 0.  (Stage 2 uses "
        "--stage2_phase instead.)"
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
    "--adapt_continuous_rollouts",
    action="store_true",
    help=(
        "[Stage 2] Continue the environment across rollout chunks instead of "
        "resetting every iteration. This permits more envs and shorter chunks "
        "at the same samples/iter without biasing data toward episode starts."
    ),
)
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
parser.add_argument(
    "--adapt_resume_checkpoint",
    type=str,
    default=None,
    help="[Stage 2] Resume adapter weights and global iteration from a periodic checkpoint.",
)
parser.add_argument(
    "--adapt_onpolicy_warmup_iters",
    type=int,
    default=50,
    help="[Stage 2] Global iteration through which predicted-latent mixing stays zero.",
)
parser.add_argument(
    "--adapt_onpolicy_ramp_iters",
    type=int,
    default=100,
    help="[Stage 2] Global iterations used to ramp predicted-latent mixing from 0 to 1.",
)
parser.add_argument(
    "--stage2_phase",
    type=str,
    default="final",
    help="[Stage 2] Curriculum phase to train the adapter under. 'final' (default) "
    "pins the last phase — the deployment regime (matches --eval_phase final); "
    "'none' leaves the counter at 0 (Phase 1, the pre-pin behaviour); an integer "
    "pins that phase index. The teacher/adapter then see the pinned phase's "
    "termination + episode length; per-env dynamics diversity still comes from DR.",
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
    if args.ppo_score_to_win is not None:
        if not math.isfinite(args.ppo_score_to_win) or args.ppo_score_to_win <= 0.0:
            raise ValueError("--ppo_score_to_win must be finite and positive")
        cfg["params"]["config"]["score_to_win"] = float(args.ppo_score_to_win)
        print(
            "[train] PPO early-stop override: "
            f"score_to_win={float(args.ppo_score_to_win):.6g}",
            flush=True,
        )
    if args.ppo_sigma_init is not None:
        continuous = cfg["params"]["network"]["space"]["continuous"]
        sigma_init = continuous.get("sigma_init")
        if not isinstance(sigma_init, dict) or sigma_init.get("name") != "const_initializer":
            raise ValueError(
                "--ppo_sigma_init requires network.space.continuous.sigma_init "
                "to be a const_initializer"
            )
        sigma_init["val"] = float(args.ppo_sigma_init)
        print(
            "[train] PPO exploration override: "
            f"sigma_init_log_std={float(args.ppo_sigma_init):.6g} "
            f"(std={math.exp(float(args.ppo_sigma_init)):.6g})",
            flush=True,
        )
    if args.ppo_zero_mu_init:
        continuous = cfg["params"]["network"]["space"]["continuous"]
        continuous["mu_init"] = {"name": "const_initializer", "val": 0.0}
        print(
            "[train] PPO actor initialization override: mu output weights=0 "
            "(bias remains 0)",
            flush=True,
        )
    if args.ppo_entropy_coef is not None:
        if args.ppo_entropy_coef < 0.0:
            raise ValueError("--ppo_entropy_coef must be non-negative")
        cfg["params"]["config"]["entropy_coef"] = float(args.ppo_entropy_coef)
    if args.ppo_sigma_override is not None:
        if not math.isfinite(args.ppo_sigma_override):
            raise ValueError("--ppo_sigma_override must be finite")
        print(
            "[train] PPO post-restore exploration override: "
            f"log_std={float(args.ppo_sigma_override):.6g} "
            f"(std={math.exp(float(args.ppo_sigma_override)):.6g})",
            flush=True,
        )

    optim_configs = (
        cfg["params"]["config"],
        cfg["params"].get("central_value_config"),
    )
    for optim_cfg in optim_configs:
        if not optim_cfg:
            continue
        if args.ppo_learning_rate is not None:
            if args.ppo_learning_rate <= 0.0:
                raise ValueError("--ppo_learning_rate must be positive")
            optim_cfg["learning_rate"] = float(args.ppo_learning_rate)
        if args.ppo_lr_schedule is not None:
            optim_cfg["lr_schedule"] = args.ppo_lr_schedule
        if args.ppo_mini_epochs is not None:
            if args.ppo_mini_epochs <= 0:
                raise ValueError("--ppo_mini_epochs must be positive")
            optim_cfg["mini_epochs"] = int(args.ppo_mini_epochs)
    if any(
        value is not None
        for value in (
            args.ppo_learning_rate,
            args.ppo_lr_schedule,
            args.ppo_mini_epochs,
            args.ppo_entropy_coef,
        )
    ):
        print(
            "[train] PPO override: "
            f"learning_rate={cfg['params']['config']['learning_rate']}, "
            f"lr_schedule={cfg['params']['config']['lr_schedule']}, "
            f"mini_epochs={cfg['params']['config']['mini_epochs']}, "
            f"entropy_coef={cfg['params']['config']['entropy_coef']}",
            flush=True,
        )

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
        from screwdriver_rl.deploy.policy import (
            canonicalize_actor_state,
            nominal_geometry_row_index,
        )

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

        deployment_row = 0
        deployment_bucket = 0
        deployment_scale = [1.0, 1.0]
        variant_table = getattr(base_env, "_variant_table", None)
        if variant_table is not None:
            geometry_scales = torch.stack(
                (variant_table.diameter_scale, variant_table.length_scale), dim=-1
            )
            deployment_variant = nominal_geometry_row_index(
                geometry_scales, geometry_scales.shape[0]
            )
            deployment_scale = (
                geometry_scales[deployment_variant].detach().cpu().tolist()
            )
            deployment_bucket = int(
                variant_table.bucket[deployment_variant].detach().cpu().item()
            )
            env_variant_idx = getattr(base_env, "_env_variant_idx", None)
            if env_variant_idx is None:
                raise ValueError(
                    "geometry-aware task has no environment variant assignment"
                )
            deployment_rows = torch.nonzero(
                env_variant_idx == deployment_variant, as_tuple=False
            ).flatten()
            if deployment_rows.numel() == 0:
                raise ValueError(
                    "nominal geometry variant is absent from the environment batch"
                )
            deployment_row = int(deployment_rows[0].item())

        def _row(attr):
            t = getattr(base_env, attr, None)
            return None if t is None else t[deployment_row].detach().cpu().tolist()

        home = _row("_home_targets")
        if home is None:
            home = _row("_default_finger_pos") or _row("_cur_targets")

        # Preserve the collision-safe approach posture used by the simulator.
        # Geometry-aware reset tables are indexed by manifest bucket, whereas
        # home/limit tensors above are indexed by environment row.
        reset_table = getattr(base_env, "_reset_joint_pos", None)
        startup_reset_targets = home
        if reset_table is not None:
            reset_parts = []
            for finger in base_env.fingers:
                values = reset_table[finger]
                if values.ndim == 2:
                    values = values[deployment_bucket]
                reset_parts.append(values.reshape(-1))
            startup_reset_targets = (
                torch.cat(reset_parts).detach().cpu().tolist()
            )

        # Serialize the exact proprioception contract used by the live task.
        # ProprioAdaptTrainer validates this against the adaptation history and
        # actor widths, so deriving it from dimensions here would risk producing
        # a bundle that trains successfully but encodes real-hand observations
        # differently from simulation.
        proprio_codec = getattr(base_env, "_proprio_codec", None)
        codec_spec = getattr(proprio_codec, "spec", None)
        if codec_spec is None or not callable(getattr(codec_spec, "as_dict", None)):
            raise ValueError("task environment does not expose an explicit ProprioCodec")

        config = {
            "task": args.task,
            "n_finger": action_dim,
            "action_delta_scale": float(getattr(env_cfg, "action_delta_scale", 0.05)),
            "finger_lower": _row("_finger_lower"),
            "finger_upper": _row("_finger_upper"),
            "home_targets": home,
            "startup_reset_targets": startup_reset_targets,
            "prop_hist_len": int(env_cfg.prop_hist_len),
            "history_obs_dim": int(env_cfg.history_obs_dim),
            "privileged_obs_dim": int(env_cfg.privileged_obs_dim),
            "observation_semantics_version": str(
                env_cfg.observation_semantics_version
            ),
            "proprio_codec": codec_spec.as_dict(),
            "deployment_env_index": deployment_row,
            "deployment_geometry_bucket": deployment_bucket,
            "deployment_geometry_scale": deployment_scale,
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
            "sigma": args.ppo_sigma_override,
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


def _assert_checkpoint_matches_task(ckpt_path: str, env_cfg) -> None:
    """Fail fast if the Stage-1 checkpoint's privileged width doesn't match this task.

    The DR and non-DR LinkerL20 phase checkpoints share a basename (differing only by
    the ``runs/<task>/`` output dir), so it is easy to feed the wrong one.  Their
    latent-encoder input differs (19 vs 21), which would otherwise crash cryptically
    deep inside ``player.restore`` with an ``env_mlp``/normaliser shape mismatch.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return  # let player.restore surface any genuine load error
    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    w = model.get("a2c_network.env_mlp.0.weight") if isinstance(model, dict) else None
    if w is None:
        return  # legacy (non-latent) checkpoint — nothing to compare
    ckpt_priv, task_priv = int(w.shape[1]), int(env_cfg.privileged_obs_dim)
    if ckpt_priv != task_priv:
        raise ValueError(
            f"Stage-1 checkpoint privileged dim {ckpt_priv} != task '{args.task}' "
            f"privileged_obs_dim {task_priv}\n  checkpoint: {ckpt_path}\n"
            "This checkpoint was trained for a different task (DR vs non-DR). "
            "Pass the --checkpoint that matches --task."
        )


def run_stage2(env_cfg, log_dir: str) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint pointing to the Stage 1 .pth is required for Stage 2.")
    _assert_checkpoint_matches_task(args.checkpoint, env_cfg)

    env_cfg.asymmetric_obs = True
    env_cfg.state_space = env_cfg.privileged_obs_dim

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)

    # Pin the curriculum phase for adaptation.  Stage 2 is a frozen actor + a pure
    # latent-MSE, so it ignores the per-phase reward weights; the only phase effect
    # on its data is termination + episode length.  Deployment == the final phase, so
    # by default we seed the global step counter past the last phase's step_start (the
    # env selects the phase from _global_steps every step).  Per-env dynamics diversity
    # still comes from domain randomisation, which is phase-independent.
    base_env = env.unwrapped
    _phases = getattr(env_cfg, "curriculum_phases", None)
    if _phases:
        _spec = str(args.stage2_phase).strip().lower()
        _idx = (len(_phases) - 1) if _spec == "final" else (
            None if _spec in ("none", "") else int(_spec)
        )
        if _idx is not None:
            if not 0 <= _idx < len(_phases):
                raise ValueError(
                    f"--stage2_phase {args.stage2_phase} out of range for "
                    f"{len(_phases)} curriculum phases (use 'final', 'none', or 0..{len(_phases) - 1})."
                )
            base_env._global_steps = int(_phases[_idx].step_start)
            base_env._update_curriculum()  # apply now so the first reset uses this phase
            print(
                f"[Stage 2] Curriculum pinned to phase {_idx + 1}/{len(_phases)} "
                f"(_global_steps={base_env._global_steps:,}, episode_length_s="
                f"{env_cfg.episode_length_s}); --stage2_phase={args.stage2_phase}",
                flush=True,
            )

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
        continuous_rollouts=args.adapt_continuous_rollouts,
        num_iters=args.adapt_iters,
        save_interval=args.adapt_save_interval,
        resume_checkpoint=args.adapt_resume_checkpoint,
        onpolicy_latent=args.adapt_onpolicy,
        onpolicy_warmup_iters=args.adapt_onpolicy_warmup_iters,
        onpolicy_ramp_iters=args.adapt_onpolicy_ramp_iters,
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
    if args.topdown_posture_search is not None:
        from screwdriver_rl.utils.linker_topdown_candidate_override import (
            apply_candidate,
            load_candidate,
        )
        posture = load_candidate(
            args.topdown_posture_search, args.topdown_posture_candidate_index
        )
        apply_candidate(
            env_cfg, posture, fixed_64mm=args.fixed_geometry_diameter_mm == 64
        )
        print(
            "[Stage 1] Top-down posture override: "
            f"{args.topdown_posture_search} "
            f"candidate={args.topdown_posture_candidate_index} "
            f"fixed_diameter_mm={args.fixed_geometry_diameter_mm}",
            flush=True,
        )
    elif args.topdown_posture_candidate_index is not None:
        raise ValueError(
            "--topdown_posture_candidate_index requires --topdown_posture_search"
        )
    elif args.fixed_geometry_diameter_mm is not None:
        raise ValueError(
            "--fixed_geometry_diameter_mm currently requires "
            "--topdown_posture_search"
        )
    if args.stage == 1 and hasattr(env_cfg, "curriculum_phases"):
        phase0 = env_cfg.curriculum_phases[0]
        if args.phase0_contact_authority_weight is not None:
            phase0.w_contact_authority = float(args.phase0_contact_authority_weight)
            print(
                "[Stage 1] Phase-0 contact-authority override: "
                f"{phase0.w_contact_authority:g}",
                flush=True,
            )
        if args.phase0_load_scale is not None:
            if not 0.0 <= args.phase0_load_scale <= 1.0:
                raise ValueError("--phase0_load_scale must be in [0, 1]")
            phase0.screwdriver_load_scale = float(args.phase0_load_scale)
            print(
                "[Stage 1] Phase-0 screwdriver-load override: "
                f"{phase0.screwdriver_load_scale:g}",
                flush=True,
            )
        if args.phase0_action_scale is not None:
            if not 0.0 < args.phase0_action_scale <= 1.0:
                raise ValueError("--phase0_action_scale must be in (0, 1]")
            phase0.action_scale_multiplier = float(args.phase0_action_scale)
            print(
                "[Stage 1] Phase-0 action-scale override: "
                f"{phase0.action_scale_multiplier:g}",
                flush=True,
            )
        if args.phase0_turn_weight is not None:
            if args.phase0_turn_weight <= 0.0:
                raise ValueError("--phase0_turn_weight must be positive")
            phase0.reward_turn_weight = float(args.phase0_turn_weight)
            print(
                f"[Stage 1] Phase-0 turn-weight override: "
                f"{phase0.reward_turn_weight:g}",
                flush=True,
            )
        if args.phase_turn_weights is not None:
            weights = tuple(float(value) for value in args.phase_turn_weights)
            if len(env_cfg.curriculum_phases) != len(weights):
                raise ValueError(
                    "--phase_turn_weights requires exactly one value per phase"
                )
            if any(value <= 0.0 for value in weights):
                raise ValueError("--phase_turn_weights values must be positive")
            for phase, value in zip(env_cfg.curriculum_phases, weights, strict=True):
                phase.reward_turn_weight = value
            print(
                "[Stage 1] Three-phase turn-weight override: "
                + " / ".join(f"{value:g}" for value in weights),
                flush=True,
            )
        if args.phase_load_scales is not None:
            scales = tuple(float(value) for value in args.phase_load_scales)
            if len(env_cfg.curriculum_phases) != len(scales):
                raise ValueError(
                    "--phase_load_scales requires exactly one value per phase"
                )
            if any(not 0.0 <= value <= 1.0 for value in scales):
                raise ValueError("--phase_load_scales values must be in [0, 1]")
            for phase, value in zip(
                env_cfg.curriculum_phases, scales, strict=True
            ):
                phase.screwdriver_load_scale = value
            print(
                "[Stage 1] Three-phase screwdriver-load override: "
                + " / ".join(f"{value:g}" for value in scales),
                flush=True,
            )
        if args.phase_drive_weights is not None:
            weights = tuple(float(value) for value in args.phase_drive_weights)
            if len(env_cfg.curriculum_phases) != len(weights):
                raise ValueError(
                    "--phase_drive_weights requires exactly one value per phase"
                )
            if any(value < 0.0 for value in weights):
                raise ValueError(
                    "--phase_drive_weights values must be non-negative"
                )
            for phase, value in zip(env_cfg.curriculum_phases, weights, strict=True):
                phase.w_drive = value
            print(
                "[Stage 1] Three-phase drive-weight override: "
                + " / ".join(f"{value:g}" for value in weights),
                flush=True,
            )
        if args.target_bound_weight is not None:
            if args.target_bound_weight < 0.0:
                raise ValueError("--target_bound_weight must be non-negative")
            env_cfg.w_target_bound = float(args.target_bound_weight)
            print(
                "[Stage 1] Target-bound penalty override: "
                f"{env_cfg.w_target_bound:g}",
                flush=True,
            )
        if args.reverse_to_turn_ratio is not None:
            if args.reverse_to_turn_ratio < 1.0:
                raise ValueError("--reverse_to_turn_ratio must be >= 1")
            env_cfg.reverse_to_turn_ratio = float(args.reverse_to_turn_ratio)
            print(
                "[Stage 1] Reverse/forward linear weight ratio: "
                f"{env_cfg.reverse_to_turn_ratio:g}",
                flush=True,
            )
        if args.turn_reward_power is not None:
            if not 1.0 <= args.turn_reward_power <= 2.0:
                raise ValueError("--turn_reward_power must be in [1, 2]")
            env_cfg.turn_reward_power = float(args.turn_reward_power)
            print(
                "[Stage 1] Target-free turn reward power: "
                f"{env_cfg.turn_reward_power:g}",
                flush=True,
            )
        if args.joint_motion_range is not None:
            if not 0.0 < args.joint_motion_range <= 1.0:
                raise ValueError("--joint_motion_range must be in (0, 1]")
            env_cfg.joint_motion_range = float(args.joint_motion_range)
            print(
                "[Stage 1] Joint-motion half-width override: "
                f"{env_cfg.joint_motion_range:g} rad (URDF-clamped)",
                flush=True,
            )
        if args.action_delta_scale is not None:
            if not 0.0 < args.action_delta_scale <= 0.2:
                raise ValueError("--action_delta_scale must be in (0, 0.2]")
            env_cfg.action_delta_scale = float(args.action_delta_scale)
            print(
                "[Stage 1] Action delta scale override: "
                f"{env_cfg.action_delta_scale:g} rad/step",
                flush=True,
            )
        if args.absolute_action_targets:
            env_cfg.absolute_action_targets = True
            print(
                "[Stage 1] Control mode: home-relative absolute targets",
                flush=True,
            )
        for option, attr, values in (
            ("--phase_excess_weights", "w_excess", args.phase_excess_weights),
            ("--phase_wrong_weights", "w_wrong", args.phase_wrong_weights),
        ):
            if values is None:
                continue
            weights = tuple(float(value) for value in values)
            if len(env_cfg.curriculum_phases) != len(weights):
                raise ValueError(f"{option} requires exactly one value per phase")
            if any(value < 0.0 for value in weights):
                raise ValueError(f"{option} values must be non-negative")
            for phase, value in zip(
                env_cfg.curriculum_phases, weights, strict=True
            ):
                setattr(phase, attr, value)
            print(
                f"[Stage 1] {attr} override: "
                + " / ".join(f"{value:g}" for value in weights),
                flush=True,
            )
        if args.phase0_fall_weight is not None:
            phase0.reward_fall_weight = float(args.phase0_fall_weight)
            print(
                f"[Stage 1] Phase-0 fall-weight override: {phase0.reward_fall_weight:g}",
                flush=True,
            )
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
