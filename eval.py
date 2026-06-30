"""Aggregate-statistics evaluator for ScrewdriverRL.

Unlike ``play.py`` (which opens a viewport and shows you a handful of
environments), this script runs **headless** over many environments for
**full-length episodes** and reports the *distribution* of the same metrics
that scroll past during training (``FwdVel``, ``TiltNorm``, ``ContactGate``,
net turns, …).  It exists to answer two questions that a single viewport
cannot:

  1. Is the loaded policy actually the policy that produced the training
     numbers?  A training log line is a mean over ~2048 envs in steady state;
     16 cold-start envs in a viewport cannot reproduce it.  Average the same
     quantities over many envs and full episodes and they become comparable.

  2. Is a good reward score hiding bad behaviour?  The report includes
     reward-validity diagnostics (see ``--rot_damping_scale`` and the
     "coasting" split) that probe the most likely reward-hacking failure
     modes for this task: free-spin coasting and wobble-scraping.

By default it matches the *training distribution* (final curriculum phase,
domain randomisation ON, observation noise ON, deterministic actions).  That
is deliberately different from ``play.py --no_domain_rand`` (which shows the
*nominal* motion).  To reproduce training numbers you want the training
conditions, not the clean ones.

Usage
-----
# Faithful comparison against the training log (DR on, final phase)
python eval.py --checkpoint <path> --num_envs 256

# Size the deterministic-vs-stochastic gap
python eval.py --checkpoint <path> --stochastic

# Reward-validity stress test: is the spin real manipulation or free-spin?
python eval.py --checkpoint <path> --no_domain_rand --rot_damping_scale 4.0
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Aggregate evaluation of a screwdriver rotation policy.")
parser.add_argument("--task", type=str, default="Isaac-Allegro-Screwdriver-Rotation-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs to average over.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Policy steps to roll out. Default: 2x the (final-phase) episode length, "
    "so every env completes ~2 full episodes.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=0,
    help="Discard this many initial steps from the per-step aggregates so the "
    "cold-start ramp does not bias the means (episode-level stats are unaffected).",
)
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample actions from the policy instead of using the deterministic mean. "
    "Training metrics come from the stochastic policy; use this to size that gap.",
)
parser.add_argument(
    "--eval_phase",
    type=str,
    default="final",
    help="Curriculum phase to evaluate under ('final', 'none', or an index). "
    "'final' matches the trained regime. Affects reward weights/termination only.",
)
parser.add_argument(
    "--no_domain_rand",
    action="store_true",
    help="Disable domain randomisation + observation noise (shows NOMINAL behaviour, "
    "which will NOT match the training average — training ran with DR on).",
)
parser.add_argument(
    "--fixed_start",
    action="store_true",
    help="Start the screwdriver at its fixed reset angle instead of a random one.",
)
parser.add_argument(
    "--no_pad_gate",
    action="store_true",
    help="Disable the pad-facing requirement in the contact gate (distance-only), "
    "to A/B the effect of pad-facing on the logged ContactGate / reward.",
)
parser.add_argument(
    "--rot_damping_scale",
    type=float,
    default=1.0,
    help="Multiply the screwdriver rotation-joint damping by this factor. >1 makes "
    "the handle resist free-spin; a policy that only works near 1.0 was exploiting a "
    "low-friction bearing rather than learning manipulation. Best used with "
    "--no_domain_rand for a clean signal.",
)
parser.add_argument(
    "--success_turns",
    type=float,
    default=2.0,
    help="An episode counts as a success if it did NOT fall over (timed out upright) "
    "and accumulated at least this many net forward turns.",
)
parser.add_argument(
    "--deploy_eval",
    action="store_true",
    help="Stage-2 deployment gate: replace the actor's ground-truth screwdriver "
    "euler with the proprioceptive-adaptation network's PREDICTION (no privileged "
    "obs), i.e. run the exact deploy-time inference path. Compare against the "
    "oracle (true-euler) run to size the sim-to-deploy gap before hardware.",
)
parser.add_argument(
    "--adapter_checkpoint",
    type=str,
    default=None,
    help="[--deploy_eval] Path to the Stage-2 deploy.pth or proprio_adapt.pth. "
    "Defaults to <checkpoint dir>/../stage2_nn/{deploy,proprio_adapt}.pth.",
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

# This script never renders.
args.headless = True
args.enable_cameras = False
args.rl_device = getattr(args, "rl_device", None) or args.device or "cuda:0"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import yaml
import gymnasium as gym
import torch

import screwdriver_rl.tasks  # noqa: F401

# Import paths differ across Isaac Lab releases (newest first).
try:
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
except ImportError:
    try:
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    except ImportError:  # legacy omni.isaac namespace
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

try:
    from rl_games.common.algo_observer import IsaacAlgoObserver as RlGamesAlgoObserver
except ImportError:  # very old Isaac Lab
    from isaaclab_tasks.utils.wrappers.rl_games import RlGamesAlgoObserver

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def _pin_eval_phase(env, phase_spec: str) -> None:
    """Pin the curriculum phase (see play.py:_apply_eval_phase for the rationale)."""
    cfg = getattr(env, "cfg", None)
    phases = getattr(cfg, "curriculum_phases", None)
    if not phases:
        return
    spec = phase_spec.strip().lower()
    if spec == "none":
        return
    idx = len(phases) - 1 if spec == "final" else None
    if idx is None:
        try:
            idx = int(spec)
        except ValueError:
            print(f"[eval] Unrecognised --eval_phase '{phase_spec}'; leaving env default.", flush=True)
            return
        if not -len(phases) <= idx < len(phases):
            print(f"[eval] --eval_phase index {idx} out of range; leaving env default.", flush=True)
            return
    target = phases[idx]
    env._curriculum_phase = target
    env._global_steps = int(target.step_start)
    cfg.episode_length_s = target.episode_length_s
    print(
        f"[eval] Curriculum phase pinned to @{target.step_start:,} "
        f"(term_threshold={target.upright_termination_threshold} rad, "
        f"episode={target.episode_length_s}s)",
        flush=True,
    )


class _RunningStat:
    """Streaming mean/std over (step x env) samples, kept on-device until the end."""

    def __init__(self, device: torch.device) -> None:
        self._n = torch.zeros((), device=device)
        self._sum = torch.zeros((), device=device)
        self._sumsq = torch.zeros((), device=device)

    def update(self, x: torch.Tensor) -> None:
        x = x.flatten().float()
        self._n += x.numel()
        self._sum += x.sum()
        self._sumsq += (x * x).sum()

    def result(self) -> tuple[float, float]:
        n = self._n.item()
        if n == 0:
            return float("nan"), float("nan")
        mean = self._sum.item() / n
        var = max(self._sumsq.item() / n - mean * mean, 0.0)
        return mean, math.sqrt(var)


def _safe(extras: dict, key: str) -> torch.Tensor | None:
    v = extras.get(key)
    return v if isinstance(v, torch.Tensor) else None


def _resolve_adapter_path(checkpoint: str, explicit: str | None) -> str | None:
    """Find the Stage-2 adapter checkpoint for --deploy_eval."""
    if explicit:
        return explicit
    # Stage-1 ckpts live in <run>/nn/*.pth; Stage-2 in <run>/stage2_nn/*.pth.
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    for name in ("deploy.pth", "proprio_adapt.pth", "proprio_adapt_last.pth"):
        cand = os.path.join(run_dir, "stage2_nn", name)
        if os.path.exists(cand):
            return cand
    return None


def _load_adapter(path: str, device: str):
    """Load a ProprioAdaptNet from a deploy.pth or proprio_adapt.pth bundle.

    Returns ``(net.eval(), euler_dim)``.  Works for both the deployable bundle
    (``{"adapter", "net_dims"}``) and the adapter-only checkpoint (``{"net",
    "net_dims"}``).
    """
    from screwdriver_rl.algos.proprio_adapt import ProprioAdaptNet

    state = torch.load(path, map_location=device)
    nd = state["net_dims"]
    sd = state.get("adapter", state.get("net"))
    net = ProprioAdaptNet(
        frame_dim=int(nd["frame_dim"]), hist_len=int(nd["hist_len"]), out_dim=int(nd["out_dim"])
    ).to(device).eval()
    net.load_state_dict(sd)
    euler_dim = int(state.get("config", {}).get("euler_dim", 3))
    return net, euler_dim


def _obs_tensor(obses):
    """Extract the actor obs tensor from the rl_games wrapper output (dict or tensor)."""
    if isinstance(obses, dict):
        return obses.get("obs", obses.get("policy"))
    return obses


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed

    if args.no_domain_rand and hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand.enabled = False
        print("[eval] Domain randomisation + observation noise: DISABLED", flush=True)
    if args.fixed_start and hasattr(env_cfg, "randomize_obj_start"):
        env_cfg.randomize_obj_start = False
        print("[eval] Screwdriver start angle: FIXED", flush=True)
    if args.no_pad_gate and hasattr(env_cfg, "require_pad_facing"):
        env_cfg.require_pad_facing = False
        print("[eval] Pad-facing contact gate: DISABLED (distance-only)", flush=True)

    if args.deploy_eval:
        # Maintain the proprio-history buffer the adapter reads (only updated when
        # asymmetric_obs is on). The actor obs dim (policy) is unchanged.
        env_cfg.asymmetric_obs = True
        env_cfg.state_space = env_cfg.privileged_obs_dim

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    _pin_eval_phase(base_env, args.eval_phase)

    # Optional rotation-damping stress test (reward-validity probe).
    if args.rot_damping_scale != 1.0 and hasattr(base_env, "_base_rotation_damping"):
        scale = float(args.rot_damping_scale)
        base_env._base_rotation_damping *= scale  # so DR (if on) scales around the new base
        damp = torch.full(
            (base_env.num_envs, 1),
            base_env._base_rotation_damping,
            device=base_env.device,
        )
        base_env.screwdriver.write_joint_damping_to_sim(damp, joint_ids=[base_env._screwdriver_z_id])
        print(f"[eval] Screwdriver rotation damping scaled x{scale:g}", flush=True)

    # ---- rl_games wrapper + player (reuses the exact obs-normalisation pipeline) ----
    import importlib

    _entry = gym.spec(args.task).kwargs["rl_games_cfg_entry_point"]
    _module_name, _, _file_name = _entry.partition(":")
    _agent_module = importlib.import_module(_module_name)
    agent_cfg_path = os.path.join(os.path.dirname(_agent_module.__file__), _file_name)
    with open(agent_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    env_section = agent_cfg["params"].get("env", {})
    clip_obs = float(env_section.get("clip_observations", 5.0))
    clip_actions = float(env_section.get("clip_actions", 1.0))
    obs_groups = env_section.get("obs_groups")
    concate_obs_group = env_section.get("concate_obs_groups", True)
    wrapped = RlGamesVecEnvWrapper(env, args.rl_device, clip_obs, clip_actions, obs_groups, concate_obs_group)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_: wrapped}
    )

    agent_cfg["params"]["config"]["num_actors"] = base_env.num_envs
    agent_cfg["params"]["config"]["device"] = args.rl_device
    agent_cfg["params"]["config"]["device_name"] = args.rl_device
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    agent_cfg["params"]["config"]["player"]["deterministic"] = not args.stochastic

    # Register the HORA-faithful latent network if the config selects it (no-op
    # for the legacy ``actor_critic`` network).  Must run before ``Runner.load``.
    from screwdriver_rl.algos.latent_network import LATENT_NETWORK_NAME, register_latent_network
    if agent_cfg["params"].get("network", {}).get("name") == LATENT_NETWORK_NAME:
        register_latent_network()

    runner = Runner(RlGamesAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args.checkpoint)

    # Sanity check: confirm the observation normaliser actually restored.  If it is
    # all-zeros (mean) / all-ones (var), normalisation did not load and behaviour
    # would diverge badly from training regardless of everything else.
    rms = getattr(getattr(player, "model", None), "running_mean_std", None)
    if rms is not None and hasattr(rms, "running_mean"):
        mean_abs = rms.running_mean.abs().mean().item()
        var_mean = rms.running_var.mean().item()
        status = "OK" if (mean_abs > 1e-6 or abs(var_mean - 1.0) > 1e-3) else "SUSPECT (looks uninitialised!)"
        print(f"[eval] obs normaliser: |mean|={mean_abs:.4f} var={var_mean:.4f} -> {status}", flush=True)

    # Stage-2 deployment gate: load the adapter that substitutes the privileged
    # signal the actor consumes at deploy.
    adapter = None
    euler_dim = 3
    # HORA-faithful latent mode: the actor consumes [proprio, latent]; the gate
    # drives it with the adapter's PREDICTED latent (vs env_mlp's true latent in
    # the oracle run).  Legacy mode: the adapter substitutes the raw euler.
    latent_mode = bool(getattr(base_env.cfg, "latent_conditioned", False))
    deploy_a2c = None
    deploy_proprio_dim = 0
    if args.deploy_eval:
        adapter_path = _resolve_adapter_path(args.checkpoint, args.adapter_checkpoint)
        if adapter_path is None:
            raise FileNotFoundError(
                "--deploy_eval needs a Stage-2 adapter; none found next to the "
                "checkpoint. Pass --adapter_checkpoint <stage2_nn/deploy.pth>."
            )
        adapter, euler_dim = _load_adapter(adapter_path, args.rl_device)
        if latent_mode:
            deploy_a2c = player.model.a2c_network
            deploy_proprio_dim = int(getattr(deploy_a2c, "proprio_dim",
                                             base_env.cfg.history_obs_dim))
            print(f"[eval] DEPLOY GATE (latent): actor driven by adapter-predicted "
                  f"latent (dim={int(adapter.head.out_features)}), no privileged obs\n"
                  f"[eval] Adapter      : {adapter_path}", flush=True)
        else:
            print(f"[eval] DEPLOY GATE: euler from adapter prediction (no privileged obs)\n"
                  f"[eval] Adapter      : {adapter_path}  (euler_dim={euler_dim})", flush=True)

    max_ep = int(base_env.max_episode_length)
    num_steps = args.steps if args.steps is not None else 2 * max_ep
    is_det = not args.stochastic

    print(
        f"\n[eval] Task        : {args.task}"
        f"\n[eval] Checkpoint  : {args.checkpoint}"
        f"\n[eval] Num envs    : {base_env.num_envs}"
        f"\n[eval] Steps       : {num_steps}  (episode = {max_ep} steps)"
        f"\n[eval] Actions     : {'deterministic' if is_det else 'stochastic'}"
        f"\n[eval] Device      : {args.rl_device}\n",
        flush=True,
    )

    device = base_env.device

    # Per-step streaming stats.  Superset across hands; whichever the running task
    # does not emit simply stays empty (reported as nan) and is skipped in the
    # layout branch below.
    step_keys = [
        # shared
        "eval_fwd_vel", "eval_rev_vel", "eval_tilt_norm", "eval_upright_gate",
        "eval_contact_gate", "eval_binary_gate",
        "eval_total_reward", "eval_turn_reward",
        # force-based contact (LinkerL20)
        "eval_drive_count", "eval_in_window", "eval_contact_force",
        "eval_index_cap_force", "eval_idle_count", "eval_wrong_surface_force",
        "eval_max_joint_dev",
        # distance/pad-based contact (Allegro)
        "eval_motion_gate", "eval_pad_gate", "eval_pad_cos",
        "eval_contact_count", "eval_avg_contact_speed", "eval_min_tip_dist",
    ]
    stats = {k: _RunningStat(device) for k in step_keys}

    # Coasting diagnostic: forward spin split by whether qualifying contact exists.
    coast = {
        "fwd_in": torch.zeros((), device=device), "n_in": torch.zeros((), device=device),
        "fwd_out": torch.zeros((), device=device), "n_out": torch.zeros((), device=device),
    }

    # Per-episode collectors (captured at the done step, before auto-reset).
    ep_net_turns: list[torch.Tensor] = []
    ep_total_turns: list[torch.Tensor] = []
    ep_failed: list[torch.Tensor] = []  # True = fell over (tilt termination)

    # Per-(step,env) tilt samples, kept to report the distribution (not just
    # mean±std) — a heavy right tail is what makes std > mean.
    tilt_samples: list[torch.Tensor] = []

    obses = player.env_reset(player.env)
    # BasePlayer.run() calls this after reset to set ``has_batch_dimension``/
    # ``batch_size`` from the obs shape.  We drive the rollout manually, so we
    # must do it too — otherwise the player treats the whole (num_envs, obs)
    # batch as a single flattened observation and the first Linear layer fails
    # with a shape-mismatch (mat1 = 1 x num_envs*obs_dim).
    player.get_batch_size(obses, 1)

    for step in range(num_steps):
        # Deployment gate: replace the privileged signal the actor consumes with
        # the adapter's proprioceptive prediction, so the actor runs on exactly
        # the signal it would have on hardware.
        if adapter is not None and latent_mode:
            # Latent gate: drive the *live* actor trunk with the predicted latent
            # (the exact hardware inference path: mu(actor_mlp([norm(proprio),
            # adapter(hist)]))).  The oracle run (no --deploy_eval) instead uses
            # env_mlp(true priv).
            obs_t = _obs_tensor(obses)
            hist = getattr(base_env, "_prop_hist_buf", None)
            with torch.no_grad():
                xn = player.model.norm_obs(obs_t)
                latent = adapter(hist)
                merged = torch.cat([xn[:, :deploy_proprio_dim], latent], dim=-1)
                mu = deploy_a2c.mu_act(deploy_a2c.mu(deploy_a2c.actor_mlp(merged)))
                actions = torch.clamp(mu, -1.0, 1.0)
        else:
            if adapter is not None:
                # Legacy euler-bridge gate: overwrite the last euler_dim columns.
                hist = getattr(base_env, "_prop_hist_buf", None)
                obs_t = _obs_tensor(obses)
                if hist is not None and obs_t is not None:
                    with torch.no_grad():
                        obs_t[:, -euler_dim:] = adapter(hist)[:, :euler_dim]
            actions = player.get_action(obses, is_deterministic=is_det)

        obses, _, _, _ = player.env_step(player.env, actions)
        ex = base_env.extras

        if step >= args.warmup_steps:
            for k, stat in stats.items():
                t = _safe(ex, k)
                if t is not None:
                    stat.update(t)

            tilt = _safe(ex, "eval_tilt_norm")
            if tilt is not None:
                tilt_samples.append(tilt.detach().clone())

            fwd = _safe(ex, "eval_fwd_vel")
            bgate = _safe(ex, "eval_binary_gate")
            if fwd is not None and bgate is not None:
                in_contact = bgate > 0.5
                coast["fwd_in"] += fwd[in_contact].sum()
                coast["n_in"] += in_contact.sum()
                coast["fwd_out"] += fwd[~in_contact].sum()
                coast["n_out"] += (~in_contact).sum()

        # Capture full-episode outcomes for envs that finished this step.
        terminated = getattr(base_env, "reset_terminated", None)
        timed_out = getattr(base_env, "reset_time_outs", None)
        if terminated is not None and timed_out is not None:
            done = terminated | timed_out
            if bool(done.any()):
                net = _safe(ex, "eval_net_turns")
                tot = _safe(ex, "eval_total_turns")
                if net is not None:
                    ep_net_turns.append(net[done].detach().clone())
                    ep_total_turns.append(tot[done].detach().clone())
                    ep_failed.append(terminated[done].detach().clone())

    # ---------------------------------------------------------------- report
    def line(label: str, key: str, fmt: str = "{:+.3f}") -> str:
        mean, std = stats[key].result()
        return f"    {label:<16} {fmt.format(mean):>10}  ± {fmt.format(std).lstrip('+'):>8}"

    print(f"\n{'='*64}")
    print("  AGGREGATE EVALUATION  (mean ± std over step x env samples)")
    print(f"{'='*64}")
    print("  Rotation")
    print(line("FwdVel (rad/s)", "eval_fwd_vel"))
    print(line("RevVel (rad/s)", "eval_rev_vel"))
    print(line("TurnRew", "eval_turn_reward"))
    print(line("TotalRew", "eval_total_reward"))
    print("  Object / upright")
    print(line("TiltNorm (rad)", "eval_tilt_norm"))
    if tilt_samples:
        tilt_all = torch.cat(tilt_samples).float()
        q = torch.quantile(tilt_all, torch.tensor([0.5, 0.9, 0.99], device=tilt_all.device))
        upright_frac = (tilt_all < 0.10).float().mean().item()
        print(f"      └ tilt  p50 {q[0].item():.3f}  p90 {q[1].item():.3f}  "
              f"p99 {q[2].item():.3f}  |  {upright_frac*100:.1f}% of steps < 0.10 rad")
    print(line("UprightGate", "eval_upright_gate"))
    print("  Contact")
    print(line("ContactGate", "eval_contact_gate"))
    print(line("BinaryGate", "eval_binary_gate"))
    # Force-based (LinkerL20) vs distance/pad-based (Allegro) contact diagnostics.
    force_based = not math.isnan(stats["eval_in_window"].result()[0])
    if force_based:
        print(line("DriveCount", "eval_drive_count", "{:.2f}"))
        print(line("InWindow", "eval_in_window"))
        print(line("ContactForce N", "eval_contact_force", "{:.3f}"))
        print(line("IndexCapForce N", "eval_index_cap_force", "{:.3f}"))
        print(line("IdleCount", "eval_idle_count", "{:.2f}"))
        print(line("WrongSurf N", "eval_wrong_surface_force", "{:.3f}"))
        print(line("MaxJointDev rad", "eval_max_joint_dev", "{:.3f}"))
    else:
        print(line("MotionGate", "eval_motion_gate"))
        print(line("PadGate", "eval_pad_gate"))
        print(line("PadCos", "eval_pad_cos"))
        print(line("ContactCount", "eval_contact_count"))
        print(line("AvgContactSpd", "eval_avg_contact_speed"))
        print(line("MinTipDist (m)", "eval_min_tip_dist"))

    # Episode-level outcomes.
    if ep_net_turns:
        net = torch.cat(ep_net_turns).float()
        tot = torch.cat(ep_total_turns).float()
        failed = torch.cat(ep_failed).bool()
        n_ep = net.numel()
        fall_rate = failed.float().mean().item()
        upright = ~failed
        success = upright & (net >= args.success_turns)
        succ_rate = success.float().mean().item()

        def pct(t: torch.Tensor, q: float) -> float:
            return torch.quantile(t, q).item() if t.numel() else float("nan")

        print(f"\n  Episodes completed : {n_ep}")
        print(f"  NetTurns/episode   : mean {net.mean().item():+.2f}  std {net.std().item():.2f}  "
              f"[p10 {pct(net,0.1):+.2f}  p50 {pct(net,0.5):+.2f}  p90 {pct(net,0.9):+.2f}]")
        print(f"  TotalTurns/episode : mean {tot.mean().item():+.2f}")
        print(f"  Fell over (tilt)   : {fall_rate*100:5.1f}%  of episodes")
        print(f"  Success (>={args.success_turns:g} turns, upright): {succ_rate*100:5.1f}%")
    else:
        print("\n  No episodes completed within the rollout (increase --steps).")

    # Reward-validity / coasting diagnostic.
    n_in = coast["n_in"].item()
    n_out = coast["n_out"].item()
    fwd_in = coast["fwd_in"].item() / n_in if n_in else float("nan")
    fwd_out = coast["fwd_out"].item() / n_out if n_out else float("nan")
    frac_in = n_in / (n_in + n_out) if (n_in + n_out) else float("nan")
    print(f"\n  Reward-validity (coasting probe)")
    print(f"    Steps with contact gate open (binary): {frac_in*100:5.1f}%")
    print(f"    Mean FwdVel  WHILE in contact        : {fwd_in:+.3f} rad/s")
    if n_out > 0:
        print(f"    Mean FwdVel  WITHOUT contact (coast) : {fwd_out:+.3f} rad/s")
        print("    (High spin without contact => free-spin/coasting, not manipulation.)")
    else:
        print("    Mean FwdVel  WITHOUT contact (coast) : n/a — fingers were in contact")
        print("    every step (no free-spin coasting observed). Confirm with --rot_damping_scale.")
    print(f"{'='*64}\n", flush=True)

    env.close()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
