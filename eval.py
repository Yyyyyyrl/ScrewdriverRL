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
import json
import math
import os
from pathlib import Path

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
    "--fixed_geometry_diameter_mm",
    type=int,
    choices=(64,),
    default=None,
    help=(
        "Pin the top-down geometry asset to the specified physical handle "
        "diameter while preserving dynamics randomisation and the trained "
        "21-D privileged-observation contract. Currently only 64 mm is "
        "supported."
    ),
)
parser.add_argument(
    "--joint_motion_range",
    type=float,
    default=None,
    help=(
        "Override the symmetric motion half-width used by the trained policy. "
        "URDF joint limits remain the final clamp."
    ),
)
parser.add_argument("--action_delta_scale", type=float, default=None)
parser.add_argument(
    "--reset_action_hold_steps",
    type=int,
    default=None,
    help=(
        "Override the post-reset action-mute window (policy steps). The v8 "
        "fall-timing trace shows falls cluster at steps 14-21, i.e. the end "
        "of the default hold(5)+ramp(10) mute: under zero-tension starts the "
        "mute leaves the handle uncaged before the policy may act. Use 0 to "
        "grant immediate authority and re-measure fall rate."
    ),
)
parser.add_argument(
    "--reset_action_ramp_steps",
    type=int,
    default=None,
    help="Override the post-reset action ramp length (policy steps).",
)
parser.add_argument("--absolute_action_targets", action="store_true")
parser.add_argument(
    "--screwdriver_load_scale",
    type=float,
    default=None,
    help=(
        "Override all curriculum-phase screwdriver load multipliers for "
        "diagnostic evaluation. Final deployment acceptance must use 1.0."
    ),
)
parser.add_argument("--topdown_posture_search", type=str, default=None)
parser.add_argument("--topdown_posture_candidate_index", type=int, default=None)
parser.add_argument(
    "--fixed_start",
    action="store_true",
    help="Start the screwdriver at its fixed reset angle instead of a random one.",
)
parser.add_argument(
    "--wrong_surface_trace_n",
    type=float,
    default=None,
    help=(
        "Record a trace event whenever any env's non-fingertip (wrong-surface) "
        "contact force exceeds this many newtons: rollout step, env, force, "
        "tilt, episode age and geometry variant, plus whether the env "
        "terminates within the next few steps. Use to localise extreme "
        "spikes (e.g. the 2020 N outlier in the v8 report) as reset-settle "
        "impulses vs fall impacts. Off by default."
    ),
)
parser.add_argument(
    "--zero_action",
    action="store_true",
    help=(
        "Physics-exploit probe: override every policy action with zeros and "
        "report the resulting metrics. A healthy environment must show "
        "net/raw turns ~0 — any sustained rotation under motionless fingers "
        "is a simulation artifact (e.g. solver creep from standing target "
        "penetration) that the reward would otherwise pay for. Run this "
        "before every training campaign (plan doc M2/M4 acceptance)."
    ),
)
parser.add_argument(
    "--root_pos_bias_mm",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 0.0, 0.0),
    help=(
        "Add a fixed world-frame hand-root position bias in millimetres. "
        "This is applied before environment creation and remains constant "
        "across auto-resets; symmetric reset DR, when enabled, is added on top."
    ),
)
parser.add_argument(
    "--root_rpy_bias_deg",
    type=float,
    nargs=3,
    metavar=("ROLL", "PITCH", "YAW"),
    default=(0.0, 0.0, 0.0),
    help=(
        "Add a fixed world-frame hand-root roll/pitch/yaw bias in degrees. "
        "Per-geometry top-down root quaternions are perturbed consistently."
    ),
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
    default=3.0,
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
    "--deploy_openloop",
    action="store_true",
    help="[--deploy_eval] Drive the actor with the ORACLE latent (in-distribution "
    "trajectory) while still recording the adapter's latent/action error. Diagnostic "
    "to separate covariate shift from a static adapter error: if the adapter is "
    "accurate here but collapses in the default closed-loop deploy run, the failure "
    "is compounding covariate shift, not wiring.",
)
parser.add_argument(
    "--adapter_checkpoint",
    type=str,
    default=None,
    help="[--deploy_eval] Path to the Stage-2 deploy.pth or proprio_adapt.pth. "
    "Defaults to <checkpoint dir>/../stage2_nn/{deploy,proprio_adapt}.pth.",
)
parser.add_argument(
    "--json_output",
    type=str,
    default=None,
    help="Optional path for a machine-readable copy of the aggregate report.",
)
parser.add_argument(
    "--calibration_output",
    type=str,
    default=None,
    help=(
        "Optional JSON path for force-free calibration statistics. Collects "
        "per-fingertip surface clearance, force, and absolute target tracking "
        "error without changing observations, rewards, or policy actions."
    ),
)
parser.add_argument(
    "--action_trace_output",
    type=str,
    default=None,
    help=(
        "Optional JSON path for a bounded per-step trace of policy actions, "
        "joint targets, finger positions, and physical task metrics. Read-only "
        "diagnostic; normal evaluation is unchanged when omitted."
    ),
)
parser.add_argument(
    "--action_trace_envs",
    type=int,
    default=3,
    help="Number of leading environments to include in --action_trace_output.",
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


def _quat_mul_wxyz(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_from_rpy_deg(
    rpy_deg: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(value) for value in rpy_deg)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _apply_fixed_root_bias(env_cfg) -> None:
    position_bias_m = tuple(float(value) / 1000.0 for value in args.root_pos_bias_mm)
    rpy_bias_deg = tuple(float(value) for value in args.root_rpy_bias_deg)
    if any(not math.isfinite(value) for value in (*position_bias_m, *rpy_bias_deg)):
        raise ValueError("fixed root biases must be finite")

    if any(value != 0.0 for value in position_bias_m):
        nominal = tuple(float(value) for value in env_cfg.robot_cfg.init_state.pos)
        env_cfg.robot_cfg.init_state.pos = tuple(
            value + offset for value, offset in zip(nominal, position_bias_m)
        )
        print(
            "[eval] Fixed hand-root position bias: "
            f"({args.root_pos_bias_mm[0]:+.3f}, "
            f"{args.root_pos_bias_mm[1]:+.3f}, "
            f"{args.root_pos_bias_mm[2]:+.3f}) mm",
            flush=True,
        )

    if any(value != 0.0 for value in rpy_bias_deg):
        delta = _quat_from_rpy_deg(rpy_bias_deg)
        nominal_quat = tuple(
            float(value) for value in env_cfg.robot_cfg.init_state.rot
        )
        env_cfg.robot_cfg.init_state.rot = _quat_mul_wxyz(delta, nominal_quat)
        bucket_quats = getattr(env_cfg, "pregrasp_root_quats_buckets", None)
        if bucket_quats is not None:
            env_cfg.pregrasp_root_quats_buckets = [
                _quat_mul_wxyz(
                    delta,
                    tuple(float(value) for value in bucket_quat),
                )
                for bucket_quat in bucket_quats
            ]
        print(
            "[eval] Fixed hand-root RPY bias: "
            f"({rpy_bias_deg[0]:+.3f}, {rpy_bias_deg[1]:+.3f}, "
            f"{rpy_bias_deg[2]:+.3f}) deg",
            flush=True,
        )


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
        # The existing fixed-geometry block below owns asset-bank pinning.
        apply_candidate(env_cfg, posture, fixed_64mm=False)
    elif args.topdown_posture_candidate_index is not None:
        raise ValueError(
            "--topdown_posture_candidate_index requires --topdown_posture_search"
        )
    _apply_fixed_root_bias(env_cfg)
    if args.joint_motion_range is not None:
        if not 0.0 < args.joint_motion_range <= 1.0:
            raise ValueError("--joint_motion_range must be in (0, 1]")
        env_cfg.joint_motion_range = float(args.joint_motion_range)
        print(
            f"[eval] Joint-motion half-width: {env_cfg.joint_motion_range:g} rad "
            "(URDF-clamped)",
            flush=True,
        )

    if args.action_delta_scale is not None:
        if not 0.0 < args.action_delta_scale <= 0.2:
            raise ValueError("--action_delta_scale must be in (0, 0.2]")
        env_cfg.action_delta_scale = float(args.action_delta_scale)
    if args.reset_action_hold_steps is not None:
        if args.reset_action_hold_steps < 0:
            raise ValueError("--reset_action_hold_steps must be >= 0")
        env_cfg.reset_action_hold_steps = int(args.reset_action_hold_steps)
    if args.reset_action_ramp_steps is not None:
        if args.reset_action_ramp_steps < 0:
            raise ValueError("--reset_action_ramp_steps must be >= 0")
        env_cfg.reset_action_ramp_steps = int(args.reset_action_ramp_steps)
        print(
            f"[eval] Action delta scale: {env_cfg.action_delta_scale:g} rad/step",
            flush=True,
        )

    if args.absolute_action_targets:
        env_cfg.absolute_action_targets = True
        print("[eval] Control mode: home-relative absolute targets", flush=True)

    if args.screwdriver_load_scale is not None:
        if not 0.0 <= args.screwdriver_load_scale <= 1.0:
            raise ValueError("--screwdriver_load_scale must be in [0, 1]")
        if not hasattr(env_cfg, "curriculum_phases"):
            raise ValueError("--screwdriver_load_scale requires curriculum phases")
        for phase in env_cfg.curriculum_phases:
            phase.screwdriver_load_scale = float(args.screwdriver_load_scale)
        print(
            f"[eval] Screwdriver load scale: {args.screwdriver_load_scale:g}",
            flush=True,
        )

    if args.fixed_geometry_diameter_mm is not None:
        if not getattr(env_cfg.domain_rand, "randomize_geometry", False):
            raise ValueError(
                "--fixed_geometry_diameter_mm requires a geometry-randomised "
                "task configuration"
            )
        diameter_mm = int(args.fixed_geometry_diameter_mm)
        if diameter_mm != 64:
            raise ValueError(f"unsupported fixed geometry diameter: {diameter_mm}")
        assets_cfg = list(env_cfg.screwdriver_cfg.spawn.assets_cfg)
        if len(assets_cfg) != 3:
            raise ValueError(
                "fixed top-down geometry expects the 60/64/68 mm three-asset bank"
            )
        env_cfg.screwdriver_cfg.spawn.assets_cfg = [assets_cfg[1]]
        env_cfg.screwdriver_variants_dir = str(
            Path(__file__).resolve().parent
            / "assets/screwdriver/topdown_variants_fixed64"
        )
        print(
            "[eval] Geometry asset: FIXED 64 mm; dynamics randomisation unchanged",
            flush=True,
        )

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
        "eval_fwd_vel", "eval_rev_vel", "eval_osc_ratio",
        "eval_tilt_norm", "eval_upright_gate",
        "eval_contact_gate", "eval_binary_gate", "eval_motion_auth",
        "eval_total_reward", "eval_turn_reward",
        "eval_contact_authority_reward",
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
    # The hard performance gate uses physical shaft motion.  Keep the
    # contact-authorized counters beside it as an anti-coasting diagnostic.
    ep_net_turns: list[torch.Tensor] = []
    ep_total_turns: list[torch.Tensor] = []
    ep_authorized_net_turns: list[torch.Tensor] = []
    ep_authorized_total_turns: list[torch.Tensor] = []
    ep_env_ids: list[torch.Tensor] = []
    ep_failed: list[torch.Tensor] = []  # True = fell over (tilt termination)

    # Per-(step,env) tilt samples, kept to report the distribution (not just
    # mean±std) — a heavy right tail is what makes std > mean.
    tilt_samples: list[torch.Tensor] = []
    wrong_surface_samples: list[torch.Tensor] = []
    wrong_surface_events: list[dict] = []
    termination_events: list[tuple[int, int]] = []  # (rollout_step, env)
    # Episode age maintained here: base_env.episode_length_buf is already
    # zeroed for done envs by the time env_step returns, so it cannot date
    # same-step events (fall timing, spike-at-fall traces).
    ep_age = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
    fall_ages: list[torch.Tensor] = []
    # Per-episode DR-corner capture (over finished envs): fell flag plus the
    # standing DR values that persist for the whole episode. Answers whether
    # residual falls concentrate in the extreme DR corners (e.g. the 13.1x
    # load-torque tail) rather than being uniform control failures.
    term_fell: list[torch.Tensor] = []
    term_load: list[torch.Tensor] = []
    term_place_mm: list[torch.Tensor] = []
    term_diam: list[torch.Tensor] = []
    latent_delta_samples: list[torch.Tensor] = []
    latent_action_error_samples: list[torch.Tensor] = []
    action_trace: list[dict] = []
    calibration_clearance_samples: list[torch.Tensor] = []
    calibration_force_samples: list[torch.Tensor] = []
    calibration_target_error_samples: list[torch.Tensor] = []
    trace_envs = min(max(int(args.action_trace_envs), 0), int(base_env.num_envs))

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
                oracle_latent = torch.tanh(
                    deploy_a2c.env_mlp(xn[:, deploy_proprio_dim:])
                )
                merged = torch.cat([xn[:, :deploy_proprio_dim], latent], dim=-1)
                mu = deploy_a2c.mu_act(deploy_a2c.mu(deploy_a2c.actor_mlp(merged)))
                oracle_merged = torch.cat(
                    [xn[:, :deploy_proprio_dim], oracle_latent], dim=-1
                )
                oracle_mu = deploy_a2c.mu_act(
                    deploy_a2c.mu(deploy_a2c.actor_mlp(oracle_merged))
                )
                # Open-loop adapter probe: drive with the ORACLE latent (keeps
                # the trajectory in-distribution) while still recording the
                # adapter's latent/action error. Isolates covariate shift (the
                # adapter's own actions drifting the state OOD) from a static
                # adapter/wiring error. Closed-loop (default) drives with mu.
                actions = torch.clamp(
                    oracle_mu if args.deploy_openloop else mu, -1.0, 1.0
                )
        else:
            if adapter is not None:
                # Legacy euler-bridge gate: overwrite the last euler_dim columns.
                hist = getattr(base_env, "_prop_hist_buf", None)
                obs_t = _obs_tensor(obses)
                if hist is not None and obs_t is not None:
                    with torch.no_grad():
                        obs_t[:, -euler_dim:] = adapter(hist)[:, :euler_dim]
            actions = player.get_action(obses, is_deterministic=is_det)

        if args.zero_action:
            actions = torch.zeros_like(actions)

        obses, _, _, _ = player.env_step(player.env, actions)
        ex = base_env.extras
        ep_age += 1

        if args.calibration_output and step >= args.warmup_steps:
            if not hasattr(base_env, "compute_surface_clearance"):
                raise RuntimeError(
                    "--calibration_output requires compute_surface_clearance()"
                )
            if not hasattr(base_env, "_read_contact_forces"):
                raise RuntimeError(
                    "--calibration_output requires _read_contact_forces()"
                )
            with torch.no_grad():
                clearance = base_env.compute_surface_clearance()
                fingertip_force = base_env._read_contact_forces()[0]
                finger_q_all = base_env.allegro.data.joint_pos[
                    :, base_env._finger_joint_ids
                ]
                target_error = (base_env._cur_targets - finger_q_all).abs()
            if clearance.shape != fingertip_force.shape:
                raise RuntimeError(
                    "fingertip clearance/force shapes differ: "
                    f"{tuple(clearance.shape)} vs {tuple(fingertip_force.shape)}"
                )
            calibration_clearance_samples.append(clearance.detach().clone())
            calibration_force_samples.append(fingertip_force.detach().clone())
            calibration_target_error_samples.append(
                target_error.detach().clone()
            )

        if args.action_trace_output and trace_envs:
            def trace_values(name: str) -> list | None:
                value = _safe(ex, name)
                if value is None:
                    return None
                return value[:trace_envs].detach().cpu().tolist()

            finger_q = base_env.allegro.data.joint_pos[
                :trace_envs, base_env._finger_joint_ids
            ]
            action_trace.append(
                {
                    "step": int(step),
                    "actions": actions[:trace_envs].detach().cpu().tolist(),
                    "finger_q": finger_q.detach().cpu().tolist(),
                    "cur_targets": base_env._cur_targets[
                        :trace_envs
                    ].detach().cpu().tolist(),
                    "fwd_vel": trace_values("eval_fwd_vel"),
                    "rev_vel": trace_values("eval_rev_vel"),
                    "binary_gate": trace_values("eval_binary_gate"),
                    "contact_gate": trace_values("eval_contact_gate"),
                    "wrong_surface_force_n": trace_values(
                        "eval_wrong_surface_force"
                    ),
                    "tilt_norm_rad": trace_values("eval_tilt_norm"),
                }
            )

        if step >= args.warmup_steps:
            for k, stat in stats.items():
                t = _safe(ex, k)
                if t is not None:
                    stat.update(t)

            tilt = _safe(ex, "eval_tilt_norm")
            if tilt is not None:
                tilt_samples.append(tilt.detach().clone())
            wrong_surface = _safe(ex, "eval_wrong_surface_force")
            if wrong_surface is not None:
                wrong_surface_samples.append(wrong_surface.detach().clone())
                if args.wrong_surface_trace_n is not None:
                    hot = wrong_surface > args.wrong_surface_trace_n
                    if bool(hot.any()) and len(wrong_surface_events) < 2000:
                        variant = getattr(base_env, "_env_variant_idx", None)
                        for env_i in torch.nonzero(hot).flatten().tolist():
                            if len(wrong_surface_events) >= 2000:
                                break
                            wrong_surface_events.append({
                                "rollout_step": int(step),
                                "env": int(env_i),
                                "wrong_n": float(wrong_surface[env_i]),
                                "tilt_rad": (
                                    float(tilt[env_i]) if tilt is not None else None
                                ),
                                "episode_age": int(ep_age[env_i]),
                                "variant": (
                                    int(variant[env_i]) if variant is not None else None
                                ),
                            })
            if adapter is not None and latent_mode:
                latent_delta_samples.append(
                    (latent - oracle_latent).detach().clone()
                )
                latent_action_error_samples.append(
                    (mu - oracle_mu).abs().mean(dim=-1).detach().clone()
                )

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
            if bool(terminated.any()):
                fall_ages.append(ep_age[terminated].detach().clone())
                if args.wrong_surface_trace_n is not None:
                    for env_i in torch.nonzero(terminated).flatten().tolist():
                        termination_events.append((int(step), int(env_i)))
            if bool(done.any()):
                fell_d = terminated[done].detach().float()
                term_fell.append(fell_d.clone())
                load_b = getattr(base_env, "_env_load_torque", None)
                base_load = float(getattr(base_env, "_base_load_torque", 0.0)) or 1.0
                term_load.append(
                    (load_b[done] / base_load).detach().clone()
                    if load_b is not None
                    else torch.full_like(fell_d, float("nan"))
                )
                place_b = getattr(base_env, "_env_reset_root_pos_noise", None)
                term_place_mm.append(
                    (torch.linalg.norm(place_b[done][:, :2], dim=-1) * 1000.0)
                    .detach()
                    .clone()
                    if place_b is not None
                    else torch.full_like(fell_d, float("nan"))
                )
                gscale = getattr(base_env, "_env_geom_scale", None)
                base_r = float(getattr(base_env.cfg, "screwdriver_handle_radius", 0.032))
                term_diam.append(
                    (gscale[done][:, 0] * base_r * 2000.0).detach().clone()
                    if gscale is not None
                    else torch.full_like(fell_d, float("nan"))
                )
                ep_age[done] = 0
            if bool(done.any()):
                authorized_net = _safe(ex, "eval_net_turns")
                authorized_tot = _safe(ex, "eval_total_turns")
                physical_net = _safe(ex, "eval_raw_net_turns")
                physical_tot = _safe(ex, "eval_raw_total_turns")
                # Non-Linker tasks do not expose separate raw counters; for
                # those tasks the public counters already are physical.
                if physical_net is None:
                    physical_net = authorized_net
                    physical_tot = authorized_tot
                if physical_net is not None and authorized_net is not None:
                    ep_net_turns.append(physical_net[done].detach().clone())
                    ep_total_turns.append(physical_tot[done].detach().clone())
                    ep_authorized_net_turns.append(
                        authorized_net[done].detach().clone()
                    )
                    ep_authorized_total_turns.append(
                        authorized_tot[done].detach().clone()
                    )
                    ep_env_ids.append(
                        torch.nonzero(done, as_tuple=False).flatten().detach().clone()
                    )
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
    print(line("AuthorityRew", "eval_contact_authority_reward"))
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
    wrong_surface_summary = None
    if wrong_surface_samples:
        wrong_all = torch.cat(wrong_surface_samples).float()
        wrong_q = torch.quantile(
            wrong_all,
            torch.tensor([0.5, 0.9, 0.99], device=wrong_all.device),
        )
        wrong_surface_summary = {
            "p50_n": float(wrong_q[0].item()),
            "p90_n": float(wrong_q[1].item()),
            "p99_n": float(wrong_q[2].item()),
            "max_n": float(wrong_all.max().item()),
            "fraction_above_0_05_n": float(
                (wrong_all > 0.05).float().mean().item()
            ),
        }
    print("  Contact")
    print(line("OscRatio", "eval_osc_ratio"))
    print(line("ContactGate", "eval_contact_gate"))
    print(line("BinaryGate", "eval_binary_gate"))
    print(line("MotionAuth", "eval_motion_auth"))
    # Force-based (LinkerL20) vs distance/pad-based (Allegro) contact diagnostics.
    force_based = not math.isnan(stats["eval_in_window"].result()[0])
    if force_based:
        print(line("DriveCount", "eval_drive_count", "{:.2f}"))
        print(line("InWindow", "eval_in_window"))
        print(line("ContactForce N", "eval_contact_force", "{:.3f}"))
        print(line("IndexCapForce N", "eval_index_cap_force", "{:.3f}"))
        print(line("IdleCount", "eval_idle_count", "{:.2f}"))
        print(line("WrongSurf N", "eval_wrong_surface_force", "{:.3f}"))
        if wrong_surface_summary is not None:
            print(
                f"      └ wrong p50 {wrong_surface_summary['p50_n']:.3f}  "
                f"p90 {wrong_surface_summary['p90_n']:.3f}  "
                f"p99 {wrong_surface_summary['p99_n']:.3f}  "
                f"max {wrong_surface_summary['max_n']:.3f} N  |  "
                f"{100.0 * wrong_surface_summary['fraction_above_0_05_n']:.1f}% > 0.05 N"
            )
        print(line("MaxJointDev rad", "eval_max_joint_dev", "{:.3f}"))
    else:
        print(line("MotionGate", "eval_motion_gate"))
        print(line("PadGate", "eval_pad_gate"))
        print(line("PadCos", "eval_pad_cos"))
        print(line("ContactCount", "eval_contact_count"))
        print(line("AvgContactSpd", "eval_avg_contact_speed"))
        print(line("MinTipDist (m)", "eval_min_tip_dist"))

    def json_number(value: float) -> float | None:
        value = float(value)
        return value if math.isfinite(value) else None

    report = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "deploy_eval": bool(args.deploy_eval),
        "zero_action_probe": bool(args.zero_action),
        "seed": int(args.seed),
        "num_envs": int(base_env.num_envs),
        "steps": int(num_steps),
        "domain_rand_enabled": bool(getattr(base_env.cfg.domain_rand, "enabled", False)),
        "geometry_randomized": bool(
            getattr(base_env.cfg.domain_rand, "randomize_geometry", False)
        ),
        "fixed_start": bool(args.fixed_start),
        "root_pos_bias_mm": [float(value) for value in args.root_pos_bias_mm],
        "root_rpy_bias_deg": [float(value) for value in args.root_rpy_bias_deg],
        "success_turns": float(args.success_turns),
        "step_metrics": {},
    }
    for key, stat in stats.items():
        mean, std = stat.result()
        report["step_metrics"][key] = {
            "mean": json_number(mean),
            "std": json_number(std),
        }
    if tilt_samples:
        report["tilt_distribution"] = {
            "p50": float(q[0].item()),
            "p90": float(q[1].item()),
            "p99": float(q[2].item()),
            "fraction_below_0_10_rad": float(upright_frac),
        }
    if wrong_surface_summary is not None:
        report["wrong_surface_distribution"] = wrong_surface_summary
    if fall_ages:
        ages_all = torch.cat(fall_ages).float()
        report["fall_age_steps"] = {
            "count": int(ages_all.numel()),
            "p10": float(ages_all.quantile(0.10)),
            "p50": float(ages_all.quantile(0.50)),
            "p90": float(ages_all.quantile(0.90)),
            "fraction_within_20_steps": float((ages_all <= 20).float().mean()),
            "fraction_within_reset_mute_window": float(
                (ages_all <= 15).float().mean()
            ),
        }
        print(
            f"\n  Fall timing: n={int(ages_all.numel())}  "
            f"p50 {float(ages_all.quantile(0.5)):.0f} steps  |  "
            f"{100.0 * float((ages_all <= 15).float().mean()):.1f}% within the "
            f"15-step reset action hold/ramp window"
        )
    if term_fell and sum(t.numel() for t in term_fell) > 30:
        fell_all = torch.cat(term_fell)
        load_all = torch.cat(term_load)
        place_all = torch.cat(term_place_mm)

        def _tercile_fall(name: str, values: torch.Tensor, unit: str) -> dict:
            finite = torch.isfinite(values)
            v = values[finite]
            f = fell_all[finite]
            if v.numel() < 30 or float(v.max() - v.min()) < 1e-9:
                return {"available": False}
            lo, hi = v.quantile(1 / 3.0), v.quantile(2 / 3.0)
            bands = {
                "low": f[v <= lo],
                "mid": f[(v > lo) & (v <= hi)],
                "high": f[v > hi],
            }
            out = {
                "available": True,
                "unit": unit,
                "edges": [float(lo), float(hi)],
            }
            for b, ff in bands.items():
                out[b] = {
                    "n": int(ff.numel()),
                    "fall_rate": float(ff.mean()) if ff.numel() else None,
                }
            print(
                f"  Fall by {name} tercile [{unit}]: "
                f"low(≤{float(lo):.2f}) {100 * float(bands['low'].mean()):.1f}%  "
                f"mid {100 * float(bands['mid'].mean()):.1f}%  "
                f"high(>{float(hi):.2f}) {100 * float(bands['high'].mean()):.1f}%"
            )
            return out

        print(
            f"\n  Fall DR-corner cross-tab (n={fell_all.numel()} episodes, "
            f"overall {100 * float(fell_all.mean()):.1f}%):"
        )
        crosstab = {
            "n_episodes": int(fell_all.numel()),
            "overall_fall_rate": float(fell_all.mean()),
            "by_load_torque_scale": _tercile_fall("load-torque×", load_all, "×base"),
            "by_placement_noise": _tercile_fall("placement", place_all, "mm"),
        }
        # High-load ∩ high-placement corner vs the rest.
        lf, pf = torch.isfinite(load_all), torch.isfinite(place_all)
        both = lf & pf
        if int(both.sum()) > 30:
            lv, pv = load_all[both], place_all[both]
            fv = fell_all[both]
            corner = (lv > lv.quantile(2 / 3.0)) & (pv > pv.quantile(2 / 3.0))
            if int(corner.sum()) > 5:
                crosstab["high_load_high_placement_corner"] = {
                    "n": int(corner.sum()),
                    "fall_rate": float(fv[corner].mean()),
                    "rest_fall_rate": float(fv[~corner].mean()),
                }
                print(
                    f"  High-load∩high-placement corner: "
                    f"{100 * float(fv[corner].mean()):.1f}% "
                    f"(n={int(corner.sum())})  vs rest "
                    f"{100 * float(fv[~corner].mean()):.1f}%"
                )
        report["fall_dr_crosstab"] = crosstab
    if args.wrong_surface_trace_n is not None:
        term_by_env: dict[int, list[int]] = {}
        for t_step, env_i in termination_events:
            term_by_env.setdefault(env_i, []).append(t_step)
        near_term = 0
        near_reset = 0
        for ev in wrong_surface_events:
            ev["near_termination"] = any(
                0 <= t - ev["rollout_step"] <= 5
                for t in term_by_env.get(ev["env"], [])
            )
            ev["near_reset"] = (
                ev["episode_age"] is not None and ev["episode_age"] <= 15
            )
            near_term += int(ev["near_termination"])
            near_reset += int(ev["near_reset"])
        n_events = len(wrong_surface_events)
        report["wrong_surface_trace"] = {
            "threshold_n": float(args.wrong_surface_trace_n),
            "event_count": n_events,
            "near_termination_fraction": (
                near_term / n_events if n_events else None
            ),
            "near_reset_fraction": near_reset / n_events if n_events else None,
            "events": sorted(
                wrong_surface_events, key=lambda e: -e["wrong_n"]
            )[:200],
        }
        print(
            f"\n  Wrong-surface trace (> {args.wrong_surface_trace_n:.1f} N): "
            f"{n_events} events  |  near-termination "
            f"{near_term}/{n_events}  |  near-reset {near_reset}/{n_events}"
        )
    if latent_delta_samples:
        delta = torch.cat(latent_delta_samples, dim=0).float()
        action_error = torch.cat(latent_action_error_samples).float()

        def latent_row(selected: torch.Tensor) -> dict:
            chosen = delta[selected]
            return {
                "count": int(selected.sum().item()),
                "latent_mse": float(chosen.square().mean().item()),
                "latent_mae": float(chosen.abs().mean().item()),
                "action_mae": float(action_error[selected].mean().item()),
                "latent_bias": chosen.mean(dim=0).cpu().tolist(),
                "latent_mae_by_channel": chosen.abs().mean(dim=0).cpu().tolist(),
            }

        all_selected = torch.ones(delta.shape[0], dtype=torch.bool, device=device)
        latent_report = latent_row(all_selected)
        latent_report["geometry_buckets"] = []
        bucket_idx = getattr(base_env, "_env_bucket_idx", None)
        geom_scale = getattr(base_env, "_env_geom_scale", None)
        if bucket_idx is not None:
            sample_buckets = bucket_idx.repeat(len(latent_delta_samples))
            for bucket in torch.unique(bucket_idx).sort().values.tolist():
                selected = sample_buckets == int(bucket)
                row = latent_row(selected)
                row["bucket"] = int(bucket)
                if geom_scale is not None:
                    first = torch.nonzero(bucket_idx == int(bucket), as_tuple=False)[0, 0]
                    row["diameter_mm"] = float(64.0 * geom_scale[first, 0].item())
                latent_report["geometry_buckets"].append(row)
        report["latent_diagnostics"] = latent_report

    # Episode-level outcomes, including per-geometry buckets when available.
    if ep_net_turns:
        # Physical progress is never contact-masked, so a policy cannot hide
        # back-drive by deliberately dropping the three-tip gate.
        net = torch.cat(ep_net_turns).float()
        tot = torch.cat(ep_total_turns).float()
        authorized_net = torch.cat(ep_authorized_net_turns).float()
        authorized_tot = torch.cat(ep_authorized_total_turns).float()
        env_ids = torch.cat(ep_env_ids).long()
        failed = torch.cat(ep_failed).bool()
        n_ep = net.numel()
        fall_rate = failed.float().mean().item()
        upright = ~failed
        success = upright & (net >= args.success_turns)
        succ_rate = success.float().mean().item()

        def pct(t: torch.Tensor, quantile: float) -> float:
            return torch.quantile(t, quantile).item() if t.numel() else float("nan")

        episode_summary = {
            "count": int(n_ep),
            "net_turns_mean": float(net.mean().item()),
            "net_turns_std": float(net.std(unbiased=False).item()),
            "net_turns_p10": float(pct(net, 0.1)),
            "net_turns_p50": float(pct(net, 0.5)),
            "net_turns_p90": float(pct(net, 0.9)),
            "total_turns_mean": float(tot.mean().item()),
            "authorized_net_turns_mean": float(authorized_net.mean().item()),
            "authorized_total_turns_mean": float(authorized_tot.mean().item()),
            "fall_rate": float(fall_rate),
            "success_rate": float(succ_rate),
            "geometry_buckets": [],
        }
        report["episodes"] = episode_summary

        print(f"\n  Episodes completed : {n_ep}")
        print(f"  PhysicalNetTurns   : mean {net.mean().item():+.2f}  std {net.std().item():.2f}  "
              f"[p10 {pct(net,0.1):+.2f}  p50 {pct(net,0.5):+.2f}  p90 {pct(net,0.9):+.2f}]")
        print(f"  PhysicalFwdTurns   : mean {tot.mean().item():+.2f}")
        print(
            f"  AuthorizedNetTurns : mean {authorized_net.mean().item():+.2f}"
        )
        print(f"  Fell over (tilt)   : {fall_rate*100:5.1f}%  of episodes")
        print(f"  Success (>={args.success_turns:g} turns, upright): {succ_rate*100:5.1f}%")

        bucket_idx = getattr(base_env, "_env_bucket_idx", None)
        geom_scale = getattr(base_env, "_env_geom_scale", None)
        if bucket_idx is not None and env_ids.numel() > 0:
            episode_buckets = bucket_idx.index_select(0, env_ids)
            base_diameter_m = 2.0 * float(
                getattr(base_env.cfg, "screwdriver_handle_radius", 0.02)
            )
            print("\n  Per geometry bucket")
            for bucket_id in torch.unique(episode_buckets).sort().values.tolist():
                selected = episode_buckets == int(bucket_id)
                bucket_net = net[selected]
                bucket_authorized_net = authorized_net[selected]
                bucket_failed = failed[selected]
                bucket_success = (~bucket_failed) & (
                    bucket_net >= args.success_turns
                )
                first_env = int(env_ids[selected][0].item())
                diameter_scale = (
                    float(geom_scale[first_env, 0].item())
                    if geom_scale is not None
                    else 1.0
                )
                diameter_mm = 1000.0 * base_diameter_m * diameter_scale
                bucket_row = {
                    "bucket": int(bucket_id),
                    "diameter_mm": float(diameter_mm),
                    "count": int(selected.sum().item()),
                    "net_turns_mean": float(bucket_net.mean().item()),
                    "authorized_net_turns_mean": float(
                        bucket_authorized_net.mean().item()
                    ),
                    "fall_rate": float(bucket_failed.float().mean().item()),
                    "success_rate": float(bucket_success.float().mean().item()),
                }
                episode_summary["geometry_buckets"].append(bucket_row)
                print(
                    f"    {diameter_mm:4.0f} mm (bucket {int(bucket_id)}): "
                    f"n={bucket_row['count']:3d}  "
                    f"net={bucket_row['net_turns_mean']:+.2f}  "
                    f"fall={100.0 * bucket_row['fall_rate']:5.1f}%  "
                    f"success={100.0 * bucket_row['success_rate']:5.1f}%"
                )
    else:
        print("\n  No episodes completed within the rollout (increase --steps).")
        report["episodes"] = {
            "count": 0,
            "geometry_buckets": [],
        }

    # Reward-validity / coasting diagnostic.
    n_in = coast["n_in"].item()
    n_out = coast["n_out"].item()
    fwd_in = coast["fwd_in"].item() / n_in if n_in else float("nan")
    fwd_out = coast["fwd_out"].item() / n_out if n_out else float("nan")
    frac_in = n_in / (n_in + n_out) if (n_in + n_out) else float("nan")
    report["coasting"] = {
        "contact_fraction": json_number(frac_in),
        "fwd_velocity_in_contact": json_number(fwd_in),
        "fwd_velocity_without_contact": json_number(fwd_out),
    }
    print(f"\n  Reward-validity (coasting probe)")
    print(f"    Steps with contact gate open (binary): {frac_in*100:5.1f}%")
    print(f"    Mean FwdVel  WHILE in contact        : {fwd_in:+.3f} rad/s")
    if n_out > 0:
        print(f"    Mean FwdVel  WITHOUT contact (coast) : {fwd_out:+.3f} rad/s")
        print("    (High spin without contact => free-spin/coasting, not manipulation.)")
    else:
        print("    Mean FwdVel  WITHOUT contact (coast) : n/a — fingers were in contact")
        print("    every step (no free-spin coasting observed). Confirm with --rot_damping_scale.")

    if args.json_output:
        json_path = Path(args.json_output).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\n  JSON report          : {json_path}")

    if args.calibration_output:
        if not calibration_clearance_samples:
            raise RuntimeError(
                "--calibration_output collected no samples; warmup must be "
                "smaller than the rollout length"
            )

        clearances = torch.cat(calibration_clearance_samples, dim=0).float()
        forces = torch.cat(calibration_force_samples, dim=0).float()
        target_errors = torch.cat(
            calibration_target_error_samples, dim=0
        ).float()
        force_floor = float(base_env.cfg.contact_f_min)
        force_present = forces >= force_floor
        required_contacts = int(
            getattr(base_env.cfg, "role_neutral_min_contact_fingers", 3)
        )
        force_gate = force_present.sum(dim=-1) >= required_contacts

        def distribution(values: torch.Tensor) -> dict:
            values = values.flatten()
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                return {"count": 0}
            probs = torch.tensor(
                [0.01, 0.05, 0.50, 0.95, 0.99],
                device=values.device,
            )
            quantiles = torch.quantile(values, probs)
            return {
                "count": int(values.numel()),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "p01": float(quantiles[0].item()),
                "p05": float(quantiles[1].item()),
                "p50": float(quantiles[2].item()),
                "p95": float(quantiles[3].item()),
                "p99": float(quantiles[4].item()),
                "max": float(values.max().item()),
            }

        def force_window_distribution(values: torch.Tensor) -> dict:
            values = values.flatten()
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                return {"count": 0}
            edges = torch.tensor(
                [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0,
                 6.0, 10.0, 20.0, 50.0, float("inf")],
                dtype=values.dtype,
                device=values.device,
            )
            counts = [
                int(((values >= low) & (values < high)).sum().item())
                for low, high in zip(edges[:-1], edges[1:])
            ]
            return {
                "count": int(values.numel()),
                "window_n": [0.5, 4.0],
                "fraction_in_0_5_to_4_n": float(
                    ((values >= 0.5) & (values <= 4.0)).float().mean().item()
                ),
                "fraction_below_0_5_n": float((values < 0.5).float().mean().item()),
                "fraction_above_4_n": float((values > 4.0).float().mean().item()),
                "histogram_edges_n": edges[:-1].cpu().tolist() + [None],
                "histogram_counts": counts,
            }

        margin_rows = []
        for margin_mm in range(0, 26):
            margin_m = 0.001 * margin_mm
            distance_present = clearances <= margin_m
            distance_gate = (
                distance_present.sum(dim=-1) >= required_contacts
            )
            true_positive = distance_present & force_present
            predicted_positive = distance_present.sum()
            actual_positive = force_present.sum()
            margin_rows.append(
                {
                    "margin_m": margin_m,
                    "per_finger_agreement": float(
                        (distance_present == force_present)
                        .float()
                        .mean()
                        .item()
                    ),
                    "per_finger_precision": float(
                        (
                            true_positive.sum()
                            / predicted_positive.clamp_min(1)
                        ).item()
                    ),
                    "per_finger_recall": float(
                        (
                            true_positive.sum()
                            / actual_positive.clamp_min(1)
                        ).item()
                    ),
                    "gate_agreement": float(
                        (distance_gate == force_gate).float().mean().item()
                    ),
                    "distance_gate_open_fraction": float(
                        distance_gate.float().mean().item()
                    ),
                    "force_gate_open_fraction": float(
                        force_gate.float().mean().item()
                    ),
                }
            )
        best_per_finger = max(
            margin_rows, key=lambda row: row["per_finger_agreement"]
        )
        best_gate = max(margin_rows, key=lambda row: row["gate_agreement"])

        finger_names = list(getattr(base_env, "fingers", ()))
        per_finger_margin_sweeps = {}
        best_margin_by_finger = {}
        for index in range(clearances.shape[1]):
            name = (
                finger_names[index]
                if index < len(finger_names)
                else f"finger_{index}"
            )
            rows = []
            actual = force_present[:, index]
            for half_mm in range(0, 61):
                margin_m = 0.0005 * half_mm
                predicted = clearances[:, index] <= margin_m
                true_positive = (predicted & actual).sum()
                rows.append(
                    {
                        "margin_m": margin_m,
                        "agreement": float(
                            (predicted == actual).float().mean().item()
                        ),
                        "precision": float(
                            (
                                true_positive
                                / predicted.sum().clamp_min(1)
                            ).item()
                        ),
                        "recall": float(
                            (
                                true_positive
                                / actual.sum().clamp_min(1)
                            ).item()
                        ),
                    }
                )
            per_finger_margin_sweeps[name] = rows
            best_margin_by_finger[name] = max(
                rows, key=lambda row: row["agreement"]
            )

        per_finger = {}
        target_joint_offset = 0
        for index in range(clearances.shape[1]):
            name = (
                finger_names[index]
                if index < len(finger_names)
                else f"finger_{index}"
            )
            present = force_present[:, index]
            joint_count = len(base_env.FINGER_JOINT_NAMES[name])
            finger_target_error = target_errors[
                :, target_joint_offset : target_joint_offset + joint_count
            ]
            target_joint_offset += joint_count
            per_finger[name] = {
                "force_n": distribution(forces[:, index]),
                "force_window_distribution": force_window_distribution(
                    forces[:, index]
                ),
                "clearance_m_force_present": distribution(
                    clearances[present, index]
                ),
                "clearance_m_force_absent": distribution(
                    clearances[~present, index]
                ),
                "target_abs_error_rad_force_present": distribution(
                    finger_target_error[present]
                ),
                "target_abs_error_rad_force_absent": distribution(
                    finger_target_error[~present]
                ),
            }

        calibrated_thresholds = torch.tensor(
            [
                best_margin_by_finger[name]["margin_m"]
                for name in finger_names
            ],
            dtype=clearances.dtype,
            device=clearances.device,
        )
        calibrated_distance_present = (
            clearances <= calibrated_thresholds.unsqueeze(0)
        )
        calibrated_distance_gate = (
            calibrated_distance_present.sum(dim=-1) >= required_contacts
        )

        calibration_report = {
            "schema_version": 1,
            "task": args.task,
            "checkpoint": str(
                Path(args.checkpoint).expanduser().resolve()
            ),
            "seed": int(args.seed),
            "num_envs": int(base_env.num_envs),
            "steps": int(num_steps),
            "warmup_steps": int(args.warmup_steps),
            "sampled_step_env_rows": int(clearances.shape[0]),
            "force_contact_floor_n": force_floor,
            "required_contact_fingers": required_contacts,
            "surface_clearance_definition": (
                "tip-axis distance minus per-env handle radius"
            ),
            "per_finger": per_finger,
            "all_fingertip_force_n": distribution(forces),
            "all_fingertip_force_window_distribution": (
                force_window_distribution(forces)
            ),
            "all_clearance_m_force_present": distribution(
                clearances[force_present]
            ),
            "all_clearance_m_force_absent": distribution(
                clearances[~force_present]
            ),
            "target_abs_error_rad": {
                "all": distribution(target_errors),
                "force_gate_open": distribution(
                    target_errors[force_gate]
                ),
                "force_gate_closed": distribution(
                    target_errors[~force_gate]
                ),
            },
            "distance_margin_sweep": margin_rows,
            "per_finger_margin_sweeps": per_finger_margin_sweeps,
            "best_margin_by_finger": best_margin_by_finger,
            "calibrated_per_finger_gate": {
                "agreement": float(
                    (calibrated_distance_gate == force_gate)
                    .float().mean().item()
                ),
            },
            "best_per_finger_agreement": best_per_finger,
            "best_three_finger_gate_agreement": best_gate,
        }
        calibration_path = (
            Path(args.calibration_output).expanduser().resolve()
        )
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(
            json.dumps(calibration_report, indent=2, sort_keys=True) + "\n"
        )
        print(f"  Calibration report   : {calibration_path}")
        print(
            "    best per-finger margin "
            f"{1000.0 * best_per_finger['margin_m']:.1f} mm, "
            f"agreement {100.0 * best_per_finger['per_finger_agreement']:.1f}%"
        )
        print(
            "    best 3-finger-gate margin "
            f"{1000.0 * best_gate['margin_m']:.1f} mm, "
            f"agreement {100.0 * best_gate['gate_agreement']:.1f}%"
        )
        print(
            "    calibrated per-finger gate agreement "
            f"{100.0 * calibration_report['calibrated_per_finger_gate']['agreement']:.1f}%"
        )

    if args.action_trace_output:
        trace_path = Path(args.action_trace_output).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        joint_order = [
            joint
            for finger in getattr(base_env, "fingers", ())
            for joint in base_env.FINGER_JOINT_NAMES[finger]
        ]
        variant_idx = getattr(base_env, "_env_variant_idx", None)
        trace_report = {
            "task": args.task,
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "seed": int(args.seed),
            "deterministic": bool(is_det),
            "domain_rand_enabled": bool(
                getattr(base_env.cfg.domain_rand, "enabled", False)
            ),
            "fixed_start": bool(args.fixed_start),
            "policy_dt_s": float(base_env._policy_dt),
            "action_delta_scale_rad": float(base_env.cfg.action_delta_scale),
            "action_joint_order": joint_order,
            "traced_env_count": trace_envs,
            "variant_ids": (
                variant_idx[:trace_envs].detach().cpu().tolist()
                if isinstance(variant_idx, torch.Tensor)
                else None
            ),
            "steps": action_trace,
        }
        trace_path.write_text(
            json.dumps(trace_report, indent=2, sort_keys=True) + "\n"
        )
        print(f"  Action trace         : {trace_path}")

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
