#!/usr/bin/env python3
"""Search a safe periodic Linker L20 top-down turning stroke in simulation.

The deployed policy integrates 10 Hz actions into joint-position targets.  That
control law makes broad Gaussian exploration a poor way to discover a gait:
large noise destroys the validated grasp, while small noise converges to a
static contact-holding policy.  This diagnostic searches *closed target-space
cycles* instead.  Each candidate is a short sequence of offsets from the
validated home target; linear transitions are converted back to the exact
delta-action contract used by training and deployment.

Candidates are evaluated in parallel on the 60/64/68 mm geometry bank.  The
objective is contact-authorized shaft progress, with explicit penalties for a
bad worst-diameter result, contact loss, non-fingertip contact, tilt, and episode
termination.  No reward gate or task physics is relaxed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/linker_l20_screwdriver_topdown/pip108_20260722"
    / "periodic_action_cycle_search.json"
)
DEFAULT_JOINTS = (
    "index_mcp_roll",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
    "thumb_cmc_yaw",
    "thumb_mcp",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--population", type=int, default=256)
parser.add_argument("--elite_count", type=int, default=32)
parser.add_argument("--iterations", type=int, default=12)
parser.add_argument("--waypoints", type=int, default=4)
parser.add_argument("--transition_steps", type=int, default=3)
parser.add_argument("--warmup_cycles", type=int, default=1)
parser.add_argument("--score_cycles", type=int, default=5)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--replicates", type=int, default=1)
parser.add_argument("--initial_std_rad", type=float, default=0.10)
parser.add_argument("--minimum_std_rad", type=float, default=0.01)
parser.add_argument("--offset_limit_rad", type=float, default=0.20)
parser.add_argument(
    "--initial_mean_json",
    type=Path,
    default=None,
    help=(
        "Optional checkpoint-derived waypoint prior. The JSON must contain "
        "waypoint_offsets_rad with shape [waypoints, len(joints)]."
    ),
)
parser.add_argument(
    "--initial_mean_iteration",
    type=int,
    default=None,
    help="Use iterations_detail[index].best from --initial_mean_json.",
)
parser.add_argument("--elite_smoothing", type=float, default=0.25)
parser.add_argument("--joints", nargs="+", default=list(DEFAULT_JOINTS))
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--posture_search", type=Path, default=None)
parser.add_argument("--posture_candidate_index", type=int, default=None)
parser.add_argument("--absolute_action_targets", action="store_true")
parser.add_argument("--angle_feedback", action="store_true")
parser.add_argument("--joint_motion_range", type=float, default=None)
parser.add_argument(
    "--fixed_geometry_diameter_mm",
    type=int,
    choices=(64,),
    default=None,
)
parser.add_argument("--domain_rand", action="store_true")
parser.add_argument("--commissioning_dr", action="store_true")
parser.add_argument("--rotate_candidate_assignment", action="store_true")
parser.add_argument("--force_soft_limit_n", type=float, default=20.0)
parser.add_argument("--force_penalty_weight", type=float, default=0.02)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: E402,F401

try:  # noqa: E402
    from isaaclab_tasks.utils import parse_env_cfg
except ImportError:  # pragma: no cover - Isaac Lab compatibility
    try:
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    except ImportError:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


DIAMETERS_MM = (
    (int(args.fixed_geometry_diameter_mm),)
    if args.fixed_geometry_diameter_mm is not None
    else (60, 64, 68)
)


def _validate_args() -> None:
    if args.population < 4:
        raise ValueError("--population must be at least 4")
    if not 1 <= args.elite_count < args.population:
        raise ValueError("--elite_count must be in [1, population)")
    if min(
        args.iterations,
        args.waypoints,
        args.transition_steps,
        args.score_cycles,
        args.settle_steps,
        args.replicates,
    ) <= 0:
        raise ValueError("iteration/trajectory/settle/replicate counts must be positive")
    if args.warmup_cycles < 0:
        raise ValueError("--warmup_cycles must be non-negative")
    if not 0.0 < args.initial_std_rad <= args.offset_limit_rad:
        raise ValueError("--initial_std_rad must be in (0, offset_limit]")
    if not 0.0 < args.minimum_std_rad <= args.initial_std_rad:
        raise ValueError("--minimum_std_rad must be in (0, initial_std]")
    if not 0.0 <= args.elite_smoothing < 1.0:
        raise ValueError("--elite_smoothing must be in [0, 1)")
    if len(set(args.joints)) != len(args.joints):
        raise ValueError("--joints contains duplicates")


def _mean_by_candidate(
    values: torch.Tensor, candidate_ids: torch.Tensor, population: int
) -> torch.Tensor:
    result = torch.zeros(population, device=values.device)
    result.scatter_add_(0, candidate_ids, values)
    counts = torch.bincount(candidate_ids, minlength=population).to(values.dtype)
    return result / counts.clamp_min(1.0)


def _extreme_by_candidate(
    values: torch.Tensor,
    candidate_ids: torch.Tensor,
    population: int,
    *,
    reduce: str,
) -> torch.Tensor:
    fill = torch.inf if reduce == "amin" else -torch.inf
    result = torch.full(
        (population,), fill, dtype=values.dtype, device=values.device
    )
    result.scatter_reduce_(
        0, candidate_ids, values, reduce=reduce, include_self=True
    )
    return result


def _reduce_buckets(
    values: torch.Tensor,
    population: int,
    replicates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = values.view(population, len(DIAMETERS_MM), replicates).mean(dim=2)
    return matrix, matrix.mean(dim=1), matrix.min(dim=1).values


def _candidate_actions(
    offsets: torch.Tensor,
    selected_indices: torch.Tensor,
    action_dim: int,
    delta_scale: float,
) -> torch.Tensor:
    """Convert closed target-offset waypoints into constant delta actions."""
    population, waypoint_count, _ = offsets.shape
    following = torch.roll(offsets, shifts=-1, dims=1)
    delta = (following - offsets) / (
        float(args.transition_steps) * float(delta_scale)
    )
    if bool((delta.abs() > 1.0 + 1.0e-5).any()):
        raise RuntimeError("projected waypoint transition exceeds action bounds")
    selected_actions = delta.clamp(-1.0, 1.0)
    actions = torch.zeros(
        population,
        waypoint_count,
        action_dim,
        dtype=offsets.dtype,
        device=offsets.device,
    )
    actions[:, :, selected_indices] = selected_actions
    return actions


def _serialize_candidate(
    index: int,
    offsets: torch.Tensor,
    actions: torch.Tensor,
    metrics: dict[str, torch.Tensor],
) -> dict:
    by_bucket = []
    for bucket, diameter in enumerate(DIAMETERS_MM):
        by_bucket.append(
            {
                "diameter_mm": diameter,
                "qualified_net_rad_per_s": float(
                    metrics["net_rate_bucket"][index, bucket].item()
                ),
                "qualified_net_turns_per_60s": float(
                    metrics["net_rate_bucket"][index, bucket].item()
                    * 60.0
                    / (2.0 * math.pi)
                ),
                "raw_net_rad_per_s": float(
                    metrics["raw_rate_bucket"][index, bucket].item()
                ),
                "contact_gate_fraction": float(
                    metrics["contact_bucket"][index, bucket].item()
                ),
                "instant_contact_fraction": float(
                    metrics["instant_contact_bucket"][index, bucket].item()
                ),
                "wrong_surface_mean_n": float(
                    metrics["wrong_bucket"][index, bucket].item()
                ),
                "wrong_surface_max_n": float(
                    metrics["wrong_max_bucket"][index, bucket].item()
                ),
                "tilt_max_rad": float(
                    metrics["tilt_max_bucket"][index, bucket].item()
                ),
                "done_fraction": float(
                    metrics["done_bucket"][index, bucket].item()
                ),
            }
        )
    return {
        "candidate_id": int(index),
        "score": float(metrics["score"][index].item()),
        "qualified_net_rad_per_s_mean": float(metrics["net_mean"][index].item()),
        "qualified_net_rad_per_s_min_bucket": float(metrics["net_min"][index].item()),
        "qualified_net_rad_per_s_min_replica": float(
            metrics.get("net_replica_min", metrics["net_min"])[index].item()
        ),
        "contact_gate_fraction_min_replica": float(
            metrics.get("contact_replica_min", metrics["contact_min"])[index].item()
        ),
        "contact_force_max_n_max_replica": float(
            metrics.get("force_replica_max", torch.zeros_like(metrics["net_min"]))[
                index
            ].item()
        ),
        "qualified_net_turns_per_60s_min_bucket": float(
            metrics["net_min"][index].item() * 60.0 / (2.0 * math.pi)
        ),
        "physical_net_rad_per_s_mean": float(
            metrics["raw_mean"][index].item()
        ),
        "physical_net_rad_per_s_min_replica": float(
            metrics["raw_replica_min"][index].item()
        ),
        "physical_net_turns_per_60s_min_replica": float(
            metrics["raw_replica_min"][index].item()
            * 60.0
            / (2.0 * math.pi)
        ),
        "contact_gate_fraction_min_bucket": float(
            metrics["contact_min"][index].item()
        ),
        "wrong_surface_mean_n_max_bucket": float(
            metrics["wrong_mean_max"][index].item()
        ),
        "done_fraction_max_bucket": float(metrics["done_max"][index].item()),
        "waypoint_offsets_rad": offsets[index].detach().cpu().tolist(),
        "transition_actions": actions[index].detach().cpu().tolist(),
        "buckets": by_bucket,
    }


def main() -> dict:
    _validate_args()

    from screwdriver_rl.tasks.linker_l20.screwdriver_rotation_env import (
        LinkerL20ScrewdriverRotationEnv,
    )

    joint_names = [
        joint
        for finger in ("index", "middle", "ring", "pinky", "thumb")
        for joint in LinkerL20ScrewdriverRotationEnv.FINGER_JOINT_NAMES[finger]
    ]
    unknown = sorted(set(args.joints).difference(joint_names))
    if unknown:
        raise ValueError(f"unknown selected joints: {unknown}")

    population = int(args.population)
    env_count = population * len(DIAMETERS_MM) * int(args.replicates)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=env_count)
    env_cfg.seed = args.seed
    env_cfg.randomize_obj_start = False
    env_cfg.domain_rand.enabled = bool(
        args.domain_rand or args.commissioning_dr
    )
    if args.commissioning_dr:
        dr = env_cfg.domain_rand
        dr.contact_friction_range = (0.8, 1.5)
        dr.rotation_damping_range = (0.7, 1.5)
        dr.tilt_damping_range = (0.7, 1.3)
        dr.screwdriver_load_torque_range = (0.8, 2.0)
        dr.reset_root_pos_noise_m = 0.002
        dr.reset_root_z_noise_m = 0.001
        dr.reset_root_tilt_noise_rad = 0.010
        dr.reset_root_yaw_noise_rad = 0.030
        dr.reset_screwdriver_tilt_noise_rad = 0.010
        dr.joint_zero_bias_rad = 0.005
    env_cfg.reset_action_hold_steps = 0
    env_cfg.reset_action_ramp_steps = 0
    env_cfg.absolute_action_targets = bool(args.absolute_action_targets)
    if args.joint_motion_range is not None:
        if not 0.0 < args.joint_motion_range <= 1.0:
            raise ValueError("--joint_motion_range must be in (0, 1]")
        env_cfg.joint_motion_range = float(args.joint_motion_range)
    if args.posture_search is not None:
        posture_data = json.loads(args.posture_search.read_text())
        if "top_candidates" in posture_data:
            rows = posture_data["top_candidates"]
            if args.posture_candidate_index is None:
                posture = rows[0]
            else:
                posture = next(
                    row
                    for row in rows
                    if int(row["candidate_index"]) == args.posture_candidate_index
                )
        else:
            posture = posture_data
        finger_widths = {
            "index": 3, "middle": 3, "ring": 3, "pinky": 3, "thumb": 4
        }
        cursor = 0
        target = {}
        joints = posture["joint_positions_independent"]
        for finger, width in finger_widths.items():
            names = joint_names[cursor : cursor + width]
            target[finger] = tuple(float(joints[name]) for name in names)
            cursor += width
        env_cfg.robot_cfg.init_state.pos = tuple(posture["root_pos_w"])
        env_cfg.robot_cfg.init_state.rot = tuple(posture["root_quat_wxyz"])
        env_cfg.pregrasp_positions = target
        if hasattr(env_cfg, "pregrasp_positions_buckets"):
            reset64 = env_cfg.reset_joint_positions_buckets[1]
            tilt64 = env_cfg.reset_screwdriver_tilt_xy_buckets[1]
            env_cfg.pregrasp_positions_buckets = [target, target, target]
            env_cfg.reset_joint_positions_buckets = [reset64, reset64, reset64]
            env_cfg.pregrasp_root_pos_offsets_buckets = [(0.0, 0.0, 0.0)] * 3
            env_cfg.pregrasp_root_quats_buckets = [
                tuple(posture["root_quat_wxyz"])
            ] * 3
            env_cfg.reset_screwdriver_tilt_xy_buckets = [tilt64, tilt64, tilt64]
    if args.fixed_geometry_diameter_mm is not None:
        assets_cfg = getattr(env_cfg.screwdriver_cfg.spawn, "assets_cfg", None)
        if assets_cfg is not None:
            assets_cfg = list(assets_cfg)
            if len(assets_cfg) != 3:
                raise ValueError("fixed 64 mm search expects the three-asset bank")
            env_cfg.screwdriver_cfg.spawn.assets_cfg = [assets_cfg[1]]
            env_cfg.screwdriver_variants_dir = str(
                REPO_ROOT / "assets/screwdriver/topdown_variants_fixed64"
            )

    env = None
    try:
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        base._log_stage = 2
        device = base.device
        env_ids = torch.arange(env_count, device=device)
        base_candidate_ids = torch.div(
            env_ids,
            len(DIAMETERS_MM) * int(args.replicates),
            rounding_mode="floor",
        )
        expected_bucket = torch.div(
            env_ids.remainder(len(DIAMETERS_MM) * int(args.replicates)),
            int(args.replicates),
            rounding_mode="floor",
        )
        variant_ids = getattr(base, "_env_variant_idx", None)
        if variant_ids is None:
            variant_ids = torch.zeros(
                env_count, device=device, dtype=torch.long
            )
        variant_ids = variant_ids.to(device=device, dtype=torch.long)
        if not torch.equal(variant_ids, expected_bucket):
            raise RuntimeError(
                "geometry variants are not in the expected cyclic 60/64/68 layout"
            )

        action_dim = int(base.cfg.action_space.shape[0])
        if action_dim != len(joint_names):
            raise RuntimeError(
                f"action dim {action_dim} != semantic joint count {len(joint_names)}"
            )
        selected_indices = torch.tensor(
            [joint_names.index(name) for name in args.joints],
            dtype=torch.long,
            device=device,
        )
        policy_dt = float(base._policy_dt)
        delta_scale = float(base.cfg.action_delta_scale)

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        parameter_shape = (
            (3, len(args.joints))
            if args.angle_feedback
            else (args.waypoints, len(args.joints))
        )
        mean = torch.zeros(parameter_shape, device=device)
        initial_mean_source = None
        if args.initial_mean_json is not None:
            initial_mean_source = json.loads(args.initial_mean_json.read_text())
            if args.initial_mean_iteration is not None:
                initial_mean_source = initial_mean_source["iterations_detail"][
                    args.initial_mean_iteration
                ]["best"]
            elif "waypoint_offsets_rad" not in initial_mean_source:
                initial_mean_source = initial_mean_source["best"]
            loaded_mean = torch.tensor(
                initial_mean_source["waypoint_offsets_rad"],
                dtype=mean.dtype,
                device=device,
            )
            if tuple(loaded_mean.shape) != tuple(parameter_shape):
                raise ValueError(
                    "--initial_mean_json waypoint shape "
                    f"{tuple(loaded_mean.shape)} != expected {tuple(parameter_shape)}"
                )
            mean.copy_(
                loaded_mean.clamp(
                    -float(args.offset_limit_rad),
                    float(args.offset_limit_rad),
                )
            )
        std = torch.full(
            parameter_shape, float(args.initial_std_rad), device=device
        )
        best_ever: dict | None = None
        iteration_rows = []

        for iteration in range(args.iterations):
            # Rotate each candidate across persistent environment/physics slots.
            # Without this, CEM overfits the same small DR sample every generation.
            if args.rotate_candidate_assignment:
                if len(DIAMETERS_MM) != 1:
                    raise ValueError(
                        "--rotate_candidate_assignment currently requires fixed geometry"
                    )
                candidate_ids = (
                    base_candidate_ids + iteration
                ).remainder(population)
            else:
                candidate_ids = base_candidate_ids
            noise = torch.randn(
                (population,) + parameter_shape,
                generator=generator,
                device=device,
            )
            # Antithetic pairs reduce ranking variance without changing the
            # population budget.  Candidate zero remains an exact static baseline.
            half = population // 2
            if half > 0:
                noise[half : 2 * half] = -noise[:half]
            offsets = (
                mean.unsqueeze(0) + std.unsqueeze(0) * noise
            ).clamp(-float(args.offset_limit_rad), float(args.offset_limit_rad))
            offsets[0] = mean if initial_mean_source is not None else 0.0
            if initial_mean_source is not None:
                reverse_mean = torch.cat(
                    (mean[:1], torch.flip(mean[1:], dims=(0,))), dim=0
                )
                if population > 1:
                    offsets[1] = reverse_mean
                if population > 2:
                    offsets[2] = -mean
                if population > 3:
                    offsets[3] = -reverse_mean
                if population > 4:
                    offsets[4] = 0.0
            if args.absolute_action_targets:
                actions = torch.zeros(
                    population, args.waypoints, action_dim,
                    dtype=offsets.dtype, device=device,
                )
                selected_range = base._joint_range[0].index_select(
                    0, selected_indices
                )
                if not args.angle_feedback:
                    actions[:, :, selected_indices] = (
                        offsets / selected_range
                    ).clamp(-1.0, 1.0)
            else:
                # Preserve exact closed-cycle semantics for accumulated deltas.
                following = torch.roll(offsets, shifts=-1, dims=1)
                largest_transition = (following - offsets).abs().amax(dim=(1, 2))
                largest_acquisition = offsets[:, 0].abs().amax(dim=1)
                largest_delta = torch.maximum(largest_transition, largest_acquisition)
                reachable_delta = float(args.transition_steps) * delta_scale
                projection = torch.minimum(
                    torch.ones_like(largest_delta),
                    reachable_delta / largest_delta.clamp_min(1.0e-9),
                )
                offsets = offsets * projection[:, None, None]
                actions = _candidate_actions(
                    offsets, selected_indices, action_dim, delta_scale
                )

            env.reset(seed=args.seed + iteration)
            zero_actions = torch.zeros(env_count, action_dim, device=device)
            for _ in range(args.settle_steps):
                env.step(zero_actions)

            # Move from home to waypoint 0.  The closed-cycle transition tensor
            # starts with last->first, so initial acquisition is computed
            # separately and excluded from scoring.
            if args.angle_feedback:
                acquisition_selected = torch.zeros(
                    population, len(args.joints), device=device
                )
            elif args.absolute_action_targets:
                acquisition_selected = actions[:, 0].index_select(
                    1, selected_indices
                )
            else:
                acquisition_selected = (
                    offsets[:, 0]
                    / (float(args.transition_steps) * delta_scale)
                ).clamp(-1.0, 1.0)
            acquisition = torch.zeros_like(actions[:, 0])
            acquisition[:, selected_indices] = acquisition_selected
            acquisition_env = acquisition.index_select(0, candidate_ids)
            done_any = torch.zeros(env_count, dtype=torch.bool, device=device)
            for _ in range(args.transition_steps):
                _, _, terminated, truncated, _ = env.step(acquisition_env)
                done_any |= terminated.to(device=device, dtype=torch.bool)
                done_any |= truncated.to(device=device, dtype=torch.bool)

            scored_steps = (
                int(args.score_cycles)
                * int(args.waypoints)
                * int(args.transition_steps)
            )
            qualified_net = torch.zeros(env_count, device=device)
            raw_net = torch.zeros_like(qualified_net)
            contact_sum = torch.zeros_like(qualified_net)
            force_contact_sum = torch.zeros_like(qualified_net)
            instant_contact_sum = torch.zeros_like(qualified_net)
            wrong_sum = torch.zeros_like(qualified_net)
            wrong_max = torch.zeros_like(qualified_net)
            force_max = torch.zeros_like(qualified_net)
            tilt_max = torch.zeros_like(qualified_net)

            total_cycles = int(args.warmup_cycles) + int(args.score_cycles)
            for cycle in range(total_cycles):
                scoring = cycle >= int(args.warmup_cycles)
                for waypoint in range(args.waypoints):
                    if args.angle_feedback:
                        coeff = offsets.index_select(0, candidate_ids)
                        z = base.screwdriver.data.joint_pos[
                            :, base._screwdriver_z_id
                        ]
                        selected_offset = (
                            coeff[:, 0]
                            + coeff[:, 1] * torch.sin(z).unsqueeze(-1)
                            + coeff[:, 2] * torch.cos(z).unsqueeze(-1)
                        )
                        action_env = torch.zeros(
                            env_count, action_dim, device=device
                        )
                        action_env[:, selected_indices] = (
                            selected_offset / selected_range
                        ).clamp(-1.0, 1.0)
                    else:
                        action_env = actions[:, waypoint].index_select(0, candidate_ids)
                    for _ in range(args.transition_steps):
                        if args.angle_feedback:
                            z = base.screwdriver.data.joint_pos[
                                :, base._screwdriver_z_id
                            ]
                            selected_offset = (
                                coeff[:, 0]
                                + coeff[:, 1] * torch.sin(z).unsqueeze(-1)
                                + coeff[:, 2] * torch.cos(z).unsqueeze(-1)
                            )
                            action_env[:, selected_indices] = (
                                selected_offset / selected_range
                            ).clamp(-1.0, 1.0)
                        _, _, terminated, truncated, _ = env.step(action_env)
                        done_any |= terminated.to(device=device, dtype=torch.bool)
                        done_any |= truncated.to(device=device, dtype=torch.bool)
                        if not scoring:
                            continue
                        extras = base.extras
                        fwd = extras["eval_fwd_vel"].to(device)
                        rev = extras["eval_rev_vel"].to(device)
                        gate = extras["eval_binary_gate"].to(device)
                        qualified_net += (fwd - rev) * gate * policy_dt
                        raw_net += (fwd - rev) * policy_dt
                        contact_sum += extras["eval_contact_gate"].to(device)
                        tip_force, _, _, _ = base._read_contact_forces()
                        force_contact_sum += (
                            (tip_force > 0.10).sum(dim=-1) >= 3
                        ).to(dtype=qualified_net.dtype)
                        instant_contact_sum += extras[
                            "eval_instant_contact_gate"
                        ].to(device)
                        wrong = extras["eval_wrong_surface_force"].to(device)
                        wrong_sum += wrong
                        wrong_max = torch.maximum(wrong_max, wrong)
                        force_max = torch.maximum(
                            force_max,
                            extras["eval_contact_force_max"].to(device),
                        )
                        tilt_max = torch.maximum(
                            tilt_max, extras["eval_tilt_norm"].to(device)
                        )

            duration_s = scored_steps * policy_dt
            net_rate_env = qualified_net / duration_s
            raw_rate_env = raw_net / duration_s
            distance_contact_env = contact_sum / float(scored_steps)
            contact_env = force_contact_sum / float(scored_steps)
            instant_contact_env = instant_contact_sum / float(scored_steps)
            wrong_env = wrong_sum / float(scored_steps)

            if args.rotate_candidate_assignment:
                net_mean = _mean_by_candidate(
                    net_rate_env, candidate_ids, population
                )
                raw_mean = _mean_by_candidate(
                    raw_rate_env, candidate_ids, population
                )
                raw_replica_min = _extreme_by_candidate(
                    raw_rate_env, candidate_ids, population, reduce="amin"
                )
                contact_mean = _mean_by_candidate(
                    contact_env, candidate_ids, population
                )
                instant_mean = _mean_by_candidate(
                    instant_contact_env, candidate_ids, population
                )
                wrong_mean = _mean_by_candidate(
                    wrong_env, candidate_ids, population
                )
                done_mean = _mean_by_candidate(
                    done_any.float(), candidate_ids, population
                )
                net_bucket = net_mean[:, None]
                raw_bucket = raw_mean[:, None]
                contact_bucket = contact_mean[:, None]
                instant_contact_bucket = instant_mean[:, None]
                wrong_bucket = wrong_mean[:, None]
                done_bucket = done_mean[:, None]
                net_min = net_mean
                raw_min = raw_mean
                contact_min = contact_mean
                net_replica_min = _extreme_by_candidate(
                    net_rate_env, candidate_ids, population, reduce="amin"
                )
                contact_replica_min = _extreme_by_candidate(
                    contact_env, candidate_ids, population, reduce="amin"
                )
                force_replica_max = _extreme_by_candidate(
                    force_max, candidate_ids, population, reduce="amax"
                )
                wrong_max_bucket = _extreme_by_candidate(
                    wrong_max, candidate_ids, population, reduce="amax"
                )[:, None]
                tilt_max_bucket = _extreme_by_candidate(
                    tilt_max, candidate_ids, population, reduce="amax"
                )[:, None]
            else:
                def bucket_metrics(values: torch.Tensor):
                    matrix = values.view(
                        population, len(DIAMETERS_MM), int(args.replicates)
                    ).mean(dim=2)
                    return matrix, matrix.mean(dim=1), matrix.min(dim=1).values

                net_bucket, net_mean, net_min = bucket_metrics(net_rate_env)
                raw_bucket, raw_mean, raw_min = bucket_metrics(raw_rate_env)
                contact_bucket, _, contact_min = bucket_metrics(contact_env)
                instant_contact_bucket, _, _ = bucket_metrics(instant_contact_env)
                wrong_bucket, _, _ = bucket_metrics(wrong_env)
                wrong_max_bucket = wrong_max.view(
                    population, len(DIAMETERS_MM), int(args.replicates)
                ).max(dim=2).values
                tilt_max_bucket = tilt_max.view(
                    population, len(DIAMETERS_MM), int(args.replicates)
                ).max(dim=2).values
                done_bucket = done_any.float().view(
                    population, len(DIAMETERS_MM), int(args.replicates)
                ).mean(dim=2)
                replica_shape = (
                    population, len(DIAMETERS_MM), int(args.replicates)
                )
                net_replica_min = net_rate_env.view(replica_shape).amin(dim=(1, 2))
                raw_replica_min = raw_rate_env.view(replica_shape).amin(dim=(1, 2))
                contact_replica_min = contact_env.view(replica_shape).amin(dim=(1, 2))
                force_replica_max = force_max.view(replica_shape).amax(dim=(1, 2))
            wrong_mean_max = wrong_bucket.max(dim=1).values
            done_max = done_bucket.max(dim=1).values
            tilt_worst = tilt_max_bucket.max(dim=1).values

            # Net forward speed is primary.  Under DR, the worst replica is
            # explicit so an easy randomized draw cannot hide reversal.
            # to the cross-bucket mean so an easy diameter cannot hide a failure.
            # Safety/contact terms only break ties and reject invalid strokes.
            score = (
                raw_mean
                + raw_replica_min
                - 0.30 * torch.relu(0.80 - contact_replica_min)
                - float(args.force_penalty_weight)
                * torch.relu(force_replica_max - float(args.force_soft_limit_n))
                - 0.05 * wrong_mean_max
                - 0.10 * torch.relu(tilt_worst - 0.5)
                - 2.0 * done_max
            )
            metrics = {
                "score": score,
                "net_rate_bucket": net_bucket,
                "net_mean": net_mean,
                "net_min": net_min,
                "net_replica_min": net_replica_min,
                "raw_mean": raw_mean,
                "raw_min": raw_min,
                "raw_replica_min": raw_replica_min,
                "contact_replica_min": contact_replica_min,
                "force_replica_max": force_replica_max,
                "raw_rate_bucket": raw_bucket,
                "contact_bucket": contact_bucket,
                "contact_min": contact_min,
                "instant_contact_bucket": instant_contact_bucket,
                "wrong_bucket": wrong_bucket,
                "wrong_max_bucket": wrong_max_bucket,
                "wrong_mean_max": wrong_mean_max,
                "tilt_max_bucket": tilt_max_bucket,
                "done_bucket": done_bucket,
                "done_max": done_max,
            }
            ranking = torch.argsort(score, descending=True)
            best_index = int(ranking[0].item())
            best = _serialize_candidate(
                best_index, offsets, actions, metrics
            )
            best["iteration"] = iteration
            if best_ever is None or best["score"] > best_ever["score"]:
                best_ever = best

            elites = offsets.index_select(0, ranking[: args.elite_count])
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False)
            smoothing = float(args.elite_smoothing)
            mean = smoothing * mean + (1.0 - smoothing) * elite_mean
            std = (
                smoothing * std + (1.0 - smoothing) * elite_std
            ).clamp_min(float(args.minimum_std_rad))

            row = {
                "iteration": iteration,
                "best": best,
                "seed_candidate": (
                    _serialize_candidate(0, offsets, actions, metrics)
                    if initial_mean_source is not None
                    else None
                ),
                "distribution_mean_abs_rad": float(mean.abs().mean().item()),
                "distribution_std_mean_rad": float(std.mean().item()),
                "elite_score_mean": float(
                    score.index_select(0, ranking[: args.elite_count]).mean().item()
                ),
            }
            iteration_rows.append(row)
            print(
                f"iter={iteration:02d} "
                f"score={best['score']:+.5f} "
                f"physical_mean={best['physical_net_rad_per_s_mean']:+.4f}rad/s "
                f"physical_min={best['physical_net_rad_per_s_min_replica']:+.4f}rad/s "
                f"turn60_min={best['physical_net_turns_per_60s_min_replica']:+.2f} "
                f"gate_min={best['contact_gate_fraction_min_bucket']:.3f} "
                f"wrong={best['wrong_surface_mean_n_max_bucket']:.3f}N "
                f"done={best['done_fraction_max_bucket']:.3f}"
            )

        assert best_ever is not None
        result = {
            "task": args.task,
            "seed": args.seed,
            "joint_order": joint_names,
            "selected_joints": list(args.joints),
            "geometry_diameters_mm": list(DIAMETERS_MM),
            "domain_randomization_enabled": bool(
                args.domain_rand or args.commissioning_dr
            ),
            "domain_randomization_profile": (
                "commissioning_narrow"
                if args.commissioning_dr
                else "training_full"
                if args.domain_rand
                else "disabled"
            ),
            "force_soft_limit_n": float(args.force_soft_limit_n),
            "force_penalty_weight": float(args.force_penalty_weight),
            "action_semantics": (
                "home-relative bias/sin(z)/cos(z) closed-loop targets"
                if args.angle_feedback
                else "home-relative absolute target waypoints"
                if args.absolute_action_targets
                else "closed target-offset waypoints converted to accumulated "
                "delta actions at the deployment action_delta_scale"
            ),
            "population": population,
            "elite_count": int(args.elite_count),
            "iterations": int(args.iterations),
            "waypoints": int(args.waypoints),
            "transition_steps": int(args.transition_steps),
            "warmup_cycles": int(args.warmup_cycles),
            "score_cycles": int(args.score_cycles),
            "settle_steps": int(args.settle_steps),
            "replicates": int(args.replicates),
            "rotate_candidate_assignment": bool(
                args.rotate_candidate_assignment
            ),
            "policy_dt_s": policy_dt,
            "action_delta_scale_rad": delta_scale,
            "offset_limit_rad": float(args.offset_limit_rad),
            "initial_std_rad": float(args.initial_std_rad),
            "minimum_std_rad": float(args.minimum_std_rad),
            "initial_mean_json": (
                None
                if args.initial_mean_json is None
                else str(args.initial_mean_json.resolve())
            ),
            "ranking_note": (
                "score = mean physical raw net rad/s + worst-replica physical "
                "raw net rad/s, with penalties for three-fingertip force contact "
                "below 0.80, non-tip contact, "
                "tilt above 0.5 rad, and any termination"
            ),
            "best": best_ever,
            "iterations_detail": iteration_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")
        return result
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    ok = False
    try:
        main()
        ok = True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        import os
        import threading

        code = 0 if ok else 1
        watchdog = threading.Timer(60.0, lambda: os._exit(code))
        watchdog.daemon = True
        watchdog.start()
        if code != 0:
            os._exit(code)
        simulation_app.close()
        os._exit(code)
