#!/usr/bin/env python3
"""Probe contact-preserving action directions for the Linker L20 top-down task.

The policy action is an accumulated joint-target delta.  A zero-mean, low-noise
policy can therefore learn to hold the validated grasp without ever discovering
a useful turning stroke.  This probe applies short, symmetric push/return pulses
from the same reset grasp across every 60/64/68 mm geometry bucket and ranks the
directions by *contact-authorized* shaft progress.

The probe is diagnostic only: it does not relax reward gates, alter the task, or
write a checkpoint.
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
    / "action_direction_probe.json"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--random_pairs", type=int, default=96)
parser.add_argument(
    "--anchor_sweep",
    action="store_true",
    help=(
        "add index_mcp_roll-minus plus every signed auxiliary axis at "
        "0.5x and 1.0x relative weight"
    ),
)
parser.add_argument("--amplitude", type=float, default=0.20)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--pulse_steps", type=int, default=4)
parser.add_argument("--cycles", type=int, default=3)
parser.add_argument("--replicates", type=int, default=1)
parser.add_argument("--top_k", type=int, default=24)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def _candidate_bank(joint_names: list[str]) -> tuple[list[str], torch.Tensor]:
    """Return zero, every signed axis, and paired random drive-hand directions."""
    action_dim = len(joint_names)
    names = ["zero"]
    vectors = [torch.zeros(action_dim, dtype=torch.float32)]

    for index, joint in enumerate(joint_names):
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            vector = torch.zeros(action_dim, dtype=torch.float32)
            vector[index] = sign
            names.append(f"axis:{joint}:{label}")
            vectors.append(vector)

    if args.anchor_sweep:
        anchor_index = joint_names.index("index_mcp_roll")
        for aux_index, joint in enumerate(joint_names):
            if aux_index == anchor_index:
                continue
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                for weight in (0.5, 1.0):
                    vector = torch.zeros(action_dim, dtype=torch.float32)
                    vector[anchor_index] = -1.0
                    vector[aux_index] = sign * weight
                    names.append(
                        "anchor:index_mcp_roll:minus"
                        f"+{joint}:{label}:w{int(100 * weight):03d}"
                    )
                    vectors.append(vector)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    index_joint_count = sum(name.startswith("index_") for name in joint_names)
    for pair in range(args.random_pairs):
        vector = torch.randn(action_dim, generator=generator)
        # The index is the cap stabilizer.  Keep it fixed during the broad search
        # so random exploration tests turning strokes instead of trivially
        # knocking the cap contact away.
        vector[:index_joint_count] = 0.0
        vector /= vector.abs().max().clamp_min(1.0e-6)
        names.extend((f"random:{pair:03d}:plus", f"random:{pair:03d}:minus"))
        vectors.extend((vector, -vector))

    return names, torch.stack(vectors) * float(args.amplitude)


def _mean(tensor: torch.Tensor, ids: torch.Tensor) -> float:
    return float(tensor.index_select(0, ids).mean().item())


def _max(tensor: torch.Tensor, ids: torch.Tensor) -> float:
    return float(tensor.index_select(0, ids).max().item())


def main() -> dict:
    if args.random_pairs < 0:
        raise ValueError("--random_pairs must be non-negative")
    if args.amplitude <= 0.0 or args.amplitude > 1.0:
        raise ValueError("--amplitude must be in (0, 1]")
    if min(args.settle_steps, args.pulse_steps, args.cycles, args.replicates) <= 0:
        raise ValueError("settle/pulse/cycles/replicates must be positive")

    # The semantic action order is the insertion order of FINGER_JOINT_NAMES.
    from screwdriver_rl.tasks.linker_l20.screwdriver_rotation_env import (
        LinkerL20ScrewdriverRotationEnv,
    )

    joint_names = [
        joint
        for finger in ("index", "middle", "ring", "pinky", "thumb")
        for joint in LinkerL20ScrewdriverRotationEnv.FINGER_JOINT_NAMES[finger]
    ]
    candidate_names, candidate_actions_cpu = _candidate_bank(joint_names)
    candidate_count = len(candidate_names)
    bucket_count = 3
    env_count = candidate_count * bucket_count * args.replicates

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=env_count)
    env_cfg.seed = args.seed
    env_cfg.randomize_obj_start = False
    env_cfg.domain_rand.enabled = False
    env_cfg.reset_action_hold_steps = 0
    env_cfg.reset_action_ramp_steps = 0

    env = None
    try:
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        base._log_stage = 2
        env.reset(seed=args.seed)

        device = base.device
        action_dim = int(base.cfg.action_space.shape[0])
        if action_dim != len(joint_names):
            raise RuntimeError(
                f"action dim {action_dim} != semantic joint count {len(joint_names)}"
            )

        env_ids = torch.arange(env_count, device=device)
        candidate_ids = torch.div(
            env_ids, bucket_count * args.replicates, rounding_mode="floor"
        )
        expected_bucket = torch.div(
            env_ids.remainder(bucket_count * args.replicates),
            args.replicates,
            rounding_mode="floor",
        )
        variant_ids = base._env_variant_idx.to(device=device, dtype=torch.long)
        if not torch.equal(variant_ids, expected_bucket):
            raise RuntimeError(
                "geometry variants are not in the expected cyclic 60/64/68 layout"
            )

        candidate_actions = candidate_actions_cpu.to(device=device)
        actions = candidate_actions.index_select(0, candidate_ids)
        zero = torch.zeros_like(actions)

        for _ in range(args.settle_steps):
            env.step(zero)

        policy_dt = float(base._policy_dt)
        total_steps = 2 * args.pulse_steps * args.cycles
        fwd_rad = torch.zeros(env_count, device=device)
        rev_rad = torch.zeros_like(fwd_rad)
        qualified_fwd_rad = torch.zeros_like(fwd_rad)
        qualified_rev_rad = torch.zeros_like(fwd_rad)
        contact_sum = torch.zeros_like(fwd_rad)
        instant_contact_sum = torch.zeros_like(fwd_rad)
        wrong_sum = torch.zeros_like(fwd_rad)
        wrong_max = torch.zeros_like(fwd_rad)
        tilt_max = torch.zeros_like(fwd_rad)
        index_cap_sum = torch.zeros_like(fwd_rad)
        drive_count_sum = torch.zeros_like(fwd_rad)
        done_any = torch.zeros(env_count, dtype=torch.bool, device=device)

        for cycle in range(args.cycles):
            for direction in (1.0, -1.0):
                for _ in range(args.pulse_steps):
                    _, _, terminated, truncated, _ = env.step(actions * direction)
                    done_any |= terminated.to(device=device, dtype=torch.bool)
                    done_any |= truncated.to(device=device, dtype=torch.bool)

                    fwd = base.extras["eval_fwd_vel"].to(device)
                    rev = base.extras["eval_rev_vel"].to(device)
                    gate = base.extras["eval_binary_gate"].to(device)
                    fwd_rad += fwd * policy_dt
                    rev_rad += rev * policy_dt
                    qualified_fwd_rad += fwd * gate * policy_dt
                    qualified_rev_rad += rev * gate * policy_dt
                    contact_sum += base.extras["eval_contact_gate"].to(device)
                    instant_contact_sum += base.extras[
                        "eval_instant_contact_gate"
                    ].to(device)
                    wrong = base.extras["eval_wrong_surface_force"].to(device)
                    wrong_sum += wrong
                    wrong_max = torch.maximum(wrong_max, wrong)
                    tilt_max = torch.maximum(
                        tilt_max, base.extras["eval_tilt_norm"].to(device)
                    )
                    index_cap_sum += base.extras["eval_index_cap_binary"].to(device)
                    drive_count_sum += base.extras["eval_drive_count"].to(device)

        rows = []
        for candidate_id, name in enumerate(candidate_names):
            bucket_rows = []
            for bucket in range(bucket_count):
                ids = torch.nonzero(
                    (candidate_ids == candidate_id) & (variant_ids == bucket),
                    as_tuple=False,
                ).flatten()
                qfwd = _mean(qualified_fwd_rad, ids)
                qrev = _mean(qualified_rev_rad, ids)
                bucket_rows.append(
                    {
                        "bucket": bucket,
                        "diameter_mm": (60, 64, 68)[bucket],
                        "qualified_fwd_rad": qfwd,
                        "qualified_rev_rad": qrev,
                        "qualified_net_rad": qfwd - qrev,
                        "raw_net_rad": _mean(fwd_rad - rev_rad, ids),
                        "contact_gate_fraction": _mean(contact_sum, ids) / total_steps,
                        "instant_contact_fraction": (
                            _mean(instant_contact_sum, ids) / total_steps
                        ),
                        "index_cap_fraction": _mean(index_cap_sum, ids) / total_steps,
                        "drive_count_mean": _mean(drive_count_sum, ids) / total_steps,
                        "wrong_surface_mean_n": _mean(wrong_sum, ids) / total_steps,
                        "wrong_surface_max_n": _max(wrong_max, ids),
                        "tilt_max_rad": _max(tilt_max, ids),
                        "done_fraction": _mean(done_any.float(), ids),
                    }
                )

            mean_net = sum(row["qualified_net_rad"] for row in bucket_rows) / bucket_count
            min_net = min(row["qualified_net_rad"] for row in bucket_rows)
            min_contact = min(row["contact_gate_fraction"] for row in bucket_rows)
            max_wrong_mean = max(row["wrong_surface_mean_n"] for row in bucket_rows)
            max_done = max(row["done_fraction"] for row in bucket_rows)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "name": name,
                    "action": [
                        float(value)
                        for value in candidate_actions_cpu[candidate_id].tolist()
                    ],
                    "qualified_net_rad_mean": mean_net,
                    "qualified_net_rad_min_bucket": min_net,
                    "contact_gate_fraction_min_bucket": min_contact,
                    "wrong_surface_mean_n_max_bucket": max_wrong_mean,
                    "done_fraction_max_bucket": max_done,
                    "buckets": bucket_rows,
                }
            )

        # Every short rollout contains posture/load settling drift, particularly
        # for the 68 mm bucket. Compare each candidate with the matched zero
        # action bucket instead of rewarding inherited drift. Candidate 0 is the
        # deliberately included zero vector.
        zero_buckets = rows[0]["buckets"]
        for row in rows:
            deltas = [
                bucket["qualified_net_rad"] - zero["qualified_net_rad"]
                for bucket, zero in zip(row["buckets"], zero_buckets, strict=True)
            ]
            delta_mean = sum(deltas) / bucket_count
            delta_min = min(deltas)
            row["qualified_net_delta_vs_zero_rad_by_bucket"] = deltas
            row["qualified_net_delta_vs_zero_rad_mean"] = delta_mean
            row["qualified_net_delta_vs_zero_rad_min_bucket"] = delta_min
            # Improvement over matched zero drift is primary. Contact loss,
            # non-tip contact, and any termination make a direction unsuitable
            # as a training bootstrap.
            row["score"] = (
                delta_mean
                + 0.5 * delta_min
                - 0.05
                * max(0.0, 0.5 - row["contact_gate_fraction_min_bucket"])
                - 0.02 * row["wrong_surface_mean_n_max_bucket"]
                - 2.0 * row["done_fraction_max_bucket"]
            )

        ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
        result = {
            "task": args.task,
            "seed": args.seed,
            "joint_order": joint_names,
            "candidate_count": candidate_count,
            "environment_count": env_count,
            "geometry_diameters_mm": [60, 64, 68],
            "domain_randomization_enabled": False,
            "action_semantics": "accumulated target delta; symmetric push then return",
            "amplitude": args.amplitude,
            "settle_steps": args.settle_steps,
            "pulse_steps": args.pulse_steps,
            "cycles": args.cycles,
            "policy_dt_s": policy_dt,
            "ranking_note": (
                "qualified progress uses the hard sustained contact gate and is "
                "measured relative to matched zero-action drift; score penalizes "
                "worst-bucket contact loss, wrong contact, and done"
            ),
            "top_candidates": ranked[: args.top_k],
            "all_candidates": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

        print(
            f"probed {candidate_count} candidates x 3 buckets x "
            f"{args.replicates} replicate(s)"
        )
        for rank, row in enumerate(ranked[: args.top_k], start=1):
            print(
                f"{rank:2d}. {row['name']:<30} "
                f"delta_mean={row['qualified_net_delta_vs_zero_rad_mean']:+.5f} "
                f"delta_min={row['qualified_net_delta_vs_zero_rad_min_bucket']:+.5f} "
                f"gate_min={row['contact_gate_fraction_min_bucket']:.3f} "
                f"wrong_maxmean={row['wrong_surface_mean_n_max_bucket']:.3f} "
                f"done={row['done_fraction_max_bucket']:.3f}"
            )
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
