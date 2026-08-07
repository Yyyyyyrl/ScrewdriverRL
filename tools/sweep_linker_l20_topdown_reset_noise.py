#!/usr/bin/env python3
"""Measure the top-down reset settle envelope for hand-root XY offsets.

Each amplitude is sampled with the final dynamics/geometry DR active, while
all other placement perturbations are zeroed so this audit isolates XY
absorption.  A sample passes when at least three fingertips satisfy the
force-free distance-contact predicate after the normal compliant settle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-LinkerL20-Screwdriver-Rotation-Topdown")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--samples_per_amplitude", type=int, default=1024)
parser.add_argument(
    "--amplitudes_mm",
    type=float,
    nargs="+",
    default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 5.0, 8.0),
)
parser.add_argument("--pass_threshold", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=73)
parser.add_argument(
    "--output",
    type=Path,
    default=REPO_ROOT
    / "artifacts/linker_l20_screwdriver_topdown/reset_xy_settle_sweep.json",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: E402,F401

try:  # noqa: E402
    from isaaclab_tasks.utils import parse_env_cfg
except ImportError:  # pragma: no cover
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _stats(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().cpu().reshape(-1)
    return {
        "min": float(flat.min()),
        "p05": float(torch.quantile(flat, 0.05)),
        "mean": float(flat.mean()),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
        "std": float(flat.std(unbiased=False)),
    }


def _quat_error_rad(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(actual * expected, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def main() -> dict:
    if args.num_envs < 1 or args.samples_per_amplitude < 1:
        raise ValueError("num_envs and samples_per_amplitude must be positive")
    amplitudes_m = sorted({float(value) / 1000.0 for value in args.amplitudes_mm})
    if not amplitudes_m or amplitudes_m[0] < 0.0:
        raise ValueError("amplitudes_mm must contain non-negative values")
    if not 0.0 < args.pass_threshold <= 1.0:
        raise ValueError("pass_threshold must be in (0, 1]")

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    dr = cfg.domain_rand
    configured_amplitude_m = float(dr.reset_root_pos_noise_m)

    # Exercise the final envelope rather than the P0 curriculum's 0.25 scale.
    for phase in cfg.curriculum_phases:
        phase.dynamics_randomization_scale = 1.0

    # Isolate XY placement absorption. Other reset/calibration perturbations
    # have their own independent range checks in the domain-randomisation audit.
    dr.reset_root_z_noise_m = 0.0
    dr.reset_root_tilt_noise_rad = 0.0
    dr.reset_root_yaw_noise_rad = 0.0
    dr.reset_screwdriver_tilt_noise_rad = 0.0
    dr.joint_zero_bias_rad = 0.0

    env = None
    try:
        env = gym.make(args.task, cfg=cfg)
        base = env.unwrapped
        base._curriculum_phase.dynamics_randomization_scale = 1.0
        results: list[dict] = []

        for amplitude_index, amplitude_m in enumerate(amplitudes_m):
            dr.reset_root_pos_noise_m = amplitude_m
            passed_chunks: list[torch.Tensor] = []
            count_chunks: list[torch.Tensor] = []
            clearance_chunks: list[torch.Tensor] = []
            score_chunks: list[torch.Tensor] = []
            xy_chunks: list[torch.Tensor] = []
            variant_chunks: list[torch.Tensor] = []
            root_pos_error_chunks: list[torch.Tensor] = []
            root_quat_error_chunks: list[torch.Tensor] = []

            collected = 0
            batch_index = 0
            while collected < args.samples_per_amplitude:
                # Paired seeds make amplitude comparisons low-variance: every
                # row sees the same geometry/dynamics draw and the same unit
                # XY sample, scaled only by the requested amplitude.
                env.reset(seed=args.seed + batch_index)
                take = min(base.num_envs, args.samples_per_amplitude - collected)
                clearance, score, present = base._compute_distance_contact()
                contact_count = present[:take].sum(dim=-1)
                passed_chunks.append(contact_count >= 3)
                count_chunks.append(contact_count)
                clearance_chunks.append(clearance[:take].detach().clone())
                score_chunks.append(score[:take].detach().clone())
                xy_chunks.append(
                    base._env_reset_root_pos_noise[:take, :2].detach().clone()
                )
                variant_chunks.append(base._env_variant_idx[:take].detach().clone())

                if amplitude_m == 0.0:
                    expected_root = base.allegro.data.default_root_state[:take, :7].clone()
                    expected_root[:, :3] += base.scene.env_origins[:take]
                    if base._pregrasp_root_offset is not None:
                        expected_root[:, :3] += base._pregrasp_root_offset[
                            base._env_bucket_idx[:take]
                        ]
                    if base._pregrasp_root_quat is not None:
                        expected_root[:, 3:7] = base._pregrasp_root_quat[
                            base._env_bucket_idx[:take]
                        ]
                    actual_root = base.allegro.data.root_state_w[:take, :7]
                    root_pos_error_chunks.append(
                        torch.linalg.norm(
                            actual_root[:, :3] - expected_root[:, :3], dim=-1
                        )
                    )
                    root_quat_error_chunks.append(
                        _quat_error_rad(actual_root[:, 3:7], expected_root[:, 3:7])
                    )

                collected += take
                batch_index += 1

            passed = torch.cat(passed_chunks)
            contact_count = torch.cat(count_chunks)
            clearance = torch.cat(clearance_chunks)
            score = torch.cat(score_chunks)
            xy = torch.cat(xy_chunks)
            variant_idx = torch.cat(variant_chunks)
            pass_rate = float(passed.float().mean())
            within_requested_range = bool(
                (xy.abs() <= amplitude_m + 1.0e-7).all()
            )

            bucket_results = {}
            for bucket in range(base._variant_table.num_variants):
                mask = variant_idx == bucket
                bucket_results[base._variant_table.files[bucket]] = {
                    "samples": int(mask.sum()),
                    "pass_rate": float(passed[mask].float().mean()),
                }

            entry = {
                "amplitude_mm": 1000.0 * amplitude_m,
                "samples": collected,
                "batches": batch_index,
                "at_least_3_contact_pass_rate": pass_rate,
                "meets_95pct_gate": pass_rate >= args.pass_threshold,
                "sampled_xy_within_requested_range": within_requested_range,
                "sampled_xy_m": _stats(xy),
                "contact_count": _stats(contact_count),
                "surface_clearance_m": _stats(clearance),
                "distance_score": _stats(score),
                "per_geometry_variant": bucket_results,
            }
            if amplitude_m == 0.0:
                root_pos_error = torch.cat(root_pos_error_chunks)
                root_quat_error = torch.cat(root_quat_error_chunks)
                entry["zero_noise_legacy_root_regression"] = {
                    "max_position_error_m": float(root_pos_error.max()),
                    "max_orientation_error_rad": float(root_quat_error.max()),
                    "exact_with_tolerance": bool(
                        (root_pos_error <= 1.0e-7).all()
                        and (root_quat_error <= 1.0e-6).all()
                        and (xy == 0.0).all()
                    ),
                }
            results.append(entry)

        passing_nonzero = [
            item["amplitude_mm"]
            for item in results
            if item["amplitude_mm"] > 0.0
            and item["meets_95pct_gate"]
            and item["sampled_xy_within_requested_range"]
        ]
        selected_amplitude_mm = max(passing_nonzero) if passing_nonzero else None
        configured_entry = min(
            results,
            key=lambda item: abs(
                item["amplitude_mm"] - 1000.0 * configured_amplitude_m
            ),
        )
        configured_amplitude_was_swept = (
            abs(
                configured_entry["amplitude_mm"]
                - 1000.0 * configured_amplitude_m
            )
            <= 1.0e-9
        )
        zero_entry = next(
            (item for item in results if item["amplitude_mm"] == 0.0),
            None,
        )
        checks = {
            "all_requested_ranges_respected": all(
                item["sampled_xy_within_requested_range"] for item in results
            ),
            "configured_amplitude_was_swept": configured_amplitude_was_swept,
            "configured_amplitude_meets_contact_gate": bool(
                configured_amplitude_was_swept
                and configured_entry["meets_95pct_gate"]
            ),
            "zero_noise_legacy_root_regression": bool(
                zero_entry is not None
                and zero_entry["zero_noise_legacy_root_regression"][
                    "exact_with_tolerance"
                ]
            ),
        }
        result = {
            "task": args.task,
            "num_envs": base.num_envs,
            "samples_per_amplitude": args.samples_per_amplitude,
            "pass_threshold": args.pass_threshold,
            "configured_amplitude_mm": 1000.0 * configured_amplitude_m,
            "selected_max_passing_amplitude_mm": selected_amplitude_mm,
            "isolated_dimensions": "hand_root_xy",
            "final_phase_dynamics_randomization_scale": 1.0,
            "results": results,
            "checks": checks,
            "sweep_pass": all(checks.values()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.output}")
        return result
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        summary = main()
        exit_code = 0 if summary["sweep_pass"] else 1
    except BaseException:
        traceback.print_exc()
        exit_code = 2
    try:
        simulation_app.close()
    except SystemExit:
        pass
    raise SystemExit(exit_code)
