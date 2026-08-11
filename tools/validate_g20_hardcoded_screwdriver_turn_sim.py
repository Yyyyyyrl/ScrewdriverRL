#!/usr/bin/env python3
"""Replay the exact hardcoded G20 turn sequence in Isaac and report gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback
import statistics

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_g20_hardcoded_screwdriver_turn import (
    ALLOWED_AMPLITUDES,
    JOINTS,
    build_phases,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", default="Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
)
parser.add_argument(
    "--amplitudes", nargs="+", type=float, default=list(ALLOWED_AMPLITUDES)
)
parser.add_argument("--cycles", type=int, default=7)
parser.add_argument("--action-mode", choices=("delta", "absolute"), default="delta")
parser.add_argument("--replicates", type=int, default=1)
parser.add_argument("--domain-rand", action="store_true")
parser.add_argument("--commissioning-dr", action="store_true")
parser.add_argument("--rate-hz", type=float, default=10.0)
parser.add_argument("--preposition-s", type=float, default=0.4)
parser.add_argument("--leg-s", type=float, default=0.4)
parser.add_argument("--hold-s", type=float, default=0.0)
parser.add_argument("--settle-steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=20260806)
parser.add_argument("--video", type=Path, default=None)
parser.add_argument("--video-width", type=int, default=960)
parser.add_argument("--video-height", type=int, default=720)
parser.add_argument("--camera-distance", type=float, default=0.42)
parser.add_argument("--camera-azimuth-deg", type=float, default=-65.0)
parser.add_argument("--camera-elevation-deg", type=float, default=32.0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = args.video is not None

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import screwdriver_rl.tasks
from isaaclab_tasks.utils import parse_env_cfg


def _vector(extras: dict, key: str, count: int) -> torch.Tensor:
    value = extras[key]
    return value.expand(count) if value.ndim == 0 else value


def main() -> dict:
    for amplitude in args.amplitudes:
        if not any(
            abs(amplitude - allowed) < 1.0e-9
            for allowed in ALLOWED_AMPLITUDES
        ):
            raise ValueError(
                f"amplitude {amplitude} must be one of {ALLOWED_AMPLITUDES}"
            )
    if not 1 <= args.cycles <= 8:
        raise ValueError("--cycles must be in 1..8")

    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    env_amplitudes = [
        amplitude
        for amplitude in args.amplitudes
        for _ in range(args.replicates)
    ]
    count = len(env_amplitudes)
    if args.video is not None and count != 1:
        raise ValueError("--video requires exactly one amplitude and one replicate")
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=count)
    if args.video is not None:
        env_cfg.viewer.resolution = (args.video_width, args.video_height)
        env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp"
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
    env_cfg.absolute_action_targets = args.action_mode == "absolute"

    assets_cfg = list(env_cfg.screwdriver_cfg.spawn.assets_cfg)
    if len(assets_cfg) != 3:
        raise RuntimeError("expected the 60/64/68 mm top-down asset bank")
    env_cfg.screwdriver_cfg.spawn.assets_cfg = [assets_cfg[1]]
    env_cfg.screwdriver_variants_dir = str(
        ROOT / "assets/screwdriver/topdown_variants_fixed64"
    )

    env = None
    video_writer = None
    try:
        make_kwargs = {"cfg": env_cfg}
        if args.video is not None:
            make_kwargs["render_mode"] = "rgb_array"
        env = gym.make(args.task, **make_kwargs)
        base = env.unwrapped
        base._log_stage = 2
        env.reset(seed=args.seed)
        zero = torch.zeros(
            count,
            int(base.cfg.action_space.shape[0]),
            device=base.device,
        )
        for _ in range(args.settle_steps):
            env.step(zero)

        joint_names = [
            joint
            for finger in ("index", "middle", "ring", "pinky", "thumb")
            for joint in base.FINGER_JOINT_NAMES[finger]
        ]
        if tuple(joint_names) != JOINTS:
            raise RuntimeError("Isaac joint order differs from hardcoded sequence")

        if args.video is not None:
            import cv2

            points = []
            for asset in (base.allegro, base.screwdriver):
                points.append(
                    asset.data.body_state_w[0, :, :3].detach().cpu().numpy()
                )
            cloud = np.concatenate(points, axis=0)
            target = 0.5 * (cloud.min(axis=0) + cloud.max(axis=0))
            azimuth = math.radians(args.camera_azimuth_deg)
            elevation = math.radians(args.camera_elevation_deg)
            horizontal = args.camera_distance * math.cos(elevation)
            eye = (
                target[0] + horizontal * math.cos(azimuth),
                target[1] + horizontal * math.sin(azimuth),
                target[2] + args.camera_distance * math.sin(elevation),
            )
            base.sim.set_camera_view(
                tuple(float(v) for v in eye),
                tuple(float(v) for v in target),
                camera_prim_path=base.cfg.viewer.cam_prim_path,
            )
            for _ in range(8):
                env.render()
            args.video.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(args.video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.rate_hz,
                (args.video_width, args.video_height),
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {args.video}")
            first_frame = np.asarray(env.render())[:, :, :3]
            for _ in range(round(args.rate_hz)):
                video_writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

        anchors = base._cur_targets.detach().clone()
        static_home = base._home_targets.detach().clone()
        joint_range = base._joint_range.detach().clone()
        plans = [
            build_phases(
                start=anchors[index].detach().cpu().tolist(),
                mode="turn",
                amplitude=float(amplitude),
                cycles=args.cycles,
                rate_hz=args.rate_hz,
                preposition_s=args.preposition_s,
                leg_s=args.leg_s,
                hold_s=args.hold_s,
            )
            for index, amplitude in enumerate(env_amplitudes)
        ]
        lengths = {len(plan) for plan in plans}
        if len(lengths) != 1:
            raise RuntimeError("parallel amplitude plans have different lengths")
        step_count = lengths.pop()

        sums = {
            name: torch.zeros(count, device=base.device)
            for name in (
                "fwd_rad", "rev_rad", "contact", "binary",
                "instant", "motion_auth", "force",
            )
        }
        mins = {
            name: torch.ones(count, device=base.device)
            for name in ("contact", "binary", "instant", "motion_auth")
        }
        maxs = {
            name: torch.zeros(count, device=base.device)
            for name in ("wrong", "force", "tilt", "target_error")
        }
        terminated_any = torch.zeros(
            count, dtype=torch.bool, device=base.device
        )
        truncated_any = torch.zeros(
            count, dtype=torch.bool, device=base.device
        )
        trace: list[dict] = []

        for step in range(step_count):
            desired = torch.tensor(
                [plan[step]["semantic"] for plan in plans],
                dtype=anchors.dtype,
                device=base.device,
            )
            if args.action_mode == "absolute":
                actions = ((desired - static_home) / joint_range).clamp(-1.0, 1.0)
            else:
                actions = (
                    (desired - base._cur_targets)
                    / float(base.cfg.action_delta_scale)
                ).clamp(-1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(actions)
            if video_writer is not None:
                import cv2

                frame = np.asarray(env.render())[:, :, :3]
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            terminated_any |= terminated
            truncated_any |= truncated
            extras = base.extras
            dt = float(base._policy_dt)
            values = {
                "fwd": _vector(extras, "eval_fwd_vel", count),
                "rev": _vector(extras, "eval_rev_vel", count),
                "contact": _vector(extras, "eval_contact_gate", count),
                "binary": _vector(extras, "eval_binary_gate", count),
                "instant": _vector(
                    extras, "eval_instant_contact_gate", count
                ),
                "motion_auth": _vector(extras, "eval_motion_auth", count),
                "force": _vector(extras, "eval_contact_force_max", count),
                "wrong": _vector(
                    extras, "eval_wrong_surface_force", count
                ),
                "tilt": _vector(extras, "eval_tilt_norm", count),
            }
            sums["fwd_rad"] += values["fwd"] * dt
            sums["rev_rad"] += values["rev"] * dt
            for name in ("contact", "binary", "instant", "motion_auth", "force"):
                sums[name] += values[name]
            for name in ("contact", "binary", "instant", "motion_auth"):
                mins[name] = torch.minimum(mins[name], values[name])
            maxs["wrong"] = torch.maximum(maxs["wrong"], values["wrong"])
            maxs["force"] = torch.maximum(maxs["force"], values["force"])
            maxs["tilt"] = torch.maximum(maxs["tilt"], values["tilt"])
            maxs["target_error"] = torch.maximum(
                maxs["target_error"],
                (base._cur_targets - desired).abs().amax(dim=1),
            )
            if plans[0][step]["gate"] or step == step_count - 1:
                trace.append(
                    {
                        "step": step,
                        "phase": plans[0][step]["phase"],
                        "qualified_net_rad": (
                            sums["fwd_rad"] - sums["rev_rad"]
                        ).detach().cpu().tolist(),
                        "contact_gate": values["contact"].detach().cpu().tolist(),
                        "motion_auth": values["motion_auth"].detach().cpu().tolist(),
                        "force_max_n": values["force"].detach().cpu().tolist(),
                        "wrong_surface_force_n": values["wrong"].detach().cpu().tolist(),
                        "tilt_rad": values["tilt"].detach().cpu().tolist(),
                    }
                )

        rows = []
        raw_net_turns = _vector(
            base.extras, "eval_raw_net_turns", count
        )
        duration = step_count * float(base._policy_dt)
        for index, amplitude in enumerate(env_amplitudes):
            net = float(
                (sums["fwd_rad"] - sums["rev_rad"])[index].item()
            )
            row = {
                "amplitude": float(amplitude),
                "cycles": args.cycles,
                "steps": step_count,
                "duration_s": duration,
                "anchor_settled_rad": anchors[index].detach().cpu().tolist(),
                "anchor_vs_static_home_max_abs_rad": float(
                    (anchors[index] - static_home[index]).abs().max().item()
                ),
                "qualified_forward_rad": float(sums["fwd_rad"][index].item()),
                "qualified_reverse_rad": float(sums["rev_rad"][index].item()),
                "qualified_net_rad": net,
                "qualified_net_rad_per_s": net / duration,
                "raw_net_turns": float(raw_net_turns[index].item()),
                "contact_gate_mean": float(
                    (sums["contact"][index] / step_count).item()
                ),
                "contact_gate_min": float(mins["contact"][index].item()),
                "binary_gate_mean": float(
                    (sums["binary"][index] / step_count).item()
                ),
                "instant_contact_mean": float(
                    (sums["instant"][index] / step_count).item()
                ),
                "motion_auth_mean": float(
                    (sums["motion_auth"][index] / step_count).item()
                ),
                "motion_auth_min": float(mins["motion_auth"][index].item()),
                "contact_force_max_n": float(maxs["force"][index].item()),
                "contact_force_mean_n": float(
                    (sums["force"][index] / step_count).item()
                ),
                "wrong_surface_force_max_n": float(maxs["wrong"][index].item()),
                "tilt_max_rad": float(maxs["tilt"][index].item()),
                "absolute_target_replay_error_max_rad": float(
                    maxs["target_error"][index].item()
                ),
                "terminated": bool(terminated_any[index].item()),
                "truncated": bool(truncated_any[index].item()),
            }
            row["pass"] = bool(
                row["qualified_net_rad"] > 0.0
                and row["contact_gate_mean"] >= 0.80
                and row["instant_contact_mean"] >= 0.80
                and row["wrong_surface_force_max_n"] <= 0.10
                and row["tilt_max_rad"] <= 0.50
                and not row["terminated"]
                and not row["truncated"]
                and row["absolute_target_replay_error_max_rad"] <= 1.0e-5
            )
            rows.append(row)

        aggregates = []
        for amplitude in args.amplitudes:
            subset = [
                row for row in rows
                if abs(row["amplitude"] - amplitude) < 1.0e-9
            ]
            nets = [row["qualified_net_rad"] for row in subset]
            aggregates.append(
                {
                    "amplitude": amplitude,
                    "replicates": len(subset),
                    "pass_rate": sum(row["pass"] for row in subset) / len(subset),
                    "qualified_net_rad_min": min(nets),
                    "qualified_net_rad_median": statistics.median(nets),
                    "contact_gate_mean_min": min(
                        row["contact_gate_mean"] for row in subset
                    ),
                    "contact_force_max_n": max(
                        row["contact_force_max_n"] for row in subset
                    ),
                    "wrong_surface_force_max_n": max(
                        row["wrong_surface_force_max_n"] for row in subset
                    ),
                    "tilt_max_rad": max(row["tilt_max_rad"] for row in subset),
                    "termination_count": sum(row["terminated"] for row in subset),
                    "truncation_count": sum(row["truncated"] for row in subset),
                }
            )

        result = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "seed": args.seed,
            "geometry_diameter_mm": 64,
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
            "replicates_per_amplitude": args.replicates,
            "joint_order16": list(JOINTS),
            "sim_target_lower_rad": (
                base._finger_lower[0].detach().cpu().tolist()
            ),
            "sim_target_upper_rad": (
                base._finger_upper[0].detach().cpu().tolist()
            ),
            "action_semantics": (
                f"exact hardware-plan targets via {args.action_mode} action mode, "
                "relative to the zero-tension settled anchor"
            ),
            "pass_thresholds": {
                "qualified_net_rad": "> 0",
                "contact_gate_mean": ">= 0.80",
                "instant_contact_mean": ">= 0.80",
                "wrong_surface_force_max_n": "<= 0.10",
                "tilt_max_rad": "<= 0.50",
                "terminated": False,
                "truncated": False,
                "absolute_target_replay_error_max_rad": "<= 1e-5",
            },
            "all_pass": all(row["pass"] for row in rows),
            "robust_pass_90pct": all(
                row["pass_rate"] >= 0.90 for row in aggregates
            ),
            "aggregates": aggregates,
            "results": rows,
            "gate_trace": trace,
            "video": str(args.video) if args.video is not None else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.output}")
        return result
    finally:
        if video_writer is not None:
            video_writer.release()
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
