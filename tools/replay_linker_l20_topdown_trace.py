#!/usr/bin/env python3
"""Replay one exported Linker L20 policy target trace open-loop in Isaac."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-LinkerL20-Screwdriver-Rotation-Topdown")
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--trace-env", type=int, default=0)
parser.add_argument("--start-step", type=int, default=0)
parser.add_argument(
    "--end-step",
    type=int,
    default=None,
    help="Exclusive source-trace end step. Defaults to the trace length.",
)
parser.add_argument(
    "--target-mode",
    choices=("direct", "anchor_delta"),
    default="direct",
    help=(
        "direct replays source targets verbatim; anchor_delta transplants their "
        "motion around the current task's post-reset target."
    ),
)
parser.add_argument(
    "--anchor-mode",
    choices=("settled", "pregrasp"),
    default="settled",
    help="Anchor for --target-mode anchor_delta.",
)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--reverse", action="store_true")
parser.add_argument(
    "--preposition-steps",
    type=int,
    default=0,
    help="Hold the replay anchor for this many policy steps before the trace.",
)
parser.add_argument("--runtime-root", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-phase", type=int, default=-1)
parser.add_argument("--trim-tail-steps", type=int, default=2)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video", type=Path, default=None)
parser.add_argument("--video-width", type=int, default=960)
parser.add_argument("--video-height", type=int, default=720)
parser.add_argument("--camera-eye", type=float, nargs=3, default=None)
parser.add_argument("--camera-target", type=float, nargs=3, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = args.video is not None

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import screwdriver_rl.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _vector(extras: dict, key: str, count: int) -> torch.Tensor | None:
    value = extras.get(key)
    if value is None:
        return None
    return value.expand(count) if value.ndim == 0 else value


def main() -> dict:
    trace = json.loads(args.trace.read_text())
    steps = trace["steps"]
    trace_env = int(args.trace_env)
    traced_count = int(trace["traced_env_count"])
    if not 0 <= trace_env < traced_count:
        raise ValueError(f"--trace-env must be in 0..{traced_count - 1}")
    start_step = int(args.start_step)
    end_step = len(steps) if args.end_step is None else int(args.end_step)
    if not 0 <= start_step < end_step <= len(steps):
        raise ValueError("--start-step/--end-step must select a non-empty trace slice")
    desired_rows = [
        row["cur_targets"][trace_env] for row in steps[start_step:end_step]
    ]
    if not 0 <= args.trim_tail_steps < len(desired_rows):
        raise ValueError("--trim-tail-steps must be in 0..len(trace)-1")
    if args.trim_tail_steps:
        desired_rows = desired_rows[: -args.trim_tail_steps]
    if args.reverse:
        desired_rows = list(reversed(desired_rows))
    joint_order = tuple(trace["action_joint_order"])

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.domain_rand.enabled = False
    env_cfg.randomize_obj_start = False
    env_cfg.reset_action_hold_steps = 0
    env_cfg.reset_action_ramp_steps = 0
    if args.video is not None:
        env_cfg.viewer.resolution = (args.video_width, args.video_height)
        env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp"

    assets_cfg = list(env_cfg.screwdriver_cfg.spawn.assets_cfg)
    if len(assets_cfg) != 3:
        raise RuntimeError("expected the release 60/64/68 mm asset bank")
    env_cfg.screwdriver_cfg.spawn.assets_cfg = [assets_cfg[1]]
    env_cfg.screwdriver_variants_dir = str(
        args.runtime_root.resolve()
        / "assets/screwdriver/topdown_variants_fixed64"
    )

    env = None
    writer = None
    try:
        make_kwargs = {"cfg": env_cfg}
        if args.video is not None:
            make_kwargs["render_mode"] = "rgb_array"
        env = gym.make(args.task, **make_kwargs)
        base = env.unwrapped
        phases = base.cfg.curriculum_phases
        phase_index = int(args.eval_phase)
        if not -len(phases) <= phase_index < len(phases):
            raise ValueError(
                f"--eval-phase must be in {-len(phases)}..{len(phases) - 1}"
            )
        phase = phases[phase_index]
        base._curriculum_phase = phase
        base._global_steps = int(phase.step_start)
        base.cfg.episode_length_s = float(phase.episode_length_s)
        base._log_stage = 2
        env.reset(seed=args.seed)

        resolved_order = tuple(
            joint
            for finger in ("index", "middle", "ring", "pinky", "thumb")
            for joint in base.FINGER_JOINT_NAMES[finger]
        )
        if resolved_order != joint_order:
            raise RuntimeError(
                f"joint order mismatch: trace={joint_order}, env={resolved_order}"
            )

        if args.video is not None:
            import cv2

            if (args.camera_eye is None) != (args.camera_target is None):
                raise ValueError("--camera-eye and --camera-target must be paired")
            if args.camera_eye is not None:
                base.sim.set_camera_view(
                    tuple(args.camera_eye),
                    tuple(args.camera_target),
                    camera_prim_path=base.cfg.viewer.cam_prim_path,
                )
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(args.video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(trace["policy_dt_s"]) ** -1,
                (args.video_width, args.video_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {args.video}")

        start_q = base.allegro.data.joint_pos[
            0, base._finger_joint_ids
        ].detach().cpu().tolist()
        start_targets = base._cur_targets[0].detach().cpu().tolist()
        source_anchor = list(desired_rows[0])
        if not 0.0 < args.amplitude <= 1.5:
            raise ValueError("--amplitude must be in (0, 1.5]")
        if args.preposition_steps < 0:
            raise ValueError("--preposition-steps must be non-negative")
        replay_anchor = (
            start_targets
            if args.anchor_mode == "settled"
            else base._home_targets[0].detach().cpu().tolist()
        )
        if args.target_mode == "anchor_delta":
            desired_rows = [
                [
                    current + args.amplitude * (value - source)
                    for current, value, source in zip(
                        replay_anchor, row, source_anchor
                    )
                ]
                for row in desired_rows
            ]
            if args.preposition_steps:
                desired_rows = [list(replay_anchor)] * args.preposition_steps + desired_rows
        elif args.preposition_steps:
            raise ValueError("--preposition-steps requires --target-mode anchor_delta")
        max_target_error = 0.0
        max_tilt = 0.0
        max_wrong = 0.0
        terminated_any = False
        truncated_any = False
        early_truncated = False
        gate_sum = 0.0
        trace_rows = []
        shaft_start = float(
            base.screwdriver.data.joint_pos[0, base._screwdriver_z_id].item()
        )

        for step_index, desired_list in enumerate(desired_rows):
            desired = torch.tensor(
                [desired_list], dtype=base._cur_targets.dtype, device=base.device
            )
            actions = (
                (desired - base._cur_targets)
                / float(base.cfg.action_delta_scale)
            ).clamp(-1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(actions)
            terminated_any |= bool(terminated[0].item())
            truncated_any |= bool(truncated[0].item())
            early_truncated |= bool(
                truncated[0].item() and step_index < len(desired_rows) - 1
            )
            target_error = float(
                (base._cur_targets - desired).abs().amax().item()
            )
            if not bool(terminated[0].item() or truncated[0].item()):
                max_target_error = max(max_target_error, target_error)
            extras = base.extras
            tilt = float(_vector(extras, "eval_tilt_norm", 1)[0].item())
            wrong = float(
                _vector(extras, "eval_wrong_surface_force", 1)[0].item()
            )
            gate = float(_vector(extras, "eval_binary_gate", 1)[0].item())
            max_tilt = max(max_tilt, tilt)
            max_wrong = max(max_wrong, wrong)
            gate_sum += gate
            if writer is not None:
                import cv2

                frame = np.asarray(env.render())[:, :, :3]
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if step_index % 50 == 0 or step_index == len(desired_rows) - 1:
                trace_rows.append(
                    {
                        "step": step_index,
                        "binary_gate": gate,
                        "tilt_rad": tilt,
                        "wrong_surface_force_n": wrong,
                        "target_error_max_rad": target_error,
                    }
                )

        shaft_end = float(
            base.screwdriver.data.joint_pos[0, base._screwdriver_z_id].item()
        )
        authorized = _vector(base.extras, "eval_net_turns", 1)
        physical = _vector(base.extras, "eval_raw_net_turns", 1)
        result = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_trace": str(args.trace.resolve()),
            "source_trace_env": trace_env,
            "source_start_step": start_step,
            "source_end_step_exclusive": end_step,
            "target_mode": args.target_mode,
            "anchor_mode": args.anchor_mode,
            "amplitude": args.amplitude,
            "reverse": bool(args.reverse),
            "preposition_steps": args.preposition_steps,
            "source_anchor_rad": source_anchor,
            "replay_anchor_rad": replay_anchor,
            "task": args.task,
            "seed": args.seed,
            "steps": len(desired_rows),
            "duration_s": len(desired_rows) * float(trace["policy_dt_s"]),
            "joint_order16": list(joint_order),
            "start_finger_q_rad": start_q,
            "start_targets_rad": start_targets,
            "shaft_joint_start_rad": shaft_start,
            "shaft_joint_end_rad": shaft_end,
            "shaft_joint_delta_rad": shaft_end - shaft_start,
            "shaft_joint_delta_turns": (shaft_end - shaft_start) / (2.0 * math.pi),
            "authorized_net_turns": (
                None if authorized is None else float(authorized[0].item())
            ),
            "physical_net_turns": (
                None if physical is None else float(physical[0].item())
            ),
            "binary_gate_mean": gate_sum / len(desired_rows),
            "max_tilt_rad": max_tilt,
            "max_wrong_surface_force_n": max_wrong,
            "max_target_replay_error_rad": max_target_error,
            "terminated": terminated_any,
            "truncated": truncated_any,
            "early_truncated": early_truncated,
            "trace": trace_rows,
            "video": None if args.video is None else str(args.video.resolve()),
        }
        net_turns = result["physical_net_turns"]
        if net_turns is None:
            net_turns = result["authorized_net_turns"]
        result["pass"] = bool(
            net_turns is not None
            and net_turns >= 0.20
            and result["max_tilt_rad"] <= 1.0
            and result["max_wrong_surface_force_n"] <= 0.10
            and not result["terminated"]
            and not result["early_truncated"]
            and result["max_target_replay_error_rad"] <= 1.0e-6
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    finally:
        if writer is not None:
            writer.release()
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
