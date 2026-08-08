#!/usr/bin/env python3
"""Replay one archived Linker L20 top-down target cycle in Isaac.

The cycle is read from a search artifact instead of being copied into this
file.  Waypoint offsets are anchored at the post-reset PhysX target state and
interpolated with the same transition count used by the search.  This makes
the video and gate report an exact commissioning rehearsal for a later
hardware hardcode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CYCLE = (
    ROOT
    / "records/g20_checkpoint_netturn_search_20260806"
    / "option_d_cycle_candidate5_third_dr16_seed20260812.json"
)
DEFAULT_CAMERA_MANIFEST = (
    ROOT
    / "records/g20_og_local_q_candidate_dynamic_validation_20260805"
    / "captures/03_thumb_opposition/sim_frames/render_manifest.json"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora",
)
parser.add_argument("--cycle-json", type=Path, default=DEFAULT_CYCLE)
parser.add_argument(
    "--candidate-source",
    choices=("best", "seed_candidate"),
    default="best",
)
parser.add_argument(
    "--iteration",
    type=int,
    default=None,
    help="Read candidate from iterations_detail[index]; default is top-level best.",
)
parser.add_argument("--cycles", type=int, default=6)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--warmup-cycles", type=int, default=None)
parser.add_argument(
    "--transition-steps",
    type=int,
    default=None,
    help="Override the archived transition count for rate-limited hardware rehearsal.",
)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--initial-hold-s", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=20260813)
parser.add_argument("--domain-rand", action="store_true")
parser.add_argument("--video", type=Path, required=True)
parser.add_argument("--video-width", type=int, default=1280)
parser.add_argument("--video-height", type=int, default=720)
parser.add_argument("--camera-manifest", type=Path, default=DEFAULT_CAMERA_MANIFEST)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(extras: dict, key: str) -> torch.Tensor:
    value = extras[key]
    return value.expand(1) if value.ndim == 0 else value


def _select_candidate(payload: dict) -> dict:
    if args.iteration is None:
        if args.candidate_source != "best":
            raise ValueError("--candidate-source seed_candidate requires --iteration")
        return payload["best"]
    row = payload["iterations_detail"][args.iteration]
    candidate = row[args.candidate_source]
    if candidate is None:
        raise ValueError(
            f"iteration {args.iteration} has no {args.candidate_source}"
        )
    return candidate


def _set_matched_view(base, manifest: dict) -> dict:
    """Apply the archived D435 pose relative to the current hand root."""
    from isaaclab.utils.math import (
        convert_camera_frame_orientation_convention,
        quat_apply,
        quat_mul,
    )
    from isaacsim.core.prims import XFormPrim
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, Usd, UsdGeom
    import omni.kit.commands

    view = manifest["cameras_hand_base"][0]
    relative_pos = torch.tensor(
        [view["pos"]], dtype=torch.float32, device=base.device
    )
    relative_quat_ros = torch.tensor(
        [view["quat"]], dtype=torch.float32, device=base.device
    )
    root_pos = base.allegro.data.root_pos_w[0:1]
    root_quat = base.allegro.data.root_quat_w[0:1]
    world_pos = root_pos + quat_apply(root_quat, relative_pos)
    world_quat_ros = quat_mul(root_quat, relative_quat_ros)
    world_quat_opengl = convert_camera_frame_orientation_convention(
        world_quat_ros, origin="ros", target="opengl"
    )

    viewport = get_active_viewport()
    nav_path = str(base.cfg.viewer.cam_prim_path)
    viewport.camera_path = nav_path
    nav_camera = XFormPrim(nav_path)
    for _ in range(10):
        base.sim.render()

    forward = quat_apply(
        world_quat_ros,
        torch.tensor([[0.0, 0.0, 1.0]], device=base.device),
    )
    target = world_pos + 0.35 * forward
    base.sim.set_camera_view(
        eye=world_pos[0].detach().cpu().tolist(),
        target=target[0].detach().cpu().tolist(),
        camera_prim_path=nav_path,
    )
    base.sim.render()

    nav_prim = viewport.stage.GetPrimAtPath(nav_path)
    usd_camera = UsdGeom.Camera(nav_prim)
    parent_world = usd_camera.ComputeParentToWorldTransform(
        Usd.TimeCode.Default()
    )
    old_local = UsdGeom.Xformable(nav_prim).GetLocalTransformation(
        Usd.TimeCode.Default()
    )
    pos_values = world_pos[0].detach().cpu().tolist()
    quat_values = world_quat_opengl[0].detach().cpu().tolist()
    exact_world = Gf.Matrix4d(1.0)
    exact_world.SetRotate(
        Gf.Quatd(
            float(quat_values[0]),
            Gf.Vec3d(*[float(value) for value in quat_values[1:]]),
        )
    )
    exact_world.SetTranslateOnly(
        Gf.Vec3d(*[float(value) for value in pos_values])
    )
    exact_local = exact_world * parent_world.GetInverse()
    omni.kit.commands.create(
        "TransformPrimCommand",
        path=nav_path,
        new_transform_matrix=exact_local,
        old_transform_matrix=old_local,
        time_code=Usd.TimeCode.Default(),
        usd_context_name=viewport.usd_context_name,
    ).do()
    nav_prim.GetAttribute("omni:kit:centerOfInterest").Set(
        Gf.Vec3d(0.0, 0.0, -0.35)
    )
    for _ in range(5):
        base.sim.render()

    # Read back the navigation transform so the report records what was used.
    actual_pos, actual_quat_opengl = nav_camera.get_world_poses()
    actual_quat_ros = convert_camera_frame_orientation_convention(
        actual_quat_opengl, origin="opengl", target="ros"
    )
    return {
        "source_manifest": str(args.camera_manifest.resolve()),
        "source_camera_hand_base": view,
        "world_pos": actual_pos[0].detach().cpu().tolist(),
        "world_quat_ros_wxyz": actual_quat_ros[0].detach().cpu().tolist(),
    }


def main() -> dict:
    if args.cycles < 1:
        raise ValueError("--cycles must be positive")
    if not 0.0 < args.amplitude <= 1.0:
        raise ValueError("--amplitude must be in (0, 1]")
    if args.settle_steps < 1:
        raise ValueError("--settle-steps must be positive")
    if args.initial_hold_s < 0.0:
        raise ValueError("--initial-hold-s must be non-negative")

    cycle_payload = json.loads(args.cycle_json.read_text(encoding="utf-8"))
    candidate = _select_candidate(cycle_payload)
    camera_manifest = json.loads(
        args.camera_manifest.read_text(encoding="utf-8")
    )
    transition_steps = (
        int(cycle_payload["transition_steps"])
        if args.transition_steps is None
        else int(args.transition_steps)
    )
    if transition_steps < 1:
        raise ValueError("--transition-steps must be positive")
    warmup_cycles = (
        int(cycle_payload["warmup_cycles"])
        if args.warmup_cycles is None
        else int(args.warmup_cycles)
    )
    if warmup_cycles < 0:
        raise ValueError("--warmup-cycles must be non-negative")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.randomize_obj_start = False
    env_cfg.domain_rand.enabled = bool(args.domain_rand)
    env_cfg.reset_action_hold_steps = 0
    env_cfg.reset_action_ramp_steps = 0
    env_cfg.absolute_action_targets = False
    env_cfg.viewer.resolution = (args.video_width, args.video_height)
    env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp"

    assets_cfg = getattr(env_cfg.screwdriver_cfg.spawn, "assets_cfg", None)
    if assets_cfg is not None:
        assets_cfg = list(assets_cfg)
        if len(assets_cfg) == 3:
            env_cfg.screwdriver_cfg.spawn.assets_cfg = [assets_cfg[1]]
            env_cfg.screwdriver_variants_dir = str(
                ROOT / "assets/screwdriver/topdown_variants_fixed64"
            )
        elif len(assets_cfg) != 1:
            raise RuntimeError(
                f"expected one fixed asset or a three-diameter bank, got {len(assets_cfg)}"
            )

    env = None
    writer = None
    try:
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        base = env.unwrapped
        base._log_stage = 2
        env.reset(seed=args.seed)

        action_dim = int(base.cfg.action_space.shape[0])
        joint_order = [
            joint
            for finger in ("index", "middle", "ring", "pinky", "thumb")
            for joint in base.FINGER_JOINT_NAMES[finger]
        ]
        if joint_order != cycle_payload["joint_order"]:
            raise RuntimeError(
                "cycle joint order does not match the Isaac action order"
            )
        selected_joints = cycle_payload["selected_joints"]
        selected_indices = torch.tensor(
            [joint_order.index(name) for name in selected_joints],
            dtype=torch.long,
            device=base.device,
        )
        offsets_selected = torch.tensor(
            candidate["waypoint_offsets_rad"],
            dtype=base._cur_targets.dtype,
            device=base.device,
        )
        if tuple(offsets_selected.shape) != (
            int(cycle_payload["waypoints"]),
            len(selected_joints),
        ):
            raise RuntimeError("archived waypoint shape is inconsistent")
        offsets = torch.zeros(
            offsets_selected.shape[0],
            action_dim,
            dtype=offsets_selected.dtype,
            device=base.device,
        )
        offsets[:, selected_indices] = float(args.amplitude) * offsets_selected

        zero = torch.zeros(1, action_dim, device=base.device)
        for _ in range(args.settle_steps):
            env.step(zero)
        anchor = base._cur_targets[0].detach().clone()
        policy_dt = float(base._policy_dt)
        fps = policy_dt ** -1

        camera_record = _set_matched_view(base, camera_manifest)
        import cv2

        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (args.video_width, args.video_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {args.video}")

        def write_frame() -> None:
            frame = np.asarray(env.render())[:, :, :3]
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        initial_frame_count = round(args.initial_hold_s * fps)
        for _ in range(initial_frame_count):
            write_frame()

        metrics = {
            "physical_net_rad": 0.0,
            "qualified_net_rad": 0.0,
            "force_contact_steps": 0,
            "contact_gate_sum": 0.0,
            "wrong_surface_force_max_n": 0.0,
            "contact_force_max_n": 0.0,
            "tilt_max_rad": 0.0,
            "terminated": False,
            "truncated": False,
        }
        trace: list[dict] = []
        scored_steps = 0

        def step_to(desired: torch.Tensor, phase: str, scoring: bool) -> None:
            nonlocal scored_steps
            actions = (
                (desired.unsqueeze(0) - base._cur_targets)
                / float(base.cfg.action_delta_scale)
            ).clamp(-1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(actions)
            write_frame()
            metrics["terminated"] |= bool(terminated[0].item())
            metrics["truncated"] |= bool(truncated[0].item())
            extras = base.extras
            tip_force, _, _, _ = base._read_contact_forces()
            force_contact = int(
                ((tip_force[0] > 0.10).sum() >= 3).item()
            )
            row = {
                "step": len(trace),
                "phase": phase,
                "scoring": scoring,
                "target_rad": desired.detach().cpu().tolist(),
                "actual_rad": base.allegro.data.joint_pos[
                    0, base._finger_joint_ids
                ].detach().cpu().tolist(),
                "physical_net_rad_per_s": float(
                    (
                        _vector(extras, "eval_fwd_vel")[0]
                        - _vector(extras, "eval_rev_vel")[0]
                    ).item()
                ),
                "contact_gate": float(
                    _vector(extras, "eval_contact_gate")[0].item()
                ),
                "force_contact": force_contact,
                "contact_force_max_n": float(
                    _vector(extras, "eval_contact_force_max")[0].item()
                ),
                "wrong_surface_force_n": float(
                    _vector(extras, "eval_wrong_surface_force")[0].item()
                ),
                "tilt_rad": float(
                    _vector(extras, "eval_tilt_norm")[0].item()
                ),
            }
            trace.append(row)
            if not scoring:
                return
            scored_steps += 1
            physical_rate = row["physical_net_rad_per_s"]
            gate = float(_vector(extras, "eval_binary_gate")[0].item())
            metrics["physical_net_rad"] += physical_rate * policy_dt
            metrics["qualified_net_rad"] += physical_rate * gate * policy_dt
            metrics["force_contact_steps"] += force_contact
            metrics["contact_gate_sum"] += row["contact_gate"]
            metrics["wrong_surface_force_max_n"] = max(
                metrics["wrong_surface_force_max_n"],
                row["wrong_surface_force_n"],
            )
            metrics["contact_force_max_n"] = max(
                metrics["contact_force_max_n"],
                row["contact_force_max_n"],
            )
            metrics["tilt_max_rad"] = max(
                metrics["tilt_max_rad"],
                row["tilt_rad"],
            )

        # Acquire waypoint zero from the settled option-D target.
        for transition in range(1, transition_steps + 1):
            alpha = transition / transition_steps
            step_to(
                anchor + alpha * offsets[0],
                phase="acquire",
                scoring=False,
            )

        waypoint_count = offsets.shape[0]
        for cycle in range(warmup_cycles + args.cycles):
            scoring = cycle >= warmup_cycles
            phase = (
                f"score_cycle_{cycle - warmup_cycles + 1:02d}"
                if scoring
                else f"warmup_cycle_{cycle + 1:02d}"
            )
            for waypoint in range(waypoint_count):
                start = offsets[waypoint]
                end = offsets[(waypoint + 1) % waypoint_count]
                for transition in range(1, transition_steps + 1):
                    alpha = transition / transition_steps
                    desired = anchor + (1.0 - alpha) * start + alpha * end
                    step_to(desired, phase=phase, scoring=scoring)

        duration_s = scored_steps * policy_dt
        metrics["scored_steps"] = scored_steps
        metrics["duration_s"] = duration_s
        metrics["physical_net_rad_per_s"] = (
            metrics["physical_net_rad"] / duration_s
        )
        metrics["physical_net_turns"] = (
            metrics["physical_net_rad"] / (2.0 * math.pi)
        )
        metrics["physical_net_turns_per_60s"] = (
            metrics["physical_net_rad_per_s"] * 60.0 / (2.0 * math.pi)
        )
        metrics["qualified_net_rad_per_s"] = (
            metrics["qualified_net_rad"] / duration_s
        )
        metrics["force_contact_fraction"] = (
            metrics.pop("force_contact_steps") / scored_steps
        )
        metrics["contact_gate_fraction"] = (
            metrics.pop("contact_gate_sum") / scored_steps
        )

        result = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "seed": args.seed,
            "domain_randomization_enabled": bool(args.domain_rand),
            "cycle_json": str(args.cycle_json.resolve()),
            "cycle_json_sha256": _sha256(args.cycle_json),
            "candidate_source": args.candidate_source,
            "candidate_iteration": args.iteration,
            "candidate_id": int(candidate["candidate_id"]),
            "source_candidate_metrics": {
                key: value
                for key, value in candidate.items()
                if key
                not in ("waypoint_offsets_rad", "transition_actions", "buckets")
            },
            "joint_order": joint_order,
            "selected_joints": selected_joints,
            "settled_anchor_rad": anchor.detach().cpu().tolist(),
            "waypoint_offsets_rad": candidate["waypoint_offsets_rad"],
            "waypoints": waypoint_count,
            "transition_steps": transition_steps,
            "warmup_cycles": warmup_cycles,
            "score_cycles": args.cycles,
            "amplitude": float(args.amplitude),
            "policy_dt_s": policy_dt,
            "camera": camera_record,
            "video": str(args.video.resolve()),
            "metrics": metrics,
            "trace": trace,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"metrics": metrics, "video": result["video"]}, indent=2))
        print(f"wrote {args.output}")
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
