#!/usr/bin/env python3
"""Isaac Lab physics gate for the 60/64/68 mm Linker L20 top-down task.

This script launches the real simulator, instantiates the registered task,
resets it, allows additional gravity/contact settling, and runs a zero-action
rollout while reading the task's filtered distal and non-fingertip contact
sensors.  It writes a machine-readable summary and exits non-zero on any gate
failure.  Static mesh validation is intentionally a separate prerequisite.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--enable-domain-rand",
    action="store_true",
    help="exercise the task per-reset dynamics randomization during the rollout",
)
parser.add_argument(
    "--dr-only",
    choices=("rotation", "mass", "load", "stiffness", "damping"),
    default=None,
    help="diagnostic: hold every dynamics DR range at nominal except one",
)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--rollout_steps", type=int, default=120)
parser.add_argument(
    "--reset_contact_steps",
    type=int,
    default=None,
    help="override the simulator reset-to-contact ramp length in 60 Hz physics steps",
)
parser.add_argument("--min_contact_force", type=float, default=0.10)
parser.add_argument("--max_contact_force", type=float, default=8.0)
parser.add_argument("--max_wrong_force", type=float, default=0.05)
parser.add_argument("--min_contact_fraction", type=float, default=0.95)
parser.add_argument(
    "--min_training_authorization_fraction",
    type=float,
    default=0.80,
    help=(
        "minimum per-environment fraction of steps with the runtime turn "
        "gate authorized"
    ),
)
parser.add_argument(
    "--max_zero_action_drift_rad_s",
    type=float,
    default=0.005,
    help="maximum absolute per-environment mean raw shaft velocity under zero action",
)
parser.add_argument(
    "--min_drive_fingers",
    type=int,
    default=3,
    help="minimum non-index body-contact roles required by the final task",
)
parser.add_argument("--max_tilt_rad", type=float, default=0.35)
parser.add_argument(
    "--output",
    type=Path,
    default=REPO_ROOT / "artifacts/linker_l20_screwdriver_topdown/physics_validation_summary.json",
)
parser.add_argument(
    "--settled_posture_output",
    type=Path,
    default=REPO_ROOT / "artifacts/linker_l20_screwdriver_topdown/settled_posture.json",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: E402,F401
from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    UrdfGeometry,
    joint_limit_margins,
)
from screwdriver_rl.utils.linker_topdown_contact_gate import (  # noqa: E402
    CRITICAL_ROLE_NAMES,
    FUNCTIONAL_ACTIVE_ROLE_FRACTION,
    FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT,
    FUNCTIONAL_MIN_CRITICAL_FRACTION,
    functional_contact_topology,
)


PIP_JOINT_NAMES = ("index_pip", "middle_pip", "ring_pip", "pinky_pip")

try:  # noqa: E402
    from isaaclab_tasks.utils import parse_env_cfg
except ImportError:  # pragma: no cover - compatibility with older Isaac Lab
    try:
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    except ImportError:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def _dynamics_snapshot(base_env) -> dict:
    """Capture the exact reset-time DR values before any automatic reset."""
    body_id = base_env._handle_body_ids[base_env._handle_base_idx]
    finger_ids = torch.as_tensor(base_env._finger_joint_ids, dtype=torch.long)
    masses = base_env.screwdriver.root_physx_view.get_masses()[:, body_id]
    stiffness = base_env.allegro.root_physx_view.get_dof_stiffnesses()[
        :, finger_ids
    ][:, 0]
    damping = base_env.allegro.root_physx_view.get_dof_dampings()[
        :, finger_ids
    ][:, 0]
    return {
        "rotation_damping": _to_list(base_env._env_rotation_damping),
        "screwdriver_body_mass": _to_list(masses),
        "screwdriver_load_torque": _to_list(base_env._env_load_torque),
        "finger_stiffness": _to_list(stiffness),
        "finger_damping": _to_list(damping),
    }


def _finite_tree(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return True


def _contact_snapshot(base_env) -> dict:
    total, body, cap, wrong = base_env._read_contact_forces()
    if base_env.cfg.role_neutral_fingertip_contact:
        role = total
    else:
        role = torch.stack(
            (cap[:, 0], body[:, 1], body[:, 2], body[:, 3], body[:, 4]),
            dim=-1,
        )
    tilt = base_env.screwdriver.data.joint_pos[:, base_env._screwdriver_euler_ids[:2]]
    joint_q = base_env.allegro.data.joint_pos[:, base_env._finger_joint_ids]
    proximal = None
    if base_env._proximal_sensor is not None:
        net = base_env._proximal_sensor.data.net_forces_w
        if net is not None:
            proximal = torch.linalg.norm(net, dim=-1).detach().clone()
    fwd_velocity = base_env.extras.get("eval_fwd_vel")
    rev_velocity = base_env.extras.get("eval_rev_vel")
    if fwd_velocity is None or rev_velocity is None:
        raw_shaft_velocity = torch.zeros(
            base_env.num_envs, dtype=torch.float32, device=base_env.device
        )
    else:
        raw_shaft_velocity = fwd_velocity - rev_velocity
    zero_extra = torch.zeros(
        base_env.num_envs, dtype=torch.float32, device=base_env.device
    )
    return {
        "total": total.detach().clone(),
        "body": body.detach().clone(),
        "cap": cap.detach().clone(),
        "role": role.detach().clone(),
        "wrong": wrong.detach().clone(),
        "proximal": proximal,
        "tilt": torch.linalg.norm(tilt, dim=-1).detach().clone(),
        "finger_q": joint_q.detach().clone(),
        "raw_shaft_velocity": raw_shaft_velocity.detach().clone(),
        "training_authorization": base_env.extras.get(
            "eval_binary_gate", zero_extra
        ).detach().clone(),
        "instant_training_authorization": base_env.extras.get(
            "eval_instant_contact_gate", zero_extra
        ).detach().clone(),
        "index_cap_binary": base_env.extras.get(
            "eval_index_cap_binary", zero_extra
        ).detach().clone(),
        "drive_count": base_env.extras.get(
            "eval_drive_count", zero_extra
        ).detach().clone(),
    }


def _aggregate(
    samples: list[dict],
    fingers: list[str],
    threshold: float,
    proximal_names: list[str],
    env_ids: torch.Tensor | None = None,
) -> dict:
    total = torch.stack([sample["total"] for sample in samples])
    body = torch.stack([sample["body"] for sample in samples])
    cap = torch.stack([sample["cap"] for sample in samples])
    wrong = torch.stack([sample["wrong"] for sample in samples])
    tilt = torch.stack([sample["tilt"] for sample in samples])
    raw_shaft_velocity = torch.stack(
        [sample["raw_shaft_velocity"] for sample in samples]
    )
    training_authorization = torch.stack(
        [sample["training_authorization"] for sample in samples]
    )
    instant_training_authorization = torch.stack(
        [sample["instant_training_authorization"] for sample in samples]
    )
    index_cap_binary = torch.stack(
        [sample["index_cap_binary"] for sample in samples]
    )
    drive_count = torch.stack([sample["drive_count"] for sample in samples])
    proximal = (
        torch.stack([sample["proximal"] for sample in samples])
        if samples[0]["proximal"] is not None
        else None
    )
    if env_ids is not None:
        total = total[:, env_ids]
        body = body[:, env_ids]
        cap = cap[:, env_ids]
        wrong = wrong[:, env_ids]
        tilt = tilt[:, env_ids]
        raw_shaft_velocity = raw_shaft_velocity[:, env_ids]
        training_authorization = training_authorization[:, env_ids]
        instant_training_authorization = instant_training_authorization[:, env_ids]
        index_cap_binary = index_cap_binary[:, env_ids]
        drive_count = drive_count[:, env_ids]
        if proximal is not None:
            proximal = proximal[:, env_ids]
    role = torch.stack([sample["role"] for sample in samples])
    if env_ids is not None:
        role = role[:, env_ids]
    raw_shaft_velocity_per_environment = raw_shaft_velocity.mean(dim=0)
    training_authorization_per_environment = training_authorization.mean(dim=0)
    return {
        "sample_count": len(samples),
        "environment_count": int(total.shape[1]),
        "fingertip_total_force_mean_n": {
            finger: float(total[:, :, i].mean().item()) for i, finger in enumerate(fingers)
        },
        "fingertip_total_force_min_n": {
            finger: float(total[:, :, i].min().item()) for i, finger in enumerate(fingers)
        },
        "fingertip_total_force_max_n": {
            finger: float(total[:, :, i].max().item()) for i, finger in enumerate(fingers)
        },
        "fingertip_contact_fraction": {
            finger: float((total[:, :, i] >= threshold).float().mean().item())
            for i, finger in enumerate(fingers)
        },
        "role_force_mean_n": {
            finger: float(role[:, :, i].mean().item()) for i, finger in enumerate(fingers)
        },
        "role_force_max_n": {
            finger: float(role[:, :, i].max().item()) for i, finger in enumerate(fingers)
        },
        "role_contact_fraction": {
            finger: float((role[:, :, i] >= threshold).float().mean().item())
            for i, finger in enumerate(fingers)
        },
        "body_force_mean_n": {
            finger: float(body[:, :, i].mean().item()) for i, finger in enumerate(fingers)
        },
        "cap_force_mean_n": {
            finger: float(cap[:, :, i].mean().item()) for i, finger in enumerate(fingers)
        },
        "wrong_surface_force_mean_n": float(wrong.mean().item()),
        "wrong_surface_force_max_n": float(wrong.max().item()),
        "proximal_force_mean_n": (
            {
                name: float(proximal[:, :, i].mean().item())
                for i, name in enumerate(proximal_names)
            }
            if proximal is not None
            else {}
        ),
        "proximal_force_max_n": (
            {
                name: float(proximal[:, :, i].max().item())
                for i, name in enumerate(proximal_names)
            }
            if proximal is not None
            else {}
        ),
        "tilt_mean_rad": float(tilt.mean().item()),
        "tilt_max_rad": float(tilt.max().item()),
        "zero_action_raw_shaft_drift_rad_s_mean": float(
            raw_shaft_velocity.mean().item()
        ),
        "zero_action_raw_shaft_drift_rad_s_min_environment_mean": float(
            raw_shaft_velocity_per_environment.min().item()
        ),
        "zero_action_raw_shaft_drift_rad_s_max_environment_mean": float(
            raw_shaft_velocity_per_environment.max().item()
        ),
        "zero_action_raw_shaft_drift_rad_s_max_abs_environment_mean": float(
            raw_shaft_velocity_per_environment.abs().max().item()
        ),
        "zero_action_raw_shaft_drift_rad_s_per_environment_mean": _to_list(
            raw_shaft_velocity_per_environment
        ),
        "training_authorization_fraction_mean": float(
            training_authorization.mean().item()
        ),
        "training_authorization_fraction_min_environment_mean": float(
            training_authorization_per_environment.min().item()
        ),
        "training_authorization_fraction_per_environment_mean": _to_list(
            training_authorization_per_environment
        ),
        "instant_training_authorization_fraction_mean": float(
            instant_training_authorization.mean().item()
        ),
        "index_cap_authorization_fraction_mean": float(
            index_cap_binary.mean().item()
        ),
        "drive_count_mean": float(drive_count.mean().item()),
    }


def _contact_checks(stats: dict, fingers: list[str]) -> dict[str, bool | int | float]:
    role_force = stats["role_force_mean_n"]
    role_force_max = stats["role_force_max_n"]
    total_force_max = stats["fingertip_total_force_max_n"]
    role_fraction = stats["role_contact_fraction"]
    persistent = {
        finger: role_fraction[finger] >= args.min_contact_fraction
        for finger in fingers
    }
    topology = functional_contact_topology(role_fraction)
    force_valid = {
        finger: (
            role_force[finger] >= args.min_contact_force
            and role_force_max[finger] <= args.max_contact_force
            and total_force_max[finger] <= args.max_contact_force
        )
        for finger in fingers
    }
    active_roles = topology["active_roles"]
    return {
        "index_cap_contact_persistently": persistent["index"],
        "minimum_drive_fingers_contact_persistently": (
            sum(persistent.values()) >= args.min_drive_fingers
        ),
        "functional_contact_topology": bool(topology["pass"]),
        "functional_critical_roles_persistently": bool(
            topology["critical_roles_pass"]
        ),
        "functional_active_role_count": int(topology["active_role_count"]),
        "functional_mean_role_contact_fraction": float(
            topology["mean_role_contact_fraction"]
        ),
        "task_role_contact_persistently": bool(topology["pass"]),
        "task_role_forces_in_range": (
            all(force_valid[finger] for finger in active_roles)
            and all(
                role_force_max[finger] <= args.max_contact_force
                and total_force_max[finger] <= args.max_contact_force
                for finger in fingers
            )
        ),
        # Strict five-role diagnostics remain visible but are not the release gate.
        "all_fingertips_contact_persistently": all(persistent.values()),
        "fingertips_contact_expected_body_or_cap": all(force_valid.values()),
        "fingertip_total_force_below_max": all(
            value <= args.max_contact_force for value in total_force_max.values()
        ),
        "fingertip_wrong_surface_contact_below_threshold": (
            stats["wrong_surface_force_max_n"] <= args.max_wrong_force
        ),
        "non_fingertip_contact_below_threshold": all(
            value <= args.max_wrong_force
            for value in stats["proximal_force_max_n"].values()
        ),
        "screwdriver_remains_upright": stats["tilt_max_rad"] <= args.max_tilt_rad,
    }


def _required_contact_checks_pass(checks: dict[str, bool]) -> bool:
    return all(
        checks[name]
        for name in (
            "task_role_contact_persistently",
            "task_role_forces_in_range",
            "fingertip_total_force_below_max",
            "fingertip_wrong_surface_contact_below_threshold",
            "non_fingertip_contact_below_threshold",
            "screwdriver_remains_upright",
        )
    )


def _settled_posture_for_env(
    base,
    env_id: int,
    final_finger_q: torch.Tensor,
    independent_names: list[str],
    hand_model: UrdfGeometry,
) -> dict:
    final_q = final_finger_q[env_id]
    target_q = base._cur_targets[env_id]
    final_joint_positions = {
        name: float(final_q[i]) for i, name in enumerate(independent_names)
    }
    final_joint_margins = joint_limit_margins(hand_model, final_joint_positions)
    hand_root = base.allegro.data.root_state_w[env_id, :7].detach().cpu()
    env_origin = base.scene.env_origins[env_id].detach().cpu()
    posture = {
        "posture_source": "Isaac zero-action rollout final state",
        "task": args.task,
        "environment_id": env_id,
        "root_pos_w": [float(hand_root[i] - env_origin[i]) for i in range(3)],
        "root_quat_wxyz": [float(value) for value in hand_root[3:7]],
        "joint_positions_independent": final_joint_positions,
        "screwdriver_joint_positions": {
            name: float(base.screwdriver.data.joint_pos[env_id, i].item())
            for i, name in enumerate(base.screwdriver.joint_names)
        },
        "joint_limit_margins_rad": final_joint_margins,
        "minimum_joint_limit_margin_rad": float(min(final_joint_margins.values())),
        "position_target_independent": {
            name: float(target_q[i]) for i, name in enumerate(independent_names)
        },
    }
    if getattr(base, "_env_variant_idx", None) is not None:
        posture["geometry_variant_index"] = int(base._env_variant_idx[env_id].item())
    if getattr(base, "_env_geom_scale", None) is not None:
        posture["geometry_scale_radius_length"] = [
            float(value) for value in base._env_geom_scale[env_id].detach().cpu()
        ]
    return posture


def main() -> dict:
    if args.max_zero_action_drift_rad_s <= 0.0:
        raise ValueError("--max_zero_action_drift_rad_s must be positive")
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.randomize_obj_start = False
    env_cfg.domain_rand.enabled = bool(args.enable_domain_rand)
    if args.dr_only is not None:
        if not args.enable_domain_rand:
            raise ValueError("--dr-only requires --enable-domain-rand")
        range_by_component = {
            "rotation": "rotation_damping_range",
            "mass": "screwdriver_mass_range",
            "load": "screwdriver_load_torque_range",
            "stiffness": "finger_stiffness_range",
            "damping": "finger_damping_range",
        }
        for component, attribute in range_by_component.items():
            if component != args.dr_only:
                setattr(env_cfg.domain_rand, attribute, (1.0, 1.0))
    if args.reset_contact_steps is not None:
        if args.reset_contact_steps <= 0:
            raise ValueError("--reset_contact_steps must be positive")
        env_cfg.reset_contact_steps = int(args.reset_contact_steps)

    env = None
    try:
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        base._log_stage = 2
        observations, _ = env.reset(seed=args.seed)
        reset_snapshot = _contact_snapshot(base)
        dynamics_snapshot = _dynamics_snapshot(base)
        initial_finger_q = reset_snapshot["finger_q"].clone()

        action_dim = int(base.cfg.action_space.shape[0])
        zero = torch.zeros((base.num_envs, action_dim), dtype=torch.float32, device=base.device)
        settle_samples: list[dict] = []
        rollout_samples: list[dict] = []
        terminated_any = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        truncated_any = torch.zeros_like(terminated_any)
        rewards_finite = True
        observations_finite = _finite_tree(observations)

        for _ in range(args.settle_steps):
            observations, reward, terminated, truncated, _ = env.step(zero)
            observations_finite &= _finite_tree(observations)
            rewards_finite &= bool(torch.isfinite(reward).all().item())
            terminated_any |= terminated.to(base.device, dtype=torch.bool)
            truncated_any |= truncated.to(base.device, dtype=torch.bool)
            settle_samples.append(_contact_snapshot(base))

        settled_finger_q = settle_samples[-1]["finger_q"].clone()
        for _ in range(args.rollout_steps):
            observations, reward, terminated, truncated, _ = env.step(zero)
            observations_finite &= _finite_tree(observations)
            rewards_finite &= bool(torch.isfinite(reward).all().item())
            terminated_any |= terminated.to(base.device, dtype=torch.bool)
            truncated_any |= truncated.to(base.device, dtype=torch.bool)
            rollout_samples.append(_contact_snapshot(base))

        fingers = list(base.fingers)
        proximal_names = (
            list(base._proximal_sensor.body_names)
            if base._proximal_sensor is not None
            else []
        )
        settle = _aggregate(
            settle_samples, fingers, args.min_contact_force, proximal_names
        )
        rollout = _aggregate(
            rollout_samples, fingers, args.min_contact_force, proximal_names
        )
        overall_contact_checks = _contact_checks(rollout, fingers)
        no_termination = not bool(terminated_any.any().item()) and not bool(truncated_any.any().item())
        joint_drift = float(
            torch.max(torch.abs(rollout_samples[-1]["finger_q"] - settled_finger_q)).item()
        )

        independent_names = [
            name
            for finger in fingers
            for name in base.FINGER_JOINT_NAMES[finger]
        ]
        hand_model = UrdfGeometry(
            REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
        )
        final_finger_q = rollout_samples[-1]["finger_q"]
        variant_idx = getattr(base, "_env_variant_idx", None)
        if variant_idx is None:
            variant_idx = torch.zeros(
                base.num_envs, dtype=torch.long, device=base.device
            )
        variant_ids = sorted(
            int(value) for value in torch.unique(variant_idx).detach().cpu().tolist()
        )
        geometry_buckets = []
        settled_postures = []
        for variant_id in variant_ids:
            env_ids = torch.nonzero(variant_idx == variant_id, as_tuple=False).flatten()
            representative_env_id = int(env_ids[0].item())
            bucket_settle = _aggregate(
                settle_samples,
                fingers,
                args.min_contact_force,
                proximal_names,
                env_ids,
            )
            bucket_rollout = _aggregate(
                rollout_samples,
                fingers,
                args.min_contact_force,
                proximal_names,
                env_ids,
            )
            bucket_checks = _contact_checks(bucket_rollout, fingers)
            bucket_checks["zero_action_rollout_has_no_done"] = not bool(
                terminated_any[env_ids].any().item()
                or truncated_any[env_ids].any().item()
            )
            bucket_checks["zero_action_raw_shaft_drift_within_limit"] = (
                bucket_rollout[
                    "zero_action_raw_shaft_drift_rad_s_max_abs_environment_mean"
                ]
                <= args.max_zero_action_drift_rad_s
            )
            bucket_checks["training_turn_authorization_persistent"] = (
                bucket_rollout[
                    "training_authorization_fraction_min_environment_mean"
                ]
                >= args.min_training_authorization_fraction
            )
            minimum_joint_margin = math.inf
            minimum_pip_joint_margin = math.inf
            for env_id in env_ids.detach().cpu().tolist():
                positions = {
                    name: float(final_finger_q[env_id, i])
                    for i, name in enumerate(independent_names)
                }
                margins = joint_limit_margins(hand_model, positions)
                minimum_joint_margin = min(
                    minimum_joint_margin, min(margins.values())
                )
                minimum_pip_joint_margin = min(
                    minimum_pip_joint_margin,
                    min(margins[name] for name in PIP_JOINT_NAMES),
                )
            bucket_checks["pip_joint_limit_margin_at_least_0_1_rad"] = (
                minimum_pip_joint_margin >= 0.10
            )
            posture = _settled_posture_for_env(
                base,
                representative_env_id,
                final_finger_q,
                independent_names,
                hand_model,
            )
            settled_postures.append(posture)
            table = getattr(base, "_variant_table", None)
            bucket_info = {
                "variant_index": variant_id,
                "environment_ids": env_ids.detach().cpu().tolist(),
                "environment_count": int(env_ids.numel()),
                "representative_environment_id": representative_env_id,
                "gravity_settling": bucket_settle,
                "zero_action_rollout": bucket_rollout,
                "minimum_joint_limit_margin_rad": float(minimum_joint_margin),
                "minimum_pip_joint_limit_margin_rad": float(
                    minimum_pip_joint_margin
                ),
                "dynamic_joint_margin_scope": list(PIP_JOINT_NAMES),
                "checks": bucket_checks,
                "physics_validation_pass": (
                    _required_contact_checks_pass(bucket_checks)
                    and bucket_checks["zero_action_rollout_has_no_done"]
                    and bucket_checks[
                        "zero_action_raw_shaft_drift_within_limit"
                    ]
                    and bucket_checks[
                        "training_turn_authorization_persistent"
                    ]
                    and bucket_checks["pip_joint_limit_margin_at_least_0_1_rad"]
                ),
            }
            if table is not None:
                bucket_info.update(
                    {
                        "asset_file": table.files[variant_id],
                        "diameter_m": float(2.0 * table.radius[variant_id]),
                        "length_m": float(table.length[variant_id]),
                        "diameter_scale": float(table.diameter_scale[variant_id]),
                        "length_scale": float(table.length_scale[variant_id]),
                    }
                )
            geometry_buckets.append(bucket_info)

        settled_posture = settled_postures[0]
        settled_posture_document = {
            "task": args.task,
            "posture_source": "Isaac zero-action rollout final state",
            "postures_by_geometry_variant": settled_postures,
        }
        args.settled_posture_output.parent.mkdir(parents=True, exist_ok=True)
        args.settled_posture_output.write_text(
            json.dumps(settled_posture_document, indent=2, sort_keys=True) + "\n"
        )

        spawn_cfg = base.cfg.screwdriver_cfg.spawn
        if hasattr(spawn_cfg, "asset_path"):
            asset_paths = [str(spawn_cfg.asset_path)]
        elif hasattr(spawn_cfg, "assets_cfg"):
            asset_paths = [
                str(asset_cfg.asset_path) for asset_cfg in spawn_cfg.assets_cfg
            ]
        else:
            raise TypeError(f"unsupported screwdriver spawn cfg {type(spawn_cfg)!r}")

        summary = {
            "task": args.task,
            "asset_paths": asset_paths,
            "num_envs": base.num_envs,
            "device": str(base.device),
            "domain_randomization_enabled": bool(env_cfg.domain_rand.enabled),
            "domain_randomization_only_component": args.dr_only,
            "reset_dynamics_per_environment": dynamics_snapshot,
            "reset_contact_steps_from_cfg": int(base.cfg.reset_contact_steps),
            "additional_settle_steps": args.settle_steps,
            "zero_action_rollout_steps": args.rollout_steps,
            "thresholds": {
                "min_contact_force_n": args.min_contact_force,
                "max_contact_force_n": args.max_contact_force,
                "min_contact_fraction": args.min_contact_fraction,
                "min_drive_fingers": args.min_drive_fingers,
                "required_contact_roles": "functional_redundant_five_finger_topology",
                "functional_critical_roles": list(CRITICAL_ROLE_NAMES),
                "functional_min_critical_fraction": FUNCTIONAL_MIN_CRITICAL_FRACTION,
                "functional_active_role_fraction": FUNCTIONAL_ACTIVE_ROLE_FRACTION,
                "functional_min_active_role_count": FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT,
                "functional_mean_role_fraction_is_diagnostic_only": True,
                "min_training_authorization_fraction": (
                    args.min_training_authorization_fraction
                ),
                "max_wrong_surface_force_n": args.max_wrong_force,
                "max_tilt_rad": args.max_tilt_rad,
                "max_zero_action_drift_rad_s": args.max_zero_action_drift_rad_s,
            },
            "reset": {
                "fingertip_total_force_n": _to_list(reset_snapshot["total"]),
                "fingertip_body_force_n": _to_list(reset_snapshot["body"]),
                "fingertip_cap_force_n": _to_list(reset_snapshot["cap"]),
                "wrong_surface_force_n": _to_list(reset_snapshot["wrong"]),
                "proximal_force_n": (
                    {
                        name: float(reset_snapshot["proximal"][0, i].item())
                        for i, name in enumerate(proximal_names)
                    }
                    if reset_snapshot["proximal"] is not None
                    else {}
                ),
                "tilt_rad": _to_list(reset_snapshot["tilt"]),
            },
            "gravity_settling": settle,
            "zero_action_rollout": rollout,
            "role_force_means_n": rollout["role_force_mean_n"],
            "role_force_maxima_n": rollout["role_force_max_n"],
            "role_contact_fractions": rollout["role_contact_fraction"],
            "geometry_buckets": geometry_buckets,
            "settled_posture_path": str(args.settled_posture_output),
            "settled_posture": settled_posture,
            "settled_postures_by_geometry_variant": settled_postures,
            "finger_joint_max_change_reset_to_settled_rad": float(
                torch.max(torch.abs(settled_finger_q - initial_finger_q)).item()
            ),
            "finger_joint_max_drift_during_rollout_rad": joint_drift,
            "checks": {
                "environment_instantiated_and_reset": True,
                "observations_finite": observations_finite,
                "rewards_finite": rewards_finite,
                **overall_contact_checks,
                "pip_joint_limit_margin_at_least_0_1_rad": all(
                    bucket["checks"]["pip_joint_limit_margin_at_least_0_1_rad"]
                    for bucket in geometry_buckets
                ),
                "zero_action_rollout_has_no_done": no_termination,
                "zero_action_raw_shaft_drift_within_limit": (
                    rollout[
                        "zero_action_raw_shaft_drift_rad_s_max_abs_environment_mean"
                    ]
                    <= args.max_zero_action_drift_rad_s
                ),
                "training_turn_authorization_persistent": (
                    rollout[
                        "training_authorization_fraction_min_environment_mean"
                    ]
                    >= args.min_training_authorization_fraction
                ),
                "all_geometry_buckets_pass": all(
                    bucket["physics_validation_pass"] for bucket in geometry_buckets
                ),
            },
        }
        summary["physics_validation_pass"] = all(
            summary["checks"][name]
            for name in (
                "environment_instantiated_and_reset",
                "observations_finite",
                "rewards_finite",
                "task_role_contact_persistently",
                "task_role_forces_in_range",
                "fingertip_total_force_below_max",
                "fingertip_wrong_surface_contact_below_threshold",
                "non_fingertip_contact_below_threshold",
                "screwdriver_remains_upright",
                "pip_joint_limit_margin_at_least_0_1_rad",
                "zero_action_rollout_has_no_done",
                "zero_action_raw_shaft_drift_within_limit",
                "training_turn_authorization_persistent",
                "all_geometry_buckets_pass",
            )
        )
        return summary
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    success = False
    try:
        result = main()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print(f"wrote {args.output}", flush=True)
        success = bool(result["physics_validation_pass"])
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # Isaac Sim can hang during teardown after all data is flushed.  Mirror
        # the repository renderer's bounded close watchdog.
        import os
        import threading

        code = 0 if success else 1
        watchdog = threading.Timer(60.0, lambda: os._exit(code))
        watchdog.daemon = True
        watchdog.start()
        if code != 0:
            os._exit(code)
        simulation_app.close()
        os._exit(code)

