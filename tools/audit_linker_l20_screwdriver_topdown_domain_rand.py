#!/usr/bin/env python3
"""Runtime audit of top-down screwdriver domain randomisation in Isaac Lab.

The audit instantiates the registered task with DR enabled, resets every
environment once, reads the authored PhysX masses/drive gains/materials back,
and compares the effective values with the configured distributions.  It is a
training preflight, not a replacement for the contact/rollout physics gate.
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
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--dynamics_randomization_scale",
    type=float,
    default=1.0,
    help="Audit this curriculum DR scale; default 1.0 exercises the final envelope.",
)
parser.add_argument(
    "--reset_samples",
    type=int,
    default=1024,
    help="Minimum number of per-episode reset/calibration samples to audit.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=REPO_ROOT
    / "artifacts/linker_l20_screwdriver_topdown/domain_rand_runtime_audit.json",
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
    values = values.detach().float().cpu().reshape(-1)
    return {
        "min": float(values.min()),
        "p05": float(torch.quantile(values, 0.05)),
        "mean": float(values.mean()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
        "std": float(values.std(unbiased=False)),
    }


def _inside(values: torch.Tensor, lower: float, upper: float, tol: float = 1.0e-5) -> bool:
    return bool(((values >= lower - tol) & (values <= upper + tol)).all())


def main() -> dict:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if not 0.0 <= args.dynamics_randomization_scale <= 1.0:
        raise ValueError("--dynamics_randomization_scale must be in [0, 1]")
    # A preflight must exercise the final training envelope, not merely the P0
    # curriculum defaults selected at environment construction.
    for phase in cfg.curriculum_phases:
        phase.dynamics_randomization_scale = args.dynamics_randomization_scale
    if not cfg.domain_rand.enabled:
        raise RuntimeError("domain randomisation is disabled in the resolved task config")

    env = None
    try:
        env = gym.make(args.task, cfg=cfg)
        base = env.unwrapped
        if args.reset_samples < 1:
            raise ValueError("--reset_samples must be positive")
        reset_chunks: dict[str, list[torch.Tensor]] = {
            "root_pos": [],
            "root_rpy": [],
            "screwdriver_tilt": [],
            "joint_bias": [],
            "contact_count": [],
            "guard_retries": [],
            "donor_repaired": [],
        }
        reset_samples_collected = 0
        reset_batch_index = 0
        observations = None
        while reset_samples_collected < args.reset_samples:
            observations, _ = env.reset(seed=args.seed + reset_batch_index)
            take = min(
                base.num_envs,
                args.reset_samples - reset_samples_collected,
            )
            reset_chunks["root_pos"].append(
                base._env_reset_root_pos_noise[:take].detach().clone()
            )
            reset_chunks["root_rpy"].append(
                base._env_reset_root_rpy_noise[:take].detach().clone()
            )
            reset_chunks["screwdriver_tilt"].append(
                base._env_reset_screwdriver_tilt_noise[:take].detach().clone()
            )
            reset_chunks["joint_bias"].append(
                base._env_joint_position_bias[:take].detach().clone()
            )
            _, _, reset_present = base._compute_distance_contact()
            reset_chunks["contact_count"].append(
                reset_present[:take].sum(dim=-1).detach().clone()
            )
            reset_chunks["guard_retries"].append(
                base._reset_contact_guard_retries[:take].detach().clone()
            )
            reset_chunks["donor_repaired"].append(
                base._reset_contact_guard_donor_repaired[:take].detach().clone()
            )
            reset_samples_collected += take
            reset_batch_index += 1
        assert observations is not None
        dr = cfg.domain_rand
        variant_idx = getattr(base, "_env_variant_idx", None)
        geom_scale = getattr(base, "_env_geom_scale", None)
        variant_table = getattr(base, "_variant_table", None)
        if variant_idx is None or geom_scale is None or variant_table is None:
            raise RuntimeError("geometry DR runtime tensors were not initialised")
        variant_idx = variant_idx.detach().clone()
        geom_scale = geom_scale.detach().clone()
        expected_variant_idx = torch.arange(
            base.num_envs, dtype=torch.long, device=base.device
        ) % variant_table.num_variants
        expected_geom_scale = torch.stack(
            (
                variant_table.diameter_scale.to(base.device)[variant_idx],
                variant_table.length_scale.to(base.device)[variant_idx],
            ),
            dim=-1,
        )
        variant_counts = torch.bincount(
            variant_idx, minlength=variant_table.num_variants
        )
        critic_obs = observations.get("critic")
        policy_obs = observations["policy"]
        critic_geom = None if critic_obs is None else critic_obs[:, -2:]
        privileged = base._compute_privileged_obs().detach().clone()
        privileged_geom = privileged[:, -2:]
        policy_geom = (
            policy_obs[:, -2:] if getattr(cfg, "latent_conditioned", False) else None
        )

        rotation = base._env_rotation_damping.detach().clone()
        load = base._env_load_torque.detach().clone()
        friction = base._env_friction.detach().clone()
        reset_root_pos = torch.cat(reset_chunks["root_pos"], dim=0)
        reset_root_rpy = torch.cat(reset_chunks["root_rpy"], dim=0)
        reset_screwdriver_tilt = torch.cat(
            reset_chunks["screwdriver_tilt"], dim=0
        )
        joint_zero_bias = torch.cat(reset_chunks["joint_bias"], dim=0)
        reset_contact_count = torch.cat(reset_chunks["contact_count"], dim=0)
        reset_guard_retries = torch.cat(reset_chunks["guard_retries"], dim=0)
        reset_donor_repaired = torch.cat(
            reset_chunks["donor_repaired"], dim=0
        )
        joint_zero_bias_episode = base._env_joint_position_bias.detach().clone()

        body_id = base._handle_body_ids[base._handle_base_idx]
        masses = base.screwdriver.root_physx_view.get_masses()[:, body_id].detach().clone()
        inertias = (
            base.screwdriver.root_physx_view.get_inertias()[:, body_id]
            .detach()
            .clone()
        )
        default_mass = float(base.screwdriver.data.default_mass[0, body_id])

        finger_ids = torch.as_tensor(base._finger_joint_ids, dtype=torch.long)
        stiffness = (
            base.allegro.root_physx_view.get_dof_stiffnesses()[:, finger_ids]
            .detach()
            .clone()
        )
        damping = (
            base.allegro.root_physx_view.get_dof_dampings()[:, finger_ids]
            .detach()
            .clone()
        )
        # One scale is broadcast across every independent finger joint per env.
        stiffness_env = stiffness[:, 0]
        damping_env = damping[:, 0]
        gains_broadcast_pass = bool(
            torch.allclose(stiffness, stiffness_env[:, None].expand_as(stiffness), atol=1.0e-5)
            and torch.allclose(damping, damping_env[:, None].expand_as(damping), atol=1.0e-5)
        )

        tilt_ids = torch.as_tensor(
            base._screwdriver_euler_ids[:2], dtype=torch.long
        )
        tilt_damping = (
            base.screwdriver.root_physx_view.get_dof_dampings()[:, tilt_ids]
            .detach()
            .clone()
        )
        materials = base.screwdriver.root_physx_view.get_material_properties().detach().clone()
        material_static = materials[:, :, 0]
        material_dynamic = materials[:, :, 1]
        friction_material = friction.to(material_static.device)

        # ``env.reset`` includes the task's 32 contact-settling physics steps, so
        # this is the sampled start angle plus a small real contact-driven motion.
        # The sampler itself is exactly [-pi, pi]; allow 0.25 rad here so the
        # runtime audit does not misclassify legitimate settling near a boundary.
        z = base.screwdriver.data.joint_pos[:, base._screwdriver_z_id].detach().clone()
        phase_dr_scale = float(
            getattr(base._curriculum_phase, "dynamics_randomization_scale", 1.0)
        )

        def scaled_range(bounds):
            return [
                1.0 + phase_dr_scale * (bounds[0] - 1.0),
                1.0 + phase_dr_scale * (bounds[1] - 1.0),
            ]

        def scaled_absolute_range(bounds, center):
            return [
                center + phase_dr_scale * (bounds[0] - center),
                center + phase_dr_scale * (bounds[1] - center),
            ]

        rotation_scale_range = scaled_range(dr.rotation_damping_range)
        mass_scale_range = scaled_range(dr.screwdriver_mass_range)
        load_scale_range = scaled_range(dr.screwdriver_load_torque_range)
        stiffness_scale_range = scaled_range(dr.finger_stiffness_range)
        damping_scale_range = scaled_range(dr.finger_damping_range)
        tilt_damping_scale_range = scaled_range(dr.tilt_damping_range)
        friction_range = scaled_absolute_range(
            dr.contact_friction_range, base._base_friction
        )
        expected = {
            "rotation_damping": [
                base._base_rotation_damping * rotation_scale_range[0],
                base._base_rotation_damping * rotation_scale_range[1],
            ],
            "body_mass": [
                default_mass * mass_scale_range[0],
                default_mass * mass_scale_range[1],
            ],
            "load_torque": [
                base._base_load_torque * load_scale_range[0],
                base._base_load_torque * load_scale_range[1],
            ],
            "finger_stiffness": [
                base._base_finger_stiffness * stiffness_scale_range[0],
                base._base_finger_stiffness * stiffness_scale_range[1],
            ],
            "finger_damping": [
                base._base_finger_damping * damping_scale_range[0],
                base._base_finger_damping * damping_scale_range[1],
            ],
            "contact_friction": friction_range,
            "tilt_damping": [
                base._base_tilt_damping * tilt_damping_scale_range[0],
                base._base_tilt_damping * tilt_damping_scale_range[1],
            ],
            "reset_root_xy_m": [
                -phase_dr_scale * dr.reset_root_pos_noise_m,
                phase_dr_scale * dr.reset_root_pos_noise_m,
            ],
            "reset_root_z_m": [
                -phase_dr_scale * dr.reset_root_z_noise_m,
                phase_dr_scale * dr.reset_root_z_noise_m,
            ],
            "reset_root_tilt_rad": [
                -phase_dr_scale * dr.reset_root_tilt_noise_rad,
                phase_dr_scale * dr.reset_root_tilt_noise_rad,
            ],
            "reset_root_yaw_rad": [
                -phase_dr_scale * dr.reset_root_yaw_noise_rad,
                phase_dr_scale * dr.reset_root_yaw_noise_rad,
            ],
            "reset_screwdriver_tilt_rad": [
                -phase_dr_scale * dr.reset_screwdriver_tilt_noise_rad,
                phase_dr_scale * dr.reset_screwdriver_tilt_noise_rad,
            ],
            "joint_zero_bias_rad": [
                -phase_dr_scale * dr.joint_zero_bias_rad,
                phase_dr_scale * dr.joint_zero_bias_rad,
            ],
        }

        joint_zero_bias_before_observation = joint_zero_bias_episode.clone()
        base._get_observations()
        joint_zero_bias_is_episode_constant = torch.equal(
            joint_zero_bias_before_observation, base._env_joint_position_bias
        )

        checks = {
            "rotation_damping_in_range": _inside(rotation, *expected["rotation_damping"]),
            "body_mass_in_range": _inside(masses, *expected["body_mass"]),
            "body_inertia_fixed_as_current_implementation": bool(
                torch.allclose(inertias, inertias[0:1].expand_as(inertias))
            ),
            "load_torque_in_range": _inside(load, *expected["load_torque"]),
            "finger_stiffness_in_range": _inside(stiffness_env, *expected["finger_stiffness"]),
            "finger_damping_in_range": _inside(damping_env, *expected["finger_damping"]),
            "finger_gains_broadcast_per_env": gains_broadcast_pass,
            "all_enabled_dynamics_distributions_have_variance": all(
                float(values.float().std(unbiased=False)) > 0.0
                for values in (
                    rotation, masses, load, stiffness_env, damping_env,
                    friction, tilt_damping[:, 0],
                )
            ),
            "contact_friction_randomisation_enabled": bool(
                dr.randomize_contact_friction
            ),
            "contact_friction_in_range": _inside(
                friction, *expected["contact_friction"]
            ),
            "contact_friction_materials_match_sample": bool(
                torch.allclose(
                    material_static,
                    friction_material[:, None].expand_as(material_static),
                )
                and torch.allclose(
                    material_dynamic,
                    friction_material[:, None].expand_as(material_dynamic),
                )
            ),
            "tilt_damping_randomisation_enabled": bool(
                dr.randomize_tilt_damping
            ),
            "tilt_damping_in_range": _inside(
                tilt_damping, *expected["tilt_damping"]
            ),
            "tilt_damping_broadcast_across_xy": bool(
                torch.allclose(tilt_damping[:, 0], tilt_damping[:, 1])
            ),
            "reset_root_xy_in_range": _inside(
                reset_root_pos[:, :2], *expected["reset_root_xy_m"]
            ),
            "reset_root_z_in_range": _inside(
                reset_root_pos[:, 2], *expected["reset_root_z_m"]
            ),
            "reset_root_tilt_in_range": _inside(
                reset_root_rpy[:, :2], *expected["reset_root_tilt_rad"]
            ),
            "reset_root_yaw_in_range": _inside(
                reset_root_rpy[:, 2], *expected["reset_root_yaw_rad"]
            ),
            "reset_screwdriver_tilt_in_range": _inside(
                reset_screwdriver_tilt,
                *expected["reset_screwdriver_tilt_rad"],
            ),
            "joint_zero_bias_in_range": _inside(
                joint_zero_bias, *expected["joint_zero_bias_rad"]
            ),
            "reset_and_bias_distributions_have_variance": all(
                float(values.float().std(unbiased=False)) > 0.0
                for values in (
                    reset_root_pos,
                    reset_root_rpy,
                    reset_screwdriver_tilt,
                    joint_zero_bias,
                )
            ),
            "joint_zero_bias_is_episode_constant": bool(
                joint_zero_bias_is_episode_constant
            ),
            "privileged_observation_width_matches": (
                privileged.shape[1] == cfg.privileged_obs_dim
            ),
            "policy_observation_width_matches": (
                policy_obs.shape[1] == cfg.observation_space.shape[0]
            ),
            "privileged_load_proxy_matches": bool(
                torch.allclose(
                    privileged[:, 13], load / base._base_load_torque
                )
            ),
            "privileged_contact_friction_matches": bool(
                torch.allclose(
                    privileged[:, 14], friction / base._base_friction
                )
            ),
            "privileged_distance_scores_bounded": _inside(
                privileged[:, 15:20], 0.0, 1.0
            ),
            "geometry_randomisation_enabled": bool(dr.randomize_geometry),
            "geometry_assignment_is_cyclic": (
                getattr(cfg, "geometry_variant_assignment", None) == "cyclic"
            ),
            "geometry_variant_indices_are_cyclic": bool(
                torch.equal(variant_idx, expected_variant_idx)
            ),
            "geometry_histogram_balanced": bool(
                int(variant_counts.max() - variant_counts.min()) <= 1
                and int((variant_counts > 0).sum()) == variant_table.num_variants
            ),
            "geometry_scales_match_manifest": bool(
                torch.allclose(geom_scale, expected_geom_scale, atol=1.0e-7, rtol=0.0)
            ),
            "geometry_diameter_has_variance": float(
                geom_scale[:, 0].std(unbiased=False)
            ) > 0.0,
            "geometry_length_fixed": bool(
                torch.allclose(geom_scale[:, 1], torch.ones_like(geom_scale[:, 1]))
            ),
            "privileged_geometry_observation_matches": bool(
                torch.allclose(privileged_geom, geom_scale, atol=1.0e-7, rtol=0.0)
            ),
            "critic_geometry_observation_matches_if_enabled": bool(
                (
                    not cfg.asymmetric_obs
                    and critic_geom is None
                )
                or (
                    critic_geom is not None
                    and torch.allclose(
                        critic_geom, geom_scale, atol=1.0e-7, rtol=0.0
                    )
                )
            ),
            "policy_geometry_observation_matches": bool(
                policy_geom is None
                or torch.allclose(policy_geom, geom_scale, atol=1.0e-7, rtol=0.0)
            ),
            "object_start_spans_both_signs": bool((z < 0).any() and (z > 0).any()),
            "object_start_post_settle_within_pi_plus_0p25": _inside(
                z, -torch.pi - 0.25, torch.pi + 0.25
            ),
            "reset_contact_guard_enabled": bool(
                cfg.reset_contact_guard_min_fingers > 0
            ),
            "all_resets_meet_contact_guard": bool(
                (reset_contact_count >= cfg.reset_contact_guard_min_fingers).all()
            ),
        }
        result = {
            "task": args.task,
            "num_envs": base.num_envs,
            "reset_samples": reset_samples_collected,
            "reset_batches": reset_batch_index,
            "seed": args.seed,
            "config": {
                "enabled": dr.enabled,
                "active_curriculum_dynamics_randomization_scale": phase_dr_scale,
                "requested_audit_dynamics_randomization_scale": (
                    args.dynamics_randomization_scale
                ),
                "rotation_damping_scale_range": list(dr.rotation_damping_range),
                "effective_rotation_damping_scale_range": rotation_scale_range,
                "effective_screwdriver_mass_scale_range": mass_scale_range,
                "effective_screwdriver_load_torque_scale_range": load_scale_range,
                "effective_finger_stiffness_scale_range": stiffness_scale_range,
                "effective_finger_damping_scale_range": damping_scale_range,
                "screwdriver_mass_scale_range": list(dr.screwdriver_mass_range),
                "screwdriver_load_torque_scale_range": list(dr.screwdriver_load_torque_range),
                "finger_stiffness_scale_range": list(dr.finger_stiffness_range),
                "finger_damping_scale_range": list(dr.finger_damping_range),
                "randomize_contact_friction": dr.randomize_contact_friction,
                "contact_friction_range": list(dr.contact_friction_range),
                "effective_contact_friction_range": friction_range,
                "randomize_tilt_damping": dr.randomize_tilt_damping,
                "tilt_damping_scale_range": list(dr.tilt_damping_range),
                "effective_tilt_damping_scale_range": tilt_damping_scale_range,
                "reset_root_pos_noise_m": dr.reset_root_pos_noise_m,
                "reset_root_z_noise_m": dr.reset_root_z_noise_m,
                "reset_root_tilt_noise_rad": dr.reset_root_tilt_noise_rad,
                "reset_root_yaw_noise_rad": dr.reset_root_yaw_noise_rad,
                "reset_screwdriver_tilt_noise_rad": (
                    dr.reset_screwdriver_tilt_noise_rad
                ),
                "joint_zero_bias_rad": dr.joint_zero_bias_rad,
                "reset_root_pose_ramp": cfg.reset_root_pose_ramp,
                "reset_contact_guard_min_fingers": (
                    cfg.reset_contact_guard_min_fingers
                ),
                "reset_contact_guard_max_resamples": (
                    cfg.reset_contact_guard_max_resamples
                ),
                "randomize_geometry": dr.randomize_geometry,
                "geometry_variant_assignment": cfg.geometry_variant_assignment,
                "geometry_variant_files": list(variant_table.files),
                "geometry_diameter_scales": [
                    float(value) for value in variant_table.diameter_scale
                ],
                "geometry_length_scales": [
                    float(value) for value in variant_table.length_scale
                ],
                "latent_conditioned": cfg.latent_conditioned,
                "asymmetric_obs": cfg.asymmetric_obs,
                "obs_noise_std": dr.obs_noise_std,
                "randomize_obj_start": cfg.randomize_obj_start,
            },
            "base_values": {
                "rotation_damping": base._base_rotation_damping,
                "screwdriver_body_mass": default_mass,
                "screwdriver_body_inertia_row_major": inertias[0].detach().cpu().tolist(),
                "screwdriver_load_torque": base._base_load_torque,
                "finger_stiffness": base._base_finger_stiffness,
                "finger_damping": base._base_finger_damping,
                "contact_friction": base._base_friction,
                "tilt_damping": base._base_tilt_damping,
                "handle_radius_m": cfg.screwdriver_handle_radius,
            },
            "expected_effective_ranges": expected,
            "sampled": {
                "rotation_damping": _stats(rotation),
                "screwdriver_body_mass": _stats(masses),
                "screwdriver_body_inertia_xx": _stats(inertias[:, 0]),
                "screwdriver_body_inertia_yy": _stats(inertias[:, 4]),
                "screwdriver_body_inertia_zz": _stats(inertias[:, 8]),
                "screwdriver_load_torque": _stats(load),
                "finger_stiffness": _stats(stiffness_env),
                "finger_damping": _stats(damping_env),
                "contact_friction": _stats(friction),
                "tilt_damping": _stats(tilt_damping[:, 0]),
                "reset_root_xy_m": _stats(reset_root_pos[:, :2]),
                "reset_root_z_m": _stats(reset_root_pos[:, 2]),
                "reset_root_tilt_rad": _stats(reset_root_rpy[:, :2]),
                "reset_root_yaw_rad": _stats(reset_root_rpy[:, 2]),
                "reset_screwdriver_tilt_rad": _stats(reset_screwdriver_tilt),
                "joint_zero_bias_rad": _stats(joint_zero_bias),
                "reset_contact_count": _stats(reset_contact_count),
                "reset_contact_guard_retries": _stats(reset_guard_retries),
                "reset_contact_guard_donor_repaired": _stats(
                    reset_donor_repaired
                ),
                "object_start_z_angle_post_settle_rad": _stats(z),
                "geometry_variant_histogram": {
                    variant_table.files[index]: int(variant_counts[index])
                    for index in range(variant_table.num_variants)
                },
                "geometry_diameter_scale": _stats(geom_scale[:, 0]),
                "geometry_length_scale": _stats(geom_scale[:, 1]),
                "effective_handle_radius_m": _stats(
                    cfg.screwdriver_handle_radius * geom_scale[:, 0]
                ),
            },
            "checks": checks,
            "runtime_audit_pass": all(checks.values()),
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
        exit_code = 0 if summary["runtime_audit_pass"] else 1
    except BaseException:
        traceback.print_exc()
        exit_code = 2
    try:
        simulation_app.close()
    except SystemExit:
        pass
    raise SystemExit(exit_code)
