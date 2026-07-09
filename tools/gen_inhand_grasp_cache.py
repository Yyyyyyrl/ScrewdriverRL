"""Generate LinkerL20 in-hand rotation grasp caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate LinkerL20 in-hand grasp cache states.")
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--num_states", type=int, default=50000)
parser.add_argument("--out", type=str, default="assets/grasp_cache")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many env steps without saving; 0 disables.")
parser.add_argument("--diagnostics_every", type=int, default=0, help="Print grasp acceptance diagnostics every N steps.")
parser.add_argument("--debug", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if args.debug:
    args.num_envs = 16
    args.headless = False
args.enable_cameras = False
args.rl_device = getattr(args, "rl_device", None) or args.device or "cuda:0"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import screwdriver_rl.tasks  # noqa: F401

try:
    from isaaclab_tasks.utils import parse_env_cfg
except ImportError:
    try:
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    except ImportError:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from screwdriver_rl.tasks.linker_l20.inhand_rotation_env_cfg import grasp_cache_filename


def _configure_scale(env_cfg, scale: float) -> None:
    # Cache generation runs on CYLINDERS only, at one pinned scale.
    env_cfg.cache_scale = float(scale)
    env_cfg.object_kind = "cylinder"
    env_cfg.object_scales = (float(scale),)
    env_cfg._rebuild_object()  # rebuilds object_cfg + object_asset_scale_idx
    env_cfg.load_grasp_cache = False
    env_cfg.enable_fingertip_sensors = True
    env_cfg.episode_length_s = 2.5
    if env_cfg.curriculum_phases:
        env_cfg.curriculum_phases[0].episode_length_s = 2.5
    env_cfg.domain_rand.random_force_prob = 0.0
    env_cfg.domain_rand.force_scale = 0.0
    env_cfg.domain_rand.pd_gain_range = (1.0, 1.0)


def _debug_report(base_env, steps: int) -> None:
    obj_z = (base_env.object.data.root_pos_w[:, 2] - base_env.scene.env_origins[:, 2]).detach()
    median_z = float(torch.median(obj_z).item())
    extras = base_env.extras
    tip_pos = base_env.hand.data.body_state_w[:, base_env._fingertip_body_ids, :3]
    obj_pos = base_env.object.data.root_pos_w.unsqueeze(1)
    tip_dist = torch.linalg.norm(tip_pos - obj_pos, dim=-1).detach()
    tip_close = tip_dist <= float(base_env.cfg.tip_dist_max)
    tip_forces = base_env._read_tip_object_forces().detach()

    def mean_extra(name: str) -> float:
        value = extras.get(name)
        if value is None:
            return float("nan")
        return float(value.float().mean().item())

    finger_lines = []
    for i, finger in enumerate(base_env.fingers):
        finger_lines.append(
            f"\n    {finger:<6} dist_mean={float(tip_dist[:, i].mean().item()):.4f}"
            f" close={float(tip_close[:, i].float().mean().item()):.3f}"
            f" contact={float((tip_forces[:, i] > 0.0).float().mean().item()):.3f}"
        )

    print(
        "\n[debug] grasp-gen diagnostics"
        f"\n  steps                    : {steps}"
        f"\n  settled obj z median     : {median_z:.4f}"
        f"\n  suggested reset threshold: {median_z - 0.03:.4f}"
        f"\n  suggested drop height    : {median_z + 0.07:.4f}"
        f"\n  palm-up hand quat        : {tuple(base_env.cfg.robot_cfg.init_state.rot)}"
        f"\n  acceptance mean          : {mean_extra('eval_grasp_acceptance'):.3f}"
        f"\n  all tips close mean      : {mean_extra('eval_tip_close'):.3f}"
        f"\n  contact fingers mean     : {mean_extra('eval_contact_fingers'):.3f}"
        f"\n  object high mean         : {mean_extra('eval_obj_high'):.3f}"
        f"\n  per-finger stats         : {''.join(finger_lines)}\n",
        flush=True,
    )


def main() -> None:
    task = "Isaac-LinkerL20-Inhand-GraspGen"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    _configure_scale(env_cfg, args.scale)
    env = gym.make(task, cfg=env_cfg, render_mode="human" if args.debug else None)
    base_env = env.unwrapped

    out = Path(args.out) / grasp_cache_filename(args.scale, env_cfg.grasp_cache_name)
    print(
        f"[grasp-cache] scale={args.scale} num_envs={base_env.num_envs} "
        f"target={args.num_states} out={out}",
        flush=True,
    )

    obs, _ = env.reset()
    del obs
    actions = torch.zeros(
        (base_env.num_envs, int(env_cfg.action_space.shape[0])),
        dtype=torch.float32,
        device=base_env.device,
    )
    steps = 0
    try:
        while True:
            env.step(actions)
            steps += 1
            diag_every = 25 if args.debug else int(args.diagnostics_every)
            if diag_every > 0 and steps % diag_every == 0:
                _debug_report(base_env, steps)
            if base_env.save_if_full(out, n=args.num_states):
                print(f"[grasp-cache] saved {args.num_states} states to {out}", flush=True)
                break
            if steps % 100 == 0:
                rows = sum(t.shape[0] for t in base_env._harvest_rows)
                print(f"[grasp-cache] steps={steps} cached={rows}/{args.num_states}", flush=True)
            if args.max_steps > 0 and steps >= args.max_steps:
                rows = sum(t.shape[0] for t in base_env._harvest_rows)
                _debug_report(base_env, steps)
                raise RuntimeError(
                    f"Reached --max_steps={args.max_steps} with {rows}/{args.num_states} cached states."
                )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
