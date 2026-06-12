#!/usr/bin/env python3
"""Smoke test: env creation, joint mapping, rollouts, DR — before any RL.

Run:
    python scripts/smoke_test.py --headless --num_envs 8
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ScrewdriverRL smoke test")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--hand", type=str, default="allegro")
parser.add_argument("--cfg", action="append", default=[], help="path.to.field=value env cfg overrides")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: F401, E402
from screwdriver_rl.tasks.continuous_turning import ContinuousTurningEnvCfg  # noqa: E402
from screwdriver_rl.utils import apply_overrides  # noqa: E402

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = ""):
    print(f"{PASS if condition else FAIL} {name} {detail}")
    if not condition:
        _failures.append(name)


def main():
    cfg = ContinuousTurningEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.hand_name = args.hand
    applied = apply_overrides(cfg, args.cfg)
    if applied:
        print(f"cfg overrides: {applied}")

    env = gym.make("Screwdriver-Continuous-Turning-v0", cfg=cfg).unwrapped
    device = env.device
    act_dim = env.num_finger_dofs

    # ---- 1. spaces ----
    obs_dict, _ = env.reset()
    expected_policy = 2 * act_dim + (3 if cfg.include_object_in_policy_obs else 0)
    check("policy obs dim", obs_dict["policy"].shape == (args.num_envs, expected_policy),
          f"shape={tuple(obs_dict['policy'].shape)}")
    check("critic obs dim", obs_dict["critic"].shape == (args.num_envs, cfg.privileged_obs_dim),
          f"shape={tuple(obs_dict['critic'].shape)} cfg={cfg.privileged_obs_dim}")
    check("proprio hist dim",
          obs_dict["proprio_hist"].shape == (args.num_envs, cfg.prop_hist_len, cfg.history_obs_dim),
          f"shape={tuple(obs_dict['proprio_hist'].shape)}")

    # ---- 2. joint mapping round-trip (BFS-order safety) ----
    spec = env._hand_spec
    ok = True
    for finger, jids in env._joint_ids_by_finger.items():
        ok &= len(jids) == len(spec.finger_joint_names[finger])
        pregrasp = torch.tensor(spec.pregrasp[finger], device=device)
        # read back the written pregrasp (within settle drift)
        q = env.hand.data.joint_pos[:, jids]
        ok &= bool(torch.all((q - pregrasp).abs() < 0.25))
    check("joint name->id mapping round-trip", ok)

    # ---- 3. fingertip bodies + initial distances ----
    tip_dist = env._fingertip_screwdriver_distances()
    print(f"  initial fingertip-to-handle-axis distances (m): {tip_dist[0].tolist()}")
    check("fingertip distances sane", bool(torch.all(tip_dist < 0.3)) and bool(torch.all(tip_dist > 0.0)))

    # ---- 4. zero-action rollout (below stagnation grace) ----
    n_zero = min(80, cfg.stagnation_grace_steps - 5)
    nan_free = True
    for _ in range(n_zero):
        obs_dict, rew, terminated, timed_out, extras = env.step(torch.zeros(args.num_envs, act_dim, device=device))
        nan_free &= not (torch.isnan(rew).any() or torch.isnan(obs_dict["policy"]).any()
                         or torch.isnan(obs_dict["critic"]).any())
    euler = env.screwdriver.data.joint_pos[:, env._euler_joint_ids]
    tilt = torch.linalg.norm(euler[:, :2], dim=-1)
    check("zero-action: no NaNs", nan_free)
    check("zero-action: screwdriver stays upright", bool(torch.all(tilt < 0.15)),
          f"max tilt={tilt.max().item():.4f} rad")
    parked_ok = True
    for finger in spec.parked_fingers:
        jids = env._joint_ids_by_finger[finger]
        q = env.hand.data.joint_pos[:, jids]
        pregrasp = torch.tensor(spec.pregrasp[finger], device=device)
        parked_ok &= bool(torch.all((q - pregrasp).abs() < 0.1))
    check("zero-action: parked fingers hold pregrasp", parked_ok)

    # ---- 5. DR variation across resets ----
    priv_a = obs_dict["critic"].clone()
    env.reset()
    obs_dict, _, _, _, _ = env.step(torch.zeros(args.num_envs, act_dim, device=device))
    priv_b = obs_dict["critic"].clone()
    dr_slice = slice(-8, -1)  # 7 DR features before mean-torque
    check("DR features vary across resets",
          bool((priv_a[:, dr_slice] - priv_b[:, dr_slice]).abs().max() > 1e-4),
          f"max delta={(priv_a[:, dr_slice] - priv_b[:, dr_slice]).abs().max().item():.5f}")

    # ---- 6. random-action rollout: terminations + resets ----
    any_termination = False
    nan_free = True
    for _ in range(200):
        actions = 2.0 * torch.rand(args.num_envs, act_dim, device=device) - 1.0
        obs_dict, rew, terminated, timed_out, extras = env.step(actions)
        any_termination |= bool(terminated.any())
        nan_free &= not (torch.isnan(rew).any() or torch.isnan(obs_dict["policy"]).any())
    check("random-action: no NaNs", nan_free)
    check("random-action: terminations fire and reset cleanly", any_termination)

    print()
    if _failures:
        print(f"{FAIL} {len(_failures)} check(s) failed: {_failures}")
    else:
        print(f"{PASS} all smoke checks passed")
    env.close()
    return 1 if _failures else 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
