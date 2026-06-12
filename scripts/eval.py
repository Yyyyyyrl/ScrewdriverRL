#!/usr/bin/env python3
"""Evaluate a stage-1 teacher or stage-2 student checkpoint.

The student is evaluated WITHOUT privileged observations (adaptation module
only) to prove deployability.

Usage:
    python scripts/eval.py --headless --stage 1 --checkpoint .../stage1_nn/best.pth
    python scripts/eval.py --headless --stage 2 --checkpoint .../stage2_nn/best.pth

Reported per finished episode: sustained turn velocity (net rad/s), net/total
turns, mean+max tilt, termination breakdown, gate uptime.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ScrewdriverRL evaluation")
parser.add_argument("--task", type=str, default="Screwdriver-Continuous-Turning-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--hand", type=str, default="allegro")
parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=512, help="Total finished episodes to evaluate")
parser.add_argument("--curriculum_phase", type=str, default="final",
                    help="'final' applies the last curriculum phase env settings; 'none' = cfg defaults")
parser.add_argument("--cfg", action="append", default=[], help="env cfg override path=value")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: F401, E402
from screwdriver_rl.algo import PPO, ProprioAdapt  # noqa: E402
from screwdriver_rl.configs import NetworkCfg, get_curriculum  # noqa: E402
from screwdriver_rl.tasks.continuous_turning import ContinuousTurningEnvCfg  # noqa: E402
from screwdriver_rl.utils import apply_overrides, cli  # noqa: E402


def build_env():
    env_cfg = ContinuousTurningEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.hand_name = args.hand
    if args.curriculum_phase == "final":
        for path, value in get_curriculum("allegro_default")[-1].env_overrides.items():
            cli.set_dotted(env_cfg, path, value)
    apply_overrides(env_cfg, args.cfg)
    return gym.make(args.task, cfg=env_cfg).unwrapped


@torch.no_grad()
def main():
    device = args.device if args.device is not None else "cuda:0"
    env = build_env()

    if args.stage == 1:
        agent = PPO(env, "/tmp/screwdriver_rl_eval", net_cfg=NetworkCfg(), device=device)
        agent.restore(args.checkpoint, resume_training_state=False)
        agent.set_eval()

        def act(obs):
            return agent.model.act_inference(
                {"obs": agent.obs_rms(obs["obs"]), "priv_info": obs["priv_info"]}
            )
    else:
        agent = ProprioAdapt(env, "/tmp/screwdriver_rl_eval", net_cfg=NetworkCfg(), device=device)
        agent.restore(args.checkpoint)
        agent.model.eval()
        agent.hist_rms.eval()

        def act(obs):
            # NO priv_info: adaptation module only (deployment mode).
            return agent.model.act_inference(
                {"obs": agent.obs_rms(obs["obs"]), "proprio_hist": agent.hist_rms(obs["proprio_hist"])}
            )

    obs_dict, _ = env.reset()
    N = env.num_envs
    policy_dt = env.cfg.decimation * env.cfg.sim.dt

    ep_net_turn = torch.zeros(N, device=device)
    ep_total_turn = torch.zeros(N, device=device)
    ep_len = torch.zeros(N, device=device)
    ep_tilt_sum = torch.zeros(N, device=device)
    ep_tilt_max = torch.zeros(N, device=device)
    ep_gate_sum = torch.zeros(N, device=device)

    done_stats: dict[str, list[float]] = {k: [] for k in (
        "sustained_radps", "net_turns", "total_turns", "mean_tilt", "max_tilt", "gate_uptime", "length")}
    n_drop, n_lost, n_stagnant, n_timeout, n_done = 0, 0, 0, 0, 0

    while n_done < args.episodes:
        obs = {
            "obs": obs_dict["policy"],
            "priv_info": obs_dict.get("critic"),
            "proprio_hist": obs_dict.get("proprio_hist"),
        }
        actions = torch.clamp(act(obs), -1.0, 1.0)
        obs_dict, _, terminated, timed_out, extras = env.step(actions)

        ep_net_turn += extras["eval_turn_delta"]
        ep_total_turn += torch.clamp(extras["eval_turn_delta"], min=0.0)
        ep_len += 1
        tilt = extras["eval_screwdriver_upright_norm"]
        ep_tilt_sum += tilt
        ep_tilt_max = torch.maximum(ep_tilt_max, tilt)
        ep_gate_sum += extras["eval_turn_gate"]

        dones = terminated | timed_out
        idx = dones.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            lengths = ep_len[idx].clamp(min=1.0)
            done_stats["sustained_radps"] += (ep_net_turn[idx] / (lengths * policy_dt)).tolist()
            done_stats["net_turns"] += (ep_net_turn[idx] / (2 * math.pi)).tolist()
            done_stats["total_turns"] += (ep_total_turn[idx] / (2 * math.pi)).tolist()
            done_stats["mean_tilt"] += (ep_tilt_sum[idx] / lengths).tolist()
            done_stats["max_tilt"] += ep_tilt_max[idx].tolist()
            done_stats["gate_uptime"] += (ep_gate_sum[idx] / lengths).tolist()
            done_stats["length"] += lengths.tolist()
            n_done += idx.numel()
            n_timeout += int(timed_out[idx].sum())
            if "eval_upright_terminated" in extras:
                n_drop += int(extras["eval_upright_terminated"][idx].sum())
            if "eval_lost_contact_terminated" in extras:
                n_lost += int(extras["eval_lost_contact_terminated"][idx].sum())
            if "eval_stagnation_terminated" in extras:
                n_stagnant += int(extras["eval_stagnation_terminated"][idx].sum())
            for buf in (ep_net_turn, ep_total_turn, ep_len, ep_tilt_sum, ep_tilt_max, ep_gate_sum):
                buf[idx] = 0.0

    print("=" * 70)
    print(f"Evaluation: stage {args.stage} | {args.checkpoint}")
    print(f"Episodes: {n_done}")
    for key, values in done_stats.items():
        t = torch.tensor(values)
        print(f"  {key:>16}: mean {t.mean():8.4f} | std {t.std():7.4f} | min {t.min():8.4f} | max {t.max():8.4f}")
    print(f"  termination: timeout {n_timeout / n_done:.1%} | tilt-drop {n_drop / n_done:.1%} | "
          f"lost-contact {n_lost / n_done:.1%} | stagnation {n_stagnant / n_done:.1%}")
    print("=" * 70)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
