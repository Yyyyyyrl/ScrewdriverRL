#!/usr/bin/env python3
"""Visualize a trained checkpoint in the Isaac Sim GUI (or livestream).

Usage:
    python scripts/play.py --stage 1 --checkpoint .../stage1_nn/best.pth --num_envs 4
    python scripts/play.py --stage 2 --checkpoint .../stage2_nn/best.pth --num_envs 4
    # zero-action posture inspection (no checkpoint):
    python scripts/play.py --zero-action --num_envs 4

If the local GUI misbehaves on this GPU, use --livestream 2.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ScrewdriverRL policy visualization")
parser.add_argument("--task", type=str, default="Screwdriver-Continuous-Turning-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--hand", type=str, default="allegro")
parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--zero-action", dest="zero_action", action="store_true",
                    help="Hold the pregrasp (no policy) — useful for posture checks")
parser.add_argument("--curriculum_phase", type=str, default="final")
parser.add_argument("--cfg", action="append", default=[], help="env cfg override path=value")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: F401, E402
from screwdriver_rl.algo import PPO, ProprioAdapt  # noqa: E402
from screwdriver_rl.configs import NetworkCfg, get_curriculum  # noqa: E402
from screwdriver_rl.tasks.continuous_turning import ContinuousTurningEnvCfg  # noqa: E402
from screwdriver_rl.utils import apply_overrides, cli  # noqa: E402


@torch.no_grad()
def main():
    device = args.device if args.device is not None else "cuda:0"
    env_cfg = ContinuousTurningEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.hand_name = args.hand
    if args.curriculum_phase == "final":
        for path, value in get_curriculum("allegro_default")[-1].env_overrides.items():
            cli.set_dotted(env_cfg, path, value)
    # No per-step noise during visualization.
    env_cfg.dr.obs_noise_std = 0.0
    env_cfg.dr.action_noise_std = 0.0
    apply_overrides(env_cfg, args.cfg)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None).unwrapped

    act = None
    if not args.zero_action:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --zero-action is set")
        if args.stage == 1:
            agent = PPO(env, "/tmp/screwdriver_rl_play", net_cfg=NetworkCfg(), device=device)
            agent.restore(args.checkpoint, resume_training_state=False)
            agent.set_eval()

            def act(obs):
                return agent.model.act_inference(
                    {"obs": agent.obs_rms(obs["obs"]), "priv_info": obs["priv_info"]}
                )
        else:
            agent = ProprioAdapt(env, "/tmp/screwdriver_rl_play", net_cfg=NetworkCfg(), device=device)
            agent.restore(args.checkpoint)
            agent.model.eval()
            agent.hist_rms.eval()

            def act(obs):
                return agent.model.act_inference(
                    {"obs": agent.obs_rms(obs["obs"]), "proprio_hist": agent.hist_rms(obs["proprio_hist"])}
                )

    obs_dict, _ = env.reset()
    step = 0
    while simulation_app.is_running():
        if act is None:
            actions = torch.zeros(env.num_envs, env.num_finger_dofs, device=device)
        else:
            obs = {
                "obs": obs_dict["policy"],
                "priv_info": obs_dict.get("critic"),
                "proprio_hist": obs_dict.get("proprio_hist"),
            }
            actions = torch.clamp(act(obs), -1.0, 1.0)
        obs_dict, _, _, _, extras = env.step(actions)
        step += 1
        if step % 50 == 0:
            net = extras.get("eval_net_turns")
            tilt = extras.get("eval_screwdriver_upright_norm")
            print(
                f"step {step} | net turns {net.mean().item() if net is not None else float('nan'):.2f} "
                f"| tilt {tilt.mean().item() if tilt is not None else float('nan'):.3f} rad"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
