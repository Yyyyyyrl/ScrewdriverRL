"""Reset-quality / policy probe for Isaac-LinkerL20-Inhand-Rotation.

Rolls out ZERO ACTIONS (default) or a trained rl_games checkpoint over many
envs under training conditions and reports per-shape episode statistics:
completed-episode length distribution, fall vs timeout fraction, and the
"doomed reset" rate (episodes that end within 10 steps — i.e. cache states
that fall at reset before the policy can act).

Use the zero-action mode after (re)generating grasp caches: healthy caches
show a single-digit doomed%% and a large timeout fraction.  Use --checkpoint
to measure how a trained policy trades hold time for rotation speed
(mean rotate-reward/step; +0.5 is the per-step clip).

Usage:
    python tools/probe_inhand_resets.py --headless
    python tools/probe_inhand_resets.py --headless --checkpoint runs/<...>/nn/<ckpt>.pth
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="rl_games .pth to evaluate; omit for the zero-action reset-quality baseline.",
)
parser.add_argument("--num_envs", type=int, default=810)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument(
    "--no_domain_rand", action="store_true",
    help="Disable domain randomisation (default keeps it ON to match training).",
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = False
args.rl_device = getattr(args, "rl_device", None) or args.device or "cuda:0"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch
import yaml

import screwdriver_rl.tasks  # noqa: F401

try:
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

TASK = "Isaac-LinkerL20-Inhand-Rotation"


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 0
    if args.no_domain_rand:
        env_cfg.domain_rand.enabled = False
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    # Pin the FINAL curriculum phase (cf. play.py): a fresh env starts in
    # phase 1 where the rotation reward has weight 0, which would zero the
    # reported rotate-reward and soften termination-relevant reward terms.
    phases = getattr(base.cfg, "curriculum_phases", None)
    if phases:
        base._curriculum_phase = phases[-1]
        base._global_steps = int(phases[-1].step_start)

    player = None
    obses = None
    if args.checkpoint:
        import importlib

        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        try:
            from rl_games.common.algo_observer import IsaacAlgoObserver as Obs
        except ImportError:
            from isaaclab_tasks.utils.wrappers.rl_games import RlGamesAlgoObserver as Obs

        from screwdriver_rl.algos.latent_network import (
            LATENT_NETWORK_NAME,
            register_latent_network,
        )

        entry = gym.spec(TASK).kwargs["rl_games_cfg_entry_point"]
        mod_name, _, file_name = entry.partition(":")
        mod = importlib.import_module(mod_name)
        with open(os.path.join(os.path.dirname(mod.__file__), file_name)) as f:
            agent_cfg = yaml.safe_load(f)
        if agent_cfg["params"].get("network", {}).get("name") == LATENT_NETWORK_NAME:
            register_latent_network()
        wrapped = RlGamesVecEnvWrapper(env, args.rl_device, 5.0, 1.0, None, True)
        vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
        env_configurations.register(
            "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_: wrapped}
        )
        agent_cfg["params"]["config"]["num_actors"] = base.num_envs
        agent_cfg["params"]["config"]["device"] = args.rl_device
        agent_cfg["params"]["config"]["device_name"] = args.rl_device
        runner = Runner(Obs())
        runner.load(agent_cfg)
        runner.reset()
        player = runner.create_player()
        player.restore(args.checkpoint)
        obses = player.env_reset(player.env)
        player.get_batch_size(obses, 1)
    else:
        env.reset()

    base.episode_length_buf[:] = 0

    n = base.num_envs
    device = base.device
    ep_len = torch.zeros(n, dtype=torch.long, device=device)
    done_lens: list[torch.Tensor] = []
    done_falls: list[torch.Tensor] = []
    done_shapes: list[torch.Tensor] = []
    rot_sum = torch.zeros((), device=device)
    rot_steps = torch.zeros((), device=device)

    shape_idx = torch.tensor(
        base.cfg.object_asset_shape_idx, dtype=torch.long, device=device
    )
    n_assets = max(len(shape_idx), 1)
    shape = shape_idx[torch.arange(n, device=device) % n_assets]

    actions = torch.zeros((n, 16), dtype=torch.float32, device=device)
    for _ in range(args.steps):
        if player is not None:
            obs_t = obses.get("obs") if isinstance(obses, dict) else obses
            with torch.no_grad():
                act = player.get_action(obs_t, is_deterministic=True)
            obses, _, dones, _ = player.env_step(player.env, act)
            terminated = dones.to(device, dtype=torch.bool)
        else:
            _, _, term, trunc, _ = env.step(actions)
            terminated = term.to(device, dtype=torch.bool) | trunc.to(device, dtype=torch.bool)
        ep_len += 1
        # Fall vs timeout from the externally tracked length (episode_length_buf
        # is already reset by the time env.step returns for done envs).
        fall = terminated & (ep_len < base.max_episode_length - 1)
        rr = base.extras.get("eval_rotate_reward")
        if rr is not None:
            rot_sum += rr.to(device).sum()
            rot_steps += n
        if terminated.any():
            ids = torch.nonzero(terminated, as_tuple=False).squeeze(-1)
            done_lens.append(ep_len[ids].clone())
            done_falls.append(fall[ids].clone())
            done_shapes.append(shape[ids].clone())
            ep_len[ids] = 0

    lens = (torch.cat(done_lens).float() if done_lens else torch.zeros(0)).cpu()
    falls = (torch.cat(done_falls).float() if done_falls else torch.zeros(0)).cpu()
    shapes = (torch.cat(done_shapes) if done_shapes else torch.zeros(0, dtype=torch.long)).cpu()
    mode = "POLICY" if player is not None else "ZERO-ACTION"
    dr = "off" if args.no_domain_rand else "on"
    print(
        f"\n=== {mode} rollout: {args.steps} steps x {n} envs, "
        f"max_ep={int(base.max_episode_length)}, DR {dr} ==="
    )
    print(f"completed episodes: {int(lens.numel())}")
    for sid, name in ((0, "cylinder"), (1, "cube"), (2, "sphere"), (None, "ALL")):
        m = torch.ones_like(shapes, dtype=torch.bool) if sid is None else shapes == sid
        if int(m.sum()) == 0:
            continue
        ls, fs = lens[m], falls[m]
        q = torch.quantile(ls, torch.tensor([0.25, 0.5, 0.9]))
        doomed = (ls <= 10).float().mean()
        print(
            f"{name:9s} n={int(m.sum()):6d}  len mean={ls.mean():6.1f} "
            f"p25/med/p90={q[0]:5.0f}/{q[1]:5.0f}/{q[2]:5.0f}"
            f"  fall%={100 * fs.mean():5.1f}  doomed(<=10)%={100 * doomed:5.1f}"
        )
    if rot_steps > 0:
        print(f"mean rotate-reward/step (clip +-0.5): {(rot_sum / rot_steps).item():+.4f}")

    env.close()


if __name__ == "__main__":
    ok = False
    try:
        main()
        ok = True
    finally:
        import threading

        watchdog = threading.Timer(60.0, lambda: os._exit(0 if ok else 1))
        watchdog.daemon = True
        watchdog.start()
        simulation_app.close()
