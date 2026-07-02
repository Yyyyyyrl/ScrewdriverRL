"""In-sim contact-band / penetration / clamp gate for the per-bucket pregrasp variants.

Instantiates the geometry-DR LinkerL20 task with enough envs to cover all manifest
buckets, resets (running the per-bucket pregrasp seeding + hand-root length offset +
the compliant settle), holds the pregrasp for a few zero-action steps, then per
``(diameter,length)`` bucket checks that the grasp is:

  * **not penetrating** — signed fingertip-to-surface clearance >= -pen_thresh
    (``env.compute_surface_clearance()``, radius-aware per variant);
  * **in contact / not floating** — per-finger force on the screwdriver
    (``env._read_contact_forces()``) above force_thresh on enough fingers;
  * **not clamped at reset** — each seeded finger target sits inside the soft-limit
    target-clamp window (so the pregrasp isn't yanked on the first step);
  * **covering every bucket** — the env->variant histogram spans all buckets.

Run (needs a PTY + sandbox off; see the isaac-sim-run-debug notes):
    python tools/validate_pregrasp_buckets.py --headless --num_envs 96
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_TASK = "Isaac-LinkerL20-Screwdriver-Rotation-DR-Direct-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK, help="Geometry-DR task id to validate.")
parser.add_argument("--num_envs", type=int, default=96, help="Envs to spawn (>= num buckets, ideally many per bucket).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle_steps", type=int, default=30, help="Zero-action steps after reset before reading contact.")
parser.add_argument("--pen_thresh", type=float, default=0.002, help="Penetration tolerance [m]: fail if clearance < -pen_thresh.")
parser.add_argument("--float_thresh", type=float, default=0.02, help="Clearance [m] above which a no-force finger is 'floating'.")
parser.add_argument("--force_thresh", type=float, default=0.05, help="Per-finger force [N] above which a finger counts as in contact.")
parser.add_argument("--min_contacts", type=int, default=3, help="Min fingers per bucket that must be in contact to pass.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import screwdriver_rl.tasks  # noqa: F401,E402

try:
    from isaaclab_tasks.utils import parse_env_cfg
except ImportError:  # pragma: no cover - version fallbacks
    try:
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    except ImportError:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _bucket_label(manifest: dict, b: int) -> str:
    for v in manifest["variants"]:
        if int(v["bucket"]) == b:
            return f"d={v['diameter_scale']:.2f} L={v['length_scale']:.2f} (r={v['radius']:.3f}, len={v['length']:.3f})"
    return "?"


def main() -> bool:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    if hasattr(env_cfg, "randomize_obj_start"):
        env_cfg.randomize_obj_start = False  # deterministic handle angle for a clean read
    assert getattr(env_cfg.domain_rand, "randomize_geometry", False), (
        f"{args.task} does not enable geometry DR — no pregrasp buckets to validate."
    )

    manifest = json.loads((Path(env_cfg.screwdriver_variants_dir) / "manifest.json").read_text())

    env = None
    ok = True
    try:
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        env.reset(seed=args.seed)

        # Hold the pregrasp for a few steps so contacts/sensors settle.
        act = torch.zeros((base.num_envs, base.action_space.shape[-1]), device=base.device)
        for _ in range(args.settle_steps):
            env.step(act)

        bucket_idx = base._env_bucket_idx
        assert bucket_idx is not None, "env has no per-bucket assignment (geometry DR off?)"
        clearance = base.compute_surface_clearance()          # (N, nf) >0 float, <0 penetrate
        F_total, _F_body, _F_cap, wrong = base._read_contact_forces()  # (N, nf), (N,)
        fingers = list(base.fingers)
        nf = len(fingers)

        # --- Reset clamping check (real soft limits) ---
        soft = base.allegro.data.soft_joint_pos_limits[:, base._finger_joint_ids]  # (N, D, 2)
        margin = float(base.cfg.joint_target_margin)
        home = base._home_targets                              # (N, D) seeded per-bucket
        below = home < soft[..., 0] + margin - 1e-6
        above = home > soft[..., 1] - margin + 1e-6
        n_clamped = int((below | above).any(dim=0).sum())     # DOFs clamped in >=1 env

        num_buckets = int(manifest["num_buckets"])
        hist = torch.bincount(bucket_idx, minlength=num_buckets)
        missing = [b for b in range(num_buckets) if int(hist[b]) == 0]

        print(f"\n=== Pregrasp bucket validation: {args.task} (n={base.num_envs}) ===")
        print(f"histogram (per bucket): {hist.tolist()}")
        print(f"fingers: {fingers}\n")
        print(f"{'bkt':>3} {'n':>4}  {label_hdr()}  {'min_clear':>10} {'pen?':>5}  {'contacts':>9} {'meanF':>7}")

        for b in range(num_buckets):
            m = bucket_idx == b
            n = int(m.sum())
            if n == 0:
                print(f"{b:>3} {n:>4}  {_bucket_label(manifest, b):<34}  (no envs)")
                ok = False
                continue
            cl = clearance[m]                                 # (n, nf)
            f = F_total[m]                                    # (n, nf)
            min_clear = float(cl.min())
            penetrating = min_clear < -args.pen_thresh
            # per-finger in-contact fraction across this bucket's envs
            in_contact = (f > args.force_thresh).float().mean(dim=0)   # (nf,)
            n_contacts = int((in_contact > 0.5).sum())        # fingers in contact for a majority of envs
            # floating = far AND no force (per finger, majority of envs)
            floating = ((cl > args.float_thresh) & (f <= args.force_thresh)).float().mean(dim=0) > 0.5
            mean_f = float(f.mean())

            bucket_ok = (not penetrating) and (n_contacts >= args.min_contacts)
            ok = ok and bucket_ok
            flags = []
            if penetrating:
                flags.append("PENETRATION")
            if n_contacts < args.min_contacts:
                fl = [fingers[i] for i in range(nf) if bool(floating[i])]
                flags.append(f"LOW-CONTACT{('/float:' + ','.join(fl)) if fl else ''}")
            tag = "ok" if bucket_ok else "FAIL " + ";".join(flags)
            print(
                f"{b:>3} {n:>4}  {_bucket_label(manifest, b):<34}  "
                f"{min_clear:>10.4f} {str(penetrating):>5}  {n_contacts:>4}/{nf:>3} {mean_f:>7.3f}  {tag}"
            )

        print("\n--- summary ---")
        print(f"buckets missing from histogram: {missing or 'none'}")
        print(f"finger DOFs clamped at reset (soft-limit window): {n_clamped}")
        print(f"max wrong-surface (non-tip) force: {float(wrong.max()):.3f}")
        if missing:
            ok = False
        if n_clamped > 0:
            ok = False
        print(f"\nOVERALL ok: {ok}")
        return ok
    finally:
        # Close the env BEFORE app teardown (teardown can deadlock and swallow errors).
        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()


def label_hdr() -> str:
    return f"{'variant':<34}"


if __name__ == "__main__":
    result = False
    try:
        result = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(0 if result else 1)
