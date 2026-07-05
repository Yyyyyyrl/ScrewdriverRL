"""Deploy-bundle contract tests: a synthetic ``deploy.pth`` (the exact schema
``train.py:_build_deploy_meta`` + ``proprio_adapt.py:_save_deploy`` write) must
load into ``DeployPolicy`` and behave like the training-side integrator.

No Isaac, no rl_games, no hardware.
Run:  python -m pytest tests/test_deploy_policy_bundle.py -q
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.algos.proprio_adapt import ProprioAdaptNet  # noqa: E402
from screwdriver_rl.deploy.policy import DeployActor, DeployPolicy  # noqa: E402
from screwdriver_rl.deploy import linker_sdk_map as sdkmap  # noqa: E402

DELTA = 0.05  # action_delta_scale used across the LinkerL20 task


def make_bundle(n_finger: int = 16, latent_dim: int = 8, hist_len: int = 30, seed: int = 0) -> dict:
    """A minimal, schema-faithful Stage-2 deploy bundle around PREGRASP_16."""
    torch.manual_seed(seed)
    proprio_dim = 2 * n_finger
    arch = {
        "mlp_units": [32, 32],
        "activation": "elu",
        "obs_dim": proprio_dim + 19,
        "proprio_dim": proprio_dim,
        "latent_dim": latent_dim,
        "action_dim": n_finger,
        "normalize_input": True,
        "clip_obs": 5.0,
    }
    actor = DeployActor(arch)
    adapter = ProprioAdaptNet(frame_dim=proprio_dim, hist_len=hist_len, out_dim=latent_dim)
    home = list(sdkmap.PREGRASP_16)[:n_finger]
    return {
        "actor": actor.state_dict(),
        "actor_arch": arch,
        "adapter": adapter.state_dict(),
        "net_dims": {"frame_dim": proprio_dim, "hist_len": hist_len, "out_dim": latent_dim},
        "config": {
            "task": "test-bundle",
            "n_finger": n_finger,
            "action_delta_scale": DELTA,
            "finger_lower": [h - 0.35 for h in home],
            "finger_upper": [h + 0.35 for h in home],
            "home_targets": home,
            "prop_hist_len": hist_len,
            "history_obs_dim": proprio_dim,
            "privileged_obs_dim": 19,
        },
    }


def test_load_from_mapping_and_file(tmp_path):
    bundle = make_bundle()
    pol_mem = DeployPolicy(bundle, device="cpu")
    path = tmp_path / "deploy.pth"
    torch.save(bundle, str(path))
    pol_file = DeployPolicy(str(path), device="cpu")
    q = list(sdkmap.PREGRASP_16)
    pol_mem.reset(q)
    pol_file.reset(q)
    assert torch.allclose(pol_mem.act(q), pol_file.act(q))


def test_act_shape_clamps_and_delta():
    pol = DeployPolicy(make_bundle(), device="cpu")
    q = list(sdkmap.PREGRASP_16)
    pol.reset(q)
    prev = pol.cur_targets.clone()
    for _ in range(50):
        t = pol.act(q)
        assert t.shape == (1, 16)
        assert bool((t >= pol.finger_lower - 1e-6).all())
        assert bool((t <= pol.finger_upper + 1e-6).all())
        step = (t - prev).abs().max().item()
        assert step <= DELTA + 1e-6, f"per-tick delta {step} exceeds action_delta_scale"
        prev = t.clone()
        q = t[0].tolist()  # perfect tracking


def test_reset_seeds_history_and_targets():
    pol = DeployPolicy(make_bundle(), device="cpu")
    q = [v + 0.05 for v in sdkmap.PREGRASP_16]
    pol.reset(q)
    home = pol.home_targets
    assert torch.allclose(pol.cur_targets, home)
    # Every history frame is [q, home].
    expect = torch.cat([torch.tensor(q).view(1, -1), home], dim=-1)
    for i in range(pol.hist_len):
        assert torch.allclose(pol.hist[0, i], expect[0])
    # First act after reset stays within one delta of home.
    t = pol.act(q)
    assert (t - home).abs().max().item() <= DELTA + 1e-6
