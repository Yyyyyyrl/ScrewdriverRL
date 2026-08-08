"""Deploy-bundle contract tests: a synthetic ``deploy.pth`` (the exact schema
``train.py:_build_deploy_meta`` + ``proprio_adapt.py:_save_deploy`` write) must
load into ``DeployPolicy`` and behave like the training-side integrator.

No Isaac, no rl_games, no hardware.
Run:  python -m pytest tests/test_deploy_policy_bundle.py -q
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.algos.proprio_adapt import ProprioAdaptNet  # noqa: E402
from screwdriver_rl.deploy.policy import DeployActor, DeployPolicy, nominal_geometry_row_index  # noqa: E402
from screwdriver_rl.deploy.codecs import inhand_linker_g20_codec_spec, mounted_linker_g20_codec_spec  # noqa: E402
from screwdriver_rl.deploy import linker_sdk_map as sdkmap  # noqa: E402

DELTA = 0.05  # action_delta_scale used across the LinkerL20 task

def test_nominal_geometry_row_selects_scale_one_not_cyclic_row_zero():
    scales = torch.tensor(
        [[0.9375, 1.0], [1.0, 1.0], [1.0625, 1.0]]
    )
    assert nominal_geometry_row_index(scales, 3) == 1
    assert nominal_geometry_row_index(None, 3) == 0



def make_bundle(n_finger: int = 16, latent_dim: int = 8, hist_len: int = 30, seed: int = 0, free_object: bool = False) -> dict:
    """A minimal, schema-faithful Stage-2 deploy bundle around PREGRASP_16."""
    torch.manual_seed(seed)
    frame_dim = 2 * n_finger
    proprio_dim = frame_dim * (3 if free_object else 1)
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
    adapter = ProprioAdaptNet(frame_dim=frame_dim, hist_len=hist_len, out_dim=latent_dim)
    home = list(sdkmap.PREGRASP_16)[:n_finger]
    lower = [h - 0.35 for h in home]
    upper = [h + 0.35 for h in home]
    codec_spec = (
        inhand_linker_g20_codec_spec(lower, upper, hist_len)
        if free_object
        else mounted_linker_g20_codec_spec(hist_len)
    )
    return {
        "actor": actor.state_dict(),
        "actor_arch": arch,
        "adapter": adapter.state_dict(),
        "net_dims": {"frame_dim": frame_dim, "hist_len": hist_len, "out_dim": latent_dim},
        "config": {
            "task": "test-bundle",
            "n_finger": n_finger,
            "action_delta_scale": DELTA,
            "finger_lower": lower,
            "finger_upper": upper,
            "home_targets": home,
            "prop_hist_len": hist_len,
            "history_obs_dim": frame_dim,
            "privileged_obs_dim": 19,
            "observation_semantics_version": "test-force-free-v1",
            "proprio_codec": codec_spec.as_dict(),
        },
    }


def test_topdown_bundle_requires_nominal_geometry_and_safe_startup_metadata():
    bundle = make_bundle()
    bundle["config"]["task"] = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"

    with pytest.raises(RuntimeError, match="missing startup_reset_targets"):
        DeployPolicy(bundle, device="cpu")

    bundle["config"]["startup_reset_targets"] = list(
        bundle["config"]["home_targets"]
    )
    with pytest.raises(RuntimeError, match="missing deployment_geometry_scale"):
        DeployPolicy(bundle, device="cpu")

    bundle["config"]["deployment_geometry_scale"] = [0.9375, 1.0]
    with pytest.raises(RuntimeError, match="requires nominal geometry scale"):
        DeployPolicy(bundle, device="cpu")

    bundle["config"]["deployment_geometry_scale"] = [1.0, 1.0]
    policy = DeployPolicy(bundle, device="cpu")
    assert policy.cfg["deployment_geometry_scale"] == [1.0, 1.0]
    assert torch.allclose(policy.startup_reset_targets, policy.home_targets)


def test_load_from_mapping_and_file(tmp_path):
    bundle = make_bundle()
    pol_mem = DeployPolicy(bundle, device="cpu")
    path = tmp_path / "deploy.pth"
    torch.save(bundle, str(path))
    pol_file = DeployPolicy(str(path), device="cpu")
    q = list(sdkmap.PREGRASP_16)
    pol_mem.reset(q, q)
    pol_file.reset(q, q)
    assert torch.allclose(pol_mem.act(q), pol_file.act(q))


def test_act_shape_clamps_and_delta():
    pol = DeployPolicy(make_bundle(), device="cpu")
    q = list(sdkmap.PREGRASP_16)
    pol.reset(q, q)
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
    effective = [v + 0.10 for v in sdkmap.PREGRASP_16]
    pol.reset(q, effective)
    effective_t = torch.tensor(effective).view(1, -1)
    assert torch.allclose(pol.cur_targets, effective_t)
    # Every history frame uses the measured pose plus the acknowledged target.
    expect = torch.cat([torch.tensor(q).view(1, -1), effective_t], dim=-1)
    for i in range(pol.hist_len):
        assert torch.allclose(pol.hist[0, i], expect[0])
    # First act after reset stays within one delta of the effective target.
    t = pol.act(q)
    assert (t - effective_t).abs().max().item() <= DELTA + 1e-6
