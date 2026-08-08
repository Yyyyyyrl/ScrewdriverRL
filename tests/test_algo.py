"""Tests for the Stage-2 adaptation algo + the deployable (HORA-faithful) bundle.

No Isaac Sim.  Covers:
  - the adaptation network shapes + supervised fit;
  - the legacy ``ProprioAdaptTrainer`` loop (priv-vector target) via a fake env;
  - the HORA-faithful latent path: trainer writes a loadable ``deploy.pth`` whose
    actor reconstructs the rl_games actor exactly, and ``DeployPolicy`` runs it
    proprioception-only;
  - the deploy actor canonicaliser + latent forward;
  - the 16→20 LinkerHand SDK map (bounds + round-trip).

The fake env mimics the ``DirectRLEnv`` interface the trainer consumes:
``reset() -> (obs_dict, info)`` and
``step(a) -> (obs_dict, rew, terminated, truncated, info)``.

rl_games is required only for the two network tests (auto-skipped if absent).

Run:  python tests/test_algo.py   (or  python -m pytest tests/ -q)
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.algos.proprio_adapt import (  # noqa: E402
    AdaptTrainCfg,
    ProprioAdaptNet,
    ProprioAdaptTrainer,
    _held_out_split_indices,
)
from screwdriver_rl.deploy.policy import (  # noqa: E402
    DeployActor,
    DeployPolicy,
    canonicalize_actor_state,
)
from screwdriver_rl.deploy import linker_sdk_map as sdkmap  # noqa: E402
from screwdriver_rl.deploy.codecs import mounted_linker_g20_codec_spec  # noqa: E402

DEVICE = "cpu"

try:
    from rl_games.algos_torch import model_builder  # noqa: E402
    from screwdriver_rl.algos.latent_network import (  # noqa: E402
        LATENT_NETWORK_NAME,
        register_latent_network,
    )
    _HAS_RLGAMES = True
except Exception:  # pragma: no cover - rl_games not installed
    _HAS_RLGAMES = False


class FakeEnv:
    """Minimal stand-in for the screwdriver env's gym interface.

    Produces the three observation groups the trainer reads (``policy``,
    ``critic``, ``proprio_hist``).  The privileged ``critic`` vector is a fixed
    linear function of the latest proprio frame (plus small noise) so the
    adaptation network has a learnable signal rather than pure noise.
    """

    def __init__(self, num_envs=16, policy_dim=27, priv_dim=17, hist_len=30,
                 frame_dim=24, act_dim=12, device=DEVICE):
        self.num_envs = num_envs
        self.device = device
        self._policy_dim = policy_dim
        self._priv_dim = priv_dim
        self._hist_len = hist_len
        self._frame_dim = frame_dim
        self._act_dim = act_dim
        self.reset_calls = 0
        # Fixed (unknown to the net) mapping latest-frame -> privileged obs.
        self._proj = torch.randn(frame_dim, priv_dim, device=device)

    def _obs(self):
        hist = torch.randn(self.num_envs, self._hist_len, self._frame_dim, device=self.device)
        critic = hist[:, -1] @ self._proj + 0.01 * torch.randn(self.num_envs, self._priv_dim, device=self.device)
        return {
            "policy": torch.randn(self.num_envs, self._policy_dim, device=self.device),
            "critic": critic,
            "proprio_hist": hist,
        }

    def reset(self):
        self.reset_calls += 1
        return self._obs(), {}

    def step(self, actions):
        rew = -actions.pow(2).sum(dim=-1)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(terminated)
        return self._obs(), rew, terminated, truncated, {}


# --------------------------------------------------------------------------- #
# Adaptation network
# --------------------------------------------------------------------------- #

def test_net_output_shapes():
    net = ProprioAdaptNet(frame_dim=24, hist_len=30, out_dim=17)
    out = net(torch.randn(8, 30, 24))
    assert out.shape == (8, 17), f"got {tuple(out.shape)}"
    # Custom dims also resolve the conv-flatten size dynamically.
    net2 = ProprioAdaptNet(frame_dim=12, hist_len=40, out_dim=9)
    assert net2(torch.randn(4, 40, 12)).shape == (4, 9)


def test_net_overfits_supervised_batch():
    """The network must be able to reduce MSE on a fixed (hist -> priv) batch."""
    torch.manual_seed(0)
    net = ProprioAdaptNet(frame_dim=24, hist_len=30, out_dim=17)
    proj = torch.randn(24, 17)
    hist = torch.randn(64, 30, 24)
    target = hist[:, -1] @ proj
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.mse_loss

    init_loss = loss_fn(net(hist), target).item()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(net(hist), target)
        loss.backward()
        opt.step()
    final_loss = loss_fn(net(hist), target).item()
    assert final_loss < 0.5 * init_loss, f"loss did not drop: {init_loss:.4f} -> {final_loss:.4f}"


def test_continuous_rollouts_reset_only_once_across_chunks():
    env = FakeEnv(num_envs=4)
    actor = lambda obs: torch.zeros(obs.shape[0], 12, device=obs.device)  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        trainer = ProprioAdaptTrainer(
            env=env, stage1_actor_fn=actor,
            cfg=AdaptTrainCfg(
                rollout_steps=2, num_iters=2, batch_size=8,
                num_epochs_per_iter=1, continuous_rollouts=True,
            ),
            out_dir=tmp, device=DEVICE, priv_obs_dim=17,
            frame_dim=24, hist_len=30,
        )
        first_h, _ = trainer._collect(1)
        second_h, _ = trainer._collect(2)
    assert env.reset_calls == 1
    assert first_h.shape == second_h.shape == (8, 30, 24)


def test_environment_group_held_out_split_has_no_trajectory_leakage():
    torch.manual_seed(7)
    group_ids = torch.arange(20).repeat(4)
    train_idx, val_idx, diagnostics = _held_out_split_indices(
        group_ids.numel(), DEVICE, group_ids
    )

    train_groups = set(group_ids[train_idx].tolist())
    val_groups = set(group_ids[val_idx].tolist())
    assert train_groups.isdisjoint(val_groups)
    assert train_groups | val_groups == set(range(20))
    assert diagnostics == {
        "validation_split": "environment_group",
        "validation_group_count": 2,
        "train_group_count": 18,
    }


def test_trainer_runs_and_saves():
    """Legacy mode (no deploy_meta / no latent): adapter-only checkpoint."""
    env = FakeEnv()
    actor = lambda obs: torch.zeros(obs.shape[0], 12, device=obs.device)  # noqa: E731
    cfg = AdaptTrainCfg(
        rollout_steps=4, num_iters=3, batch_size=32,
        num_epochs_per_iter=2, log_interval=100,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = ProprioAdaptTrainer(
            env=env, stage1_actor_fn=actor, cfg=cfg, out_dir=tmp, device=DEVICE,
            priv_obs_dim=17, frame_dim=24, hist_len=30,
        )
        ckpt = trainer.train()
        assert Path(ckpt).exists(), "trainer did not save a checkpoint"
        # No deploy_meta -> no deploy.pth in legacy mode.
        assert not (Path(tmp) / "deploy.pth").exists()
        state = torch.load(ckpt, map_location=DEVICE)
        net = ProprioAdaptNet(frame_dim=24, hist_len=30, out_dim=17)
        net.load_state_dict(state["net"])


def test_trainer_resumes_from_saved_global_iteration():
    """A weights-only checkpoint resumes at its global iteration, not at one."""
    env = FakeEnv(num_envs=4)
    actor = lambda obs: torch.zeros(obs.shape[0], 12, device=obs.device)  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        first = ProprioAdaptTrainer(
            env=env,
            stage1_actor_fn=actor,
            cfg=AdaptTrainCfg(
                rollout_steps=2, num_iters=2, batch_size=8, num_epochs_per_iter=1,
            ),
            out_dir=tmp, device=DEVICE,
            priv_obs_dim=17, frame_dim=24, hist_len=30,
        ).train()
        assert torch.load(first, map_location=DEVICE)["iter"] == 2

        resumed = ProprioAdaptTrainer(
            env=env,
            stage1_actor_fn=actor,
            cfg=AdaptTrainCfg(
                rollout_steps=2, num_iters=3, batch_size=8, num_epochs_per_iter=1,
                resume_checkpoint=first,
            ),
            out_dir=tmp, device=DEVICE,
            priv_obs_dim=17, frame_dim=24, hist_len=30,
        ).train()
        assert torch.load(resumed, map_location=DEVICE)["iter"] == 3


# --------------------------------------------------------------------------- #
# Deploy: canonicaliser + latent actor + bundle round-trip
# --------------------------------------------------------------------------- #

def test_canonicalize_drops_envmlp_and_slices_normalizer():
    """The canonicaliser keeps actor_mlp/mu + a proprio-sliced normaliser and
    drops env_mlp / critic / value heads."""
    state = {
        "a2c_network.actor_mlp.0.weight": torch.randn(8, 40),
        "a2c_network.mu.weight": torch.randn(16, 8),
        "a2c_network.env_mlp.0.weight": torch.randn(64, 19),   # must be dropped
        "a2c_network.value.weight": torch.randn(1, 8),         # must be dropped
        "a2c_network.sigma": torch.zeros(16),                  # must be dropped
        "running_mean_std.running_mean": torch.randn(51),
        "running_mean_std.running_var": torch.rand(51) + 0.5,
        "running_mean_std.count": torch.tensor(10.0),
        "value_mean_std.running_mean": torch.randn(1),         # must be dropped
    }
    out = canonicalize_actor_state(state, proprio_dim=32)
    assert "actor_mlp.0.weight" in out and "mu.weight" in out
    assert not any("env_mlp" in k or "value" in k or "sigma" in k for k in out)
    assert out["running_mean_std.running_mean"].shape == (32,)
    assert out["running_mean_std.running_var"].shape == (32,)
    assert out["running_mean_std.count"].item() == 10.0


def test_deploy_actor_latent_forward():
    arch = {"mlp_units": [64, 32], "activation": "elu", "obs_dim": 51,
            "proprio_dim": 32, "latent_dim": 8, "action_dim": 16,
            "normalize_input": True, "clip_obs": 5.0}
    actor = DeployActor(arch).eval()
    proprio = torch.randn(5, 32)
    latent = torch.randn(5, 8)
    out = actor(proprio, latent)
    assert out.shape == (5, 16)
    # actor_mlp must accept proprio + latent.
    first = [m for m in actor.actor_mlp if isinstance(m, torch.nn.Linear)][0]
    assert first.in_features == 32 + 8


def _make_latent_bundle(proprio_dim=32, latent_dim=8, action_dim=16, hist_len=30):
    """A synthetic deploy.pth bundle (no rl_games): the 'actor' state is taken
    from a fresh DeployActor so it loads 1:1."""
    arch = {"mlp_units": [64, 32], "activation": "elu", "obs_dim": proprio_dim + 19,
            "proprio_dim": proprio_dim, "latent_dim": latent_dim, "action_dim": action_dim,
            "normalize_input": True, "clip_obs": 5.0}
    actor_state = DeployActor(arch).state_dict()
    adapter = ProprioAdaptNet(frame_dim=proprio_dim, hist_len=hist_len, out_dim=latent_dim)
    return {
        "actor": actor_state,
        "actor_arch": arch,
        "adapter": adapter.state_dict(),
        "net_dims": {"frame_dim": proprio_dim, "hist_len": hist_len, "out_dim": latent_dim},
        "config": {"task": "smoke", "n_finger": action_dim, "action_delta_scale": 0.05,
                   "finger_lower": [-1.0] * action_dim, "finger_upper": [1.0] * action_dim,
                   "home_targets": [0.0] * action_dim, "prop_hist_len": hist_len,
                   "history_obs_dim": proprio_dim, "privileged_obs_dim": 19,
                   "proprio_codec": mounted_linker_g20_codec_spec(hist_len).as_dict()},
    }


def test_deploy_policy_roundtrip_latent():
    bundle = _make_latent_bundle()
    pol = DeployPolicy(bundle, device=DEVICE)
    fq = torch.zeros(16)
    pol.reset(fq, fq)
    t = pol.act(fq)
    assert t.shape == (1, 16)
    assert bool((t >= -1).all() and (t <= 1).all()), "targets out of [lo,hi]"
    # Determinism: same seed of state -> same action.
    pol.reset(fq, fq); a = pol.act(fq)
    pol.reset(fq, fq); b = pol.act(fq)
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# SDK map (16 policy joints -> 20 LinkerHand command slots, 0..255)
# --------------------------------------------------------------------------- #

def test_sdk_map_bounds_and_roundtrip():
    # Forward: 16 rad -> 20 ints in [0,255].
    rng = sdkmap.joints16_to_sdk_range([0.0] * 16)
    assert len(rng) == 20
    assert all(isinstance(v, int) and 0 <= v <= 255 for v in rng)
    # Inverse: 20 SDK values -> 16 finite joint targets.
    j = sdkmap.sdk_range_to_joints16([128] * 20)
    assert len(j) == 16 and all(torch.isfinite(torch.tensor(j)))
    # Round-trip from in-range joints (built via the inverse) is quantisation-only.
    r2 = sdkmap.joints16_to_sdk_range(j)
    j2 = sdkmap.sdk_range_to_joints16(r2)
    err = max(abs(a - b) for a, b in zip(j, j2))
    assert err < 0.1, f"SDK round-trip error too large: {err:.4f} rad"


# --------------------------------------------------------------------------- #
# Stage-2 latent path: trainer writes a deployable deploy.pth
# --------------------------------------------------------------------------- #

def test_trainer_latent_mode_writes_deploy():
    # Historical LinkerL20 widths plus the final force-free 22-D width.  The
    # bundle must record the right width, yet the deploy actor stays a fixed 40-D
    # [proprio(32), latent(K)] regardless (env_mlp is dropped at deploy).
    for V in (19, 21, 22):
        torch.manual_seed(0)
        P, K, A = 32, 4, 16
        env = FakeEnv(num_envs=16, policy_dim=P + V, priv_dim=V, hist_len=30,
                      frame_dim=P, act_dim=A)
        frozen_actor = lambda obs: torch.zeros(obs.shape[0], A, device=obs.device)  # noqa: E731
        W = torch.randn(P + V, K)
        teacher_latent_fn = lambda policy_obs, _W=W: torch.tanh(policy_obs @ _W)  # noqa: E731

        arch = {"mlp_units": [64, 32], "activation": "elu", "obs_dim": P + V,
                "proprio_dim": P, "latent_dim": K, "action_dim": A,
                "normalize_input": True, "clip_obs": 5.0}
        deploy_meta = {
            "actor": DeployActor(arch).state_dict(),
            "actor_arch": arch,
            "config": {"task": "smoke", "n_finger": A, "action_delta_scale": 0.05,
                       "finger_lower": [-1.0] * A, "finger_upper": [1.0] * A,
                       "home_targets": [0.0] * A, "prop_hist_len": 30,
                       "history_obs_dim": P, "privileged_obs_dim": V,
                       "proprio_codec": mounted_linker_g20_codec_spec(30).as_dict()},
        }
        cfg = AdaptTrainCfg(rollout_steps=4, num_iters=3, batch_size=32, num_epochs_per_iter=2)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ProprioAdaptTrainer(
                env=env, stage1_actor_fn=frozen_actor, cfg=cfg, out_dir=tmp, device=DEVICE,
                priv_obs_dim=V, frame_dim=P, hist_len=30,
                deploy_meta=deploy_meta, latent_dim=K, teacher_latent_fn=teacher_latent_fn,
            )
            trainer.train()
            deploy_path = Path(tmp) / "deploy.pth"
            assert deploy_path.exists(), "Stage-2 latent mode did not write deploy.pth"
            bundle = torch.load(deploy_path, map_location=DEVICE)
            assert bundle["net_dims"]["out_dim"] == K, "adapter should regress the K-D latent"
            assert (Path(tmp) / "adaptation_validation.json").exists()
            validation = bundle["adaptation_validation"]
            assert validation["sample_count"] > 0
            assert validation["adapter_latent_mse"] >= 0.0
            assert validation["raw_privileged_mse"] >= 0.0
            assert len(validation["raw_privileged_mse_by_channel"]) == V
            # Only the final force-free layout may carry semantic channel names.
            if V == 22:
                assert validation["screw_relative_position_mse"] >= 0.0
                assert validation["distance_contact_score_mse"] >= 0.0
                assert validation["raw_privileged_channel_layout"][
                    "distance_contact_scores"
                ] == [15, 20]
            else:
                assert "distance_contact_score_mse" not in validation
            assert "priv_probe" not in bundle, "diagnostic probe leaked into deploy bundle"
            # The bundle records the task's privileged width...
            assert bundle["config"]["privileged_obs_dim"] == V
            assert bundle["actor_arch"]["obs_dim"] == P + V
            # ...but the deployable actor is the same 40-D input for both widths.
            pol = DeployPolicy(bundle, device=DEVICE)
            assert pol.actor.proprio_dim + pol.actor.latent_dim == P + K
            pol.reset(torch.zeros(A), torch.zeros(A))
            out = pol.act(torch.zeros(A))
            assert out.shape == (1, A)


# --------------------------------------------------------------------------- #
# Custom rl_games latent network (skipped if rl_games is unavailable)
# --------------------------------------------------------------------------- #

def _build_latent_model(proprio=32, priv=19, K=8, act=16, mlp=(256, 128)):
    register_latent_network()
    params = {
        "model": {"name": "continuous_a2c_logstd"},
        "network": {"name": LATENT_NETWORK_NAME, "separate": False,
            "proprio_dim": proprio, "latent_dim": K, "priv_mlp_units": [256, 128],
            "space": {"continuous": {"mu_activation": "None", "sigma_activation": "None",
                "mu_init": {"name": "default"}, "sigma_init": {"name": "const_initializer", "val": 0},
                "fixed_sigma": True}},
            "mlp": {"units": list(mlp), "activation": "elu", "d2rl": False,
                    "initializer": {"name": "default"}, "regularizer": {"name": "None"}}}}
    model = model_builder.ModelBuilder().load(params)
    model = model.build({"actions_num": act, "input_shape": (proprio + priv,), "num_seqs": 1,
                         "value_size": 1, "normalize_input": True, "normalize_value": True})
    return model.eval()


def test_latent_network_forward_shapes():
    if not _HAS_RLGAMES:
        print("[SKIP] rl_games not available")
        return
    # Both LinkerL20 shapes: non-DR (priv 19 -> obs 51) and geometry-DR (priv 21 -> obs 53).
    for priv, obs_dim in [(19, 51), (21, 53)]:
        model = _build_latent_model(priv=priv)
        out = model({"obs": torch.randn(5, obs_dim), "is_train": False, "prev_actions": None})
        assert out["mus"].shape == (5, 16)
        assert out["values"].shape == (5, 1)
        first = [m for m in model.a2c_network.actor_mlp if isinstance(m, torch.nn.Linear)][0]
        assert first.in_features == 32 + 8, "actor_mlp must consume [proprio, latent]"
        # The privileged tail (and thus env_mlp input) is derived, not hardcoded.
        assert model.a2c_network.priv_dim == priv, f"priv_dim must be {priv} for obs {obs_dim}"


def test_deploy_actor_reconstructs_rlgames_actor():
    """The canonicalised deploy actor must reproduce the rl_games actor's
    deterministic mu exactly (the core deployability guarantee) — for BOTH the
    non-DR (obs 51) and geometry-DR (obs 53) privileged widths."""
    if not _HAS_RLGAMES:
        print("[SKIP] rl_games not available")
        return
    for OBS in (51, 53):  # priv 19 (non-DR) and priv 21 (geometry-DR)
        torch.manual_seed(0)
        P, K, A = 32, 8, 16
        model = _build_latent_model(proprio=P, priv=OBS - P, K=K, act=A)
        a2c = model.a2c_network
        with torch.no_grad():  # exercise a non-identity normaliser
            model.running_mean_std.running_mean.copy_(torch.randn(OBS) * 0.3)
            model.running_mean_std.running_var.copy_(torch.rand(OBS) * 0.5 + 0.5)

        obs = torch.randn(7, OBS)
        with torch.no_grad():
            ref_mu = model({"obs": obs, "is_train": False, "prev_actions": None})["mus"]

        arch = {"mlp_units": [256, 128], "activation": "elu", "obs_dim": OBS,
                "proprio_dim": P, "latent_dim": K, "action_dim": A,
                "normalize_input": True, "clip_obs": 5.0}
        deploy_actor = DeployActor(arch).eval()
        miss, _ = deploy_actor.load_state_dict(
            canonicalize_actor_state(model.state_dict(), proprio_dim=P), strict=False)
        assert not [m for m in miss if not m.startswith("running_mean_std")]

        with torch.no_grad():
            xn = model.norm_obs(obs)
            latent = torch.tanh(a2c.env_mlp(xn[:, P:]))
            dep_mu = deploy_actor(obs[:, :P], latent)
        err = (ref_mu - dep_mu).abs().max().item()
        assert err < 1e-4, f"deploy actor mismatch vs rl_games (obs={OBS}): {err:.3e}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    raise SystemExit(1 if failures else 0)
