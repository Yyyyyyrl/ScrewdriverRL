# Stage 2 Deployability — Implementation (LinkerL20)

> Mirrors the Notion task "Stage 2 Deployability Gap: HORA vs ScrewdriverRL".
> **Implemented as the HORA-faithful latent redesign** (2026-06-30), superseding
> the earlier euler-bridge sketch. Deployment doc: `docs/3-deployment.md`.

## Problem

Stage 2 previously saved only a standalone `ProprioAdaptNet` — an *estimator*
(`proprio_hist → privileged_obs`), not a controller — and at HEAD it was actually
broken (`train.py` passed a `deploy_meta=` the trainer didn't accept → `TypeError`).
HORA instead ships a self-contained `stage2_nn/best.pth` = `{model,
running_mean_std, sa_mean_std}` its deploy script loads directly. We close that
gap, the HORA-faithful way.

## Design: latent-conditioned actor + proprioceptive adapter

Make the LinkerL20 actor consume a **learned latent** instead of the raw euler —
exactly HORA's structure — so the deployable policy is the frozen actor + the
adapter that reproduces the latent proprioceptively (pure RMA, no sensor).

```
actor input = [ proprio(32) , latent(K=8) ]
latent      = tanh(env_mlp(privileged(19)))      # privileged = euler+angvel+rel_pos
                                                 #   +quat+friction+per-finger force
Stage 1: actor trained on the TRUE latent (custom rl_games network).
Stage 2: freeze actor; adapter(history 30x32) → latent, MSE to the teacher latent;
         on-policy refinement drives the frozen actor with the predicted latent.
Deploy : latent = adapter(history); action = mu(actor_mlp([norm(proprio), latent])).
```

The actor MLP + `mu` head are identical across Stage 1 / Stage 2 / deploy; only
the latent source changes (`env_mlp(priv)` → `adapter(history)`).

## What was built

1. **Env latent mode** (`cfg.latent_conditioned`, LinkerL20 default True):
   actor obs = `[finger_q, cur_targets, privileged(19)]` (= 51-D; raw euler
   dropped). `base/...env.py:_get_observations`, linker cfg `observation_space` +
   `__post_init__`. Obs noise applied only to the proprio block.

2. **Custom rl_games network** `screwdriver_rl/algos/latent_network.py`
   (`PrivLatentA2CBuilder`): subclasses `A2CBuilder`, encodes the privileged obs
   tail into a `latent_dim` bottleneck and feeds `[proprio, latent]` to the MLP.
   Registered in `train.py:_register_rl_games` / `eval.py`; YAML
   `network.name: priv_latent_actor_critic` + `proprio_dim/latent_dim/priv_mlp_units`.
   The asymmetric `central_value_config` critic is unchanged.

3. **Stage 2 trainer** (`screwdriver_rl/algos/proprio_adapt.py`): adapter
   `out_dim = latent_dim`; regresses the teacher latent (closures built from the
   live `player.model` in `run_stage2`); `AdaptTrainCfg.onpolicy_latent/_warmup/
   _ramp` on-policy refinement; writes `deploy.pth` (+ rolling `deploy_last.pth`).
   Fixes the HEAD `TypeError`.

4. **Deploy bundle** (`train.py:_build_deploy_meta` + `deploy/policy.py`):
   `canonicalize_actor_state(state, proprio_dim)` keeps actor_mlp/mu and slices
   the obs normaliser to the proprio block (drops `env_mlp`/critic/value).
   `DeployActor.forward(proprio, latent)` and `DeployPolicy.act(finger_q)` run the
   latent path (pure RMA).

5. **Sim gate** (`eval.py --deploy_eval`): drives the live actor trunk with the
   adapter's predicted latent vs the env_mlp oracle.

6. **SDK map + live node** (`linker_sdk_map.py`, `deploy_linker.py`): **unchanged**
   — independent of the actor's internal obs. Thumb mapping still PROVISIONAL.

7. **Tests** (`tests/test_algo.py`, no Isaac/ROS): custom network forward; **deploy
   actor reproduces the rl_games `mu` exactly**; canonicaliser; `DeployPolicy`
   latent inference; Stage-2 latent mode writes a loadable `deploy.pth`; SDK map.
   All pass.

## Verify / next

- CPU: `python -m pytest tests/test_algo.py -q` (or `python tests/test_algo.py`).
- Isaac (user's rig): **retrain Stage 1** with the latent network → Stage 2 →
  `deploy.pth`; then the `--deploy_eval` sim gate (predicted vs true latent) is the
  go/no-go before hardware. The new-architecture pipeline has **not** been run end
  to end in Isaac yet (env obs change + custom-net registration + central-value
  with the 51-D actor obs are validated only at the unit level).
- Hardware: `deploy_linker.py --dry-run`, then verify the provisional thumb mapping.

## Open item

The thumb 4-DOF → SDK slot pairing (`{0,5,10,15}`) and per-joint sign/range need
physical verification (one editable block in `linker_sdk_map.py:_OUR_JOINTS`).

## Alternatives (not taken)

- **Euler bridge** (keep the actor on raw euler; adapter predicts euler): lighter,
  no Stage-1 retrain, but diverges from HORA's latent-conditioned structure.
- **External orientation tracker** → inject a measured pose: not needed under pure
  RMA (chosen).
