# Stage 3 — Deployment (LinkerHand L20/G20)

This stage turns a trained two-stage policy into a hardware controller. Stage 2
now writes a self-contained **`deploy.pth`** that is deployable exactly like
HORA's `stage2_nn/best.pth` (`../allegro_inhand_rotation/`).

See also `docs/stage2-deployability-plan.md` (design rationale) and
`docs/2-stage-training.md` (Stages 1–2).

## HORA-faithful latent design (what changed and why)

HORA's actor never observes the object directly: it consumes proprioception plus
an 8-D **privileged latent** produced by an encoder `env_mlp`, and a Stage-2
adaptation module learns to reproduce that latent from proprioceptive history. So
the *deployable* policy is the frozen actor + the adapter — no privileged input.

ScrewdriverRL now mirrors this exactly. The LinkerL20 actor consumes

```
actor input = [ proprio(32) , latent(K=8) ]            # NOT the raw euler
proprio     = [ finger_q(16) , cur_targets(16) ]       # joint encoders + self-state
latent      = tanh(env_mlp(privileged(20/22)))         # learned bottleneck
privileged  = euler(3)+angvel(3)+rel_pos(3)+quat(4)+load(1)
              +contact_friction(1)+distance_contact_score(5)+geometry(0/2)
```

- **Stage 1** (rl_games PPO) trains the actor on the *true* latent
  `tanh(env_mlp(privileged))`. The encoder + concat live in a custom rl_games
  network (`screwdriver_rl/algos/latent_network.py`, the port of HORA's
  `ActorCritic._actor_critic`). The asymmetric critic (`central_value_config`)
  sees the full 20-D (22-D with geometry DR) force-free privileged state.
- **Stage 2** freezes the actor and trains `ProprioAdaptNet` to reproduce the
  **teacher latent** from the `(30,32)` proprio history (MSE). On-policy
  refinement (after a warmup) drives the frozen actor with the *predicted* latent
  so the collected history matches deployment. By default it runs under the
  **final curriculum phase** (the deployment regime — strict upright termination,
  full episode length); use `--stage2_phase {final,none,<idx>}` to change this. The
  per-env dynamics the adapter must infer come from domain randomization, which is
  phase-independent, so no phase *mixing* is needed.
- **Deploy** runs the frozen actor on the adapter's predicted latent — **pure
  proprioceptive RMA**, no privileged state, no external orientation sensor.

The actor MLP + `mu` head are byte-identical across Stage 1 / Stage 2 / deploy;
only the source of `latent` changes (`env_mlp(priv)` → `adapter(history)`).

> The env exposes this via `cfg.latent_conditioned = True` (LinkerL20 default).
> Legacy tasks (`= False`) keep the old `[finger_q, cur_targets, euler]` actor obs.

## The deployable bundle: `stage2_nn/deploy.pth`

Written by `ProprioAdaptTrainer._save_deploy` (alongside `proprio_adapt.pth`).
Self-contained, HORA-parity:

| key          | contents                                                              |
|--------------|-----------------------------------------------------------------------|
| `actor`      | actor trunk + `mu` head + input normaliser, **sliced to the proprio block**, re-keyed for the env-free `DeployActor` |
| `actor_arch` | `mlp_units`, `activation`, `proprio_dim`, `latent_dim`, `action_dim`, `normalize_input`, `clip_obs` |
| `adapter`    | `ProprioAdaptNet` weights (history → `latent_dim`-D latent)           |
| `net_dims`   | `frame_dim` (32), `hist_len` (30), `out_dim` (= `latent_dim`)         |
| `config`     | action/history dimensions, explicit `proprio_codec`, `observation_semantics_version`, joint bounds/home targets, privileged width, and task |
| `adaptation_validation` | held-out latent MSE plus diagnostic raw privileged-channel errors; no diagnostic probe weights |

`train.py:_build_deploy_meta` extracts the actor + normaliser from the restored
rl_games `player.model`; `deploy/policy.py:canonicalize_actor_state(state,
proprio_dim)` re-keys them and slices the obs normaliser to the proprio block
(the latent is concatenated *after* normalisation, matching Stage 1). `env_mlp`
is intentionally **not** in the bundle — deploy gets the latent from the adapter.

## Inference contract: `DeployPolicy`

`screwdriver_rl/deploy/policy.py` — env-free (no Isaac, no rl_games):

```python
from screwdriver_rl.deploy.policy import DeployPolicy
pol = DeployPolicy("runs/<task>/stage2_nn/deploy.pth", device="cpu")
pol.reset(finger_q, effective_target)  # seed from acknowledged hardware target
targets = pol.act(finger_q)    # 16 absolute joint targets (radians)
```

Each `act` (pure RMA):
1. push `[finger_q, cur_targets]` into the `(30,32)` history;
2. `latent = adapter(history)`  ← **the bridge** (no euler, no sensor);
3. `obs = normalize(clip(proprio))`; `action = clamp(mu(actor_mlp([obs, latent])), ±1)`;
4. `cur_targets = clamp(cur_targets + 0.05·action, lower, upper)`.

Never consumes privileged/simulation-only state — same property as HORA's
`act_inference({obs, proprio_hist})`.

## Hardware mapping: `linker_sdk_map.py`

Unchanged by the latent redesign (it maps 16 joint targets → 20 SDK command
slots, independent of the actor's internal obs). Reuses the real SDK at
`/home/user/linkerhand-ros-sdk` (`linker_range_arc.arc_to_range_left`), falling
back to a vendored copy of the constants when the SDK is absent (so tests run
standalone).

SDK 20-slot layout (left hand): `0-4` root-flex, `5-9` abduction, `10` thumb
rotation, `11-14` reserved, `15-19` finger-bend. The 5 mimic joints (dips,
thumb_ip) are not sent; the hand couples them mechanically (matching our
`COUPLED_JOINTS`).

> ⚠️ **Thumb mapping is provisional.** The 4 thumb DOFs → slots `{0,5,10,15}` and
> their sign/range need verification against the SDK URDF on hardware. They are
> isolated in `_OUR_JOINTS`; the four finger chains are well-determined. Always
> bring up a new hand with `--dry-run` first.

## Live node: `deploy_linker.py`

Ties it together (ROS/SDK imports are lazy). 10 Hz loop = read joint state →
`DeployPolicy.act` → `joints16_to_sdk_range` → publish/`finger_move`.

```bash
python -m screwdriver_rl.deploy.deploy_linker \
    --checkpoint runs/<task>/stage2_nn/deploy.pth --side left --transport ros --dry-run
python -m screwdriver_rl.deploy.deploy_linker \
    --checkpoint runs/<task>/stage2_nn/deploy.pth --transport can --dry-run
```

`--dry-run` computes/prints the 0..255 command without sending it. **Cannot be
tested without the physical hand.**

## Sim validation gate (run before hardware)

`eval.py --deploy_eval` runs the *exact* deploy-time inference in sim: it drives
the live actor trunk with the adapter's **predicted latent** (no privileged obs)
and reports the usual rotation-progress metrics. Compare against the oracle
(true-latent) run to size the sim-to-deploy gap.

```bash
# Oracle (true latent via env_mlp):
python eval.py --task <task> --checkpoint runs/<task>/<run>/nn/<stage1>.pth --num_envs 256
# Deploy gate (predicted latent, no privileged obs):
python eval.py --task <task> --checkpoint runs/<task>/<run>/nn/<stage1>.pth \
    --deploy_eval --adapter_checkpoint runs/<task>/stage2_nn/deploy.pth --num_envs 256
```

A small, modest degradation in NetTurns / fall-rate is the go/no-go signal. (Per
the repo's Isaac run/debug notes: PTY, sandbox off, watch the teardown-hang.)

## What is and isn't tested here

- **Unit-tested, no Isaac/ROS** (`tests/test_algo.py`): the custom latent network
  forward; **the deploy actor reproduces the rl_games actor's `mu` exactly**; the
  canonicaliser (drops `env_mlp`, slices the normaliser); `DeployPolicy` latent
  inference; Stage-2 latent mode writing a loadable `deploy.pth`; the SDK map
  (bounds + round-trip).
- **Requires Isaac**: the Stage-1 retrain, the Stage-2 run, and the
  `eval.py --deploy_eval` sim gate.
- **Requires the physical hand**: `deploy_linker.py` and confirming the
  provisional thumb mapping.
