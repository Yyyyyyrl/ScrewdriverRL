# ScrewdriverRL

Isaac Lab training environment for **continuous in-hand screwdriver rotation** with a dexterous robot hand.  The policy must spin the screwdriver around its own axial direction while keeping it upright, using fingertip contacts only.

Training follows a two-stage **RMA** (Rapid Motor Adaptation) recipe on top of RL-Games PPO:

- **Stage 1 — Teacher.** An asymmetric actor-critic where the actor sees the policy observation and the critic additionally sees a privileged observation (exact screwdriver pose/velocity, friction, fingertip distances). The deployment policy is the actor alone.
- **Stage 2 — Adaptation.** A small temporal-conv network learns to predict the privileged observation from a history of proprioception, so the policy can run without privileged sensors. See [`docs/2-stage-training.md`](docs/2-stage-training.md).

The task logic is **hand-agnostic** (shared base env + reward), and each hand is a thin subclass. Two hands are configured today: the **Allegro** (right) hand and the **Linker Hand L20** (left).

---

## Configured tasks (`--task <id>`)

Every entry script (`train.py`, `play.py`, `eval.py`, `calibrate_pad.py`, `render_posture.py`) takes `--task <id>`. Registered ids:

| `--task` id | Hand | Fingers | Action | Obs (policy / privileged) | Turn dir | Self-collision |
|---|---|---|---|---|---|---|
| `Isaac-Allegro-Screwdriver-Rotation-Direct-v0` | Allegro (right) | index, middle, thumb (3) | 12 | 27 / 17 | −z (CCW from above) | on |
| `Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0` | Linker Hand L20 (left) | index, middle, ring, pinky, thumb (5) | 16 | 35 / 19 | +z (mirror of Allegro) | on (pair-filtered) |

`train.py`/`play.py`/`eval.py`/`calibrate_pad.py` **default to the Allegro task** if `--task` is omitted.

---

## Installation

```bash
# Activate your Isaac Lab conda environment (this repo's is `env_isaac`)
conda activate env_isaac

# Install this package in editable mode
cd ScrewdriverRL
pip install -e .
```

Environments register automatically when `screwdriver_rl.tasks` is imported (the entry scripts do this).

**Assets** (hand + screwdriver URDFs and meshes) are bundled under [assets/](assets/), so the project has no external file dependency. Default asset root is `<repo>/assets`; override with:

```bash
export SCREWDRIVER_RL_ASSET_ROOT=/path/to/your/assets
```

---

## Training

Both stages use `train.py`. Pick the hand with `--task` (see the table above).

### Stage 1 — Teacher (PPO with asymmetric critic)

```bash
# Allegro
python train.py --task Isaac-Allegro-Screwdriver-Rotation-Direct-v0 --stage 1 --num_envs 2048 --headless

# Linker Hand L20
python train.py --task Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0 --stage 1 --num_envs 2048 --headless

# Resume from a checkpoint
python train.py --task <id> --stage 1 --headless \
  --checkpoint runs/<id>/<run-name>/nn/<name>.pth
```

### Stage 2 — Adaptation (run after Stage 1 converges)

```bash
python train.py --task <id> --stage 2 --headless \
  --checkpoint runs/<id>/<run-name>/nn/<name>.pth
```

### Useful flags

| Flag | Default | Purpose |
|---|---|---|
| `--task ID` | Allegro | Which hand/task to run (see table). |
| `--stage {1,2}` | `1` | Teacher PPO (1) or adaptation network (2). |
| `--num_envs N` | task cfg (2048) | Parallel environments. |
| `--checkpoint PATH` | — | Resume Stage 1, or the frozen teacher for Stage 2 (required for Stage 2). |
| `--output DIR` | `runs/<task>` | Where checkpoints, tensorboard logs and videos go. |
| `--max_epochs N` | cfg (8000) | [Stage 1] Cap RL-Games epochs (smoke tests). |
| `--save_interval_steps N` | 2 M | [Stage 1] Env-step interval between checkpoints. |
| `--init_global_steps N` | 0 | [Stage 1] Seed the curriculum step counter so a resumed run starts in a later phase (the counter is process state, not saved in the checkpoint). |
| `--seed N` | 42 | RNG seed. |
| `--video` | off | Record training videos (`--video_interval` controls frequency). |
| `--adapt_iters` / `--adapt_rollout_steps` | 500 / 512 | [Stage 2] Adaptation training schedule. |

> **Minibatch note:** the agent config uses `minibatch_size: 8192`, which needs `num_envs × horizon_length(32) ≥ 8192` (i.e. `num_envs ≥ 256`). For smaller runs `train.py` shrinks the minibatch to fit.

> **Throughput:** env-step throughput scales with `--num_envs`. On a 32 GB GPU the Linker fits **~16384 envs (~15.5 GB)** with `convex_hull` colliders; use a high count for fast training (the default 2048 is conservative). Allegro scales similarly.

### Output layout

```
runs/<task>/                       # (override with --output DIR)
├── <run-name>/                    # Stage 1, timestamped by RL-Games
│   ├── nn/                        #   checkpoints (.pth)
│   └── summaries/                 #   tensorboard
└── stage2_nn/
    └── proprio_adapt.pth          # Stage 2 adaptation network
```

---

## Evaluation & tooling

All take `--task <id>` and run in `env_isaac`. Isaac boots in ~2–3 min; run headless when you don't need the viewport.

### `play.py` — viewport playback / quick stats
Loads a Stage 1 checkpoint and runs the deterministic policy.
```bash
python play.py --task <id> --checkpoint <path.pth> --num_envs 16            # viewport
python play.py --task <id> --checkpoint <path.pth> --num_envs 512 --headless
python play.py --task <id> --checkpoint <path.pth> --video --video_length 300
```
Key flags: `--eval_phase {final,none,<idx>}` (pin curriculum phase), `--no_domain_rand`, `--fixed_start`, `--output DIR`.

### `eval.py` — headless aggregate statistics
Runs many envs over full episodes and reports the **distribution** of training metrics (FwdVel, TiltNorm, ContactGate, PadGate, PadCos, net turns, per-episode success/fall), plus reward-validity probes.
```bash
# Faithful comparison vs the training log (DR on, final phase)
python eval.py --task <id> --checkpoint <path.pth> --num_envs 256

# Reward-validity stress test: real manipulation or free-spin coasting?
python eval.py --task <id> --checkpoint <path.pth> --no_domain_rand --rot_damping_scale 4.0
```
Key flags: `--stochastic`, `--no_pad_gate`, `--eval_phase`, `--fixed_start`, `--success_turns`.

### `calibrate_pad.py` — recover the fingertip pad-normal axis
Policy-independent: resets to the zero-action pregrasp and measures, per fingertip, the local-frame direction toward the handle (= the pad normal). Use the printed value for `fingertip_pad_axis_local` in the hand cfg. **Only valid once the pregrasp actually contacts the handle.**
```bash
python calibrate_pad.py --task <id> --num_envs 64 --headless
```

### `render_posture.py` — headless posture snapshots
Spawns the hand + screwdriver at the task's `init_state`, settles, and saves front/side/top/iso PNGs + a 2×2 composite (default into `post_render/`). Also prints a stability line (max joint velocity) — handy for checking `init_state.pos/rot/joint_pos` or actuator/collision changes without opening the viewport.
```bash
python render_posture.py --task <id> --headless
```
Key flags: `--out PREFIX`, `--self_collisions {cfg,on,off}`, `--cam_dist`, `--settle`, `--width/--height`.

---

## Curriculum

Training proceeds through three automatic phases based on the global step counter (shared across hands; per-phase values live in
`screwdriver_rl/tasks/base/screwdriver_rotation_env_cfg.py`):

| Phase | Steps | Focus | Turn wt | Contact gate | Screw load | Proximal pen. | Near wt |
|---|---|---|---|---|---|---|---|
| **0 — Reach & grasp** | 0 → 40 M | Surround the handle with fingertips | 120 | 0.10 m | 0× | off | 0.8 |
| **1 — Contact rotation** | 40 M → 90 M | Turn while keeping fingertip contact | 180 | 0.07 m | 0.5× | 3 | 0.3 |
| **2 — Sustained rotation** | 90 M → | Fingertip-only style, long episodes | 200 | 0.05 m | 1.0× | 5 | 0.15 |

Phase transitions print a `═══` banner. (A resumed run restarts the curriculum at Phase 0 unless you pass `--init_global_steps`.)

---

## Reward design

All components and weights are documented inline in
`screwdriver_rl/tasks/base/screwdriver_rotation_env_cfg.py` (shared) and the maths lives in `screwdriver_rl/core/rewards.py` (pure-torch, unit-tested).

Failure modes and countermeasures:

- **Flick-and-coast** → contact gate requires ≥2 fingertips within `contact_distance` AND moving ≥ `min_fingertip_speed`; screwdriver rotation damping stops it within one policy step.
- **Tilt-and-scrape** → multiplicative Gaussian upright gate `exp(-(tilt/0.25)²)` suppresses the turn reward (an additive penalty can't beat a large positive term).
- **Oscillation** → reverse penalty (220) slightly exceeds the turn reward and shares the same contact gate, so contact only pays when moving forward.
- **Wrong-surface contact** → (a) per-step proximal-link proximity penalty (Phase 1+); (b) a **pad-facing** factor that only credits a contact whose fingertip pad faces the handle (ramped lenient→strict by the curriculum).
- **Free-spin reward hacking** → a curriculum-ramped resistive **screw load torque** on the rotation joint, so feeble nudging doesn't transfer; probe it with `eval.py --rot_damping_scale`.

---

## Self-collision (sim-to-real)

Self-collision is **enabled** for both hands so the policy can't learn finger configurations that pass through each other / the palm.

For the Linker, the per-link convex-hull collision shapes are *inflated* relative to the real geometry, so non-adjacent links (palm↔proximals, sibling metacarpals, the thumb's nested CMC chain) falsely overlap at the grasp pose and destabilise the very light joints. This is fixed by **collision-pair filtering**: a `SELF_COLLISION_FILTER_PAIRS` list (applied via `pxr` `FilteredPairsAPI` in `_setup_scene`, before env cloning) that excludes only those physically-impossible pairs while **keeping** the deployment-critical collisions (fingertip↔fingertip, thumb↔fingers, finger-crossing). Verified stable (max |joint vel| ≈ 0.01 rad/s) across all envs.

> **Collision geometry / performance.** The hand uses cheap `convex_hull` colliders (one shape per link). An earlier `convex_decomposition` attempt fit the geometry tighter but was **~10× slower per step and ~2.6× heavier in memory** for no stability benefit once pair-filtering was in place — so it was dropped. With `convex_hull`, the Linker runs comfortably at high env counts (e.g. `--num_envs 16384` ≈ 15.5 GB), which dominates throughput.

---

## Monitoring

The terminal logger prints a structured summary every ~500 environment steps (FwdTurns/NetTurns/OscRatio, TiltNorm/UprightGate, ContactGate/PadGate/PadCos/MinTipDist/FwdVel, and a reward breakdown). Inline warnings flag failure modes:
`⚠ OSCILLATION` (OscRatio > 0.35), `⚠ TILT` (TiltNorm > 0.4 rad), `⚠ NO-CONTACT` (BinaryGate < 0.15), `⚠ BACKWARD` (RevVel > FwdVel).

---

## Adding a new hand

The env is hand-agnostic; a new hand is a thin subclass of `ScrewdriverRotationEnv` + a cfg subclass of `ScrewdriverRotationEnvCfg`.

1. Create `screwdriver_rl/tasks/<hand>/` mirroring `linker_l20/`.
2. **Env subclass** — set the class-attribute name maps:
   ```python
   from screwdriver_rl.tasks.base.screwdriver_rotation_env import ScrewdriverRotationEnv

   class MyHandEnv(ScrewdriverRotationEnv):
       FINGERTIP_BODY_NAMES = {"index": "index_tip", ...}          # finger -> fingertip body
       PROXIMAL_BODY_PATTERNS = [r"^palm$", r"^.*_proximal$", ...] # non-fingertip links to penalise
       FINGER_JOINT_NAMES = {"index": ("j0", "j1", "j2"), ...}     # finger -> INDEPENDENT joint names
       COUPLED_JOINTS = {"index_dip": ("index_pip", 0.89, 0.0)}    # mimic followers (optional)
       SELF_COLLISION_FILTER_PAIRS = [("palm", "index_proximal")]  # phantom-overlap pairs (optional)
   ```
3. **Cfg subclass** — set `robot_cfg` (your URDF `ArticulationCfg`), `fingers`, `pregrasp_positions`, the gym `observation_space`/`action_space`, `privileged_obs_dim`, `history_obs_dim`, `fingertip_pad_axis_local`, and (if a mirror hand) `turn_direction`.
4. **Register** the gym id in `screwdriver_rl/tasks/<hand>/__init__.py` and add `agents/rl_games_ppo_cfg.yaml`.
5. **Import** the package in `screwdriver_rl/tasks/__init__.py`.
6. Verify dims/joints with `tests/`, then `render_posture.py` / `calibrate_pad.py` to tune the pregrasp and pad axis.

---

## Repository structure

```
ScrewdriverRL/
├── train.py                # Two-stage training entry point (--stage 1|2, --task)
├── play.py                 # Viewport playback / quick stats
├── eval.py                 # Headless aggregate-statistics evaluator
├── calibrate_pad.py        # Recover fingertip pad-normal axis from the pregrasp
├── render_posture.py       # Headless multi-view posture snapshots (-> post_render/)
├── pyproject.toml
├── docs/
│   └── 2-stage-training.md
├── tests/                  # CPU-only (no Isaac Sim)
│   ├── test_rewards.py     #   reward/geometry/quaternion primitives
│   ├── test_algo.py        #   Stage 2 adaptation network + trainer
│   └── test_linker_cfg.py  #   Linker URDF joint/mimic inventory guard
└── screwdriver_rl/
    ├── core/rewards.py             # Pure-torch reward / geometry / quaternion primitives
    ├── algos/proprio_adapt.py      # Stage 2 adaptation network + trainer
    ├── utils/logging.py            # Formatted terminal logger
    └── tasks/
        ├── __init__.py             # Imports hand sub-packages (triggers gym.register)
        ├── base/                   # Hand-agnostic shared env + cfg (no gym id)
        │   ├── screwdriver_rotation_env.py
        │   └── screwdriver_rotation_env_cfg.py
        ├── allegro/                # Allegro hand task (thin subclass + cfg + agents)
        │   ├── __init__.py  screwdriver_rotation_env.py  screwdriver_rotation_env_cfg.py
        │   └── agents/rl_games_ppo_cfg.yaml
        └── linker_l20/             # Linker Hand L20 task (thin subclass + cfg + agents)
            ├── __init__.py  screwdriver_rotation_env.py  screwdriver_rotation_env_cfg.py
            └── agents/rl_games_ppo_cfg.yaml
```

---

## Tests

Pure-PyTorch / pure-Python — **no Isaac Sim required**:

```bash
python -m pytest tests/ -q
```

---

## Linker Hand L20 — bring-up status

The Linker task is wired, stable, and trainable. A few items still benefit from viewport tuning before/while training (all noted in `screwdriver_rl/tasks/linker_l20/screwdriver_rotation_env_cfg.py`):

- **`fingertip_pad_axis_local`** is a placeholder `(0,0,1)` — recover the true axis with `calibrate_pad.py` once the pregrasp forms a proper contact grasp.
- **Pregrasp / hand pose** (`init_state.pos/rot`, `pregrasp_positions`) are seeded from the MFR reference and tuned so the screwdriver stays upright at reset; refine the thumb opposition visually with `render_posture.py` if training struggles to close the grasp.
- The 5 mimic distal joints (`*_dip`, `thumb_ip`) are driven via `COUPLED_JOINTS`; per-phase contact distances are inherited from the Allegro curriculum and may want re-tuning for the Linker tip geometry.
