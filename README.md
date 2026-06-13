# ScrewdriverRL

Isaac Lab training environment for **continuous in-hand screwdriver rotation** with a dexterous robot hand.  The policy must spin the screwdriver around its own axial direction while keeping it upright, using fingertip contacts only.

Training follows a two-stage **RMA** (Rapid Motor Adaptation) recipe on top of RL-Games PPO:

- **Stage 1 — Teacher.** An asymmetric actor-critic where the actor sees the 27-D policy observation and the critic additionally sees a 17-D privileged observation (exact screwdriver pose/velocity, friction, fingertip distances). The deployment policy is the actor alone.
- **Stage 2 — Adaptation.** A small temporal-conv network learns to predict the privileged observation from a history of proprioception, so the policy can run without privileged sensors. See [`docs/2-stage-training.md`](docs/2-stage-training.md) for details.

> **Current support:** Allegro hand (4-finger, 3-finger configuration).  The architecture is designed to extend to other hands — see [Adding a new hand](#adding-a-new-hand).

---

## Task description

| Requirement | Implementation |
|---|---|
| Screwdriver stays roughly upright | Multiplicative upright gate kills turn reward at tilt > ~14°; additive upright cost penalises every step of tilt |
| Continuous rotation, one direction | HORA-style signed delta reward; reverse penalty slightly above forward reward |
| No oscillation (net ≈ forward turns) | Reverse penalty gated identically to forward reward; oscillation ratio logged |
| Fingertip contact only | Proximal-link proximity penalty (Phase 2+); fingertip contact gate on turn reward |
| No flick-and-coast | Contact gate requires ≥2 fingertips near handle AND moving; screwdriver damping 0.15 stops it in < 1 policy step |
| Realistic physics | Friction 1.5, rotation damping 0.15, tilt damping 0.001 |

---

## Installation

```bash
# Activate your Isaac Lab conda environment
conda activate env_isaaclab

# Install this package in editable mode
cd ScrewdriverRL
pip install -e .
```

The environments are registered automatically when `screwdriver_rl.tasks` is imported.  The `train.py` script does this for you.

**Assets** are loaded from the sibling `MFR_benchmark/MFR_benchmark/assets/` directory by default.  Override with:

```bash
export SCREWDRIVER_RL_ASSET_ROOT=/path/to/your/assets
```

---

## Training

### Stage 1 — Teacher (PPO with asymmetric critic)

```bash
# Phase 1→3 curriculum, 2048 envs, headless
python train.py --stage 1 --num_envs 2048 --headless

# Resume from a checkpoint
python train.py --stage 1 --headless \
  --checkpoint runs/Isaac-Allegro-Screwdriver-Rotation-Direct-v0/<run-name>/nn/allegro_screwdriver_rotation.pth
```

### Stage 2 — Adaptation (run after Stage 1 converges)

```bash
python train.py --stage 2 --headless \
  --checkpoint runs/Isaac-Allegro-Screwdriver-Rotation-Direct-v0/<run-name>/nn/allegro_screwdriver_rotation.pth
```

### Useful flags

| Flag | Default | Purpose |
|---|---|---|
| `--stage {1,2}` | `1` | Teacher PPO (1) or adaptation network (2). |
| `--num_envs N` | task cfg (2048) | Parallel environments. |
| `--checkpoint PATH` | — | Resume Stage 1, or the frozen teacher for Stage 2 (required). |
| `--output DIR` | `runs/<task>` | Where checkpoints, tensorboard logs and videos are written. |
| `--max_epochs N` | cfg (8000) | Cap RL-Games epochs (handy for smoke tests). |
| `--seed N` | `42` | RNG seed. |
| `--video` | off | Record training videos (`--video_interval` controls frequency). |

> **Note:** the production agent config uses `minibatch_size: 8192`, which needs `num_envs × horizon_length(32) ≥ 8192` (i.e. `num_envs ≥ 256`). For smaller runs `train.py` automatically shrinks the minibatch to fit.

### Output layout

By default everything is written under `runs/<task>/` (override with `--output DIR`):

```
<output>/
├── <run-name>/            # Stage 1, timestamped by RL-Games
│   ├── nn/                #   checkpoints (.pth)
│   └── summaries/         #   tensorboard
└── stage2_nn/
    └── proprio_adapt.pth  # Stage 2 adaptation network
```

---

## Evaluation / playback

`play.py` loads a Stage 1 checkpoint and runs the deterministic policy.

```bash
# Visual playback (opens the Isaac Sim viewport)
python play.py --checkpoint <path-to.pth> --num_envs 16

# Headless statistics over many episodes
python play.py --checkpoint <path-to.pth> --num_envs 512 --num_episodes 20 --headless

# Record an mp4
python play.py --checkpoint <path-to.pth> --video --video_length 300
```

`--output DIR` relocates the player logs and recorded videos (default `runs/<task>/`).

---

## Tests

Unit tests are pure PyTorch and run on CPU — **no Isaac Sim required**:

```bash
python -m pytest tests/ -q
```

- `tests/test_rewards.py` — reward, gate, distance and quaternion primitives in `screwdriver_rl/core/rewards.py`.
- `tests/test_algo.py` — the Stage 2 adaptation network and trainer.

---

## Curriculum

Training proceeds through three automatic phases based on `global_step_counter`:

| Phase | Steps | Description | Key changes |
|---|---|---|---|
| **1 — Reach & grasp** | 0 → 15 M | Learn to approach and surround the handle | Near-reward dominant; contact gate OFF; turn reward weak |
| **2 — Contact rotation** | 15 M → 60 M | Learn to turn while maintaining fingertip contact | Contact gate ON (0.10 m); proximal penalty active; turn reward 150 |
| **3 — Sustained fingertip rotation** | 60 M → | Polish fingertip-only style; long episodes | Contact gate tight (0.05 m); proximal penalty strong; turn reward 200 |

Phase transitions are printed to the terminal with a `═══` banner.

---

## Reward design

All reward components and their weights are documented in
`screwdriver_rl/tasks/allegro/screwdriver_rotation_env_cfg.py`.

Key failure modes and their countermeasures:

**Flick-and-coast** — policy knocks the screwdriver and retreats.
→ Contact gate: requires ≥2 fingertips at ≤ `contact_distance` AND moving at ≥ `min_fingertip_speed`.  Screwdriver damping 0.15 ensures it stops within one policy step without active finger force.

**Tilt-and-scrape** — policy tilts the screwdriver to the side and uses gravity/friction to spin it.
→ Multiplicative Gaussian gate `exp(-(tilt/0.25)²)` directly suppresses the turn reward.  An additive penalty cannot win against a large positive term; a multiplicative gate can.

**Oscillation** — policy alternates forward/backward, accumulating forward_turns but not net_turns.
→ Reverse penalty (220) slightly exceeds turn reward (200) and shares the same contact gate, so the expected value of any contact is positive only when moving forward.

**Proximal contact** — policy uses palm, knuckle, or finger-back to push.
→ Proximal-link distance penalty (Phase 2+) penalises any of 9 named links being within 5 cm of the handle axis.

---

## Monitoring

The terminal logger prints a structured summary every ~500 environment steps:

```
════════════════════════════════════════════════════════════════════════
  ScrewdriverRL — Allegro Continuous Rotation Training
  Colour guide: good  ok  bad
════════════════════════════════════════════════════════════════════════
────────────────────────────────────────────────────────────────────────
  Step       15,000,000  Elapsed 02h14m  SPS    45,312  Curriculum Ph@15,000,000
  Progress
    FwdTurns       1.842  NetTurns       1.531  OscRatio       0.169
  Object state
    TiltNorm       0.121  UprightGate    0.812
  Contact quality
    ContactGate    0.734  BinaryGate     0.891  MotionGate     0.824
    MinTipDist     0.031  FwdVel         0.412  RevVel         0.089
  Reward breakdown
    TurnRew      61.213  RevCost      13.401  NearRew      0.087  ProxCost  0.002
    UprightCost  12.301  ActionCost    1.443  TotalRew     34.153
```

Warnings are appended inline when failure modes are detected:
- `⚠ OSCILLATION` — OscRatio > 0.35
- `⚠ TILT` — TiltNorm > 0.4 rad
- `⚠ NO-CONTACT` — BinaryGate < 0.15
- `⚠ BACKWARD` — RevVel > FwdVel

---

## Adding a new hand

1. Create `screwdriver_rl/tasks/<hand_name>/` mirroring the `allegro/` structure.
2. Override the three name tables in your env subclass:
   ```python
   from screwdriver_rl.tasks.allegro.screwdriver_rotation_env import AllegroScrewdriverRotationEnv

   class MyHandScrewdriverRotationEnv(AllegroScrewdriverRotationEnv):
       # Map finger names → fingertip body names in your URDF
       _FINGERTIP_BODY_NAMES = {"index": "index_tip", ...}
       # Regex patterns for proximal/medial links to penalise
       _PROXIMAL_BODY_PATTERNS = [r"^index_proximal$", ...]
       # Map finger names → 4-tuple of joint names
       _FINGER_JOINT_NAMES = {"index": ("j0", "j1", "j2", "j3"), ...}
   ```
3. Provide an `ArticulationCfg` for your hand's URDF and update `robot_cfg` in your config.
4. Register the new gymnasium env in `screwdriver_rl/tasks/<hand_name>/__init__.py`.
5. Import it in `screwdriver_rl/tasks/__init__.py`.

---

## Repository structure

```
ScrewdriverRL/
├── train.py                          # Two-stage training entry point (--stage 1|2)
├── play.py                           # Evaluation / playback entry point
├── pyproject.toml
├── docs/
│   └── 2-stage-training.md           # RMA teacher/student details
├── tests/                            # CPU-only unit tests (no Isaac Sim)
│   ├── test_rewards.py
│   └── test_algo.py
└── screwdriver_rl/
    ├── core/
    │   └── rewards.py                # Pure-torch reward / geometry / quaternion primitives
    ├── algos/
    │   └── proprio_adapt.py          # Stage 2 adaptation network + trainer
    ├── tasks/
    │   ├── __init__.py               # Imports all hand sub-packages
    │   └── allegro/
    │       ├── __init__.py           # Gymnasium registration
    │       ├── screwdriver_rotation_env_cfg.py   # All hyperparameters + justifications
    │       ├── screwdriver_rotation_env.py       # DirectRLEnv implementation
    │       └── agents/
    │           └── rl_games_ppo_cfg.yaml
    └── utils/
        └── logging.py                # Formatted terminal logger
```
