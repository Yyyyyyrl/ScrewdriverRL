# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Isaac Lab training environment for **continuous in-hand screwdriver rotation** with dexterous robot hands (Allegro right hand, Linker Hand L20 left hand). Training follows a two-stage **RMA** (Rapid Motor Adaptation) recipe on RL-Games PPO: Stage 1 trains a teacher with an asymmetric actor-critic (the critic sees privileged sim-only state), Stage 2 trains a temporal-conv network to predict that privileged state from proprioceptive history. See `docs/2-stage-training.md` for the full design rationale.

## Commands

Everything except the unit tests requires an Isaac Lab conda environment (this repo's is `env_isaac`) and a GPU. Isaac Sim boots in ~2–3 min; run headless unless a viewport is needed.

```bash
pip install -e .                      # install (editable)

# Tests — pure PyTorch, NO Isaac Sim required, run anywhere
python -m pytest tests/ -q
python -m pytest tests/test_rewards.py::test_wrap_to_pi_across_boundary -q   # single test

# Stage 1 training (teacher PPO)
python train.py --task <id> --stage 1 --num_envs 2048 --headless

# Stage 2 training (adaptation net; requires a Stage 1 checkpoint)
python train.py --task <id> --stage 2 --headless --checkpoint runs/<task>/<run>/nn/<name>.pth

# Viewport playback / quick stats
python play.py --task <id> --checkpoint <path.pth> --num_envs 16

# Headless aggregate statistics + reward-validity probes
python eval.py --task <id> --checkpoint <path.pth> --num_envs 256
```

Registered task ids (all entry scripts take `--task`, defaulting to the first one):

| `--task` id | Hand |
|---|---|
| `Isaac-Allegro-Screwdriver-Rotation-Direct-v0` | Allegro, 3 active fingers (index, middle, thumb) |
| `Isaac-Allegro-4F-Screwdriver-Rotation-Direct-v0` | Allegro, 4 fingers (adds ring) |
| `Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0` | Linker L20, 5 fingers |
| `Isaac-LinkerL20-Screwdriver-Rotation-Top-Grasp-Direct-v0` | Linker L20, top-down initial grasp |

Useful flags: `--max_epochs` (smoke tests), `--output DIR` (default `runs/<task>`), `--init_global_steps N` (resume the curriculum in a later phase — the curriculum step counter is **process state, not saved in checkpoints**, so a plain `--checkpoint` resume restarts at Phase 0), `--seed`, `--video`. For eval: `--no_domain_rand`, `--rot_damping_scale`, `--eval_phase {final,none,<idx>}`, `--stochastic`, `--fixed_start`.

Assets (hand + screwdriver URDFs/meshes) are bundled under `assets/`; override the root with `SCREWDRIVER_RL_ASSET_ROOT`.

## Architecture

### Hand-agnostic base + thin per-hand subclasses

The task logic lives in `screwdriver_rl/tasks/base/`:

- `screwdriver_rotation_env.py` — `ScrewdriverRotationEnv(DirectRLEnv)`: full reward, observation, reset, domain-randomisation, proprio-history and curriculum logic (~1000 lines). Hand specifics are **class attributes** overridden by subclasses: `FINGER_JOINT_NAMES`, `FINGERTIP_BODY_NAMES`, `PROXIMAL_BODY_PATTERNS`, `COUPLED_JOINTS` (mimic/coupled distal joints), `SELF_COLLISION_FILTER_PAIRS`.
- `screwdriver_rotation_env_cfg.py` — `ScrewdriverRotationEnvCfg`: shared sim/screwdriver/DR/reward config. Hand-specific fields are declared `MISSING` and must be set by the subclass cfg: `observation_space`, `action_space`, `fingers`, `pregrasp_positions`, `privileged_obs_dim`, `history_obs_dim`, `robot_cfg`. The base has no gym id — only hand packages register environments.

Each hand package (`tasks/allegro/`, `tasks/linker_l20/`) contains the env subclass, one or more cfg modules, and `agents/rl_games_ppo_cfg.yaml`. Gym registration happens in each package's `__init__.py`; importing `screwdriver_rl.tasks` triggers all registrations (the entry scripts do this). Variant tasks (Allegro 4F, Linker top-grasp) are just alternate cfg classes registered under new ids — same env class.

**The two hands intentionally diverge in design.** Allegro uses the base implementation: geometric contact (fingertip distance to the handle axis) with distance/motion/pad-facing gates. The Linker L20 overrides reward/contact/curriculum/privileged-obs in its own env class: per-fingertip `ContactSensor`s with a trapezoidal force window, full screw load from step 0, prescribed finger roles, joint-range clamping around the pregrasp. When changing task logic, keep Allegro-only changes in the base/allegro modules and Linker-only changes in `linker_l20/` — the Linker subclass exists specifically so the Allegro task is left untouched.

### Curriculum

Training advances through phases driven by a **global step counter** (`_global_steps`, process state). Each hand owns its own `curriculum_phases` list in its cfg module — the list in the base cfg is only a generic fallback; edit a hand's curriculum in that hand's cfg, not the base. Phase transitions print a `═══` banner, and `PhaseCheckpointObserver` (`utils/rl_games_observer.py`) saves a tagged `<name>_phase{N}.pth` checkpoint at each boundary plus a final-phase checkpoint at the end of training.

### Reward math is isolated and unit-tested

All reward/geometry/quaternion primitives live in `screwdriver_rl/core/rewards.py` as **pure-torch functions with no Isaac/USD imports**, so they run and are tested on CPU (`tests/test_rewards.py`). New reward math belongs there with a test, not inlined in the env. Quaternions follow the Isaac Lab convention `(w, x, y, z)`, but the helpers are re-implemented locally to avoid the `pxr` dependency.

Reward weights carry inline justifications in the cfg docstrings — keep numbers traceable the same way when tuning. The design defends against specific reward hacks (flick-and-coast, tilt-and-scrape, oscillation, free-spin farming); the README's "Reward design" section maps each failure mode to its countermeasure, and `eval.py --rot_damping_scale` probes them.

### Stage 2 adaptation

`screwdriver_rl/algos/proprio_adapt.py` holds `ProprioAdaptNet` (HORA-style temporal conv) and `ProprioAdaptTrainer` (supervised MSE regression on frozen-actor rollouts) — also pure-torch and tested in `tests/test_algo.py` against a `FakeEnv`. Output goes to `<output>/stage2_nn/proprio_adapt.pth`.

### RL-Games integration details

- The asymmetric critic activates when `train.py --stage 1` sets `env_cfg.asymmetric_obs = True` and `env_cfg.state_space = privileged_obs_dim` at runtime AND the agent YAML has a `central_value_config` block. Obs dims are hand-specific (Allegro 27/17, Linker 35/19).
- The agent YAML is resolved from the gym registration's `rl_games_cfg_entry_point`, never hardcoded.
- `train.py` auto-shrinks `minibatch_size` when `num_envs × horizon_length(32)` is smaller than the configured 8192, and converts `--save_interval_steps` into RL-Games epoch counts.
- This rl_games version restores checkpoints only from the `"checkpoint"` key in `runner.run()`'s args dict — `params["load_checkpoint"]/["load_path"]` are ignored on the train path.

## Entry-script conventions

`train.py`, `play.py`, `eval.py` all follow the Isaac Lab pattern: parse args and create `AppLauncher` **before** importing anything that depends on Isaac Sim; all isaaclab/rl_games/task imports come after `app_launcher = AppLauncher(args)`. Isaac Lab import paths vary across releases, so wrappers are imported through try/except fallback chains (`isaaclab_rl.rl_games` → `isaaclab_tasks.utils.wrappers.rl_games` → legacy `omni.isaac.lab_tasks`) — preserve these when touching imports. Always `env.close()` before `simulation_app.close()` (skipping it can deadlock teardown), and print tracebacks before shutdown so Isaac's teardown hang doesn't swallow the real error.

## Known documentation drift

The README references some tooling that is not currently in the repo: `calibrate_pad.py`, `render_posture.py`, a `tools/` directory, and `tests/test_linker_cfg.py`. Don't assume these files exist; the actual entry points are `train.py`, `play.py`, `eval.py` and the actual tests are `tests/test_rewards.py` and `tests/test_algo.py`.
