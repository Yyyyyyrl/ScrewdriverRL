# HORA full-training report — 2026-08-07/08

## Technical summary

**PASS: the Stage-2 deployed policy satisfies every simulator acceptance gate and is the release candidate for hardware integration.** The decisive closed-loop comparison used 256 environments, seed 0, the final curriculum phase, 1,200 policy steps, and 512 completed episodes per row. Under native DR the adapter recorded 0.8% falls and 3.33 physical net turns versus the oracle's 0.4% and 3.37. Under bench DR it recorded 0.6% falls and 3.16 turns versus 1.0% and 3.21. These are not a repeat of the historical 7%→64% adapter collapse.

Deploy bundle:

`/home/fresh/Workspace/dex-forge/runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth`

SHA-256: `3079f57c71b8de3a39911daf65eaf28597c4ff6c884da0f5ccf0e392532235c0`

Teacher checkpoint:

`/home/fresh/Workspace/dex-forge/runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/linker_l20_screwdriver_rotation_07-23-29-15/nn/last_linker_l20_screwdriver_rotation_ep_760_rew_22314.447.pth`

SHA-256: `301250559c0d3d0fb79e2e01e048932fe26fe0b962b223882d6929ce03f33741`

This is a simulator acceptance result, not permission to bypass the hardware rail-lock, joint-limit, emergency-stop, or supervised commissioning checks.

## Configuration and run contract

| Item | Value |
|---|---|
| Git branch / commit | `rand` / `3f0b26d7a05ee72fbcb7cb6e1cbc90bdfd0f2dc7` |
| Task | `Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora` |
| Seed | `20260807` training; `0` evaluation |
| Stage-1 environments | 12,288 |
| Stage-1 horizon | 32 |
| Stage-1 target / completed steps | 300,000,000 / 300,023,808 |
| Stage-1 terminal epoch | 763 |
| Stage-2 environments / iterations | 2,048 / 150 |
| Stage-2 mode | HORA latent, 8-D, teacher-latent adaptation |
| Stage-2 rollout | 512 steps per iteration |
| Selected teacher | epoch 760 |
| Acceptance cohort | 256 envs × 1,200 steps = 512 completed episodes per row |

Both Stage-1 and Stage-2 logs contain the required contract line:

`[train] proprio_dim 32 -> 96 (32 x 3 frames, from the task cfg)`

No unknown/unrecognized-parameter warning was found in either training log. Both supervisors ended with return code 0 and `hang_count: 0`.

## Environment verification

The fresh-tree test result was `244 passed, 2 skipped, 1 xfailed`. This is equivalent to the handoff's `245 passed, 1 skipped, 1 xfailed`: the extra skip is `test_tables_match_sdk`, because `/home/user/linkerhand-ros-sdk` is absent on this machine. Open3D 0.19.0 was installed to restore the intended geometry backend; NumPy 1.26.0, SciPy 1.15.3, and trimesh 4.5.1 match the known-good environment. `pip check` passed after restoring `psutil==5.9.8` and pinning IPython 8.39.0.

## Stage-1 reached a stable Phase-3 policy

Curriculum boundaries are defined by global steps, not an invariant epoch number. The handoff's epoch 153 and 344 labels assume 8,192 environments. At 12,288 environments, the same 40M and 90M boundaries occur at approximately epochs 102 and 229. The table therefore uses the nearest complete logged blocks around the actual boundaries.

| Epoch | Steps | Phase | EpNetTurns | Fall rate | OscRatio | UprightGate | ContactGate | Mean reward |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 74 | 29.5M | 1/3 | 1.131 | 0.0% | 0.016 | 1.000 | 0.994 | 37.554 |
| 112 | 44.2M | 2/3 | 2.373 | 0.0% | 0.029 | 1.000 | 0.994 | 49.073 |
| 224 | 88.5M | 2/3 | 2.295 | 0.7% | 0.018 | 1.000 | 0.987 | 50.711 |
| 262 | 103.2M | 3/3 | 2.122 | 11.5% | 0.073 | 1.000 | 0.950 | 17.689 |
| 299 | 118.0M | 3/3 | 2.393 | 3.3% | 0.060 | 1.000 | 0.979 | 34.516 |
| 374 | 147.5M | 3/3 | 2.375 | 1.5% | 0.053 | 1.000 | 0.983 | 34.387 |
| 455 | 178.9M | 3/3 | 2.340 | 0.0% | 0.262 | 1.000 | 0.904 | 1.082 |
| 530 | 208.4M | 3/3 | 2.378 | 0.1% | 0.237 | 1.000 | 0.920 | 32.345 |
| 605 | 237.9M | 3/3 | 2.489 | 0.0% | 0.141 | 1.000 | 0.922 | 42.763 |
| 680 | 267.4M | 3/3 | 2.531 | 0.3% | 0.087 | 1.000 | 0.953 | 25.823 |
| 755 | 296.9M | 3/3 | 2.434 | 5.0% | 0.060 | 1.000 | 0.921 | 16.583 |

The Phase-3 transition produced the expected temporary fall-rate increase, then recovered from 11.5% to 3.3% in the next logged block. Later oscillation rose during the resumed run but subsequently fell from 0.292 at epoch 492 to 0.060 at epoch 755. The late training reward and fall rate were not monotonic, so the final checkpoint was not selected blindly.

The run was manually stopped once at epoch 382 after the initial interpretation of the handoff's oscillation red flag. The user clarified that a transition spike is acceptable when it recovers, so training resumed from epoch 380 / global step 149,422,080. This was an operator stop/resume, not an automatic supervisor hang; each supervisor invocation reports zero hangs. Approximate Stage-1 training wall time was seven hours across the two invocations.

### Pinned checkpoint comparison

All candidates below used `--eval_phase 1`, 256 environments, seed 0, and 512 episodes, so checkpoint selection did not mix curriculum regimes.

| Checkpoint | Falls | Authorized net turns | Physical net turns | Success |
|---:|---:|---:|---:|---:|
| ep605 | 0.0% | 2.360 | 2.924 | 48.8% |
| ep680 | 0.0% | 2.357 | 2.895 | 44.1% |
| ep725 | 0.0% | 2.358 | 2.878 | 42.8% |
| **ep760** | **0.0%** | **2.385** | **2.913** | 43.4% |
| ep763 | 0.0% | 2.381 | 2.903 | 43.2% |

Epoch 760 had the best authorized net turns and was therefore selected as the Stage-2 teacher. Its coasting probe measured 0.404 rad/s with contact versus 0.075 rad/s without contact; object-specific wrong-surface force was exactly 0.

## Stage-2 adaptation converged

Training loss fell from 0.094020 at iteration 1 to 0.012833 at iteration 20, 0.006653 at iteration 120, and 0.006126 at iteration 150. The held-out, environment-group validation result was:

| Metric | Result |
|---|---:|
| Adapter target MSE | 0.006721 |
| Adapter target MAE | 0.044296 |
| Validation samples | 104,448 |
| Training samples | 944,128 |
| Final iteration | 150 |

The supervised target error is below the handoff's 0.02 convergence guide. More importantly, the adapter also passed the closed-loop behavior gate below.

## Closed-loop deployment gate passes in native and bench regimes

`net turns` below means physical net turns per completed episode. `Authorized net turns` includes the task's motion/contact authorization. Latent MAE is reported only for adapter rows and measures the closed-loop adapter/oracle latent difference along the adapter-driven trajectory; it is not the held-out supervised validation MAE above.

| Regime | Driver | Falls | Physical net turns | Authorized net turns | Success | Closed-loop latent MAE |
|---|---|---:|---:|---:|---:|---:|
| Native DR | Oracle | 0.4% | 3.370 | 2.651 | 72.7% | — |
| Native DR | Adapter | 0.8% | 3.332 | 2.610 | 73.6% | 0.266 |
| Bench DR, fixed 64 mm | Oracle | 1.0% | 3.208 | 2.600 | 68.8% | — |
| Bench DR, fixed 64 mm | Adapter | 0.6% | 3.160 | 2.555 | 67.4% | 0.306 |

Gate arithmetic:

- Native fall delta: +0.39 percentage points; adapter/oracle net-turn ratio: 98.9%.
- Bench fall delta: −0.39 percentage points; adapter/oracle net-turn ratio: 98.5%.
- Both adapter fall rates are below 40%, both are within oracle +5 points, and both net-turn ratios exceed 70%.
- Bench oracle quality is 1.0% falls and 3.208 turns, passing the ≤10% and ≥1.5 secondary requirements.

Therefore the main deployment-gap gate and the secondary policy-quality gate both pass.

## Reward validity and contact safety remain intact

| Regime / driver | Contact fraction | Forward velocity with contact | Without contact | Object wrong-surface force |
|---|---:|---:|---:|---:|
| Native oracle | 99.1% | 0.358 rad/s | 0.174 rad/s | 0.000 N |
| Native adapter | 99.2% | 0.356 rad/s | 0.141 rad/s | 0.000 N |
| Bench oracle | 99.2% | 0.347 rad/s | 0.115 rad/s | 0.000 N |
| Bench adapter | 99.2% | 0.342 rad/s | 0.115 rad/s | 0.000 N |

Native no-contact speed is above the historical ~0.085 rad/s reference, but no-contact steps are less than 1% of the rollout and contact speed remains more than twice as high. The scored behavior is therefore not dominated by free coasting. The non-object `WrongSurf` statistic contains rare hand self-contact/fall-impact spikes, while the safety-relevant `vs object` force is exactly zero in every acceptance row.

Action saturation and hardware rail-lock behavior cannot be established from these simulator runs; those remain commissioning checks.

## Visual inspection artifacts

All videos use seed 0, fixed start, no domain randomization, final curriculum phase, 300 policy steps, 1280×720 at 10 fps. Stage-1 uses the oracle latent. Stage-2 uses the self-contained deploy bundle and adapter-predicted latent. The combined overview is 1,920×720; its top row is Stage-1 and bottom row is Stage-2, with oblique, side, and overhead columns.

At 10 s and 25 s, both policies remained upright with no visible drop. The strict overhead view is partially occluded by the hand back, so tool/contact inspection should rely primarily on the oblique and side views.

## Deviations, incidents, and limitations

- The requested 8,192 environments were increased to 12,288 after resource checks. A 16,384 configuration was not used because host RAM, rather than GPU memory, became the limiting resource.
- Open3D was initially absent. The fallback trimesh backend changed only a backend-sensitive nearest-target test label; installing Open3D 0.19.0 restored the intended geometry test result without modifying geometry, posture, architecture, or reward.
- A manual Stage-1 stop produced an Isaac/Carb mutex assertion during teardown. The resumed training process completed normally; no intrinsic training crash or supervisor hang occurred.
- After Stage-2, an unattended NVIDIA userspace update temporarily mismatched the kernel driver. Evaluation was held until reboot restored driver 580.173.02; no training artifact was changed.
- `play.py` locally required the same frame-stack-derived `proprio_dim=96` logic already present in `train.py` and `eval.py`. The playback-only fix lets the 104-D latent actor load and does not change policy weights, environment dynamics, architecture, or reward.
- Simulator acceptance does not cover real sensor bias, actuator latency/backlash, thermal limits, rail-lock behavior, or emergency-stop integration.

## Release decision and next steps

**Release decision: simulator PASS; `deploy.pth` is the deployable policy candidate.** Keep the exact epoch-760 teacher and deploy bundle together. Do not substitute epoch 763 or regenerate only one half of the pair.

Before unsupervised hardware operation:

1. Verify bundle metadata, task ID, action scale, joint ordering, and normalization at load time.
2. Run low-force, low-speed supervised commissioning with joint-limit and rail-lock telemetry enabled.
3. Confirm the deployment process never reads privileged observations and reproduces the Stage-2 adapter history length/frame ordering.
4. Record hardware fall/abort rate, authorized turns, max joint deviation, and rail-lock events against the simulator acceptance table.
5. Stop on any constant saturated action, unexpected object-side contact, or rail-lock trigger; do not compensate by changing reward or architecture without a new controlled experiment.

## Evidence files

- Stage-1 training log: `/home/fresh/Workspace/dex-forge/stage1_env12288.log`
- Stage-2 training log: `/home/fresh/Workspace/dex-forge/stage2.log`
- Stage-1 pinned evaluations: `/home/fresh/Workspace/dex-forge/artifacts/full_training_20260808/stage1_pinned_oracle/`
- Stage-2 acceptance JSON/logs: `/home/fresh/Workspace/dex-forge/artifacts/full_training_20260808/stage2_acceptance/`
- Stage-2 validation: `/home/fresh/Workspace/dex-forge/runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/adaptation_validation.json`
- Policy videos: `/home/fresh/Workspace/dex-forge/artifacts/policy_video_comparison_20260808/`
