# Top-down force-free refactor execution record (M1–M5)

> **SUPERSEDED (2026-08-08).** Execution log (M1-M5) for the force-free refactor planned in [topdown-force-free-refactor-plan.md](topdown-force-free-refactor-plan.md). Superseded by the HORA path -- see [TRAINING_REPORT_20260807.md](../../TRAINING_REPORT_20260807.md).

Date: 2026-07-27 (America/Los_Angeles)

Scope: execute `docs/topdown-force-free-refactor-plan.md` through M5 and stop
before M6 hardware policy-performance trials. This record deliberately
distinguishes implemented/verified software from full-training acceptance and
from physical evidence that does not exist in the repository.

## Current state

| Milestone | State | Evidence |
|---|---|---|
| M1 reward/contact refactor | implemented and scoped-tested | distance-window contact drives gate/reward/progress; target-penetration proxy replaces force ceiling; fingertip force is diagnostic-only; wrong-surface is a binary safety predicate |
| M2 observation/deploy contract | implemented and scoped-tested | final top-down spaces are policy 54-D / privileged 22-D; package format v2 carries `observation_semantics_version=linker-l20-force-free-reset-dr-v1`; deploy path contains no force input |
| M3 reset pose + DR | simulator acceptance complete under operator-approved proxies | final-scale 1,024-reset audit passes every check; XY support reaches ±8 mm, load reaches 0.589 N.m, and every reset has at least three distance contacts; independent 0/5/8 mm settle bins are each 100% |
| M4 full Stage 1 | updated software and GPU smoke ready; full training not started | post-proxy 1,024-env × 2-epoch smoke completed (65,536 samples), 54-D actor / 22-D critic, no NaN/exception; fixed root-position/tilt eval and oscillation/force-histogram reporting run in Isaac |
| M5 full Stage 2 | updated software and GPU smoke ready; full training not started | post-proxy 8-env × 2-iteration latent-adapter smoke writes held-out diagnostics, adapter checkpoint and deploy bundle; diagnostic privileged probe is absent from deploy bundle |

No M6 hardware policy-performance evaluation has been started.

## M1 calibration and compatibility decision

The frozen pre-refactor calibration used 256 environments and 1,200 steps.
The best shared distance margin reproduced the old force gate at 84.2%, below
the 90% acceptance target. Per-finger margins were therefore selected:

| Finger | Distance margin |
|---|---:|
| index | 7.0 mm |
| middle | 6.0 mm |
| ring | 7.0 mm |
| pinky | 9.5 mm |
| thumb | 29.0 mm |

The calibrated per-finger three-contact gate agreement is 90.158%, so the M1
gate passes without a new geometric pad-facing model. Force readings remain in
`eval.py --calibration_output` solely for before/after diagnostics.

## M3 simulator evidence

The final top-down DR contract includes contact friction `(0.6, 2.5)`, tilt
damping `(0.5, 2.0)`, root XY ±8 mm, root Z ±2 mm, root tilt ±0.025 rad, root
yaw ±0.09 rad, screwdriver tilt ±0.03 rad, and observed joint-zero bias
±0.015 rad. Phase scaling is applied per episode. The hand root moves from the
nominal pose to the sampled pose during the 32-step compliant settle. A
three-finger distance-contact guard resamples failed environment rows up to 64
times and raises on exhaustion, so an episode cannot silently start outside the
contact contract. The accepted XY distribution is therefore uniform proposals
conditioned on initial contact, not an unconditional uniform square.

The paired-seed settle sweep (1,024 samples per amplitude) measured:

| XY half-width | Three-finger settle rate |
|---:|---:|
| 0 mm | 94.629% |
| 0.25 mm | 95.508% |
| 0.5 mm | 95.801% |
| 0.75 mm | 95.996% |
| 1.0 mm | 95.801% |
| 1.5 mm | 93.262% |
| 2.0 mm | 90.625% |
| 4.0 mm | 83.594% |
| 5.0 mm | 80.957% |
| 8.0 mm | 71.777% |

Those rates describe the original teleport-then-settle reset and motivated the
new contact-preserving ramp/guard instead of narrowing the requested DR. The
operator explicitly approved a range above ±5 mm provided initial contact is
guaranteed; ±8 mm is the plan's analysed upper setting, paired with a 0.15 rad
home-deviation deadband. The guarded final-scale 1,024-reset audit passed every
check: sampled XY reached -7.994/+7.997 mm, load torque reached
0.02295--0.58913 N.m, contact count was 3--5, and guard retries averaged 0.89
with maximum 12. An independent 1,024-sample-per-bin settle sweep measured
100% three-finger contact at 0, ±5, and ±8 mm. Subset reset settling now freezes
the complement envs; this removed a detected hidden screw-angle drift and kept
all post-settle start angles inside [-pi, pi]. A zero-noise reset had previously
reproduced the legacy root pose to 1.49e-8 m maximum position error and zero
orientation error.

Volatile run evidence currently lives under
`/tmp/dex-forge-force-free-baseline/`, notably
`m3_runtime_audit_final_1024.json` and
`m3_settle_sweep_final_1024.json`. The post-proxy evidence is
`m3_guarded_final_runtime_audit_1024_v2.json` and
`m3_guarded_settle_sweep_0_5_8mm_1024.json` in the same directory.

## M4/M5 evaluation tooling

`eval.py` now supports persistent, world-frame hand-root biases:

```bash
--root_pos_bias_mm X Y Z
--root_rpy_bias_deg ROLL PITCH YAW
```

It also persists `eval_osc_ratio` and force-window histograms. The latter are
diagnostics and never policy inputs or reward gates.

`tools/evaluate_topdown_force_free_policy.py` runs a resumable matrix:

- oracle final-DR baseline and rotation-damping ×4 probe;
- symmetric X/Y biases at ±1/±2/±4/±5/±8 mm;
- roll/pitch biases at ±1.5 degrees;
- the same bias matrix with the Stage-2 adapter;
- pre-refactor median/fall/oscillation comparison;
- adapter/oracle median ratio, held-out latent MSE, and 3-D relative-position /
  5-D distance-contact-score errors.

The hard 80% retention gate now applies through the contact-conditioned ±8 mm
training envelope. All symmetric ±1/±2/±4/±5/±8 mm cases remain explicit so
performance loss versus displacement is visible rather than hidden by one
aggregate score.

Stage 2 now reserves a true held-out split on every collected batch. The deploy
adapter is evaluated against the teacher latent. A separate diagnostic probe
reports raw privileged-channel MSE but is not stored in `deploy.pth`.

## Formal training command

The plan's 16,384-env command exceeds the measured local multi-asset capacity:
that point was killed by the host OOM killer before PPO startup. The bounded
runner uses the validated ceiling of 8,192 environments and preserves a larger
300,154,880-sample Stage-1 budget (8,192 × horizon 32 × 1,145 epochs), then the
full 500 × 512-step Stage-2 schedule:

```bash
/home/user/miniconda3/envs/env_isaaclab/bin/python \
  tools/run_linker_topdown_training.py \
  --python /home/user/miniconda3/envs/env_isaaclab/bin/python \
  --output runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown/force_free_m1_m5
```

## Operator-approved proxy assumptions (no hardware data)

The operator reported that neither rig-repeatability nor real screw-torque data
can be collected before training. They approved broad DR provided initial
contact is guaranteed, and approved a common-screw proxy. Accordingly:

1. Placement uses contact-conditioned ±8 mm XY support. This does not prove a
   particular rig is in distribution, but it removes the former ±1 mm blocker
   without admitting contactless initial states.
2. The torque proxy is an M2 property-class 8.8 metric screw. Bossard's 2025
   VDI-2230 reference table gives maximum tightening torques of 31.7--39.2 Ncm
   (0.317--0.392 N.m) over friction coefficients 0.10--0.14. The top-down load
   range is widened to base 0.045 N.m × `(0.5, 13.1)` =
   `[0.0225, 0.5895]` N.m: it preserves a low-resistance probe and covers the
   0.392 N.m proxy plus a 50% margin. Source:
   https://assets.eu.ctfassets.net/0vp0u5uh75zd/3S40LEUM235Qk3rJk2phR1/c715f9f45232756c12b59aa681c7fede/060_074_Preload_tightening_torques_Fastening_EN_01_2025.pdf

Tightening torque is not the same quantity as the eventual fixture's sliding or
breakaway torque; this is a deliberately conservative training surrogate, not
hardware validation. Actual measurements remain an M6 preflight item rather
than an M4/M5 blocker under the operator's stated constraints.

## Verification executed

- 95 scoped M1–M5 CPU tests passed.
- Python compilation passed for the changed training/eval/adaptation tools.
- M4 post-proxy Stage-1 Isaac smoke: 1,024 envs, 2 epochs, 65,536 samples,
  clean exit; checkpoint under
  `/tmp/dex-forge-force-free-baseline/m4_guarded_proxy_smoke_1024e2/`.
- M4 fixed-bias Isaac eval smoke: +1/-1 mm and ±1.5-degree root offsets, clean exit.
- M5 post-proxy Stage-2 Isaac smoke: 8 envs, 2 iterations, 22-D privileged
  diagnostics, all three artifacts written, clean exit; held-out latent MSE
  0.04684 after two iterations (pipeline evidence only).
- M5 adapter-latent Isaac eval smoke: no privileged actor input, +1 mm root
  offset, clean exit.

The smoke held-out latent MSE was 0.09486 after only two iterations; it is a
pipeline check, not the M5 `<0.01` acceptance result. That result, the complete
oracle/adapter bias matrix, and the final force-distribution comparison require
the formal M4/M5 training run.
