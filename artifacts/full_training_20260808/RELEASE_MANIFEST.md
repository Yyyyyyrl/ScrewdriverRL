# HORA policy release manifest — 2026-08-08

This manifest identifies the minimal policy pair and evidence committed for deployment and independent simulator validation. Intermediate checkpoints and full training logs are intentionally excluded.

## Policy artifacts

| Role | Repository path | SHA-256 |
|---|---|---|
| Self-contained deployed actor + adapter | `runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth` | `3079f57c71b8de3a39911daf65eaf28597c4ff6c884da0f5ccf0e392532235c0` |
| Stage-1 oracle teacher, epoch 760 | `runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/linker_l20_screwdriver_rotation_07-23-29-15/nn/last_linker_l20_screwdriver_rotation_ep_760_rew_22314.447.pth` | `301250559c0d3d0fb79e2e01e048932fe26fe0b962b223882d6929ce03f33741` |
| Held-out Stage-2 validation | `runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/adaptation_validation.json` | `6fee5e7abf0d59e1453dd72720526a183d7455dd9465bcc81b8fefa95b24f62e` |

`deploy.pth` is the deployment artifact. The Stage-1 checkpoint is retained only for oracle replay and adapter-vs-oracle validation. The separate `proprio_adapt*.pth` files are not required because the deploy bundle already contains the actor, adapter, normalization, dimensions, task metadata, and action scale.

## Validation evidence

- `stage2_acceptance/`: native/bench × oracle/adapter JSON reports, 512 completed episodes per row.
- `stage1_pinned_oracle/`: five pinned checkpoint comparisons used to select epoch 760.
- `run_stage2_acceptance.sh`: exact four-way acceptance invocation.
- `TRAINING_REPORT_20260807.md`: full gate arithmetic, caveats, and release decision.

## Video layout

The six raw recordings are under `artifacts/policy_video_comparison_20260808/{oblique,side,top}_{stage1_oracle,stage2_adapter}/eval_videos/`. The four curated comparisons are under `comparisons/`:

- `all_angles_stage1_top_stage2_bottom.mp4`: top row Stage-1, bottom row Stage-2; columns oblique, side, overhead.
- The other three comparison videos place Stage-1 on the left and Stage-2 on the right.

All final recordings contain 301 frames at 10 fps. They use seed 0, fixed start, no domain randomization, and the final curriculum phase.

## Revalidation

Run `PYTHON=/path/to/isaaclab/python artifacts/full_training_20260808/run_stage2_acceptance.sh` from the repository root. The script expects the policy artifacts at the repository paths listed above and writes fresh JSON/logs into `artifacts/full_training_20260808/stage2_acceptance/`.
