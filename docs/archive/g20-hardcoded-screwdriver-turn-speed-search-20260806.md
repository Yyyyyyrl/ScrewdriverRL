# G20 hardcoded screwdriver turn speed search — 2026-08-06

> **SUPERSEDED (2026-08-08).** A CEM search for a hardcoded turn sequence, run as a fallback while the learned policy was unproven. The HORA ep-760 policy replaced it -- see [TRAINING_REPORT_20260807.md](../../TRAINING_REPORT_20260807.md).

## Status

**Not promoted to hardware.** An eight-phase open-loop cycle can overfit a small
Isaac search set to about `1.53 turns/min`, but independent replay and DR16
validation fall below `0.9 turns/min` and exceed the preferred `16 N` contact
limit. The existing hardware script remains pinned to the previously gated slow
candidate.

This record prevents the search-set result from being mistaken for a deployment
gate pass.

## Requested target and gate

The observed real-hand hardcoded motion was about `0.4 turns/min`. The requested
target was `1.5–2.0 turns/min`, equivalent to `0.1571–0.2094 rad/s` net physical
screwdriver rotation.

The simulation gate used in this search was:

- no episode termination or fall;
- zero wrong-surface contact;
- contact-gate fraction preferably at least `0.8`;
- contact-force soft limit `16 N`;
- report both physical net rate and force-qualified net rate;
- validate a selected candidate with an independent random seed and 16 DR
  replicas before any real-hand command.

## What was tested

Simple time compression of the prior cycle was rejected first. At two seconds
per leg it produced only `0.0138 turns/min`; faster playback destroyed useful
contact rather than increasing rotation.

The search was then expanded to an eight-waypoint, 16-active-joint periodic
cycle. Each waypoint transition used three policy steps. CEM continuation and
four-replica robust optimization were run with the commissioned 64 mm top-down
asset and commissioning domain randomization.

| Stage | Physical rate | Qualified rate | Contact | Peak force | Wrong/fall |
|---|---:|---:|---:|---:|---:|
| first 8-phase search | `0.865 rpm` | `0.326 rpm` | `0.948` | `11.49 N` | `0 / 0` |
| continued nominal search | `1.146 rpm` | `0.863 rpm` | `0.933` | `13.63 N` | `0 / 0` |
| larger nominal search | `1.456 rpm` | `1.085 rpm` | `0.944` | `14.71 N` | `0 / 0` |
| DR4 robust search | mean `1.417`, worst `1.400 rpm` | mean `0.922`, worst `0.855 rpm` | `0.868` | `14.33 N` | `0 / 0` |
| final DR4 search set | mean `1.529`, worst `1.509 rpm` | mean `1.015`, worst `0.997 rpm` | `0.931` | `15.15 N` | `0 / 0` |
| independent DR16, seed `20260830` | mean `0.857`, worst `0.667 rpm` | mean `0.458`, worst `0.361 rpm` | `0.799` | `18.71 N` | `0 / 0` |
| separate single-env rendered replay | `0.859 rpm` | `0.435 rpm` | `0.500` | `18.30 N` | `0 / no termination` |

The final search-set number therefore does **not** pass the independent gate.
The rendered replay also shows only small hand-shape changes, without a clear
push–unload–regrasp sequence. Its measured result agrees with the failed DR16
gate rather than with the search-set score.

## Primary artifacts

- final DR4 search-set candidate:
  `records/g20_hardcoded_screwdriver_turn_v2_20260806/high_speed/rolling8_t3_robust_dr4_final2_256x12.json`
- independent DR16 gate:
  `records/g20_hardcoded_screwdriver_turn_v2_20260806/high_speed/rolling8_t3_1p5rpm_dr16_validation.json`
- rendered replay report:
  `records/g20_hardcoded_screwdriver_turn_v2_20260806/high_speed/rolling8_t3_1p53rpm_nominal_sim.json`
- rendered replay video (the filename identifies the search candidate; the
  video's measured rate is `0.859 rpm`, not `1.53 rpm`):
  `records/g20_hardcoded_screwdriver_turn_v2_20260806/high_speed/rolling8_t3_1p53rpm_nominal_sim.mp4`
- visual review contact sheet:
  `records/g20_hardcoded_screwdriver_turn_v2_20260806/high_speed/rolling8_t3_candidate_contact_sheet.jpg`

## Deployment decision

Do not update `tools/run_g20_hardcoded_screwdriver_turn.py` to this candidate and
do not send it to the real hand. The independent validation shows substantial
environment-slot overfitting and a contact-force excursion above the preferred
limit.

Reaching a credible `1.5–2.0 rpm` requires feedback, not further open-loop time
compression. The next implementation should use screwdriver angle/velocity and
contact state to trigger push, unload, and regrasp phases, or fine-tune a HORA
checkpoint on the promoted calibrated asset. The same independent DR16 gate and
then a staged real-hand commissioning gate must be repeated before deployment.
