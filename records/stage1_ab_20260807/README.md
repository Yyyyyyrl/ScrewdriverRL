# Stage-1 A/B on the refit posture — baseline vs additive tilt price

2026-08-07. First training on the posture rebuilt for the promoted
hardware-visual calibration (`records/topdown_posture_refit_20260806/`).

Both arms: 8192 envs x 32 horizon x 140 epochs = 36.7 M samples, seed 20260807,
identical in everything but the reward switch.

## Protocol

Evaluated at a **pinned curriculum phase**, 256 envs, 512 episodes per
checkpoint, four checkpoints per arm.

The phase is `--eval_phase 1`, which is a **0-based index**, so it is the
*middle* of the three phases: `step_start` 40 M, DR scale 0.60, 45 s episodes,
upright termination 1.2 rad. The console banner labels the same phase "Phase
2/3" because the display is 1-based. Worth stating plainly, because both arms
stop at 36.7 M samples and so trained entirely inside phase index 0 (DR 0.25,
25 s). Every number below is therefore measured one difficulty step above where
the policy was trained — a mild generalisation test, not a training-set score.

Both parts of that matter and both come from past mistakes on this task:

* Pinning: the same ep330 checkpoint read 6.8% fall at its native Phase 2 and
  21.6% at Phase 3. An unpinned comparison once inverted a conclusion.
* Four checkpoints, not one: an earlier run degraded monotonically across
  Phase 3, with authorised turns going 0.726 → 0.465 → 0.215. The last
  checkpoint alone would have told the opposite story.

Decision rule, fixed before the numbers were read: primary metric is authorised
net turns, subject to fall rate ≤ 1%. A gap below 5% counts as no difference and
keeps the baseline, since changing the default needs positive evidence.

## Baseline arm

| checkpoint | fall | authorised net turns | success (≥3 turns, upright) |
|---|---|---|---|
| ep_112 | 0.2% | 2.60 | 85.2% |
| ep_120 | 0.4% | 2.54 | 82.8% |
| ep_128 | 0.0% | 2.56 | 83.2% |
| ep_140 | 0.0% | **2.61** | **85.2%** |

Spread across the four is 2.7%, so 2.6 is the level of this configuration
rather than one lucky checkpoint.

ep_140 in detail: physical net turns 3.76, drive count 3.78 (of four loaded
fingers), oscillation ratio 0.017, tilt p50 0.057 rad with 92.1% of steps under
0.10 rad. Falls are 0.0% in every DR corner separately — load-torque terciles,
placement terciles, and the high-load ∩ high-placement corner (n=63).

### The rotation is driven by contact, not by coasting

| | ep_140 |
|---|---|
| steps with the contact gate open | 95.8% |
| forward velocity while in contact | 0.522 rad/s |
| forward velocity without contact | 0.091 rad/s |

The 0.091 rad/s figure is essentially the environment's own background: the
posture's measured zero-action drift is 0.085 rad/s, the handle turning under
its load torque with the hand doing nothing. So the policy adds 0.522 rad/s of
contact-driven rotation on top of a floor it did not create.

This is the metric that motivated the whole posture rebuild. The complaint about
the previous policy was that it was not driving with the fingertips, and the
measurement then was ~0.046 rad/s of coasting against ~0.09 total, about half.

## Additive tilt price arm

| checkpoint | fall | authorised net turns | success (≥3 turns, upright) |
|---|---|---|---|
| ep_112 | 0.2% | 2.76 | 92.0% |
| ep_120 | 0.0% | 2.81 | 91.8% |
| ep_128 | 0.0% | 2.85 | 93.0% |
| ep_140 | 0.0% | **2.89** | 92.2% |

## Verdict: additive tilt price

Applying the rule as written. Fall rate is ≤ 0.2% throughout, so the gate is
satisfied and authorised net turns decide it. Averaged over the four
checkpoints, 2.58 → 2.83, a gap of **+9.7%** against a 5% threshold. The arm
also wins at every individual checkpoint, so this is not a lucky pick.

Success rate moves with it, 82.8–85.2% → 91.8–93.0%, and that metric took no
part in the decision rule — it is independent corroboration rather than a second
bite at the same measurement.

The reward scales are not comparable between arms, since one arm has an extra
reward term; raw reward is therefore never used above. Authorised net turns,
fall rate and success rate are all computed from trajectories and are
reward-independent, which is why the protocol was built on them.

One observation outside the decision rule that matters more for a 300 M-step run
than the verdict itself: **the additive arm is still improving at the budget
limit** (2.76 → 2.81 → 2.85 → 2.89, monotone) while the baseline is flat
(2.60 / 2.54 / 2.56 / 2.61). At 36.7 M samples the baseline has converged for
this phase and the additive arm has not. That says nothing certain about
behaviour at Phase 2 and 3, but it is the opposite of a saturating arm.

## Comparison with history

Protocols differ, so this is an order-of-magnitude comparison, not a like-for-like one.

| | fall | authorised net turns | budget |
|---|---|---|---|
| ep330 (last deployable bundle) | 6.8% | 1.014 | 300 M steps, full curriculum |
| ep645 (best Stage-1 ever, adapter collapsed) | 8.41% | 1.951 | 300 M steps, full curriculum |
| baseline ep_140 | **0.0%** | **2.61** | 36.7 M steps, Phase 1 only |

## What this does not establish

* **Phase 2 and 3 are untouched.** 36.7 M steps does not reach the 40 M Phase-2
  boundary, and Phase 3 is where the previous three-frame run destabilised
  (oscillation ratio 0.25 → 0.44, reward +4989 → −2158).
* **Stage 2 has not run.** ep645 also looked strong at Stage 1 and its adapter
  went from 7% to 64% fall in closed loop. Stage-1 quality has never been
  sufficient evidence of deployability on this task.
## The 120.8 N `WrongSurf` peak is hand self-contact, not the handle

Listed as an open diagnostic when this file was first written. Resolved, and the
answer was not the expected one.

Tracing spikes above 1 N attributed none of them to the two suspected causes:
76 events, 0/76 within five steps of a termination (so not fall impacts) and
6/76 within 15 steps of a reset (so not settle impulses). They happen mid-episode.

The cause is the instrument. The wrong-surface figure was read from
`net_forces_w` on an **unfiltered** contact sensor, which sums contact from every
source, while self-collision is deliberately enabled between the palm and each
proximal phalanx. A finger folding against the palm is indistinguishable from a
proximal link pressing the handle.

Reading the same links filtered to the screwdriver settles it:

| | mean ± std over the rollout |
|---|---|
| all contact sources (what the reward sees) | 0.005 ± 0.217 N |
| screwdriver contacts only | **0.000 ± 0.000 N** |

Exactly zero, not merely small: with ~180 k samples a single 120 N reading would
put the std near 0.28, which is what the unfiltered column shows. **No
non-fingertip link touches the screwdriver at any point.** That is a
deployment-relevant property that had never actually been checked, and the alarm
was measuring the hand touching itself.

Both readings are now reported side by side (`eval_wrong_surface_object_force`).

The reward still consumes the unfiltered one, deliberately: it is what every run
to date optimised against, and changing it would break comparability with the
A/B above. In practice the effect is small either way — the penalty is binary
above 1 mN, so 120.8 N and 1 mN score identically, and p99 is 0.000. What remains
is a genuine if minor mis-incentive: the policy is charged for harmless
finger-to-palm contact under a term whose name promises otherwise. Candidate for
the next iteration, not for this one.

Attaching the filters is inert, which is what licenses carrying the A/B verdict
forward into a full run built on the changed code. Same checkpoint and seed,
before and after: `WrongSurf` 0.005 ± 0.217 both times, physical net turns 3.76
± 0.78 with identical percentiles, authorised net turns 2.61, falls 0.0%, and
the same 76 / 0-of-76 / 6-of-76 trace counts. Every reported statistic matches,
so the extra reporting channel changed no physics and no reward.
