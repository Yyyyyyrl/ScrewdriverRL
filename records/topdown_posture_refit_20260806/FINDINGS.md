# Top-down grasp refit on the promoted calibration — findings

2026-08-06/07. Rebuilding the 64 mm top-down grasp after the hardware-visual
calibration was promoted (`assets/linker_hand_l20/linkerhand_l20_left.urdf`
sha256 `1fe1f1db…`). Recorded because several of these are counter-intuitive and
cost a lot of iterations to establish.

## What the calibration actually changed

Nine of the sixteen actuated joints were tightened. The two that dominate the
grasp problem:

| joint | before | after |
|---|---|---|
| `thumb_cmc_yaw` | 0 … 1.40 | 0 … **1.12** |
| `pinky_mcp_pitch` | 0 … 1.40 | 0 … **1.14** |

The superseded posture bound `thumb_cmc_yaw` at 1.2407, which is 0.12 rad
outside the reviewed limit, so it was not recoverable by editing one number.

## The reference configuration contacts with the BACKS of its fingers

The posture with the best measured fall rate and net turns on this task
(physics candidate 117) has pad-facing cosines of −0.48 / −1.00 / −0.22 / −0.59
on index / middle / ring / pinky, and +0.06 on the thumb. Four of five digits
bear on the dorsal side.

Consequences:

* The historical scores were **not** obtained by driving with fingertip pads.
* Calibrating new gates against that reference is safe for *balance* criteria
  and unsafe for *contact-quality* criteria. An early revision of
  `linker_topdown_grasp_wrench` enforced a height band and an opposition-height
  limit that the reference violates by 55–61 mm; that would have rejected a
  working grasp in favour of one that tipped the handle 1.458 rad in 30 steps.

## The handle hangs on a universal joint, not a bearing

There is nothing to react a net lateral force, so radial imbalance becomes tilt.
`radial_closure` (‖mean radial unit vector‖) is the criterion that survived
contact with evidence:

| configuration | closure | outcome |
|---|---|---|
| reference reset / target | 0.218 / 0.364 | works |
| accepted posture | 0.386 | tilt 0.032–0.046 rad, upright |
| mid-height variant | 0.532 | tilted 1.458 rad, terminated in 30 steps |

Contact *height* is not the discriminator; several rim-height grasps are fine.

## `reset_zero_tension_targets` releases the grasp

The shared top-down config snaps finger targets onto the settled measured
positions at the end of every reset, to stop standing target penetration from
feeding solver creep. Measured on this posture (64 envs, DR off, zero action):

| retained tension | index | middle | ring | pinky | thumb | tilt | drift rad/s |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.53 | **8.05** | 1.26 | 0.10 | 0.01 | 0.032 | 0.08457 |
| 0.40 | 0.37 | 6.71 | 1.71 | 0.85 | 1.27 | 0.037 | 0.08456 |
| 1.00 | 1.52 | 3.73 | 1.58 | 1.84 | 3.46 | 0.046 | 0.08456 |

The creep it exists to prevent moves in the sixth significant figure. The
self-rotation that remains is the handle turning under its own load torque
because a released grasp cannot hold it. What the snap does change is the load
distribution: at zero tension one wedged finger carries 8.05 N against 0.01 N on
the thumb, a ratio of 805. Disabled for the Hora task only.

## Preload must exceed the controller's own tracking error

Steady-state PD error at the settled grasp is ~0.14 rad, about 7 mm of fingertip
travel, while the static fit places contacts to micron precision. Preloads of
0.05–0.16 rad — the same order as the error — leave a single finger carrying
everything. Raising the closure to ~0.32 rad took the functional physics gate
from 0/20 to 20/20 with all five fingers at 1.0–3.6 N.

## Instrument failures, in order

Every wrong conclusion in this session came from measuring a proxy instead of
the quantity being judged. Recorded so the pattern is recognisable:

1. Non-distal clearance priced as a *soft* hinge under `soft_l1`; the solver sold
   0.6 mm of pinky clearance for a better contact score.
2. Penetration sampled over 256 fingertip vertices; the objective believed it had
   reached 0.1 mm while the full mesh was at 1.03 mm.
3. Contact *height* taken from tip markers rather than contact points; the
   fingers slid to 99.9% of handle height while the marker targets read fine.
4. Driving contacts classified by *link name*; the cap is a 1 mm disc of the same
   radius, so its side wall is a continuation of the body's and three genuine
   side-wall contacts were excluded.
5. Driving contacts then classified by contact *radius*; the outer rim of the cap
   top face sits at exactly the handle radius, so axial contacts passed.
6. Fingertip penetration checked only against the *nearest* handle part; once a
   tip intersects both, both surface distances are zero and the tie hid 4.86 mm
   of interpenetration in the other part. Fixed, with a regression fixture.
7. Pad-versus-back judged on the *reset* state rather than the settled one.
8. Then judged from an offline reconstruction that ignored the domain-randomised
   wrist offset, then one that ignored the object's own settled tilt; the last
   still attributed 37.6 N to a fingertip it placed 7.4 mm clear of the object.
9. Contact presence judged by raw surface clearance against a flat 1 mm
   threshold, ignoring the calibrated per-finger margins that exist because
   `compute_surface_clearance` measures from the distal body frame.
10. Contact presence judged by those margins when force was the ground truth: a
    finger carrying 1.34 N was reported as not touching.

The fix that ended it was to stop reconstructing anything offline and measure
inside the simulator — contact force vectors from the per-fingertip sensors, and
the pad direction as a per-link constant rotated by that body's own orientation
(`PAD_AXIS_LOCAL`, invariant to 0.0–0.3° across flexion, sign anchored by a fist
test). See `_compute_pad_facing` and `_compute_pad_orientation`.

## Five pad contacts are not reachable on this hand

The operator's requirement was five pad contacts at the training start, with the
posture keeping the screwdriver upright. Four are reachable; five are not, and
the obstruction is the middle finger. It has exactly three placements at this
wrist and each was measured:

| middle placement | pad facing | force |
|---|---|---|
| as fitted, on the cap top face | **−0.898** (back) | 3.72 N |
| pad on the lateral wall (`mcp_pitch` 0.425 / `pip` 0.945) | +0.712 | 0.27 N nominal, **26.1 N peak under DR** |
| retracted 4.1 mm clear | — | 0 N |

The wall placement is genuinely reachable — an early 2-D sweep called it
infeasible only because it sampled ±0.30 rad around the fitted value and the
solution needs −0.315 and −0.379, outside on both axes. But that configuration
is kinematically stiff along the direction of the ±8 mm reset placement
randomisation, so it absorbs the offset as force: 26.1 N peak on the middle and
9.7 N on the pinky over 256 replicas, against an 8 N ceiling. The same mechanism
produced 15–33 N as a near-straight strut, 30 N at zero reset interference, and
*higher* force when the middle's own preload was capped — four variants, one
cause.

Shrinking the handle does not help. Sweeping 64 → 52 mm moves the pad-contact
window only from `pip` 0.52 to 0.63, still far inside strut territory.

So the accepted posture retracts the middle. Measured at the training start with
tension retained:

| | option A (middle on cap) | option B (middle retracted) |
|---|---|---|
| loaded contacts | 5 | 4 |
| pad contacts | 4 | **4** |
| back contacts | **1** (3.72 N) | **0** |
| peak force under DR | 6.44 N | **6.03 N** |
| tilt | 0.046 | **0.026** |
| upright for a full episode | yes | yes |

B is not a compromise on any measured axis except total grip (10.8 N vs 11.6 N):
it has the same number of pad contacts, no back contact, lower peak force and
less tilt. The fifth contact A gains is precisely the one the operator excluded.

The static validator's contact count was relaxed from five fingers to four to
admit this, and a compensating check added: a fingertip that is not bearing on
the handle must be at least 2 mm clear. Hovering between contact and clearance
is rejected, so a posture cannot shed a finger unnoticed.

## Known, accepted costs

* `middle_pip` home is 1.372 rad, so the deploy envelope clips 0.17 rad off its
  upward action range. Not dialable: reducing that PIP by 0.10 rad already drives
  the fingertip 5.6 mm into the handle.
* All four `mcp_roll` joints have 0.34 rad of total travel against a 0.70 rad
  intended motion range, so their envelopes are limit-bound for any posture.
* Zero-action shaft drift is ~0.085 rad/s (one revolution per ~74 s) and is
  independent of posture and preload. It is the handle turning under its load
  torque, and the policy has to hold it.
