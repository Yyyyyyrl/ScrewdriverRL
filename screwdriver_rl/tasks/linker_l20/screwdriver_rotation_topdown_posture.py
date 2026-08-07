"""Validated reset state and compliant target for the 64 mm top-down task.

``TOPDOWN_RESET_*`` is the only pose written directly into simulation. It passed
the full URDF-mesh, convex-hull, non-distal, self-collision and joint-margin
validator. ``TOPDOWN_TARGET_*`` is a position-controller preload target applied
during the inherited reset settling steps; PhysX contact constraints determine
the actual settled pose. This module has no Isaac imports so CPU tests can audit
both states.

Refit 2026-08-06 against the promoted hardware-visual calibration
(``assets/linker_hand_l20/linkerhand_l20_left.urdf`` sha256 1fe1f1db...).  The
previous posture bound ``thumb_cmc_yaw`` at 1.2407 rad, which the calibration
put 0.12 rad outside the reviewed upper limit of 1.12; it is not recoverable by
lowering that one joint, because the thumb opposition sets every other contact.

The earlier rebuild attempts recorded in
``records/g20_og_local_q_asset_promotion_20260805/training_revalidation/``
stalled at 1.068 mm of ``pinky_middle`` clearance against a 1.5 mm gate.  The
cause was not an infeasible geometry and not under-sampled multi-starts (64
restarts from a fresh seed reproduce the same 8.847 optimum): non-distal
clearance enters ``fit_linker_l20_screwdriver_topdown`` as a *soft* hinge that
``soft_l1`` further flattens, so the solver profitably sold 0.6 mm of pinky
clearance for a better contact-role score.  With ``pinky_mcp_pitch`` cut from
1.40 to 1.14 by the calibration -- the largest reduction of any joint -- the
pinky can only reach the shaft by bringing its middle phalanx in, and the fit
took that trade every time.

This posture comes from the same objective with two changes: the non-distal term
is priced as a hard constraint, and candidate selection is ordered
feasibility-first rather than by objective score alone.

A second change was needed after the first feasible fit reached Isaac.  That fit
drove the *nearest* tip vertex to zero clearance and only penalised penetration
beyond 0.5 mm, so the deepest vertex reached 0.96 mm.  Under a stiff PD that is
9-11 N per finger against the 8 N functional ceiling, and the reaction drove the
two weak chains off the object entirely: thumb and pinky read 0.0 N while
``thumb_cmc_pitch`` settled at its 0.0 rad lower rail, 0.117 rad below command.
The grasp had collapsed onto index/middle/ring with no thumb opposition.

So the residual now targets the *deepest* vertex at 0.1 mm, sampling 3000
fingertip vertices instead of 256 -- with the coarse sample the objective
believed it had achieved 0.1 mm while the full mesh was at 1.03 mm.  The reset is
now a light valid touch (0.10-0.15 mm on all five tips, 0.1-2.5 um surface gap),
and grip force is meant to come from the TARGET preload rather than from reset
interpenetration.

Measured against the superseded posture: non-distal clearance 4.673 mm (was
1.068 mm and failing), joint margin 0.11504 rad with no joint railed at the
guard, 36 of 96 restarts feasible.

Contact distribution, from the corrected validator: four lateral drivers
(middle/ring/pinky/thumb, all at radius 0.032 on the cylindrical surface),
``radial_closure`` 0.313, thumb opposing low at 56.2% of handle height.  For
comparison the reference configuration -- the last posture with measured
fall-rate and net-turn evidence -- sits at 0.364.

This posture was briefly REPLACED and then restored, and the detour is recorded
because the reasoning that caused it is easy to repeat.  Its four fingers contact
near the top edge, which was diagnosed as "cannot produce torque" and rejected.
That diagnosis was wrong on two counts.  First, the reference configuration also
contacts at 87-100%, so a top-edge contact is not disqualifying.  Second, and the
part that actually misled: the cap is a 1 mm disc of the *same radius* stacked on
the body, so its side wall continues the body's side wall.  A contact at 100%
height and radius 0.032 is a lateral contact that drives normally; only a contact
that has pulled inside the radius is on the flat top face pressing axially.
Judging by height alone conflated the two.

The replacement -- contacts moved to 54-78% of handle height -- looked correct and
tipped the handle 1.458 rad within 30 simulation steps.  The handle hangs on a
universal joint, not a rigid bearing, so it cannot react a net lateral force; the
replacement's ``radial_closure`` was 0.532 against this posture's 0.313.  Lateral
balance, not contact height, is what keeps the handle upright.

Evidence: ``records/topdown_posture_refit_20260806/``.
"""

from __future__ import annotations


TOPDOWN_HANDLE_RADIUS_M = 0.032
TOPDOWN_PHYSICS_CANDIDATE_INDEX = 624
TOPDOWN_ROOT_YAW_RAD = 0.0826937795259024

# Isaac and URDF quaternions are scalar-first (w, x, y, z). The palm local +X
# normal maps exactly to world -Z.
# The registered diameter task adds its per-bucket offset to this shared base.
TOPDOWN_ROOT_POS_W = (-0.023818780134293113, 0.1742023017880395, 1.4570639296862542)
TOPDOWN_ROOT_QUAT_WXYZ = (
    0.520240224512027,
    0.47890511461006113,
    0.520240224512027,
    -0.47890511461006113,
)
# Provisional: the previous (-0.1334, 0.0154) was the settled equilibrium of the
# superseded grasp and does not transfer. The replacement comes from the settled
# state of the Isaac physics search, not from this file.
TOPDOWN_SCREWDRIVER_TILT_XY = (
    0.0,
    0.0,
)

# Collision-resolved state from the 64 mm feasibility-first fit, written directly.
TOPDOWN_RESET_JOINT_POSITIONS = {
    "index_mcp_roll": 0.05414624541908073,
    "index_mcp_pitch": 0.5084262399237393,
    "index_pip": 0.9184325362520711,
    "index_dip": 0.8189662925759719,
    "middle_mcp_roll": 0.0006154768077150623,
    "middle_mcp_pitch": 0.7395448187059293,
    "middle_pip": 1.4238588934462029,
    "middle_dip": 1.2696549752859791,
    "ring_mcp_roll": -0.015733775224370507,
    "ring_mcp_pitch": 0.6191580734478584,
    "ring_pip": 0.6431715319878482,
    "ring_dip": 0.5735160550735642,
    "pinky_mcp_roll": 0.0458935652917647,
    "pinky_mcp_pitch": 0.5746047799136738,
    "pinky_pip": 0.8562345763625466,
    "pinky_dip": 0.7635043717424829,
    "thumb_cmc_yaw": 0.89291068971524,
    "thumb_cmc_roll": 0.886153707318344,
    "thumb_cmc_pitch": 0.14556519674274296,
    "thumb_mcp": 0.15875990711102864,
    "thumb_ip": 0.18446313607230416,
}

TOPDOWN_RESET_POSITIONS = {
    "index": (0.05414624541908073, 0.5084262399237393, 0.9184325362520711),
    "middle": (0.0006154768077150623, 0.7395448187059293, 1.4238588934462029),
    "ring": (-0.015733775224370507, 0.6191580734478584, 0.6431715319878482),
    "pinky": (0.0458935652917647, 0.5746047799136738, 0.8562345763625466),
    "thumb": (
        0.89291068971524,
        0.886153707318344,
        0.14556519674274296,
        0.15875990711102864,
    ),
}


# Isaac physics-search candidate 217, the lowest-cost candidate passing the
# functional contact gate over a 128-candidate preload bank at 6 replicas with
# domain randomisation on the fixed 64 mm geometry.  Settled: index 6.08 N,
# middle 3.20 N, thumb 1.95 N, all three at contact fraction 1.00, tilt 0.070
# rad, zero wrong-surface force, no termination.  Forces sit under the gate's
# 8 N ceiling, against 9-11 N for the zero-preload target it replaces.
#
# Two limitations of this grasp are known and deliberately accepted:
#
# 1. The pinky never contacts (0.00 N in all 128 candidates).  It is
#    geometrically short, not under-preloaded: commanded 0.092 rad closed it
#    settled 0.042 rad *past* its own target and still touched nothing, i.e. it
#    is closing into free space.  This traces to the calibration cutting
#    pinky_mcp_pitch from 1.40 to 1.14, the largest reduction of any joint.
#    More closure would only curl it until its middle phalanx fouls the handle.
# 2. thumb_cmc_pitch saturates: commanded 0.127, it settles at -0.0008, pinned
#    against its 0.0 lower limit, and the thumb holds through cmc_roll instead.
#    One of sixteen actuators therefore starts with no downward authority.  This
#    is worth watching on hardware -- a previous run froze after saturating
#    joints against their limits within 1.5 s.
#
# The result is a three-contact tripod (index on the cap, middle and thumb
# opposing on the body).  That satisfies the repo's release criterion, which is
# role-neutral by construction: FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT is 3 and
# CRITICAL_ROLE_NAMES is empty.  It also clears turn_motion_min_fingers = 2.0,
# so the co-motion authorization gate remains satisfiable.
TOPDOWN_TARGET_JOINT_POSITIONS = {
    "index_mcp_roll": 0.05414624541908073,
    "index_mcp_pitch": 0.5109091635356399,
    "index_pip": 0.9230436801027436,
    "index_dip": 0.8230780495476165,
    "middle_mcp_roll": 0.0006154768077150623,
    "middle_mcp_pitch": 0.7395448187059293,
    "middle_pip": 1.4238588934462029,
    "middle_dip": 1.2696549752859791,
    "ring_mcp_roll": -0.015733775224370507,
    "ring_mcp_pitch": 0.6325933668850876,
    "ring_pip": 0.6681227912284166,
    "ring_dip": 0.5957650929383791,
    "pinky_mcp_roll": 0.0458935652917647,
    "pinky_mcp_pitch": 0.6654397009052274,
    "pinky_pip": 1.0249280010611463,
    "pinky_dip": 0.9139282985462243,
    "thumb_cmc_yaw": 0.89291068971524,
    "thumb_cmc_roll": 1.073946524576329,
    "thumb_cmc_pitch": 0.20816280249540464,
    "thumb_mcp": 0.32568685578479317,
    "thumb_ip": 0.37841555773635116,
}

TOPDOWN_TARGET_POSITIONS = {
    "index": (0.05414624541908073, 0.5109091635356399, 0.9230436801027436),
    "middle": (0.0006154768077150623, 0.7395448187059293, 1.4238588934462029),
    "ring": (-0.015733775224370507, 0.6325933668850876, 0.6681227912284166),
    "pinky": (0.0458935652917647, 0.6654397009052274, 1.0249280010611463),
    "thumb": (
        0.89291068971524,
        1.073946524576329,
        0.20816280249540464,
        0.32568685578479317,
    ),
}


TOPDOWN_PREGRASP_POSITIONS = {
    finger: tuple(values) for finger, values in TOPDOWN_TARGET_POSITIONS.items()
}

# Backward-compatible name used by CPU tests and render tooling.
TOPDOWN_JOINT_POSITIONS = TOPDOWN_RESET_JOINT_POSITIONS
