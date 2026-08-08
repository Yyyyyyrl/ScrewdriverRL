"""Strict palm-down Linker L20 rotation task with a 64 mm screwdriver handle.

Everything not explicitly listed here is inherited from
``LinkerL20ScrewdriverRotationEnvCfg``: observations, actions, curriculum,
non-geometry domain randomisation, controllers, simulation settings, and PPO
configuration. The task changes are the thick-handle geometry, validated
reset/pregrasp, and role-neutral fingertip contact semantics.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .screwdriver_rotation_env_cfg import ASSET_ROOT, LinkerL20ScrewdriverRotationEnvCfg
from .screwdriver_rotation_topdown_posture import (
    TOPDOWN_HANDLE_RADIUS_M,
    TOPDOWN_JOINT_POSITIONS,
    TOPDOWN_PREGRASP_POSITIONS,
    TOPDOWN_RESET_POSITIONS,
    TOPDOWN_ROOT_POS_W,
    TOPDOWN_ROOT_QUAT_WXYZ,
    TOPDOWN_SCREWDRIVER_TILT_XY,
)

# Task-specific PhysX contact envelope.  The imported colliders otherwise leave
# these unauthored and inherit a much wider PhysX default that reports
# non-distal contact across the validated 4.9 mm ring-middle clearance.
TOPDOWN_HAND_CONTACT_OFFSET_M: float = 0.00025
TOPDOWN_SCREWDRIVER_CONTACT_OFFSET_M: float = 0.001
TOPDOWN_REST_OFFSET_M: float = 0.0

# The force-free plan specifies a 1:2:3 binary wrong-surface curriculum times a
# calibrated coefficient.  A formal 8,192-env run showed that the uncalibrated
# 1/2/3 values let the policy accept wrong-link contact on 55.5% of Phase-3
# samples: the resulting 1.666 mean cost was negligible beside 47.959 mean turn
# reward.  A 30x coefficient prices the final binary safety violation at 90 per
# step, making avoidance competitive with the exploit while preserving the
# planned phase ratio.  This override is topdown-only; the lateral task retains
# its historical weights.
TOPDOWN_WRONG_SURFACE_WEIGHTS: tuple[float, float, float] = (30.0, 60.0, 90.0)



@configclass
class LinkerL20ScrewdriverRotationTopdownEnvCfg(LinkerL20ScrewdriverRotationEnvCfg):
    """Palm-down task with thick-handle geometry and role-neutral contact."""

    # Used by radius-aware diagnostic geometry in the shared environment.  The
    # baseline task has no such field and therefore retains its 0.020 m default.
    screwdriver_handle_radius: float = TOPDOWN_HANDLE_RADIUS_M

    def __post_init__(self) -> None:
        # Geometry DR would replace the dedicated 64 mm asset with the baseline's
        # 34/40/46 mm variant bank, invalidating this task's fitted posture.  The
        # inherited baseline default is False; fail loudly if a caller changes it.
        if self.domain_rand.randomize_geometry:
            raise ValueError(
                "Isaac-LinkerL20-Screwdriver-Rotation-Topdown requires its fixed "
                "64 mm handle asset; geometry randomisation must remain disabled."
            )

        super().__post_init__()
        # Force-free starts no longer have passive target penetration to hold
        # the handle during the historical 5+10-step action-silence window.
        # A final-checkpoint 64 mm DR A/B showed that allowing policy response
        # immediately (with a short three-step ramp) cuts reset-edge falls from
        # 46.7% to 11.7% while improving authorized progress. Keep this
        # topdown-only so the lateral task retains its validated reset contract.
        self.reset_action_hold_steps = 0
        self.reset_action_ramp_steps = 3
        # The index-cap/thumb-opposition roles belong to the historical lateral
        # grasp. In the palm-down task any fingertip may work on any screwdriver
        # part; three simultaneous contacts authorize motion and fingers may
        # exchange roles during a turn.
        self.role_neutral_fingertip_contact = True
        self.role_neutral_min_contact_fingers = 3
        for phase, weight in zip(
            self.curriculum_phases, TOPDOWN_WRONG_SURFACE_WEIGHTS, strict=True
        ):
            phase.w_wrong = weight
        # Fixed distal-body-frame/pad offsets, not force targets; the shared
        # 12 mm distance-score ramp is preserved around each calibrated edge.
        #
        # Recalibrated 2026-08-06 for the refit posture by
        # ``tools/calibrate_linker_l20_topdown_contact_margins.py`` (128 envs,
        # 400 steps, DR on, 51,200 samples), choosing per finger the margin that
        # maximises agreement between the distance predicate and a >0.10 N
        # fingertip-force predicate.  Mean agreement 0.845.
        #
        # This is not housekeeping.  The superseded table was fitted to the old
        # fingertip positions, and against the new controller target the pinky's
        # frame clearance is 0.00936 m versus its old 0.0095 margin -- 0.14 mm of
        # slack -- so the distance model would have called the pinky "in contact"
        # while physics measured 0.00 N on it, inflating drive_count and
        # contact_gate and paying turn reward for contact that does not exist.
        # The recalibrated pinky and thumb edges both move *tighter*
        # (0.0095->0.0085, 0.0290->0.0229), which is the direction that removes
        # false positives.
        #
        # Caveat: the sample comes from uniform random actions at 0.35 scale, a
        # broader distribution than the force-gated teacher rollout behind the
        # superseded 90.16% figure, so the two agreement numbers are not directly
        # comparable.
        self.contact_d_margin_by_finger = {
            "index": 0.0098,
            "middle": 0.0108,
            "ring": 0.0125,
            "pinky": 0.0085,
            "thumb": 0.0229,
        }
        # M3 deployability DR: absolute contact friction and the universal-joint
        # tilt damping vary independently. Reset placement and encoder-bias
        # values are topdown-specific so shared tasks retain zero-noise resets.
        self.domain_rand.randomize_contact_friction = True
        self.domain_rand.contact_friction_range = (0.6, 2.5)
        # Bring the Stage-1 training support closer to the damping-x4 probe.
        # The probe retained 79.1% median turns with zero falls; widening only
        # the upper DR bound reduces that distribution shift without changing
        # the shared Linker task default.
        self.domain_rand.rotation_damping_range = (0.5, 3.0)
        self.domain_rand.randomize_tilt_damping = True
        self.domain_rand.tilt_damping_range = (0.5, 2.0)
        # No hardware torque traces are available.  Keep the low-resistance
        # probe while conservatively covering an M2 class-8.8 tightening-torque
        # proxy (0.317--0.392 N.m, Bossard 2025) plus 50% upper margin:
        # 0.045 N.m * (0.5, 13.1) = (0.0225, 0.5895) N.m.  Tightening torque is
        # intentionally documented as a proxy, not claimed as measured sliding
        # or breakaway torque for the eventual hardware joint.
        self.domain_rand.screwdriver_load_torque_range = (0.5, 13.1)
        # Operator-approved wide placement support.  The root is ramped from
        # nominal to the sampled pose during compliant settle; a fail-closed
        # distance guard then resamples only rows with fewer than three contacts.
        # Consequently the final distribution is contact-conditioned within the
        # ±8 mm envelope rather than an unconditional uniform square.
        self.domain_rand.reset_root_pos_noise_m = 0.008
        self.domain_rand.reset_root_z_noise_m = 0.002
        self.domain_rand.reset_root_tilt_noise_rad = 0.025
        self.domain_rand.reset_root_yaw_noise_rad = 0.09
        self.domain_rand.reset_screwdriver_tilt_noise_rad = 0.03
        self.domain_rand.joint_zero_bias_rad = 0.015
        self.reset_root_pose_ramp = True
        self.reset_contact_guard_min_fingers = 3
        self.reset_contact_guard_max_resamples = 64
        self.home_deviation_deadband = 0.15
        # Back-drive is physical loss even during a regrasp/contact dropout.
        # Scope this anti-gaming rule to topdown; lateral tasks stay unchanged.
        self.penalize_reverse_outside_contact = True
        # Price physical reverse displacement at the active forward-progress
        # weight. This keeps the objective target-free and linear: only true
        # positive net shaft progress remains profitable.
        self.match_reverse_weight_to_turn_weight = True
        # Optimize true upright shaft progress through finger exchanges. Contact
        # quality remains an auxiliary objective/diagnostic, not a hard mask on
        # the physical progress term.
        self.reward_physical_progress_outside_contact = True
        # Zero-tension episode start: without the snap, the standing pregrasp
        # target penetration (~0.19 rad measured on the validated posture) both
        # feeds the solver-creep artifact that self-rotates the mounted handle
        # and keeps the plain hold beyond pen_deadband, taxing grasp maintenance.
        self.reset_zero_tension_targets = True
        # Physical progress must be earned by fingers that move with the handle
        # surface; solver creep under motionless fingers pays nothing.  See the
        # co-motion gate in the Linker reward.
        self.turn_motion_authorized = True

        # Isaac Sim 5.1 imports collision subtrees as instance prims, which
        # prevents Isaac Lab's normal nested collision_props pass from reaching
        # the shapes.  De-instance only the source collision subtrees, author
        # the task-specific offsets, and then clone that validated source.
        from screwdriver_rl.utils.noninstance_collision_spawner import (
            spawn_urdf_with_authorable_collisions,
        )

        self.robot_cfg.spawn.func = spawn_urdf_with_authorable_collisions
        self.screwdriver_cfg.spawn.func = spawn_urdf_with_authorable_collisions

        self.screwdriver_cfg.spawn.asset_path = str(
            ASSET_ROOT / "screwdriver/screwdriver_64mm_handle.urdf"
        )

        # Keep the contact envelope smaller than the statically validated
        # non-distal/self clearances.  This override is intentionally scoped to
        # this config instance; the baseline task retains its imported defaults.
        self.robot_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=TOPDOWN_HAND_CONTACT_OFFSET_M,
            rest_offset=TOPDOWN_REST_OFFSET_M,
        )
        self.screwdriver_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=TOPDOWN_SCREWDRIVER_CONTACT_OFFSET_M,
            rest_offset=TOPDOWN_REST_OFFSET_M,
        )

        # Preserve the baseline's angular-speed semantics after r=20 -> 32 mm:
        # v_tangent = radius * omega.  Every other controller/reward scalar,
        # including the explicitly fixed 0.045 N.m load torque, is inherited.
        self.drive_full_tangential_speed = TOPDOWN_HANDLE_RADIUS_M

        self.robot_cfg.init_state.pos = TOPDOWN_ROOT_POS_W
        self.robot_cfg.init_state.rot = TOPDOWN_ROOT_QUAT_WXYZ
        self.robot_cfg.init_state.joint_pos = dict(TOPDOWN_JOINT_POSITIONS)
        self.reset_joint_positions = {
            finger: tuple(values)
            for finger, values in TOPDOWN_RESET_POSITIONS.items()
        }
        self.reset_target_ramp = True
        self.reset_pin_screwdriver_upright = True
        self.reset_screwdriver_tilt_xy = TOPDOWN_SCREWDRIVER_TILT_XY
        self.pregrasp_positions = {
            finger: tuple(values)
            for finger, values in TOPDOWN_PREGRASP_POSITIONS.items()
        }
