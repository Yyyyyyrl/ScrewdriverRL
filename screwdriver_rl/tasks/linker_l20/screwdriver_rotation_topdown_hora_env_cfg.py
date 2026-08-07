"""HORA-faithful extrinsics-split validation variant of the top-down task.

Identical to ``LinkerL20ScrewdriverRotationTopdownEnvCfg`` (fixed 64 mm handle,
full deployability DR except geometry) except the ACTOR's latent encoder sees
only slow, per-episode-constant extrinsics (load proxy + contact friction)
instead of the full 20-D privileged vector.  The asymmetric critic keeps the
full state.

Purpose: test the hypothesis that routing fast object state through the latent
is what causes the Stage-2 adapter's closed-loop covariate-shift collapse.  If a
policy trained here closes the oracle-vs-adapter deployment gap, the fix is
architectural (this split), and we scale it to the full geometry-DR task.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from screwdriver_rl.tasks.linker_l20.screwdriver_rotation_topdown_env_cfg import (
    LinkerL20ScrewdriverRotationTopdownEnvCfg,
)


@configclass
class LinkerL20ScrewdriverRotationTopdownHoraEnvCfg(
    LinkerL20ScrewdriverRotationTopdownEnvCfg
):
    """Top-down 64 mm task with a HORA-faithful slow-extrinsics actor latent."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Bench-realistic load torque.  The inherited (0.5, 13.1) upper bound was
        # sized for a driven screw's tightening torque (~0.59 N.m); this bench is
        # a printed shaft in a socket base, whose resistance is far lower.  The
        # fall cross-tab confirmed load torque does not drive falls, so narrowing
        # is a faithfulness + adapter-inference win, not a stability fix.  Upper
        # bound is an estimate — refine if a torque-gauge / hanging-weight
        # measurement of the actual rig resistance becomes available.
        self.domain_rand.screwdriver_load_torque_range = (0.5, 4.0)

        # ---- Keep the pregrasp tension at episode start -----------------------
        # The shared top-down config snaps finger targets onto the settled
        # measured positions at the end of every reset.  Its stated reason is
        # that standing target penetration feeds the PhysX contact solver and
        # slowly self-rotates the mounted handle -- the "it turns by itself"
        # behaviour.  Measured on this task's posture, that reason does not hold
        # and the cost is large.
        #
        # Sweeping the retained fraction of pregrasp tension from 0 to 1
        # (64 envs, DR off, zero action, 250 steps, contact forces read from the
        # per-fingertip filtered sensors):
        #
        #   retain   index  middle   ring  pinky  thumb    tilt    drift rad/s
        #   0.00      0.53    8.05   1.26   0.10   0.01   0.032      0.08457
        #   0.40      0.37    6.71   1.71   0.85   1.27   0.037      0.08456
        #   1.00      1.52    3.73   1.58   1.84   3.46   0.046      0.08456
        #
        # The creep the snap exists to prevent moves in the sixth significant
        # figure, i.e. not at all; the self-rotation here is the handle turning
        # under its own load torque because a released grasp cannot hold it.
        # What the snap does change is the grasp: at zero tension the load piles
        # onto a single wedged finger (8.05 N on the middle against 0.01 N on the
        # thumb, a ratio of 805), while at full tension all five fingers carry
        # 1.5-3.7 N with every contact pad-facing (+0.165 to +0.874) and the
        # handle upright at 0.046 rad.
        #
        # Retained here only.  The shared top-down task keeps the snap: this is a
        # posture-specific measurement, not a refutation of the original finding.
        self.reset_zero_tension_targets = False

        # A 2026-08-06 experiment pinned both actor-latent inputs (contact
        # friction and load torque) to make the 2-D latent a per-episode
        # constant, which would have reduced Stage 2 to a trivial constant
        # regression that cannot suffer covariate-shift collapse.  REVERTED on
        # the ground that decides it: neither quantity has been measured on the
        # actual bench, so pinning them would tune the policy to a guessed
        # operating point.  Unknown hardware values are precisely the case that
        # randomisation plus online latent inference exists to cover, so both
        # stay randomised and Stage 2 stays a real training stage.
        #   contact friction: inherited (0.6, 2.5) from the top-down task
        #   load torque:      narrowed to (0.5, 4.0) just above, on bench realism
        # If a torque-gauge or friction measurement of the rig becomes
        # available, narrowing around the measurement -- not around a guess --
        # becomes the right move.
        # Actor latent = slow extrinsics only (load proxy + contact friction);
        # critic keeps the full privileged state.  Re-runs the observation-space
        # finalisation in the base __post_init__ via the flag.
        self.slow_extrinsics_only = True
        # HORA stacks 3 proprio frames for the actor.  With the object state out
        # of the latent, a single frame left the policy no temporal signal; the
        # first full run plateaued at NetTurns ~0.35 with UprightGate stuck at
        # 0.79, and on hardware it saturated every joint against its limit
        # within 1.5 s and froze (targets at the rail -> proprio constant ->
        # latent constant -> action constant).
        self.actor_frame_count = 3
        # Phase 3's turn weight destabilises the frame-stacked actor.  The first
        # 3-frame run led the single-frame baseline through Phase 1 and 2 (peak
        # NetTurns 0.427 vs 0.313, then 0.755 vs 0.463) and then degraded for the
        # 32M steps after the Phase 3 switch — OscRatio 0.25→0.44, reward
        # +4989→−2158 — while the single-frame run *improved* across the same
        # switch.  Eval agreed: authorized turns 0.726→0.465→0.215 over Phase 3
        # checkpoints, against a flat 0.85–0.98 across five Phase 2 checkpoints.
        # Hold the weight at Phase 2's value so the policy can finish the
        # curriculum (longer episodes, tighter termination) without the extra
        # turn pressure it cannot absorb.
        self.curriculum_phases[2].reward_turn_weight = (
            self.curriculum_phases[1].reward_turn_weight
        )

        # ---- A/B switch: price tilt additively instead of through the gate ----
        # 2026-08-04.  A broad reward redesign was tried here and REVERTED: it
        # froze the reward weights across phases, deleted the one-shot fall
        # penalty, widened the upright gate, retuned w_wrong and dropped the
        # milestone.  It raised the visible training metric (FwdVel peaked 0.214
        # vs the baseline's 0.170) and was, measured on the metrics that decide
        # deployment, far worse: ep450 scored bench fall 32.7% / authorized
        # turns 0.470 against ep330's 21.6% / 0.786 under this same file's
        # original reward, and ep645's 8.4% / 1.951.  The lesson is recorded
        # because it is easy to repeat: FwdVel is in the training log and
        # fall_rate is not, so four rounds of tuning optimised the visible
        # quantity and regressed the decisive one.
        #
        # Exactly one piece of that work survived on its own evidence.  A
        # three-arm probe (60M steps each, same seed) isolated the upright gate:
        # at std 0.35 the policy drifted into tilt and collapsed through Phase 2
        # (FwdVel 0.208 -> 0.113, UprightGate 0.932 -> 0.896); at std 0.20 it
        # held 0.02 rad of tilt but turned at a third of the rate; with the gate
        # effectively removed and the tilt price moved into the additive term at
        # 800, it matched the tight gate's tilt discipline (0.048 vs 0.041 rad)
        # while turning 2.1x faster, and recovered rather than degraded across
        # the Phase-2 boundary.  The reason is dimensional: at 0.10 rad the
        # additive cost is 200*0.01 = 2/step against a turn reward near 34/step,
        # so it cannot bind and the multiplicative gate was carrying the whole
        # trade-off — and a gate discounts turning rather than pricing tilt, so
        # no setting of it avoids both failures.
        #
        # That probe ran with the rest of the reverted redesign in place (no
        # fall penalty, stationary weights), so it was NOT established under this
        # file's original reward.  So it was made a switch, and the A/B was run:
        # `records/stage1_ab_20260807/`, two arms of 36.7 M samples on the same
        # seed, evaluated at a pinned phase over four checkpoints each.
        #
        # It won, and the switch is now the default.  Authorised net turns
        # averaged 2.58 -> 2.83 (+9.7% against a 5% decision threshold), winning
        # at all four checkpoints; fall rate stayed at or below 0.2%; success
        # rate 82.8-85.2% -> 91.8-93.0%.  It was also still improving at the
        # budget limit (2.76 -> 2.81 -> 2.85 -> 2.89) where the baseline had gone
        # flat, which is the property that matters for a 300 M-step run.
        #
        # The env var now selects the OLD behaviour, so the good configuration is
        # what you get by forgetting to set anything.  A long run must not depend
        # on an operator remembering to export a variable.
        if os.environ.get("DEXFORGE_HORA_GATED_TILT_PRICE") != "1":
            self.turn_upright_gate_std = 10.0
            self.reward_upright_weight = 800.0

        # ---- Diagnostic switch: physics rate, policy rate unchanged ----------
        # Watching a rollout showed the handle turning while the fingers barely
        # move: the coasting probe puts mean FwdVel WITHOUT contact at
        # 0.046 rad/s against a total of ~0.09, i.e. about half the credited
        # rotation happens with nothing touching it.  That is the signature of
        # PhysX solver creep on convex-hull colliders at 60 Hz, not of
        # manipulation.  Doubling only the physics rate (the policy stays at
        # 10 Hz, so the deployment contract and the proprio-history semantics
        # are untouched) tests whether the creep is a solver artefact.  Applied
        # at eval time on an already-trained policy: creep is a property of the
        # simulation, not of the weights, so it does not need a retrain to
        # measure.
        if os.environ.get("DEXFORGE_HORA_PHYSICS_120HZ") == "1":
            self.sim.dt = 1.0 / 120.0
            self.decimation = 12
            self.sim.render_interval = self.decimation

        # Re-finalise the actor observation space now that the flag is set (the
        # base __post_init__ ran before this assignment).
        import gymnasium as gym
        import numpy as np

        self.actor_extrinsics_dim = 2 + (
            2 if self.domain_rand.randomize_geometry else 0
        )
        obs_dim = self.history_obs_dim * self.actor_frame_count + self.actor_extrinsics_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
