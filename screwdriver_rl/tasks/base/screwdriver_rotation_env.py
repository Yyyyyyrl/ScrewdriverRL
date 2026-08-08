"""Hand-agnostic continuous screwdriver rotation environment.

Task goal
---------
The screwdriver starts roughly vertical and the hand must spin it
continuously in one direction (negative-z by default) using fingertip
contacts only.  The screwdriver must remain upright throughout.

Failure modes explicitly penalised
------------------------------------
- Flick / slap / knock: contact gate requires fingertips near the handle AND
  moving; the screwdriver cannot coast for reward after contact is lost.
- Oscillation: reverse penalty (slightly above turn reward, same gates)
  makes back-and-forth net-zero; logged as the oscillation ratio.
- Tilt: multiplicative upright gate kills turn reward at moderate tilt.
- Proximal / palm contact: per-step penalty on non-fingertip link proximity.
- Thumb flip / flail: covered by action-rate penalty and joint clamping.

Adding a new hand
-----------------
Subclass :class:`ScrewdriverRotationEnv`, set the class attributes
``FINGER_JOINT_NAMES``, ``FINGERTIP_BODY_NAMES``, ``PROXIMAL_BODY_PATTERNS``
(and ``COUPLED_JOINTS`` for mimic/coupled distal joints), and supply a matching
``ScrewdriverRotationEnvCfg`` subclass with the hand's articulation, pregrasp,
gym spaces, and pad axis.  See ``tasks/allegro`` and ``tasks/linker_l20``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz

from screwdriver_rl.core import rewards
from screwdriver_rl.deploy.codecs import ProprioCodec, mounted_linker_g20_codec_spec
from .screwdriver_rotation_env_cfg import CurriculumPhaseCfg, ScrewdriverRotationEnvCfg


# ---------------------------------------------------------------------------
# Screwdriver body / joint constants (shared across hands — same asset)
# ---------------------------------------------------------------------------

# Screwdriver bodies used for distance queries (handle segment).
_SCREWDRIVER_HANDLE_BODIES = ("screwdriver_stick", "screwdriver_body", "screwdriver_cap")

# Joint names for the 3-DOF screwdriver mounting (Euler representation).
_SCREWDRIVER_EULER_JOINTS = (
    "table_screwdriver_joint_1",
    "table_screwdriver_joint_2",
    "table_screwdriver_joint_3",
)
_SCREWDRIVER_CAP_JOINT = "screwdriver_body_cap_joint"


class ScrewdriverRotationEnv(DirectRLEnv):
    """Continuous screwdriver rotation, hand-agnostic.

    Extends ``DirectRLEnv`` directly (no MFR dependency) and implements the full
    reward, observation, reset, and curriculum logic in one class.  Hand-specific
    joint/body maps are class attributes overridden by per-hand subclasses.
    """

    cfg: ScrewdriverRotationEnvCfg

    # -- Hand-specific maps (overridden by subclasses) ----------------------
    # Fingertip (distal pad) bodies — only these should touch the handle.
    FINGERTIP_BODY_NAMES: dict[str, str] = {}
    # Proximal/medial links to penalise when close to the handle (the links
    # BEHIND the fingertip: if they touch, the policy is using the wrong surface).
    PROXIMAL_BODY_PATTERNS: list[str] = []
    # Per-finger independent joint names (variable DOF per finger), semantic order.
    FINGER_JOINT_NAMES: dict[str, tuple[str, ...]] = {}
    # Coupled (mimic) joints: follower -> (master, multiplier, offset).  The
    # follower is driven each step as ``master_target * multiplier + offset``.
    # Empty for hands without mimic coupling (e.g. Allegro).  Robust to the URDF
    # importer either keeping the followers as independent DOFs (we drive them)
    # or collapsing them into PhysX constraints (the followers fail to resolve
    # and this becomes a no-op while PhysX enforces the coupling).
    COUPLED_JOINTS: dict[str, tuple[str, float, float]] = {}
    # Link pairs to exclude from self-collision checking.  Inflated convex-hull
    # collision shapes make non-adjacent links overlap at the grasp pose even
    # though the real geometry never touches (palm <-> own proximals, sibling
    # metacarpals); filtering those pairs lets self-collision be ON without the
    # spurious penetration-recovery instability, while keeping the physically
    # meaningful collisions (fingertip <-> fingertip, finger crossing).  Each
    # tuple is ``(link_a, link_b)`` (link names, order irrelevant).  Empty = none.
    SELF_COLLISION_FILTER_PAIRS: list[tuple[str, str]] = []

    def __init__(
        self,
        cfg: ScrewdriverRotationEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Curriculum state — resolved before super().__init__ so the
        # first call to episode_length_s uses Phase-1 value.
        self._curriculum_phase: CurriculumPhaseCfg = cfg.curriculum_phases[0]
        self._global_steps: int = 0

        super().__init__(cfg, render_mode, **kwargs)

        # ---- Finger joints ----
        self.fingers: tuple[str, ...] = tuple(cfg.fingers)
        self._finger_joint_ids_by_name: dict[str, list[int]] = self._resolve_finger_joints()
        self._finger_joint_ids: list[int] = [
            jid
            for finger in self.fingers
            for jid in self._finger_joint_ids_by_name[finger]
        ]
        self.num_finger_dofs: int = len(self._finger_joint_ids)

        # ---- Coupled (mimic) follower joints ----
        self._resolve_coupled_joints()

        # ---- Screwdriver joints ----
        self._screwdriver_euler_ids: list[int] = self._find_joints(
            self.screwdriver, _SCREWDRIVER_EULER_JOINTS
        )
        self._screwdriver_z_id: int = self._screwdriver_euler_ids[2]

        # ---- Body IDs ----
        self._fingertip_body_ids: list[int] = self._resolve_fingertip_bodies()
        self._proximal_body_ids: list[int] = self._resolve_proximal_bodies()
        self._handle_body_ids: list[int] = self._resolve_handle_bodies()
        # Indices into _handle_body_ids for axis computation (handle base, cap).
        self._handle_base_idx: int = 1   # screwdriver_body
        self._handle_cap_idx: int = 2    # screwdriver_cap
        self._shaft_idx: int = 0         # screwdriver_stick (shaft axis ref)

        # Thumb index within active fingers for near-score weighting.
        self._thumb_tip_idx: int | None = (
            self.fingers.index("thumb") if "thumb" in self.fingers else None
        )
        self._non_thumb_tip_idxs: list[int] = [
            i for i, f in enumerate(self.fingers) if f != "thumb"
        ]

        # ---- Geometry-variant identification ----
        # Must precede the pregrasp / home-target build below (and any subclass
        # post-super construction) so per-env postures can be gathered by bucket.
        self._identify_geometry_variants()

        # ---- Finger target and pregrasp defaults ----
        self._default_finger_pos: torch.Tensor = self._make_default_finger_pos()
        self._cur_targets: torch.Tensor = self._default_finger_pos.clone()
        finger_limits = self.allegro.data.soft_joint_pos_limits[:, self._finger_joint_ids]
        margin = float(cfg.joint_target_margin)
        self._finger_lower = finger_limits[..., 0] + margin
        self._finger_upper = finger_limits[..., 1] - margin

        # Per-finger reset posture.  When geometry DR is on with a per-bucket
        # table this is ``(num_buckets, n_joints)`` and reset gathers each env's
        # row by bucket; otherwise it is the single shared ``(n_joints,)`` vector.
        self._pregrasp_pos: dict[str, torch.Tensor] = self._make_pregrasp_table()
        # Optional collision-safe state distinct from the contact/home target.
        # None preserves the legacy reset path exactly for every existing task.
        self._reset_joint_pos: dict[str, torch.Tensor] | None = (
            self._make_reset_joint_table()
        )

        # Per-bucket hand-root offset (world xyz) for handle-length compensation.
        # ``(num_buckets, 3)`` under geometry DR with a cfg table, else ``None``.
        self._pregrasp_root_offset: torch.Tensor | None = (
            self._make_pregrasp_root_offset_table()
        )
        self._pregrasp_root_quat: torch.Tensor | None = (
            self._make_pregrasp_root_quat_table()
        )
        self._reset_screwdriver_tilt_xy: torch.Tensor = (
            self._make_reset_screwdriver_tilt_xy_table()
        )

        # ---- Continuous-turn tracking ----
        self._policy_dt: float = float(cfg.decimation) * float(cfg.sim.dt)
        self._prev_z = self.screwdriver.data.joint_pos[
            :, self._screwdriver_z_id
        ].detach().clone()
        self._prev_tilt_xy = self.screwdriver.data.joint_pos[
            :, self._screwdriver_euler_ids[:2]
        ].detach().clone()
        self._total_turn = torch.zeros(self.num_envs, device=self.device)
        self._net_turn = torch.zeros(self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(
            (self.num_envs, self.num_finger_dofs), device=self.device
        )
        self._prev_milestone_count = torch.zeros(self.num_envs, device=self.device)
        self._prev_shaft_quat: torch.Tensor | None = None

        # ---- Episode-outcome history ----
        # fall_rate and per-episode authorized net turns are the metrics that
        # decide deployment, and they are episode-level: neither can be read off
        # a per-step mean.  Without them in the training log, checkpoint choice
        # falls back to FwdVel, which has already once selected a checkpoint
        # that was 5x worse on fall rate.  Sampled at reset, kept in a ring
        # buffer, published as scalars.
        self._ep_outcome_capacity = 1024
        self._ep_fall_hist = torch.zeros(self._ep_outcome_capacity, device=self.device)
        self._ep_turn_hist = torch.zeros(self._ep_outcome_capacity, device=self.device)
        self._ep_outcome_cursor = 0
        self._ep_outcome_filled = 0
        self._last_terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # ---- Per-env domain randomisation state ----
        # Tracks each env's current rotation damping so _compute_privileged_obs
        # can expose the actual value rather than a fixed constant.
        _base_rot_damp = cfg.screwdriver_cfg.actuators["rotation"].damping
        self._env_rotation_damping = torch.full(
            (self.num_envs,), _base_rot_damp, dtype=torch.float32, device=self.device
        )
        self._base_rotation_damping: float = _base_rot_damp

        # Per-env screwdriver rotational load (Coulomb friction of a screw).
        self._base_load_torque: float = float(cfg.screwdriver_load_torque)
        self._env_load_torque = torch.full(
            (self.num_envs,), self._base_load_torque, dtype=torch.float32, device=self.device
        )

        # Base finger PD gains (from the actuator cfg) for domain randomisation.
        _fing_act = cfg.robot_cfg.actuators["fingers"]
        self._base_finger_stiffness: float = float(_fing_act.stiffness)
        self._base_finger_damping: float = float(_fing_act.damping)

        # Per-env real contact friction (init to the task's base friction); the
        # privileged obs exposes this when contact-friction DR is enabled.
        _base_friction = float(getattr(cfg, "friction_coefficient", 1.5))
        self._base_friction: float = _base_friction
        self._env_friction = torch.full(
            (self.num_envs,), _base_friction, dtype=torch.float32, device=self.device
        )
        # Base tilt-joint (universal-joint bearing) damping for tilt-damping DR.
        self._base_tilt_damping: float = float(
            cfg.screwdriver_cfg.actuators["tilt"].damping
        )

        # Per-episode, phase-scaled placement/calibration errors. These buffers
        # are sampled once at reset and remain constant until the next reset.
        self._env_reset_root_pos_noise = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._env_reset_root_rpy_noise = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._env_reset_screwdriver_tilt_noise = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self._env_joint_position_bias = torch.zeros(
            (self.num_envs, self.num_finger_dofs),
            dtype=torch.float32,
            device=self.device,
        )

        # ---- RMA / asymmetric observations ----
        self._prop_hist_buf = torch.zeros(
            (self.num_envs, cfg.prop_hist_len, cfg.history_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self._proprio_codec = ProprioCodec(
            mounted_linker_g20_codec_spec(
                cfg.prop_hist_len, int(getattr(cfg, "actor_frame_count", 1))
            )
        )
        if round(self._policy_dt * 1_000_000_000) != self._proprio_codec.spec.control_period_ns:
            raise ValueError("mounted policy rate does not match its ProprioCodec")

        # Stagger episode starts to avoid synchronised reset artifacts.
        self.episode_length_buf = torch.randint(
            0, self.max_episode_length, (self.num_envs,),
            device=self.device, dtype=self.episode_length_buf.dtype,
        )

        # ---- Logging helper ----
        # ``_current_epoch`` is written each epoch by the rl_games
        # PhaseCheckpointObserver; the env itself has no epoch concept.
        self._current_epoch: int = 0
        # ``_log_stage`` selects the terminal-log layout: 1 = Stage-1 PPO
        # (per-step, full reward breakdown), 2 = Stage-2 adaptation (compact,
        # once-per-iter, driven by ProprioAdaptTrainer).  In Stage 2 the env's
        # per-step logging is suppressed and the trainer drives the logger.
        self._log_stage: int = 1
        self._stage2_loss: float = float("nan")
        from screwdriver_rl.utils.logging import RotationTrainingLogger
        # The logger counts aggregate environment samples. Scale by num_envs so
        # large vectorized jobs do not print once per policy step.
        self._logger = RotationTrainingLogger(
            log_interval_steps=max(2000, 1200 * self.num_envs)
        )

    # -----------------------------------------------------------------------
    # Scene
    # -----------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self.allegro = Articulation(self.cfg.robot_cfg)
        self.screwdriver = Articulation(self.cfg.screwdriver_cfg)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.cfg.friction_coefficient,
                    dynamic_friction=self.cfg.friction_coefficient,
                )
            ),
        )
        # Clone the environments + apply self-collision / inter-env collision
        # filtering (the replicate-physics vs per-env-geometry paths differ).
        self._finalize_scene()
        self.scene.articulations["allegro"] = self.allegro
        self.scene.articulations["screwdriver"] = self.screwdriver
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _finalize_scene(self) -> None:
        """Clone environments and apply collision filtering.

        ``replicate_physics=True`` (homogeneous): apply the self-collision filters
        to ``env_0`` and clone — the cloner replicates both the assets and the
        filters to every env.

        ``replicate_physics=False`` (per-env geometry DR): ``InteractiveScene``
        already cloned the env *xforms* in its ``__init__`` and the multi-asset
        spawner populated each env with its own variant when the Articulations
        were created above.  Calling ``clone_environments`` again would copy
        ``env_0`` over all envs and **collapse every env to one variant**, so we
        must NOT re-clone.  Instead apply the self-collision filters to *every*
        env (they cannot ride the cloner) and filter inter-env collisions.
        """
        if self.scene.cfg.replicate_physics:
            self._apply_self_collision_filters(env_indices=(0,))
            self.scene.clone_environments(copy_from_source=False)
        else:
            self._apply_self_collision_filters(env_indices=range(self.scene.num_envs))
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _apply_self_collision_filters(self, env_indices=(0,)) -> None:
        """Exclude ``SELF_COLLISION_FILTER_PAIRS`` from self-collision checking.

        ``env_indices`` selects which envs to author the filters on: ``(0,)`` for
        the replicate-physics path (the cloner copies them to the rest) or every
        env for the per-env-geometry path (no cloning happens afterwards).  Uses
        USD ``FilteredPairsAPI`` (no high-level Isaac Lab helper exists).  No-op
        when the list is empty.
        """
        if not self.SELF_COLLISION_FILTER_PAIRS:
            return
        import omni.usd
        from pxr import Sdf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        env_indices = list(env_indices)
        applied = 0
        for i in env_indices:
            # robot_cfg.prim_path is e.g. "/World/envs/env_.*/LinkerHand".
            base_path = self.cfg.robot_cfg.prim_path.replace(".*", str(i))
            for link_a, link_b in self.SELF_COLLISION_FILTER_PAIRS:
                a_path = f"{base_path}/{link_a}"
                b_path = f"{base_path}/{link_b}"
                prim_a = stage.GetPrimAtPath(a_path)
                if not prim_a.IsValid() or not stage.GetPrimAtPath(b_path).IsValid():
                    if i == env_indices[0]:
                        print(
                            f"[self-collision-filter] WARN: missing prim {a_path} or {b_path}",
                            flush=True,
                        )
                    continue
                api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
                api.CreateFilteredPairsRel().AddTarget(Sdf.Path(b_path))
                applied += 1
        print(
            f"[self-collision-filter] applied {applied} filtered pairs over "
            f"{len(env_indices)} env(s)",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Curriculum
    # -----------------------------------------------------------------------

    def _update_curriculum(self) -> None:
        """Select the curriculum phase based on global step count.

        Called at the start of each ``_pre_physics_step`` so that the
        phase switches take effect at the same step they are logged.
        """
        phases = self.cfg.curriculum_phases
        active = phases[0]
        for phase in phases:
            if self._global_steps >= phase.step_start:
                active = phase
        if active is not self._curriculum_phase:
            old_name = f"Phase @{self._curriculum_phase.step_start:,}"
            new_name = f"Phase @{active.step_start:,}"
            print(
                f"\n{'='*60}\n"
                f"  CURRICULUM TRANSITION: {old_name}  →  {new_name}\n"
                f"  Global steps : {self._global_steps:,}\n"
                f"  turn_weight  : {self._curriculum_phase.reward_turn_weight}"
                f"  →  {active.reward_turn_weight}\n"
                f"  contact_dist : {self._curriculum_phase.turn_reward_contact_distance}"
                f"  →  {active.turn_reward_contact_distance}\n"
                f"  episode_s    : {self._curriculum_phase.episode_length_s}"
                f"  →  {active.episode_length_s}\n"
                f"{'='*60}\n",
                flush=True,
            )
            self._curriculum_phase = active
            # Extend the episode length for the new phase.  ``max_episode_length``
            # is a read-only property derived from ``cfg.episode_length_s``, so we
            # update the config field and let the property recompute (with the
            # same math.ceil the base env uses).
            self.cfg.episode_length_s = active.episode_length_s

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._global_steps += self.num_envs
        self._update_curriculum()

        if self.cfg.action_clip > 0.0:
            actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        self.actions = actions.clone()

        # HORA-style delta: target accumulates, action=0 holds current grip.
        target = self._cur_targets + self.cfg.action_delta_scale * actions
        target = torch.clamp(target, self._finger_lower, self._finger_upper)
        self._cur_targets = target

        # NOTE: the RMA history buffer is appended in _get_observations(), not
        # here.  Appending pre-physics would store the joint positions from
        # *before* this target was applied, while ``DeployPolicy.act`` appends
        # the freshly measured joints — a ~0.03 rad sim/deploy skew in every
        # history frame the adapter (and, with frame stacking, the actor) reads.

    def _apply_action(self) -> None:
        self.allegro.set_joint_position_target(
            self._cur_targets, joint_ids=self._finger_joint_ids
        )
        self._apply_coupled_joint_targets()
        self._apply_screwdriver_load()

    def _apply_coupled_joint_targets(self) -> None:
        """Drive mimic/coupled followers from their master's current target.

        No-op when the hand has no coupled joints, or when the URDF importer
        collapsed the mimic joints into PhysX constraints (the followers did not
        resolve as independent DOFs and PhysX enforces the coupling itself).
        """
        if self._coupled_mult is None:
            return
        masters = self._cur_targets.index_select(1, self._coupled_master_cols_t)
        follower_targets = masters * self._coupled_mult + self._coupled_offset
        self.allegro.set_joint_position_target(
            follower_targets, joint_ids=self._coupled_follower_ids
        )

    def _apply_screwdriver_load(self) -> None:
        """Apply a resistive torque to the screwdriver rotation joint, modelling
        the friction of driving a real screw.

        Recomputed every physics substep from the current joint velocity and
        injected as a feed-forward joint effort (the implicit actuator adds it
        on top of its bearing damping).  The Coulomb term is smoothed through
        zero with ``tanh`` to avoid solver chatter at standstill.  No-op when
        both load components are disabled.
        """
        if self._base_load_torque <= 0.0 and self.cfg.screwdriver_load_viscous <= 0.0:
            return
        # The curriculum ramps the load in (phase scale 0 → 1) so the policy is
        # not crushed by resistance before it can rotate at all.
        phase_scale = float(self._curriculum_phase.screwdriver_load_scale)
        if phase_scale <= 0.0:
            return
        omega = self.screwdriver.data.joint_vel[:, self._screwdriver_z_id]  # (N,)
        eps = max(float(self.cfg.screwdriver_load_omega_eps), 1e-6)
        coulomb = self._env_load_torque * torch.tanh(omega / eps)
        viscous = float(self.cfg.screwdriver_load_viscous) * omega
        load = -phase_scale * (coulomb + viscous).unsqueeze(-1)  # oppose motion, (N, 1)
        self.screwdriver.set_joint_effort_target(load, joint_ids=[self._screwdriver_z_id])

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _observed_finger_q(
        self, finger_q: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the episode-constant encoder zero bias to policy proprioception."""
        bias = getattr(self, "_env_joint_position_bias", None)
        if bias is None:
            return finger_q
        return finger_q + (bias if env_ids is None else bias[env_ids])

    def _get_observations(self) -> dict[str, torch.Tensor]:
        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        observed_finger_q = self._observed_finger_q(finger_q)
        dr = self.cfg.domain_rand

        # Append the history frame here (post-physics), so the newest frame pairs
        # the joints just measured with the target that produced them — exactly
        # what ``DeployPolicy.act`` appends on hardware.
        #
        # The buffer feeds the Stage-2 adapter (asymmetric_obs) AND, once frames
        # are stacked, the actor itself.  Gating it on asymmetric_obs alone would
        # hand a frame-stacked actor three copies of a frozen reset frame in any
        # run that leaves asymmetric_obs off (e.g. plain oracle eval), which looks
        # like a catastrophically bad policy rather than a missing update.
        if self.cfg.asymmetric_obs or int(getattr(self.cfg, "actor_frame_count", 1)) > 1:
            self._update_prop_hist()

        if getattr(self.cfg, "latent_conditioned", False):
            # HORA-faithful deployable mode: actor obs = [proprio, privileged].
            # The raw euler is dropped from the actor obs; it lives inside the
            # privileged tail, which the custom network encodes into a latent.
            # Observation noise is applied only to the proprioceptive block (the
            # real sensors); the privileged tail is fed clean so it stays the
            # exact quantity the critic sees and the Stage-2 adapter regresses.
            # Frame stacking: the actor sees the last ``actor_frame_count``
            # proprio frames (HORA stacks 3).  A single frame carries no velocity
            # or phase information, so with the object state removed from the
            # latent the policy had no temporal signal at all.  Assembled from
            # the same history buffer ``DeployPolicy`` uses, via the same codec,
            # so sim and hardware build this vector identically.
            k = int(getattr(self.cfg, "actor_frame_count", 1))
            if k > 1:
                proprio = self._proprio_codec.assemble_actor_input(self._prop_hist_buf)
            else:
                proprio = self._proprio_codec.encode_frame(
                    observed_finger_q, self._cur_targets
                )
            if dr.enabled and dr.obs_noise_std > 0.0:
                proprio = proprio + torch.randn_like(proprio) * dr.obs_noise_std
            priv = self._compute_privileged_obs()
            # HORA-faithful split: the actor's latent encoder sees only slow
            # extrinsics; the asymmetric critic still gets the full state.
            if getattr(self.cfg, "slow_extrinsics_only", False):
                actor_priv = self._compute_actor_extrinsics()
            else:
                actor_priv = priv
            result: dict[str, torch.Tensor] = {
                "policy": torch.cat([proprio, actor_priv], dim=-1)
            }
            if self.cfg.asymmetric_obs:
                result["critic"] = priv
                result["proprio_hist"] = self._prop_hist_buf.clone()
            return result

        # Legacy mode: the 3-D screwdriver euler is part of the actor obs.
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        obs = torch.cat([observed_finger_q, self._cur_targets, euler], dim=-1)
        if dr.enabled and dr.obs_noise_std > 0.0:
            obs = obs + torch.randn_like(obs) * dr.obs_noise_std
        result = {"policy": obs}
        if self.cfg.asymmetric_obs:
            result["critic"] = self._compute_privileged_obs()
            result["proprio_hist"] = self._prop_hist_buf.clone()
        return result

    # -----------------------------------------------------------------------
    # Rewards
    # -----------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        phase = self._curriculum_phase
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        z_curr = euler[:, 2]

        # ---- Rotation delta ----
        raw_delta_z = self.cfg.turn_direction * (z_curr - self._prev_z)
        # Wrap to (−π, π] so coordinate resets don't produce giant deltas.
        delta_z = rewards.wrap_to_pi(raw_delta_z)
        self._prev_z = z_curr.detach().clone()

        # Prefer true shaft-axis spin over Euler-z (which includes precession).
        if self.cfg.use_shaft_spin_measure:
            shaft_delta = self._compute_shaft_spin_delta()
            if shaft_delta is not None:
                delta_z = shaft_delta

        turn_vel, fwd_vel, rev_vel = rewards.turn_velocities(
            delta_z, self._policy_dt, self.cfg.turn_velocity_clip
        )

        # ---- Upright gate (multiplicative — see cfg for rationale) ----
        tilt_norm = torch.linalg.norm(euler[:, :2], dim=-1)
        upright_gate = rewards.upright_gate(tilt_norm, self.cfg.turn_upright_gate_std)

        # ---- Contact gate ----
        contact_gate = self._compute_contact_gate(phase)

        combined_gate = contact_gate * upright_gate

        # ---- Core turn/reverse rewards ----
        turn_reward = phase.reward_turn_weight * fwd_vel * combined_gate
        reverse_cost = self.cfg.reward_reverse_weight * rev_vel * combined_gate

        # ---- Progress tracking ----
        self._total_turn += torch.clamp(delta_z, min=0.0).detach()
        self._net_turn += delta_z.detach()
        milestone_reward = self._compute_milestone_reward(gate=combined_gate)

        # ---- Upright cost ----
        tilt_xy = euler[:, :2]
        upright_cost = self.cfg.reward_upright_weight * torch.sum(tilt_xy ** 2, dim=-1)

        tilt_vel = (tilt_xy - self._prev_tilt_xy) / self._policy_dt
        self._prev_tilt_xy = tilt_xy.detach().clone()
        tilt_vel_cost = self.cfg.reward_tilt_velocity_weight * torch.linalg.norm(tilt_vel, ord=1, dim=-1)

        # ---- Regularisation ----
        action_cost = self.cfg.reward_action_weight * torch.sum(self.actions ** 2, dim=-1)
        action_rate_cost = self.cfg.reward_action_rate_weight * torch.mean(
            (self.actions - self._prev_actions) ** 2, dim=-1
        )
        self._prev_actions = self.actions.detach().clone()

        finger_vel = self.allegro.data.joint_vel[:, self._finger_joint_ids]
        finger_vel_cost = self.cfg.reward_finger_velocity_weight * torch.mean(finger_vel ** 2, dim=-1)

        # ---- Near-reward (fingertip proximity to handle axis) ----
        tip_dist = self._compute_fingertip_axis_distances()     # (N, num_fingers)
        near_reward = self._compute_near_reward(tip_dist, phase.near_reward_weight)

        # ---- Proximal-link penalty ----
        proximal_cost = self._compute_proximal_penalty(phase.reward_proximal_penalty_weight)

        reward = (
            turn_reward
            + milestone_reward
            + near_reward
            - reverse_cost
            - upright_cost
            - tilt_vel_cost
            - action_cost
            - action_rate_cost
            - finger_vel_cost
            - proximal_cost
        )

        # ---- Logging extras ----
        osc_ratio = (self._total_turn - self._net_turn.clamp(min=0.0)) / (self._total_turn + 1e-6)
        self.extras.update({
            # Progress
            "eval_total_turns":    (self._total_turn / (2.0 * math.pi)).detach(),
            "eval_net_turns":      (self._net_turn / (2.0 * math.pi)).detach(),
            "eval_osc_ratio":      osc_ratio.detach(),
            "eval_turn_vel":       turn_vel.detach(),
            "eval_fwd_vel":        fwd_vel.detach(),
            "eval_rev_vel":        rev_vel.detach(),
            # Gates
            "eval_upright_gate":   upright_gate.detach(),
            "eval_contact_gate":   contact_gate.detach(),
            # Tilt
            "eval_tilt_norm":      tilt_norm.detach(),
            "eval_upright_cost":   upright_cost.detach(),
            "eval_tilt_vel_cost":  tilt_vel_cost.detach(),
            # Reward breakdown
            "eval_turn_reward":    turn_reward.detach(),
            "eval_reverse_cost":   reverse_cost.detach(),
            "eval_milestone":      milestone_reward.detach(),
            "eval_near_reward":    near_reward.detach(),
            "eval_proximal_cost":  proximal_cost.detach(),
            "eval_action_cost":    action_cost.detach(),
            "eval_action_rate":    action_rate_cost.detach(),
            "eval_total_reward":   reward.detach(),
            # Contact
            "eval_mean_tip_dist":  tip_dist.mean(dim=-1).detach() if tip_dist.numel() > 0 else torch.zeros(self.num_envs, device=self.device),
            "eval_min_tip_dist":   tip_dist.min(dim=-1).values.detach() if tip_dist.numel() > 0 else torch.zeros(self.num_envs, device=self.device),
            # Curriculum — emit a human 1-indexed phase number and the total
            # phase count so the logger can show "Phase n/total".
            "eval_curriculum_phase": torch.full(
                (self.num_envs,),
                float(self.cfg.curriculum_phases.index(self._curriculum_phase) + 1),
                device=self.device,
            ),
            "eval_num_phases": torch.full(
                (self.num_envs,), float(len(self.cfg.curriculum_phases)), device=self.device
            ),
        })

        # Periodic terminal log.  ``_current_epoch`` is synced by the rl_games
        # PhaseCheckpointObserver (the true epoch lives at the Runner level).
        # Stage 2 suppresses the per-step log — ProprioAdaptTrainer drives the
        # logger once per adaptation iter instead.
        if self._log_stage == 1:
            self._logger.log(self._global_steps, self.extras, epoch=self._current_epoch)

        return torch.nan_to_num(reward, nan=-1.0e6)

    # -----------------------------------------------------------------------
    # Dones
    # -----------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        tilt_norm = torch.linalg.norm(euler[:, :2], dim=-1)
        threshold = float(self._curriculum_phase.upright_termination_threshold)
        terminated = tilt_norm > threshold
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["eval_tilt_terminated"] = terminated.detach()
        self._last_terminated = terminated.detach()

        # Episode-level outcomes are published HERE, not from _get_rewards.
        # LinkerL20 overrides _get_rewards wholesale without calling super(), so
        # anything written to extras there never reaches that task's logger --
        # which is exactly what happened: the block printed on schedule with the
        # EpFallRate line silently absent.  _get_dones is base-only, runs every
        # step for every task, and already owns _last_terminated.
        if self._ep_outcome_filled > 0:
            window = slice(0, self._ep_outcome_filled)
            ep_fall_rate = self._ep_fall_hist[window].mean()
            ep_net_turns = self._ep_turn_hist[window].mean()
        else:
            ep_fall_rate = torch.zeros((), device=self.device)
            ep_net_turns = torch.zeros((), device=self.device)
        self.extras.update({
            "eval_ep_fall_rate": ep_fall_rate.detach(),
            "eval_ep_net_turns": ep_net_turns.detach(),
            "eval_ep_outcome_n": torch.tensor(
                float(self._ep_outcome_filled), device=self.device
            ),
        })
        return terminated, timed_out

    def _record_episode_outcomes(self, env_ids: torch.Tensor) -> None:
        """Push finished episodes' fall flag and authorized net turns into the ring.

        Called from ``_reset_idx`` before the per-env accumulators are cleared.
        Envs with ``episode_length_buf == 0`` are startup resets, not finished
        episodes, and are skipped.
        """
        if env_ids.numel() == 0:
            return
        real = env_ids[self.episode_length_buf[env_ids] > 0]
        if real.numel() == 0:
            return
        fell = self._last_terminated[real].to(dtype=torch.float32)
        turns = (self._net_turn[real] / (2.0 * math.pi)).to(dtype=torch.float32)
        count = int(real.numel())
        capacity = self._ep_outcome_capacity
        # Only the most recent `capacity` entries can survive; drop any excess
        # so the wrap-around index arithmetic stays a single contiguous slice.
        if count > capacity:
            fell, turns, count = fell[-capacity:], turns[-capacity:], capacity
        start = self._ep_outcome_cursor
        index = (torch.arange(count, device=self.device) + start) % capacity
        self._ep_fall_hist[index] = fell
        self._ep_turn_hist[index] = turns
        self._ep_outcome_cursor = (start + count) % capacity
        self._ep_outcome_filled = min(self._ep_outcome_filled + count, capacity)

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _sample_reset_calibration_noise(self, env_ids: torch.Tensor) -> None:
        """Sample phase-scaled placement and encoder errors once per episode."""
        dr = self.cfg.domain_rand
        limits = (
            dr.reset_root_pos_noise_m,
            dr.reset_root_z_noise_m,
            dr.reset_root_tilt_noise_rad,
            dr.reset_root_yaw_noise_rad,
            dr.reset_screwdriver_tilt_noise_rad,
            dr.joint_zero_bias_rad,
        )
        if any(float(value) < 0.0 for value in limits):
            raise ValueError("reset/calibration randomisation limits must be non-negative")

        self._env_reset_root_pos_noise[env_ids] = 0.0
        self._env_reset_root_rpy_noise[env_ids] = 0.0
        self._env_reset_screwdriver_tilt_noise[env_ids] = 0.0
        self._env_joint_position_bias[env_ids] = 0.0
        if not dr.enabled:
            return

        scale = float(
            getattr(
                self._curriculum_phase,
                "dynamics_randomization_scale",
                1.0,
            )
        )
        if not 0.0 <= scale <= 1.0:
            raise ValueError("dynamics_randomization_scale must be in [0, 1]")
        n = int(env_ids.numel())

        pos_limits = scale * torch.tensor(
            [
                dr.reset_root_pos_noise_m,
                dr.reset_root_pos_noise_m,
                dr.reset_root_z_noise_m,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        rpy_limits = scale * torch.tensor(
            [
                dr.reset_root_tilt_noise_rad,
                dr.reset_root_tilt_noise_rad,
                dr.reset_root_yaw_noise_rad,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._env_reset_root_pos_noise[env_ids] = (
            torch.empty((n, 3), device=self.device).uniform_(-1.0, 1.0)
            * pos_limits
        )
        self._env_reset_root_rpy_noise[env_ids] = (
            torch.empty((n, 3), device=self.device).uniform_(-1.0, 1.0)
            * rpy_limits
        )
        self._env_reset_screwdriver_tilt_noise[env_ids] = (
            torch.empty((n, 2), device=self.device).uniform_(-1.0, 1.0)
            * (scale * float(dr.reset_screwdriver_tilt_noise_rad))
        )
        self._env_joint_position_bias[env_ids] = (
            torch.empty((n, self.num_finger_dofs), device=self.device)
            .uniform_(-1.0, 1.0)
            * (scale * float(dr.joint_zero_bias_rad))
        )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.allegro._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)

        # Before super() clears episode_length_buf and the accumulators below.
        if hasattr(self, "_ep_fall_hist"):
            self._record_episode_outcomes(env_ids)

        super()._reset_idx(env_ids)
        if hasattr(self, "_env_reset_root_pos_noise"):
            self._sample_reset_calibration_noise(env_ids)

        # ---- Hand ----
        root = self.allegro.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        # Per-bucket length compensation: shift the whole hand (world frame) so the
        # grasp tracks each env's handle length; the cap rises with a longer handle.
        if self._pregrasp_root_offset is not None:
            root[:, :3] += self._pregrasp_root_offset[self._env_bucket_idx[env_ids]]
        if self._pregrasp_root_quat is not None:
            root[:, 3:7] = self._pregrasp_root_quat[self._env_bucket_idx[env_ids]]
        nominal_root_pose = root[:, :7].clone()
        if hasattr(self, "_env_reset_root_pos_noise"):
            root[:, :3] += self._env_reset_root_pos_noise[env_ids]
            rpy = self._env_reset_root_rpy_noise[env_ids]
            delta_quat = quat_from_euler_xyz(
                rpy[:, 0],
                rpy[:, 1],
                rpy[:, 2],
            )
            # World-frame calibration perturbation around the resolved nominal
            # or per-geometry root orientation.
            root[:, 3:7] = rewards.quat_mul(delta_quat, root[:, 3:7])
        ramp_root_pose = bool(
            self.cfg.reset_root_pose_ramp and self.cfg.reset_contact_steps > 0
        )
        initial_root_pose = nominal_root_pose if ramp_root_pose else root[:, :7]
        self.allegro.write_root_pose_to_sim(initial_root_pose, env_ids=env_ids)
        self.allegro.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

        jpos = self.allegro.data.default_joint_pos[env_ids].clone()
        jvel = torch.zeros_like(self.allegro.data.default_joint_vel[env_ids])
        # Per-bucket geometry DR: gather each env's diameter/length-bucket posture;
        # otherwise broadcast the single shared posture.  The compliant 32-step
        # settle below then closes the fingers to contact, absorbing residual
        # mismatch between the seeded posture and the env's actual handle.
        bucket_of = (
            self._env_bucket_idx[env_ids] if self._env_bucket_idx is not None else None
        )
        target_jpos = jpos.clone()
        for finger, jids in self._finger_joint_ids_by_name.items():
            target_posture = self._pregrasp_pos[finger]
            target_jpos[:, jids] = (
                target_posture[bucket_of] if bucket_of is not None else target_posture
            )
            if self._reset_joint_pos is None:
                jpos[:, jids] = target_jpos[:, jids]
            else:
                reset_posture = self._reset_joint_pos[finger]
                jpos[:, jids] = (
                    reset_posture[bucket_of]
                    if bucket_of is not None and reset_posture.ndim == 2
                    else reset_posture
                )
        # Set mimic/coupled followers consistently in both state and target.
        if self._coupled_mult is not None:
            target_masters = target_jpos[:, self._coupled_master_joint_ids]
            target_jpos[:, self._coupled_follower_ids] = (
                target_masters * self._coupled_mult + self._coupled_offset
            )
            reset_masters = jpos[:, self._coupled_master_joint_ids]
            jpos[:, self._coupled_follower_ids] = (
                reset_masters * self._coupled_mult + self._coupled_offset
            )
        ramp_reset_target = bool(
            self.cfg.reset_target_ramp and self.cfg.reset_contact_steps > 0
        )
        initial_target_jpos = jpos if ramp_reset_target else target_jpos
        self.allegro.set_joint_position_target(initial_target_jpos, env_ids=env_ids)
        self.allegro.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)

        # ---- Screwdriver ----
        sd_root = self.screwdriver.data.default_root_state[env_ids].clone()
        sd_root[:, :3] += self.scene.env_origins[env_ids]
        self.screwdriver.write_root_pose_to_sim(sd_root[:, :7], env_ids=env_ids)
        self.screwdriver.write_root_velocity_to_sim(sd_root[:, 7:], env_ids=env_ids)

        sd_jpos = torch.zeros_like(self.screwdriver.data.default_joint_pos[env_ids])
        reset_tilt_xy = self._reset_screwdriver_tilt_xy
        sd_jpos[:, self._screwdriver_euler_ids[:2]] = (
            reset_tilt_xy[bucket_of]
            if bucket_of is not None and reset_tilt_xy.ndim == 2
            else reset_tilt_xy
        )
        if hasattr(self, "_env_reset_screwdriver_tilt_noise"):
            sd_jpos[:, self._screwdriver_euler_ids[:2]] += (
                self._env_reset_screwdriver_tilt_noise[env_ids]
            )
        if self.cfg.randomize_obj_start:
            sd_jpos[:, self._screwdriver_z_id] = (
                2.0 * math.pi * (torch.rand(len(env_ids), device=self.device) - 0.5)
            )
        sd_jvel = torch.zeros_like(sd_jpos)
        self.screwdriver.write_joint_state_to_sim(sd_jpos, sd_jvel, env_ids=env_ids)

        # ---- Reset tracking buffers ----
        finger_q = jpos[:, self._finger_joint_ids]
        finger_target_q = target_jpos[:, self._finger_joint_ids]
        self._cur_targets[env_ids] = finger_target_q
        self._prev_actions[env_ids] = 0.0
        self._total_turn[env_ids] = 0.0
        self._net_turn[env_ids] = 0.0
        self._prev_milestone_count[env_ids] = 0.0
        self._prev_z[env_ids] = sd_jpos[:, self._screwdriver_z_id].detach()
        self._prev_tilt_xy[env_ids] = sd_jpos[:, self._screwdriver_euler_ids[:2]].detach()
        if self._prev_shaft_quat is not None:
            shaft_quat = self._get_shaft_quat()
            if shaft_quat is not None:
                self._prev_shaft_quat[env_ids] = shaft_quat[env_ids].detach()

        # ---- Domain randomisation (applied after state is written to sim) ----
        if self.cfg.domain_rand.enabled:
            self._randomise_dynamics(env_ids)

        # ---- RMA history ----
        if self.cfg.asymmetric_obs:
            observed_finger_q = self._observed_finger_q(finger_q, env_ids)
            frame = self._proprio_codec.encode_frame(
                observed_finger_q, self._cur_targets[env_ids]
            )
            self._prop_hist_buf[env_ids] = frame.unsqueeze(1).expand(
                -1, self.cfg.prop_hist_len, -1
            )

        # Settle physics contacts.
        if self.cfg.reset_contact_steps > 0:
            # PhysX advances the entire scene even when only a subset is being
            # reset.  Snapshot the complement and write it back on every hidden
            # settle step so an asynchronous reset (or contact-guard retry)
            # cannot advance unrelated episodes or previously accepted rows.
            stable_mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            stable_mask[env_ids] = False
            stable_ids = self.allegro._ALL_INDICES[stable_mask]
            if stable_ids.numel() > 0:
                stable_hand_jpos = self.allegro.data.joint_pos[stable_ids].clone()
                stable_hand_jvel = self.allegro.data.joint_vel[stable_ids].clone()
                stable_sd_jpos = self.screwdriver.data.joint_pos[stable_ids].clone()
                stable_sd_jvel = self.screwdriver.data.joint_vel[stable_ids].clone()
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            pin_screwdriver = bool(self.cfg.reset_pin_screwdriver_upright)
            for step in range(self.cfg.reset_contact_steps):
                if ramp_root_pose:
                    alpha = float(step + 1) / float(self.cfg.reset_contact_steps)
                    ramp_root = root[:, :7].clone()
                    ramp_root[:, :3] = torch.lerp(
                        nominal_root_pose[:, :3], root[:, :3], alpha
                    )
                    q0 = nominal_root_pose[:, 3:7]
                    q1 = root[:, 3:7]
                    # Quaternions q and -q encode the same orientation.  Pick
                    # the short arc, then use normalized linear interpolation;
                    # reset perturbations are deliberately small (< 6 deg).
                    q1 = torch.where(
                        (torch.sum(q0 * q1, dim=-1, keepdim=True) < 0.0),
                        -q1,
                        q1,
                    )
                    q = torch.lerp(q0, q1, alpha)
                    ramp_root[:, 3:7] = q / torch.linalg.norm(
                        q, dim=-1, keepdim=True
                    ).clamp_min(1.0e-8)
                    self.allegro.write_root_pose_to_sim(
                        ramp_root, env_ids=env_ids
                    )
                if ramp_reset_target:
                    alpha = float(step + 1) / float(self.cfg.reset_contact_steps)
                    ramp_target = torch.lerp(jpos, target_jpos, alpha)
                    self.allegro.set_joint_position_target(ramp_target, env_ids=env_ids)
                if pin_screwdriver:
                    self.screwdriver.write_joint_state_to_sim(
                        sd_jpos, sd_jvel, env_ids=env_ids
                    )
                if stable_ids.numel() > 0:
                    self.allegro.write_joint_state_to_sim(
                        stable_hand_jpos, stable_hand_jvel, env_ids=stable_ids
                    )
                    self.screwdriver.write_joint_state_to_sim(
                        stable_sd_jpos, stable_sd_jvel, env_ids=stable_ids
                    )
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
            if stable_ids.numel() > 0:
                # The last simulated step can integrate away from the written
                # snapshot; restore it once more before exposing observations.
                self.allegro.write_joint_state_to_sim(
                    stable_hand_jpos, stable_hand_jvel, env_ids=stable_ids
                )
                self.screwdriver.write_joint_state_to_sim(
                    stable_sd_jpos, stable_sd_jvel, env_ids=stable_ids
                )
            if pin_screwdriver:
                # Release the reset fixture from a static state. The target and
                # measured joint positions are preserved; only ramp-induced hand
                # velocity is cleared before the first policy step.
                hand_jpos = self.allegro.data.joint_pos[env_ids].clone()
                hand_jvel = torch.zeros_like(
                    self.allegro.data.default_joint_vel[env_ids]
                )
                self.allegro.write_joint_state_to_sim(
                    hand_jpos, hand_jvel, env_ids=env_ids
                )
                self.screwdriver.write_joint_state_to_sim(
                    sd_jpos, sd_jvel, env_ids=env_ids
                )
                self.scene.write_data_to_sim()
                self.sim.forward()
                self.scene.update(dt=self.physics_dt)
            elif stable_ids.numel() > 0:
                self.scene.write_data_to_sim()
                self.sim.forward()
                self.scene.update(dt=self.physics_dt)

            if self.cfg.reset_zero_tension_targets:
                self._snap_targets_to_settled_state(env_ids)

    def _snap_targets_to_settled_state(self, env_ids: torch.Tensor) -> None:
        """Zero-tension start: align finger targets with the settled joint state.

        See ``reset_zero_tension_targets``.  Rewrites ``_cur_targets``, the sim
        position targets (with coupled followers kept consistent), and the RMA
        history frame so the policy's first observation matches what it can
        actually measure: targets equal to positions, no standing penetration.
        """
        settled_jpos = self.allegro.data.joint_pos[env_ids]
        settled_finger_q = settled_jpos[:, self._finger_joint_ids]
        self._cur_targets[env_ids] = torch.clamp(
            settled_finger_q,
            self._finger_lower[env_ids],
            self._finger_upper[env_ids],
        )
        target_full = self.allegro.data.joint_pos_target[env_ids].clone()
        target_full[:, self._finger_joint_ids] = self._cur_targets[env_ids]
        if self._coupled_mult is not None:
            masters = target_full[:, self._coupled_master_joint_ids]
            target_full[:, self._coupled_follower_ids] = (
                masters * self._coupled_mult + self._coupled_offset
            )
        self.allegro.set_joint_position_target(target_full, env_ids=env_ids)
        if self.cfg.asymmetric_obs:
            observed_finger_q = self._observed_finger_q(settled_finger_q, env_ids)
            frame = self._proprio_codec.encode_frame(
                observed_finger_q, self._cur_targets[env_ids]
            )
            self._prop_hist_buf[env_ids] = frame.unsqueeze(1).expand(
                -1, self.cfg.prop_hist_len, -1
            )

    # -----------------------------------------------------------------------
    # Domain randomisation
    # -----------------------------------------------------------------------

    def _randomise_dynamics(self, env_ids: torch.Tensor) -> None:
        """Per-reset physics randomisation.

        Four independent parameters are randomised:
          1. Screwdriver rotation damping — the "friction proxy".
          2. Screwdriver body mass — changes inertia and required push force.
          3. Finger joint stiffness — simulates actuator manufacturing variation.
          4. Finger joint damping — simulates gear/tendon damping variation.

        The same scale is applied to all finger joints within one env so that
        the grasp character is consistent within an episode.  The damping/mass
        values are stored so _compute_privileged_obs can expose them.
        """
        n = len(env_ids)
        dr = self.cfg.domain_rand
        scale = float(
            getattr(self._curriculum_phase, "dynamics_randomization_scale", 1.0)
        )
        if not 0.0 <= scale <= 1.0:
            raise ValueError("dynamics_randomization_scale must be in [0, 1]")

        def scaled_range(
            bounds: tuple[float, float], center: float = 1.0
        ) -> tuple[float, float]:
            return (
                center + scale * (float(bounds[0]) - center),
                center + scale * (float(bounds[1]) - center),
            )

        def varies(bounds: tuple[float, float], center: float = 1.0) -> bool:
            lower, upper = scaled_range(bounds, center)
            return (
                abs(lower - center) > 1.0e-8
                or abs(upper - center) > 1.0e-8
            )

        env_ids_cpu = env_ids.detach().to("cpu")

        # 1. Rotation damping (written to screwdriver z-joint for this env batch).
        # Do not rewrite nominal properties: on non-replicated multi-assets even
        # a value-preserving PhysX setter can perturb contact solver state.
        if varies(dr.rotation_damping_range):
            rot_scale = torch.empty(n, device=self.device).uniform_(
                *scaled_range(dr.rotation_damping_range)
            )
            new_rot_damp = self._base_rotation_damping * rot_scale
            self._env_rotation_damping[env_ids] = new_rot_damp
            self.screwdriver.write_joint_damping_to_sim(
                new_rot_damp.unsqueeze(-1),
                joint_ids=[self._screwdriver_z_id],
                env_ids=env_ids,
            )
        else:
            self._env_rotation_damping[env_ids] = self._base_rotation_damping

        # 2. Screwdriver body mass.  This Isaac Lab release exposes no
        # write_body_mass_to_sim helper, so use the PhysX view only when the
        # configured effective range actually varies.
        if varies(dr.screwdriver_mass_range):
            base_body_id = self._handle_body_ids[self._handle_base_idx]
            mass_scale = torch.empty(len(env_ids_cpu)).uniform_(
                *scaled_range(dr.screwdriver_mass_range)
            )
            masses = self.screwdriver.root_physx_view.get_masses()
            default_base_mass = self.screwdriver.data.default_mass[
                env_ids_cpu, base_body_id
            ].to("cpu")
            masses[env_ids_cpu, base_body_id] = default_base_mass * mass_scale
            self.screwdriver.root_physx_view.set_masses(masses, env_ids_cpu)

        # 3 & 4. Finger PD gains (one scale per env, broadcast across joints).
        n_fj = len(self._finger_joint_ids)
        if varies(dr.finger_stiffness_range):
            stiff_scale = torch.empty(n, 1, device=self.device).uniform_(
                *scaled_range(dr.finger_stiffness_range)
            )
            self.allegro.write_joint_stiffness_to_sim(
                (self._base_finger_stiffness * stiff_scale).expand(-1, n_fj),
                joint_ids=self._finger_joint_ids,
                env_ids=env_ids,
            )
        if varies(dr.finger_damping_range):
            damp_scale = torch.empty(n, 1, device=self.device).uniform_(
                *scaled_range(dr.finger_damping_range)
            )
            self.allegro.write_joint_damping_to_sim(
                (self._base_finger_damping * damp_scale).expand(-1, n_fj),
                joint_ids=self._finger_joint_ids,
                env_ids=env_ids,
            )

        # 5. Screwdriver rotational load (Coulomb) is a tensor-side parameter,
        # not a PhysX property write, so setting nominal explicitly is safe.
        if self._base_load_torque > 0.0:
            if varies(dr.screwdriver_load_torque_range):
                load_scale = torch.empty(n, device=self.device).uniform_(
                    *scaled_range(dr.screwdriver_load_torque_range)
                )
                self._env_load_torque[env_ids] = (
                    self._base_load_torque * load_scale
                )
            else:
                self._env_load_torque[env_ids] = self._base_load_torque

        # 6. Contact friction (real) on both object and hand materials.
        if dr.randomize_contact_friction:
            if varies(dr.contact_friction_range, self._base_friction):
                fr = torch.empty(len(env_ids_cpu)).uniform_(
                    *scaled_range(dr.contact_friction_range, self._base_friction)
                )
                self._env_friction[env_ids] = fr.to(self.device)
                for art in (self.screwdriver, self.allegro):
                    mats = art.root_physx_view.get_material_properties()
                    mats[env_ids_cpu, :, 0] = fr[:, None]
                    mats[env_ids_cpu, :, 1] = fr[:, None]
                    art.root_physx_view.set_material_properties(mats, env_ids_cpu)
            else:
                self._env_friction[env_ids] = self._base_friction

        # 7. Tilt-joint bearing damping.
        if dr.randomize_tilt_damping and varies(dr.tilt_damping_range):
            tilt_ids = self._screwdriver_euler_ids[:2]
            tilt_scale = torch.empty(n, 1, device=self.device).uniform_(
                *scaled_range(dr.tilt_damping_range)
            )
            self.screwdriver.write_joint_damping_to_sim(
                (self._base_tilt_damping * tilt_scale).expand(-1, len(tilt_ids)),
                joint_ids=tilt_ids,
                env_ids=env_ids,
            )

    # -----------------------------------------------------------------------
    # Shaft spin (HORA-style, prevents wobble-scraping reward)
    # -----------------------------------------------------------------------

    def _get_shaft_quat(self) -> torch.Tensor | None:
        if not self._handle_body_ids:
            return None
        return self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._shaft_idx], 3:7]

    def _compute_shaft_spin_delta(self) -> torch.Tensor | None:
        """Signed per-step rotation about the screwdriver's own shaft axis.

        Projects the inter-step quaternion delta onto the current shaft-body
        axis.  Precession of a tilted shaft does not contribute because the
        axis vector is updated each step.  Returns None on first call (the
        reward falls back to Euler-z delta for that step only).
        """
        shaft_quat = self._get_shaft_quat()
        if shaft_quat is None:
            return None
        if self._prev_shaft_quat is None:
            self._prev_shaft_quat = shaft_quat.detach().clone()
            return None

        spin = rewards.shaft_spin_delta(
            shaft_quat, self._prev_shaft_quat, self.cfg.turn_direction
        )
        self._prev_shaft_quat = shaft_quat.detach().clone()
        return spin

    # -----------------------------------------------------------------------
    # Contact gate
    # -----------------------------------------------------------------------

    def _compute_contact_gate(self, phase: CurriculumPhaseCfg) -> torch.Tensor:
        """Binary × continuous gate: contact proximity × fingertip speed.

        Gate = 0 when fewer than ``min_contact_fingers`` are inside
        ``turn_reward_contact_distance``, or all in-contact fingertip speeds
        are below ``turn_reward_min_fingertip_speed``.

        Both the turn reward and the reverse penalty are multiplied by this
        gate (see cfg for the asymmetric-penalty failure mode it prevents).
        """
        threshold = float(phase.turn_reward_contact_distance)
        if threshold <= 0.0:
            return torch.ones(self.num_envs, device=self.device)

        tip_dist = self._compute_fingertip_axis_distances()  # (N, n_fingers)
        if tip_dist.shape[1] == 0:
            return torch.zeros(self.num_envs, device=self.device)

        contact_mask = tip_dist <= threshold  # (N, n_fingers) bool
        contact_count = contact_mask.sum(dim=-1)
        min_c = max(1, phase.turn_reward_min_contact_fingers)
        binary_gate = (contact_count >= min_c).float()

        # Motion gate: average speed of in-contact fingertips.
        fingertip_vel = self.allegro.data.body_state_w[:, self._fingertip_body_ids, 7:10]
        tip_speed = torch.linalg.norm(fingertip_vel, dim=-1)  # (N, n_fingers)
        w = contact_mask.float()
        denom = w.sum(dim=-1).clamp(min=1.0)
        avg_contact_speed = (tip_speed * w).sum(dim=-1) / denom

        motion_gate = rewards.motion_gate(
            avg_contact_speed,
            float(phase.turn_reward_min_fingertip_speed),
            float(phase.turn_reward_full_fingertip_speed),
        )

        gate = binary_gate * motion_gate
        self.extras["eval_contact_count"] = contact_count.detach()
        self.extras["eval_binary_gate"] = binary_gate.detach()
        self.extras["eval_motion_gate"] = motion_gate.detach()
        self.extras["eval_avg_contact_speed"] = avg_contact_speed.detach()
        return gate

    # -----------------------------------------------------------------------
    # Distance helpers
    # -----------------------------------------------------------------------

    def _compute_fingertip_axis_distances(self) -> torch.Tensor:
        """Per-fingertip distance to the handle axis segment.

        Returns (N, n_fingers) tensor.  Uses the handle body origin and cap
        body origin to define a line segment and computes the closest point
        distance, which equals ~handle_radius (0.02 m) when a fingertip pad
        is in contact.
        """
        if not self._fingertip_body_ids or not self._handle_body_ids:
            return torch.empty((self.num_envs, 0), device=self.device)

        tip_pos = self.allegro.data.body_state_w[:, self._fingertip_body_ids, :3]
        base = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_base_idx], :3]
        top = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_cap_idx], :3]
        return rewards.point_segment_distance(tip_pos, base, top)

    def compute_surface_clearance(self, pad_offset: float = 0.0) -> torch.Tensor:
        """Per-fingertip signed clearance to the handle **surface**, ``(N, n_fingers)``.

        ``clearance = axis_dist - handle_radius - pad_offset``: ≈0 at contact,
        >0 floating, <0 penetrating.  This is the correct grasp-acceptance metric
        (the raw axis distance from ``_compute_fingertip_axis_distances`` is
        ≈ handle_radius at contact, not 0).  The handle radius is **per-env** when
        geometry DR is on (from the identified variant's diameter scale), else the
        base radius.  Intended for the acceptance-gate validation (see plan §3e/3f),
        not the training loop.
        """
        from screwdriver_rl.utils.variants import BASE_RADIUS

        axis_dist = self._compute_fingertip_axis_distances()  # (N, n_fingers)
        if axis_dist.shape[1] == 0:
            return axis_dist
        # Keep the canonical task at BASE_RADIUS while allowing a fixed-geometry
        # subclass to declare its own base radius. Geometry-DR scales remain
        # relative to whichever base is active.
        base_radius = float(getattr(self.cfg, "screwdriver_handle_radius", BASE_RADIUS))
        if self._env_geom_scale is not None:
            radius = (self._env_geom_scale[:, 0] * base_radius).unsqueeze(-1)  # (N, 1)
        else:
            radius = base_radius
        return axis_dist - radius - pad_offset

    def _compute_near_reward(self, tip_dist: torch.Tensor, weight: float) -> torch.Tensor:
        """Dense fingertip proximity reward with thumb/non-thumb split.

        Non-thumb fingers: top-k nearest get averaged (prevents all fingers
        clustering on one side of the handle).
        Thumb: treated separately since it opposes from the other side.
        Final score = 0.5 × (thumb_score + non_thumb_avg).
        """
        if tip_dist.shape[1] == 0:
            return torch.zeros(self.num_envs, device=self.device)

        near = torch.exp(-tip_dist / max(self.cfg.near_reward_std, 1e-6))
        score = rewards.near_contact_score(
            near, self._thumb_tip_idx, self._non_thumb_tip_idxs, self.cfg.near_reward_top_k
        )
        return weight * score

    def _compute_proximal_penalty(self, weight: float) -> torch.Tensor:
        """Penalises proximal/medial links being close to the handle.

        Any of the listed proximal bodies within 0.05 m of the handle axis
        incurs a penalty proportional to proximity.  This shapes the policy
        away from palm-pressing, knuckle-dragging, or using the finger back.

        ``weight = 0`` skips the computation entirely (Phase 0).
        """
        if weight <= 0.0 or not self._proximal_body_ids or not self._handle_body_ids:
            return torch.zeros(self.num_envs, device=self.device)

        prox_pos = self.allegro.data.body_state_w[:, self._proximal_body_ids, :3]
        base = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_base_idx], :3]
        top = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_cap_idx], :3]
        d = rewards.point_segment_distance(prox_pos, base, top)  # (N, n_proximal)

        # Penalty activates within 0.05 m; linear in proximity.
        penalty_threshold = 0.05
        penalty = torch.clamp(penalty_threshold - d, min=0.0).sum(dim=-1)
        return weight * penalty

    # -----------------------------------------------------------------------
    # Milestone (sparse progress bonus)
    # -----------------------------------------------------------------------

    def _compute_milestone_reward(self, gate: torch.Tensor) -> torch.Tensor:
        if self.cfg.milestone_angle <= 0.0 or self.cfg.milestone_bonus <= 0.0:
            return torch.zeros(self.num_envs, device=self.device)

        net_fwd = self._net_turn.clamp(min=0.0)
        count = torch.floor(net_fwd / self.cfg.milestone_angle)
        new = (count - self._prev_milestone_count).clamp(min=0.0)
        self._prev_milestone_count = torch.maximum(self._prev_milestone_count, count.detach())
        return self.cfg.milestone_bonus * new * gate

    # -----------------------------------------------------------------------
    # Privileged observations (RMA)
    # -----------------------------------------------------------------------

    def _compute_privileged_obs(self) -> torch.Tensor:
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        angvel = self.screwdriver.data.joint_vel[:, self._screwdriver_euler_ids]
        rel_pos = self.screwdriver.data.root_pos_w - self.allegro.data.root_pos_w
        quat = self.screwdriver.data.root_quat_w
        # Friction proxy: expose whichever resistance dominates.  With a screw
        # load present it is the Coulomb torque ratio; otherwise the bearing
        # damping ratio.  Both are normalised to 1.0 at the base value.
        if self._base_load_torque > 0.0:
            friction = (self._env_load_torque / self._base_load_torque).unsqueeze(-1)
        else:
            friction = (self._env_rotation_damping / self._base_rotation_damping).unsqueeze(-1)

        tip_dist = self._compute_fingertip_axis_distances()  # (N, n_fingers)
        # One slot per active fingertip (= len(self.fingers)); pad if the
        # distance query returned fewer (e.g. unresolved bodies).
        n_finger_slots = len(self.fingers)
        if tip_dist.shape[1] >= n_finger_slots:
            tip_dist_fixed = tip_dist[:, :n_finger_slots]
        else:
            pad = torch.full((self.num_envs, n_finger_slots - tip_dist.shape[1]), 1.0, device=self.device)
            tip_dist_fixed = torch.cat([tip_dist, pad], dim=-1)

        return torch.cat([euler, angvel, rel_pos, quat, friction, tip_dist_fixed], dim=-1)

    def _update_prop_hist(self) -> None:
        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        observed_finger_q = self._observed_finger_q(finger_q)
        frame = self._proprio_codec.encode_frame(observed_finger_q, self._cur_targets)
        # Truncate/pad to history_obs_dim.
        dim = self.cfg.history_obs_dim
        if frame.shape[1] > dim:
            frame = frame[:, :dim]
        elif frame.shape[1] < dim:
            frame = torch.cat([frame, torch.zeros(self.num_envs, dim - frame.shape[1], device=self.device)], dim=-1)
        self._prop_hist_buf = torch.roll(self._prop_hist_buf, shifts=-1, dims=1)
        self._prop_hist_buf[:, -1] = frame

    # -----------------------------------------------------------------------
    # Joint / body resolution helpers
    # -----------------------------------------------------------------------

    def _find_joints(self, articulation: Articulation, names: Sequence[str]) -> list[int]:
        patterns = [f"^{re.escape(n)}$" for n in names]
        ids, found = articulation.find_joints(patterns, preserve_order=True)
        if len(ids) != len(names):
            raise RuntimeError(
                f"Could not find joints {names} on {articulation.cfg.prim_path}. "
                f"Found: {found}"
            )
        return ids

    def _resolve_finger_joints(self) -> dict[str, list[int]]:
        unknown = set(self.fingers) - set(self.FINGER_JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unknown finger names: {sorted(unknown)}")
        return {
            finger: self._find_joints(self.allegro, self.FINGER_JOINT_NAMES[finger])
            for finger in self.FINGER_JOINT_NAMES  # resolve all for reset, use subset for policy
        }

    def _resolve_coupled_joints(self) -> None:
        """Resolve mimic/coupled follower joints driven from a master joint.

        Builds ``_coupled_follower_ids`` (articulation joint ids), the matching
        master columns into ``_cur_targets`` (``_coupled_master_cols_t``) and
        master articulation ids (``_coupled_master_joint_ids``), plus the
        multiplier/offset tensors.  Skips any follower whose master finger is
        inactive, or that the URDF importer collapsed into a PhysX constraint
        (i.e. it did not resolve as an independent joint).  Sets
        ``_coupled_mult = None`` when nothing to drive.
        """
        self._coupled_follower_ids: list[int] = []
        self._coupled_master_joint_ids: list[int] = []
        self._coupled_master_cols_t: torch.Tensor | None = None
        self._coupled_mult: torch.Tensor | None = None
        self._coupled_offset: torch.Tensor | None = None
        if not self.COUPLED_JOINTS:
            return

        # Column of each active finger joint within _cur_targets / _finger_joint_ids.
        name_to_col: dict[str, int] = {}
        col = 0
        for finger in self.fingers:
            for jname in self.FINGER_JOINT_NAMES[finger]:
                name_to_col[jname] = col
                col += 1

        master_cols: list[int] = []
        mults: list[float] = []
        offs: list[float] = []
        for follower, (master, mult, off) in self.COUPLED_JOINTS.items():
            if master not in name_to_col:
                continue  # master finger not active in this config
            found_ids, _ = self.allegro.find_joints(
                [f"^{re.escape(follower)}$"], preserve_order=True
            )
            if len(found_ids) != 1:
                continue  # importer collapsed the mimic into a PhysX constraint
            self._coupled_follower_ids.append(found_ids[0])
            c = name_to_col[master]
            master_cols.append(c)
            self._coupled_master_joint_ids.append(self._finger_joint_ids[c])
            mults.append(float(mult))
            offs.append(float(off))

        if self._coupled_follower_ids:
            self._coupled_master_cols_t = torch.tensor(
                master_cols, dtype=torch.long, device=self.device
            )
            self._coupled_mult = torch.tensor(
                mults, dtype=torch.float32, device=self.device
            ).view(1, -1)
            self._coupled_offset = torch.tensor(
                offs, dtype=torch.float32, device=self.device
            ).view(1, -1)

    def _resolve_bodies(
        self, articulation: Articulation, names: Sequence[str]
    ) -> list[int]:
        ids = []
        for name in names:
            found_ids, found_names = articulation.find_bodies(
                [f"^{re.escape(name)}$"], preserve_order=True
            )
            if len(found_ids) != 1:
                raise RuntimeError(
                    f"Expected exactly one body named {name!r} on "
                    f"{articulation.cfg.prim_path}. Found: {found_names}"
                )
            ids.append(found_ids[0])
        return ids

    def _resolve_fingertip_bodies(self) -> list[int]:
        return self._resolve_bodies(
            self.allegro,
            [self.FINGERTIP_BODY_NAMES[f] for f in self.fingers],
        )

    def _resolve_proximal_bodies(self) -> list[int]:
        ids = []
        for pattern in self.PROXIMAL_BODY_PATTERNS:
            found_ids, _ = self.allegro.find_bodies([pattern], preserve_order=True)
            ids.extend(found_ids)
        return list(dict.fromkeys(ids))  # deduplicate, preserve order

    def _resolve_handle_bodies(self) -> list[int]:
        return self._resolve_bodies(self.screwdriver, list(_SCREWDRIVER_HANDLE_BODIES))

    def _make_default_finger_pos(self) -> torch.Tensor:
        """Per-env reset/home posture over the *active* fingers, ``(N, D)``.

        With per-bucket geometry DR each env gets its diameter/length bucket's
        posture (gathered by ``_env_bucket_idx``); otherwise every env shares the
        single ``cfg.pregrasp_positions`` posture.
        """
        buckets = self._pregrasp_bucket_dicts()
        if buckets is not None:
            table = torch.tensor(
                [[v for f in self.fingers for v in bucket[f]] for bucket in buckets],
                dtype=torch.float32,
                device=self.device,
            )  # (num_buckets, D)
            return table[self._env_bucket_idx]  # (N, D)
        pos = [v for f in self.fingers for v in self.cfg.pregrasp_positions[f]]
        return torch.tensor(pos, dtype=torch.float32, device=self.device).expand(self.num_envs, -1).clone()

    # -----------------------------------------------------------------------
    # Geometry-variant identification + per-bucket pregrasp
    # -----------------------------------------------------------------------

    def _pregrasp_bucket_dicts(self) -> list | None:
        """Return the per-bucket pregrasp dicts iff geometry DR + a table are
        active and the env has been assigned buckets; else ``None``."""
        buckets = getattr(self.cfg, "pregrasp_positions_buckets", None)
        if buckets is None or getattr(self, "_env_bucket_idx", None) is None:
            return None
        return buckets

    def _identify_geometry_variants(self) -> None:
        """Recover each env's geometry variant from the handle's un-randomised
        ``(default_mass, default_izz)`` signature (see utils/variants.py).

        Sets ``_variant_table`` plus per-env ``_env_variant_idx`` /
        ``_env_bucket_idx`` / ``_env_geom_scale`` (``[diameter, length]`` scale).
        No-op (all set to ``None``) when geometry DR is off.
        """
        self._variant_table = None
        self._env_variant_idx = None
        self._env_bucket_idx = None
        self._env_geom_scale = None
        if not self.cfg.domain_rand.randomize_geometry:
            return

        from screwdriver_rl.utils.variants import identify_variants, load_variant_table

        manifest_path = Path(self.cfg.screwdriver_variants_dir) / "manifest.json"
        table = load_variant_table(manifest_path)
        assignment = str(getattr(self.cfg, "geometry_variant_assignment", "signature"))
        if assignment == "cyclic":
            # MultiAssetSpawnerCfg(random_choice=False) uses this exact mapping;
            # the same contract is already used by the Linker in-hand task.
            vidx = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            vidx = vidx % table.num_variants
            bidx = table.bucket.to(self.device)[vidx]
            gscale = torch.stack(
                (
                    table.diameter_scale.to(self.device)[vidx],
                    table.length_scale.to(self.device)[vidx],
                ),
                dim=-1,
            )
        elif assignment == "signature":
            body_id = self._handle_body_ids[self._handle_base_idx]  # screwdriver_body
            masses = self.screwdriver.data.default_mass[:, body_id].to(self.device)
            # default_inertia rows are flattened 3x3; element 8 is izz (zz).
            izz = self.screwdriver.data.default_inertia[:, body_id, 8].to(self.device)
            vidx, bidx, gscale = identify_variants(masses, izz, table)
        else:
            raise ValueError(f"unsupported geometry_variant_assignment {assignment!r}")
        self._variant_table = table
        self._env_variant_idx = vidx
        self._env_bucket_idx = bidx
        self._env_geom_scale = gscale
        counts = torch.bincount(vidx, minlength=table.num_variants).tolist()
        print(f"[geometry-DR] env→variant histogram (n={self.num_envs}): {counts}")

    def _make_pregrasp_table(self) -> dict[str, torch.Tensor]:
        """Per-finger reset posture tensors for *all* reset fingers.

        Per-bucket ``(num_buckets, n_joints)`` when geometry DR + a table are
        active; otherwise the single shared ``(n_joints,)`` vector.
        """
        buckets = self._pregrasp_bucket_dicts()
        if buckets is not None:
            return {
                finger: torch.tensor(
                    [bucket[finger] for bucket in buckets],
                    dtype=torch.float32,
                    device=self.device,
                )  # (num_buckets, n_joints)
                for finger in self.FINGER_JOINT_NAMES
            }
        return {
            finger: torch.tensor(
                self.cfg.pregrasp_positions[finger], dtype=torch.float32, device=self.device
            )
            for finger in self.FINGER_JOINT_NAMES
        }

    def _make_reset_joint_table(self) -> dict[str, torch.Tensor] | None:
        """Optional reset-state posture; targets remain ``pregrasp_positions``."""

        positions = getattr(self.cfg, "reset_joint_positions", None)
        if positions is None:
            return None
        if self.cfg.domain_rand.randomize_geometry:
            buckets = getattr(self.cfg, "reset_joint_positions_buckets", None)
            if buckets is None:
                raise ValueError(
                    "reset_joint_positions with geometry DR requires "
                    "reset_joint_positions_buckets"
                )
            if len(buckets) != self._variant_table.num_buckets:
                raise ValueError(
                    "reset_joint_positions_buckets length "
                    f"{len(buckets)} != manifest num_buckets "
                    f"{self._variant_table.num_buckets}"
                )
            for bucket_index, bucket in enumerate(buckets):
                missing = set(self.FINGER_JOINT_NAMES).difference(bucket)
                if missing:
                    raise ValueError(
                        "reset_joint_positions_buckets"
                        f"[{bucket_index}] missing fingers: {sorted(missing)}"
                    )
            return {
                finger: torch.tensor(
                    [bucket[finger] for bucket in buckets],
                    dtype=torch.float32,
                    device=self.device,
                )
                for finger in self.FINGER_JOINT_NAMES
            }
        missing = set(self.FINGER_JOINT_NAMES).difference(positions)
        if missing:
            raise ValueError(f"reset_joint_positions missing fingers: {sorted(missing)}")
        return {
            finger: torch.tensor(
                positions[finger], dtype=torch.float32, device=self.device
            )
            for finger in self.FINGER_JOINT_NAMES
        }

    def _make_pregrasp_root_offset_table(self) -> torch.Tensor | None:
        """Per-bucket hand-root offset (world xyz), ``(num_buckets, 3)`` or ``None``.

        Active only under geometry DR with a ``pregrasp_root_pos_offsets_buckets``
        cfg table and per-env bucket assignments; reset gathers each env's row by
        ``_env_bucket_idx`` and shifts the hand root so the grasp tracks the
        per-variant handle length.
        """
        offsets = getattr(self.cfg, "pregrasp_root_pos_offsets_buckets", None)
        if offsets is None or getattr(self, "_env_bucket_idx", None) is None:
            return None
        return torch.tensor(offsets, dtype=torch.float32, device=self.device)


    def _make_reset_screwdriver_tilt_xy_table(self) -> torch.Tensor:
        """Shared ``(2,)`` or per-bucket ``(num_buckets, 2)`` reset tilt."""

        buckets = getattr(self.cfg, "reset_screwdriver_tilt_xy_buckets", None)
        if self.cfg.domain_rand.randomize_geometry and buckets is not None:
            if len(buckets) != self._variant_table.num_buckets:
                raise ValueError(
                    "reset_screwdriver_tilt_xy_buckets length "
                    f"{len(buckets)} != manifest num_buckets "
                    f"{self._variant_table.num_buckets}"
                )
            table = torch.tensor(buckets, dtype=torch.float32, device=self.device)
            if table.shape != (self._variant_table.num_buckets, 2):
                raise ValueError(
                    "reset_screwdriver_tilt_xy_buckets must contain (x, y) tuples"
                )
            return table
        shared = torch.tensor(
            self.cfg.reset_screwdriver_tilt_xy,
            dtype=torch.float32,
            device=self.device,
        )
        if shared.shape != (2,):
            raise ValueError("reset_screwdriver_tilt_xy must be an (x, y) tuple")
        return shared


    def _make_pregrasp_root_quat_table(self) -> torch.Tensor | None:
        """Per-bucket hand-root quaternion ``(w, x, y, z)`` or ``None``."""

        quats = getattr(self.cfg, "pregrasp_root_quats_buckets", None)
        if quats is None or getattr(self, "_env_bucket_idx", None) is None:
            return None
        if len(quats) != self._variant_table.num_buckets:
            raise ValueError(
                "pregrasp_root_quats_buckets length "
                f"{len(quats)} != manifest num_buckets "
                f"{self._variant_table.num_buckets}"
            )
        table = torch.tensor(quats, dtype=torch.float32, device=self.device)
        norms = torch.linalg.vector_norm(table, dim=-1)
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5, rtol=0.0)):
            raise ValueError("pregrasp_root_quats_buckets must contain unit quaternions")
        return table
