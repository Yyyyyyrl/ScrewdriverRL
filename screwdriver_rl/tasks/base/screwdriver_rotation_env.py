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
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

from screwdriver_rl.core import rewards
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

        # Map the contact-sensor body order to the active-finger order so the
        # net-force columns align with the fingertip distances (self.fingers).
        self._contact_body_order: torch.Tensor | None = None
        if self._fingertip_contact_sensor is not None:
            distal_names = [self.FINGERTIP_BODY_NAMES[f] for f in self.fingers]
            order_ids, _ = self._fingertip_contact_sensor.find_bodies(
                [f"^{re.escape(n)}$" for n in distal_names], preserve_order=True
            )
            self._contact_body_order = torch.tensor(
                order_ids, dtype=torch.long, device=self.device
            )

        # Fingertip pad-normal (outward, toward a grasped object) in the
        # fingertip link local frame, used by the pad-facing contact gate.
        self._pad_axis_local = torch.tensor(
            cfg.fingertip_pad_axis_local, dtype=torch.float32, device=self.device
        ).view(1, 1, 3)

        # Thumb index within active fingers for near-score weighting.
        self._thumb_tip_idx: int | None = (
            self.fingers.index("thumb") if "thumb" in self.fingers else None
        )
        self._non_thumb_tip_idxs: list[int] = [
            i for i, f in enumerate(self.fingers) if f != "thumb"
        ]

        # ---- Finger target and pregrasp defaults ----
        self._default_finger_pos: torch.Tensor = self._make_default_finger_pos()
        self._cur_targets: torch.Tensor = self._default_finger_pos.clone()
        finger_limits = self.allegro.data.soft_joint_pos_limits[:, self._finger_joint_ids]
        margin = float(cfg.joint_target_margin)
        self._finger_lower = finger_limits[..., 0] + margin
        self._finger_upper = finger_limits[..., 1] - margin

        self._pregrasp_pos: dict[str, torch.Tensor] = {
            finger: torch.tensor(
                cfg.pregrasp_positions[finger], dtype=torch.float32, device=self.device
            )
            for finger in self.FINGER_JOINT_NAMES  # all fingers for reset
        }

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

        # ---- RMA / asymmetric observations ----
        self._prop_hist_buf = torch.zeros(
            (self.num_envs, cfg.prop_hist_len, cfg.history_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )

        # Stagger episode starts to avoid synchronised reset artifacts.
        self.episode_length_buf = torch.randint(
            0, self.max_episode_length, (self.num_envs,),
            device=self.device, dtype=self.episode_length_buf.dtype,
        )

        # ---- Logging helper ----
        # ``_current_epoch`` is written each epoch by the rl_games
        # PhaseCheckpointObserver; the env itself has no epoch concept.
        self._current_epoch: int = 0
        from screwdriver_rl.utils.logging import RotationTrainingLogger
        self._logger = RotationTrainingLogger(log_interval_steps=2000)

    # -----------------------------------------------------------------------
    # Scene
    # -----------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self.allegro = Articulation(self.cfg.robot_cfg)
        self.screwdriver = Articulation(self.cfg.screwdriver_cfg)

        # Fingertip contact-force sensor (true-touch gate).  Built from the
        # hand's fingertip body names so it stays hand-agnostic.  ``self.fingers``
        # is not set yet (this runs inside super().__init__), so read the cfg
        # directly.  Requires ``activate_contact_sensors`` on the hand spawn.
        self._fingertip_contact_sensor: ContactSensor | None = None
        if self.cfg.use_contact_force_gate:
            distal_names = [self.FINGERTIP_BODY_NAMES[f] for f in self.cfg.fingers]
            body_regex = "(" + "|".join(distal_names) + ")"
            sensor_cfg = ContactSensorCfg(
                prim_path=f"{self.cfg.robot_cfg.prim_path}/{body_regex}",
                history_length=0,
                update_period=0.0,
                track_air_time=False,
            )
            self._fingertip_contact_sensor = ContactSensor(sensor_cfg)
            self.scene.sensors["fingertip_contact"] = self._fingertip_contact_sensor

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.cfg.friction_coefficient,
                    dynamic_friction=self.cfg.friction_coefficient,
                )
            ),
        )
        # Apply self-collision pair filters on the source env BEFORE cloning so
        # the cloner replicates them to every env.
        self._apply_self_collision_filters()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["allegro"] = self.allegro
        self.scene.articulations["screwdriver"] = self.screwdriver
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_self_collision_filters(self) -> None:
        """Exclude ``SELF_COLLISION_FILTER_PAIRS`` from self-collision checking.

        Applied to the source env (``env_0``) prims before ``clone_environments``
        so the filtering is replicated to all envs.  Uses USD ``FilteredPairsAPI``
        (no high-level Isaac Lab helper exists).  No-op when the list is empty.
        """
        if not self.SELF_COLLISION_FILTER_PAIRS:
            return
        import omni.usd
        from pxr import Sdf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        # robot_cfg.prim_path is e.g. "/World/envs/env_.*/LinkerHand" -> env_0 source.
        base_path = self.cfg.robot_cfg.prim_path.replace(".*", "0")
        applied = 0
        for link_a, link_b in self.SELF_COLLISION_FILTER_PAIRS:
            a_path = f"{base_path}/{link_a}"
            b_path = f"{base_path}/{link_b}"
            prim_a = stage.GetPrimAtPath(a_path)
            if not prim_a.IsValid() or not stage.GetPrimAtPath(b_path).IsValid():
                print(f"[self-collision-filter] WARN: missing prim {a_path} or {b_path}", flush=True)
                continue
            api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
            api.CreateFilteredPairsRel().AddTarget(Sdf.Path(b_path))
            applied += 1
        print(f"[self-collision-filter] applied {applied} filtered pairs under {base_path}", flush=True)

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

        # Update RMA proprioceptive history buffer.
        if self.cfg.asymmetric_obs:
            self._update_prop_hist()

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

    def _get_observations(self) -> dict[str, torch.Tensor]:
        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        obs = torch.cat([finger_q, self._cur_targets, euler], dim=-1)
        dr = self.cfg.domain_rand
        if dr.enabled and dr.obs_noise_std > 0.0:
            obs = obs + torch.randn_like(obs) * dr.obs_noise_std
        result: dict[str, torch.Tensor] = {"policy": obs}
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
        # ``fwd_vel`` (forward handle angular speed) feeds the rolling-consistency
        # factor so the gate only opens for spin the fingertips actually drive.
        contact_gate = self._compute_contact_gate(phase, fwd_vel)

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

        # ---- Contact-engagement reward (bridge hover -> press; per-phase weight) ----
        contact_reward = self._compute_contact_engagement_reward(
            phase.reward_contact_weight, tip_dist, float(phase.turn_reward_contact_distance)
        )

        # ---- Grip-force penalty (discourage crushing the handle; per-phase weight) ----
        grip_force_cost = self._compute_grip_force_penalty(phase.reward_contact_force_weight)

        # ---- Idle/abandonment penalty (fingers parked off the handle; per-phase) ----
        if tip_dist.numel() > 0 and phase.reward_finger_abandon_weight > 0.0:
            abandon_cost = phase.reward_finger_abandon_weight * (
                (tip_dist - self.cfg.finger_abandon_distance).clamp(min=0.0).sum(dim=-1)
            )
        else:
            abandon_cost = torch.zeros(self.num_envs, device=self.device)

        reward = (
            turn_reward
            + milestone_reward
            + near_reward
            + contact_reward
            - reverse_cost
            - upright_cost
            - tilt_vel_cost
            - action_cost
            - action_rate_cost
            - finger_vel_cost
            - proximal_cost
            - grip_force_cost
            - abandon_cost
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
            "eval_contact_reward": contact_reward.detach(),
            "eval_grip_force_cost": grip_force_cost.detach(),
            "eval_abandon_cost":   abandon_cost.detach(),
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
        return terminated, timed_out

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.allegro._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        # ---- Hand ----
        root = self.allegro.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.allegro.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.allegro.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

        jpos = self.allegro.data.default_joint_pos[env_ids].clone()
        jvel = torch.zeros_like(self.allegro.data.default_joint_vel[env_ids])
        for finger, jids in self._finger_joint_ids_by_name.items():
            jpos[:, jids] = self._pregrasp_pos[finger]
        # Set mimic/coupled followers consistently with their masters.
        if self._coupled_mult is not None:
            masters = jpos[:, self._coupled_master_joint_ids]
            jpos[:, self._coupled_follower_ids] = masters * self._coupled_mult + self._coupled_offset
        self.allegro.set_joint_position_target(jpos, env_ids=env_ids)
        self.allegro.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)

        # ---- Screwdriver ----
        sd_root = self.screwdriver.data.default_root_state[env_ids].clone()
        sd_root[:, :3] += self.scene.env_origins[env_ids]
        self.screwdriver.write_root_pose_to_sim(sd_root[:, :7], env_ids=env_ids)
        self.screwdriver.write_root_velocity_to_sim(sd_root[:, 7:], env_ids=env_ids)

        sd_jpos = torch.zeros_like(self.screwdriver.data.default_joint_pos[env_ids])
        if self.cfg.randomize_obj_start:
            sd_jpos[:, self._screwdriver_z_id] = (
                2.0 * math.pi * (torch.rand(len(env_ids), device=self.device) - 0.5)
            )
        sd_jvel = torch.zeros_like(sd_jpos)
        self.screwdriver.write_joint_state_to_sim(sd_jpos, sd_jvel, env_ids=env_ids)

        # ---- Reset tracking buffers ----
        finger_q = jpos[:, self._finger_joint_ids]
        self._cur_targets[env_ids] = finger_q
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
            frame = torch.cat([finger_q, self._cur_targets[env_ids]], dim=-1)
            self._prop_hist_buf[env_ids] = frame.unsqueeze(1).expand(
                -1, self.cfg.prop_hist_len, -1
            )

        # Settle physics contacts.
        if self.cfg.reset_contact_steps > 0:
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            for _ in range(self.cfg.reset_contact_steps):
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)

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

        # 1. Rotation damping (written to screwdriver z-joint for this env batch)
        rot_scale = torch.empty(n, device=self.device).uniform_(*dr.rotation_damping_range)
        new_rot_damp = self._base_rotation_damping * rot_scale           # (n,)
        self._env_rotation_damping[env_ids] = new_rot_damp
        self.screwdriver.write_joint_damping_to_sim(
            new_rot_damp.unsqueeze(-1),                                  # (n, 1)
            joint_ids=[self._screwdriver_z_id],
            env_ids=env_ids,
        )

        # 2. Screwdriver body mass.  This Isaac Lab release exposes no
        #    write_body_mass_to_sim helper, so we go through the PhysX view
        #    directly (the same idiom isaaclab.envs.mdp.events uses).  The
        #    masses buffer lives on CPU and the setter only touches the rows
        #    for the supplied env indices.  We randomise on the *default* mass
        #    so repeated resets don't compound the scaling.
        base_body_id = self._handle_body_ids[self._handle_base_idx]
        env_ids_cpu = env_ids.detach().to("cpu")
        mass_scale = torch.empty(len(env_ids_cpu)).uniform_(*dr.screwdriver_mass_range)
        masses = self.screwdriver.root_physx_view.get_masses()           # (num_envs, num_bodies) on CPU
        default_base_mass = self.screwdriver.data.default_mass[env_ids_cpu, base_body_id].to("cpu")
        masses[env_ids_cpu, base_body_id] = default_base_mass * mass_scale
        self.screwdriver.root_physx_view.set_masses(masses, env_ids_cpu)

        # 3 & 4. Finger PD gains (one scale per env, broadcast across joints)
        n_fj = len(self._finger_joint_ids)
        stiff_scale = torch.empty(n, 1, device=self.device).uniform_(*dr.finger_stiffness_range)
        damp_scale = torch.empty(n, 1, device=self.device).uniform_(*dr.finger_damping_range)
        self.allegro.write_joint_stiffness_to_sim(
            (self._base_finger_stiffness * stiff_scale).expand(-1, n_fj),
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self.allegro.write_joint_damping_to_sim(
            (self._base_finger_damping * damp_scale).expand(-1, n_fj),
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )

        # 5. Screwdriver rotational load (Coulomb) — the dominant friction proxy
        #    once a load is present.  Randomised around the base value so the
        #    adaptation network sees a range of screw resistances.
        if self._base_load_torque > 0.0:
            load_scale = torch.empty(n, device=self.device).uniform_(*dr.screwdriver_load_torque_range)
            self._env_load_torque[env_ids] = self._base_load_torque * load_scale

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

    def _compute_contact_gate(
        self, phase: CurriculumPhaseCfg, fwd_omega: torch.Tensor
    ) -> torch.Tensor:
        """Binary × continuous gate: true contact × finger-driven rolling.

        A fingertip counts as in contact only when it is inside
        ``turn_reward_contact_distance`` AND (if the contact-force gate is on)
        registering net contact force ≥ ``fingertip_contact_force_threshold``.
        Gate = 0 when fewer than ``min_contact_fingers`` are in contact, when the
        in-contact fingertips are not *driving* the handle forward (the
        rolling-consistency factor — forward fingertip tangential speed vs. the
        handle surface speed ``fwd_omega * rolling_ref_radius``; a standing squeeze
        scores ~0 even while a damped handle spins), or when the pads do not face
        the handle.

        Both the turn reward and the reverse penalty are multiplied by this
        gate (see cfg for the asymmetric-penalty failure mode it prevents).
        """
        threshold = float(phase.turn_reward_contact_distance)
        if threshold <= 0.0:
            return torch.ones(self.num_envs, device=self.device)

        tip_dist = self._compute_fingertip_axis_distances()  # (N, n_fingers)
        if tip_dist.shape[1] == 0:
            return torch.zeros(self.num_envs, device=self.device)

        contact_mask = tip_dist <= threshold  # (N, n_fingers) bool — distance

        # True-touch gate: AND the distance proxy with measured net contact
        # force.  The distance test localises the force to the handle (it is the
        # only thing near the tip), so a fingertip hovering off the surface — even
        # if inside the distance threshold — is rejected for lack of force.
        force = self._compute_fingertip_contact_forces()
        if force is not None:
            contact_mask = contact_mask & (
                force >= float(self.cfg.fingertip_contact_force_threshold)
            )
            self.extras["eval_contact_force"] = force.mean(dim=-1).detach()
            self.extras["eval_contact_force_max"] = force.max(dim=-1).values.detach()

        contact_count = contact_mask.sum(dim=-1)
        min_c = max(1, phase.turn_reward_min_contact_fingers)
        binary_gate = (contact_count >= min_c).float()

        # Rolling factor: credit only handle spin the in-contact fingertips
        # actually DRIVE.  Measure each near fingertip's SIGNED tangential speed in
        # the forward (turn_direction) sense — a static squeeze (~0 finger motion),
        # a radial press, or a back-driving finger contributes nothing — then
        # compare the average forward fingertip speed to the handle surface speed
        # (fwd_omega * rolling_ref_radius).  Genuine rolling (fingertip orbits at
        # >= surface speed) saturates the factor to 1; a standing squeeze that
        # spins the damped joint drives it to 0.  This replaces the old absolute
        # motion gate, which saturated at a trivial speed and let micro-jitter open
        # the gate while the handle "spun by itself".
        tip_pos = self.allegro.data.body_state_w[:, self._fingertip_body_ids, :3]
        fingertip_vel = self.allegro.data.body_state_w[:, self._fingertip_body_ids, 7:10]
        base = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_base_idx], :3]
        top = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_cap_idx], :3]
        tip_speed_signed = rewards.signed_tangential_speed(
            tip_pos, fingertip_vel, base, top, self.cfg.turn_direction
        )  # (N, n_fingers) — forward-positive
        tip_fwd_speed = tip_speed_signed.clamp(min=0.0)
        w = contact_mask.float()
        denom = w.sum(dim=-1).clamp(min=1.0)
        avg_contact_speed = (tip_fwd_speed * w).sum(dim=-1) / denom
        # Absolute anti-noise floor: ignore sub-threshold tremor entirely so
        # sensor/solver jitter cannot open the gate at standstill.
        min_speed = float(phase.turn_reward_min_fingertip_speed)
        if min_speed > 0.0:
            avg_contact_speed = torch.where(
                avg_contact_speed < min_speed,
                torch.zeros_like(avg_contact_speed),
                avg_contact_speed,
            )
        motion_gate = rewards.rolling_consistency(
            fwd_omega, avg_contact_speed, float(self.cfg.rolling_ref_radius)
        )

        # Pad-facing factor: SOFT (not a hard mask) so a near-but-mis-oriented
        # fingertip still receives gradient toward facing the handle instead of
        # a zero-reward cliff.  Per finger: 0 credit at cos <= thr-width, full
        # credit at cos >= thr; averaged over the near fingers.
        pad_cos = self._compute_fingertip_pad_facing()  # (N, n_fingers), ~1 = pad on
        pad_thr = float(phase.pad_facing_cos_threshold)
        width = max(float(self.cfg.pad_facing_soft_width), 1e-6)
        pad_soft = ((pad_cos - (pad_thr - width)) / width).clamp(0.0, 1.0)
        pad_factor = (pad_soft * w).sum(dim=-1) / denom
        if not self.cfg.require_pad_facing:
            pad_factor = torch.ones_like(pad_factor)

        gate = binary_gate * motion_gate * pad_factor
        # Store as float: it is a Long (bool-mask sum) and downstream consumers
        # (terminal logger, tensorboard observer) take a float mean of it.
        self.extras["eval_contact_count"] = contact_count.float().detach()
        self.extras["eval_binary_gate"] = binary_gate.detach()
        self.extras["eval_motion_gate"] = motion_gate.detach()
        self.extras["eval_avg_contact_speed"] = avg_contact_speed.detach()
        self.extras["eval_pad_cos"] = pad_cos.mean(dim=-1).detach()
        self.extras["eval_pad_gate"] = pad_factor.detach()
        return gate

    # -----------------------------------------------------------------------
    # Distance helpers
    # -----------------------------------------------------------------------

    def _compute_fingertip_contact_forces(self) -> torch.Tensor | None:
        """Per-fingertip net contact-force magnitude (N), in ``self.fingers``
        order.  Returns ``None`` when the contact-force gate is disabled or the
        sensor is unavailable, so callers fall back to the distance-only gate.
        """
        if self._fingertip_contact_sensor is None or self._contact_body_order is None:
            return None
        net = self._fingertip_contact_sensor.data.net_forces_w  # (N, n_bodies, 3)
        if net is None:
            return None
        net = net.index_select(1, self._contact_body_order)     # -> self.fingers order
        return torch.linalg.norm(net, dim=-1)                   # (N, n_fingers)

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

    def _compute_fingertip_pad_facing(self) -> torch.Tensor:
        """Cosine between each fingertip's pad normal and the tip→handle-axis
        direction.  ~1 means the pad squarely faces the handle; ≤0 means the
        back or a grazing side faces it.  Returns (N, n_fingers).
        """
        if not self._fingertip_body_ids or not self._handle_body_ids:
            return torch.empty((self.num_envs, 0), device=self.device)

        tip_state = self.allegro.data.body_state_w[:, self._fingertip_body_ids, :7]
        tip_pos = tip_state[..., :3]                 # (N, F, 3)
        tip_quat = tip_state[..., 3:7]               # (N, F, 4) wxyz
        n_f = tip_pos.shape[1]

        base = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_base_idx], :3].unsqueeze(1)
        top = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_cap_idx], :3].unsqueeze(1)
        ab = top - base
        t = ((tip_pos - base) * ab).sum(-1, keepdim=True) / (ab * ab).sum(-1, keepdim=True).clamp(min=1e-9)
        closest = base + t.clamp(0.0, 1.0) * ab      # (N, F, 3)
        to_handle = closest - tip_pos
        to_handle = to_handle / torch.linalg.norm(to_handle, dim=-1, keepdim=True).clamp(min=1e-9)

        # World-space pad normal: rotate the local pad axis by each tip's quat.
        pad_world = quat_rotate(
            tip_quat.reshape(-1, 4),
            self._pad_axis_local.expand(self.num_envs, n_f, 3).reshape(-1, 3),
        ).reshape(self.num_envs, n_f, 3)
        return (pad_world * to_handle).sum(dim=-1)   # (N, F)

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

    def _compute_grip_force_penalty(self, weight: float) -> torch.Tensor:
        """Penalises fingertip contact force above ``cfg.contact_force_target``.

        Discourages crushing the handle: with a hard squeeze (~26 N) sub-mm finger
        motions transfer large torque (the high-force micro-gait that looks like the
        handle "spinning by itself"), and it is unrealistic for hardware.  ``weight``
        is the per-phase ``reward_contact_force_weight`` (ramped in over the
        curriculum).  Returns zeros when disabled or the contact sensor is
        unavailable.
        """
        weight = float(weight)
        if weight <= 0.0:
            return torch.zeros(self.num_envs, device=self.device)
        force = self._compute_fingertip_contact_forces()  # (N, n_fingers) or None
        if force is None:
            return torch.zeros(self.num_envs, device=self.device)
        return weight * rewards.over_force_penalty(force, float(self.cfg.contact_force_target))

    def _compute_contact_engagement_reward(
        self, weight: float, tip_dist: torch.Tensor, contact_distance: float
    ) -> torch.Tensor:
        """Dense reward for fingertips pressing the handle (force up to target).

        Bridges the "hover near the handle but never press" local optimum: the
        distance-based near-reward saturates once the tips reach the surface, so
        without a contact signal the policy parks there and never opens the contact
        gate.  ``weight`` is the per-phase ``reward_contact_weight`` (high in P0 to
        bootstrap, tapering later).  Masked to fingertips within ``contact_distance``
        so finger-finger forces don't count.  Returns zeros when disabled or the
        contact sensor is unavailable.
        """
        weight = float(weight)
        if weight <= 0.0 or tip_dist.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        force = self._compute_fingertip_contact_forces()  # (N, n_fingers) or None
        if force is None:
            return torch.zeros(self.num_envs, device=self.device)
        near_mask = (tip_dist <= contact_distance).float()
        return weight * rewards.contact_engagement(
            force, near_mask, float(self.cfg.contact_force_target)
        )

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
        frame = torch.cat([finger_q, self._cur_targets], dim=-1)
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
        pos = [v for f in self.fingers for v in self.cfg.pregrasp_positions[f]]
        return torch.tensor(pos, dtype=torch.float32, device=self.device).expand(self.num_envs, -1).clone()
