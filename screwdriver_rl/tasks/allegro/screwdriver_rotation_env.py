"""Allegro hand continuous screwdriver rotation environment.

Task goal
---------
The screwdriver starts roughly vertical and the hand must spin it
continuously in one direction (negative-z) using fingertip contacts only.
The screwdriver must remain upright throughout.

Failure modes explicitly penalised
------------------------------------
- Flick / slap / knock: contact gate requires fingertips near the handle AND
  moving; the screwdriver cannot coast for reward after contact is lost.
- Oscillation: reverse penalty (slightly above turn reward, same gates)
  makes back-and-forth net-zero; logged as the oscillation ratio.
- Tilt: multiplicative upright gate kills turn reward at moderate tilt.
- Proximal / palm contact: per-step penalty on non-fingertip link proximity.
- Thumb flip / flail: covered by action-rate penalty and joint clamping.
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
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from screwdriver_rl.core import rewards
from .screwdriver_rotation_env_cfg import AllegroScrewdriverRotationEnvCfg, CurriculumPhaseCfg


# ---------------------------------------------------------------------------
# Allegro body name constants
# ---------------------------------------------------------------------------

# Fingertip (distal pad) bodies — only these should touch the handle.
_FINGERTIP_BODY_NAMES: dict[str, str] = {
    "index":  "hitosashi_ee",
    "middle": "naka_ee",
    "ring":   "kusuri_ee",
    "thumb":  "oya_ee",
}

# Proximal and medial phalange links to penalise when close to the handle.
# These are the links BEHIND the fingertip: if they touch the handle the
# policy is using the finger body rather than the fingertip pad.
_PROXIMAL_BODY_PATTERNS: list[str] = [
    r"^allegro_hand_base_link$",                         # palm
    r"^allegro_hand_hitosashi_finger_finger_link_0$",    # index proximal
    r"^allegro_hand_hitosashi_finger_finger_link_1$",    # index medial
    r"^allegro_hand_naka_finger_finger_link_4$",         # middle proximal
    r"^allegro_hand_naka_finger_finger_link_5$",         # middle medial
    r"^allegro_hand_kusuri_finger_finger_link_8$",       # ring proximal
    r"^allegro_hand_kusuri_finger_finger_link_9$",       # ring medial
    r"^allegro_hand_oya_finger_link_12$",                # thumb proximal
    r"^allegro_hand_oya_finger_link_13$",                # thumb medial
]

# Screwdriver bodies used for distance queries (handle segment).
_SCREWDRIVER_HANDLE_BODIES = ("screwdriver_stick", "screwdriver_body", "screwdriver_cap")

# Joint names for the 3-DOF screwdriver mounting (Euler representation).
_SCREWDRIVER_EULER_JOINTS = (
    "table_screwdriver_joint_1",
    "table_screwdriver_joint_2",
    "table_screwdriver_joint_3",
)
_SCREWDRIVER_CAP_JOINT = "screwdriver_body_cap_joint"

# Per-finger joint name tuples (4 DOF each, semantic order).
_FINGER_JOINT_NAMES: dict[str, tuple[str, str, str, str]] = {
    "index": (
        "allegro_hand_hitosashi_finger_finger_joint_0",
        "allegro_hand_hitosashi_finger_finger_joint_1",
        "allegro_hand_hitosashi_finger_finger_joint_2",
        "allegro_hand_hitosashi_finger_finger_joint_3",
    ),
    "middle": (
        "allegro_hand_naka_finger_finger_joint_4",
        "allegro_hand_naka_finger_finger_joint_5",
        "allegro_hand_naka_finger_finger_joint_6",
        "allegro_hand_naka_finger_finger_joint_7",
    ),
    "ring": (
        "allegro_hand_kusuri_finger_finger_joint_8",
        "allegro_hand_kusuri_finger_finger_joint_9",
        "allegro_hand_kusuri_finger_finger_joint_10",
        "allegro_hand_kusuri_finger_finger_joint_11",
    ),
    "thumb": (
        "allegro_hand_oya_finger_joint_12",
        "allegro_hand_oya_finger_joint_13",
        "allegro_hand_oya_finger_joint_14",
        "allegro_hand_oya_finger_joint_15",
    ),
}


class AllegroScrewdriverRotationEnv(DirectRLEnv):
    """Continuous screwdriver rotation with Allegro hand.

    Extends ``DirectRLEnv`` directly (no MFR dependency) and implements
    the full reward, observation, reset, and curriculum logic in one class.
    To add support for a different hand, subclass this env, override
    ``_FINGER_JOINT_NAMES``, ``_FINGERTIP_BODY_NAMES``, and
    ``_PROXIMAL_BODY_PATTERNS``, and supply the matching URDF/articulation
    config.
    """

    cfg: AllegroScrewdriverRotationEnvCfg

    def __init__(
        self,
        cfg: AllegroScrewdriverRotationEnvCfg,
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
            for finger in _FINGER_JOINT_NAMES  # all 4 fingers for reset
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
        from screwdriver_rl.utils.logging import RotationTrainingLogger
        self._logger = RotationTrainingLogger(log_interval_steps=500)

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
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["allegro"] = self.allegro
        self.scene.articulations["screwdriver"] = self.screwdriver
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        obs = torch.cat([finger_q, self._cur_targets, euler], dim=-1)  # (N, 27)
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
            # Curriculum
            "eval_curriculum_phase": torch.full((self.num_envs,), float(self._curriculum_phase.step_start), device=self.device),
        })

        # Periodic terminal log
        self._logger.log(self._global_steps, self.extras)

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
            (6.0 * stiff_scale).expand(-1, n_fj),
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self.allegro.write_joint_damping_to_sim(
            (1.0 * damp_scale).expand(-1, n_fj),
            joint_ids=self._finger_joint_ids,
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
        """Penalises proximal/medial Allegro links being close to the handle.

        Any of the listed proximal bodies within 0.05 m of the handle axis
        incurs a penalty proportional to proximity.  This shapes the policy
        away from palm-pressing, knuckle-dragging, or using the finger back.

        ``weight = 0`` skips the computation entirely (Phase 1).
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
        friction = (self._env_rotation_damping / self._base_rotation_damping).unsqueeze(-1)  # (N,1) normalised to 1.0 baseline

        tip_dist = self._compute_fingertip_axis_distances()  # (N, n_fingers)
        # Pad or truncate to 3 dims for a fixed-size privileged obs.
        n_finger_slots = 3
        if tip_dist.shape[1] >= n_finger_slots:
            tip_dist_fixed = tip_dist[:, :n_finger_slots]
        else:
            pad = torch.full((self.num_envs, n_finger_slots - tip_dist.shape[1]), 1.0, device=self.device)
            tip_dist_fixed = torch.cat([tip_dist, pad], dim=-1)

        return torch.cat([euler, angvel, rel_pos, quat, friction, tip_dist_fixed], dim=-1)

    def _update_prop_hist(self) -> None:
        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        frame = torch.cat([finger_q, self._cur_targets], dim=-1)  # (N, 24)
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
        unknown = set(self.fingers) - set(_FINGER_JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unknown finger names: {sorted(unknown)}")
        return {
            finger: self._find_joints(self.allegro, _FINGER_JOINT_NAMES[finger])
            for finger in _FINGER_JOINT_NAMES  # resolve all 4 for reset, use subset for policy
        }

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
            [_FINGERTIP_BODY_NAMES[f] for f in self.fingers],
        )

    def _resolve_proximal_bodies(self) -> list[int]:
        ids = []
        for pattern in _PROXIMAL_BODY_PATTERNS:
            found_ids, _ = self.allegro.find_bodies([pattern], preserve_order=True)
            ids.extend(found_ids)
        return list(dict.fromkeys(ids))  # deduplicate, preserve order

    def _resolve_handle_bodies(self) -> list[int]:
        return self._resolve_bodies(self.screwdriver, list(_SCREWDRIVER_HANDLE_BODIES))

    def _make_default_finger_pos(self) -> torch.Tensor:
        pos = [v for f in self.fingers for v in self.cfg.pregrasp_positions[f]]
        return torch.tensor(pos, dtype=torch.float32, device=self.device).expand(self.num_envs, -1).clone()
