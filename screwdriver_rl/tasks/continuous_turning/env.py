"""Continuous screwdriver turning environment (Isaac Lab DirectRLEnv).

A dexterous hand (selected via the HandSpec registry) must spin a mounted
screwdriver about its shaft axis indefinitely while keeping it upright. The
reward is HORA-style signed spin progress, gated by a fingertip contact/motion
proxy and an uprightness gate; see rewards.py and docs/reward_design.md.

The env emits an observation dict:
    policy       (N, 2*act_dim [+3])  -- proprio-only by default (deployable)
    critic       (N, priv_dim)        -- privileged obs for the teacher/critic
    proprio_hist (N, hist_len, 2*act_dim) -- history for the stage-2 student
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

from ...core import rewards as R
from ...robots import get_hand_spec
from .env_cfg import ContinuousTurningEnvCfg
from .randomization import DomainRandomizer

SCREWDRIVER_EULER_JOINT_NAMES = (
    "table_screwdriver_joint_1",  # tilt x
    "table_screwdriver_joint_2",  # tilt y
    "table_screwdriver_joint_3",  # spin z
)
SCREWDRIVER_CAP_JOINT_NAME = "screwdriver_body_cap_joint"
# Order matters: (shaft/stick, handle/body, cap).
SCREWDRIVER_BODY_NAMES = ("screwdriver_stick", "screwdriver_body", "screwdriver_cap")


class ContinuousTurningEnv(DirectRLEnv):
    """Hand-agnostic continuous screwdriver turning task."""

    cfg: ContinuousTurningEnvCfg

    def __init__(self, cfg: ContinuousTurningEnvCfg, render_mode: str | None = None, **kwargs: Any):
        # Resolve the hand before super().__init__ — _setup_scene needs it and
        # the spaces must match the selected hand.
        self._hand_spec = get_hand_spec(cfg.hand_name, controlled_fingers=tuple(cfg.controlled_fingers))
        cfg.resolve_for_hand(self._hand_spec)
        super().__init__(cfg, render_mode, **kwargs)

        spec = self._hand_spec
        self.controlled_fingers = tuple(spec.controlled_fingers)
        self.num_finger_dofs = spec.num_action_dofs

        # ---- joint / body resolution (always by name: Isaac Lab BFS joint
        # order interleaves fingers, positional indexing is wrong) ----
        self._joint_ids_by_finger = {
            finger: self._find_ordered_joints(self.hand, names)
            for finger, names in spec.finger_joint_names.items()
        }
        self._controlled_joint_ids = [
            jid for finger in self.controlled_fingers for jid in self._joint_ids_by_finger[finger]
        ]
        self._parked_joint_ids = [
            jid for finger in spec.parked_fingers for jid in self._joint_ids_by_finger[finger]
        ]

        self._euler_joint_ids = self._find_ordered_joints(self.screwdriver, SCREWDRIVER_EULER_JOINT_NAMES)
        self._z_joint_id = self._euler_joint_ids[2]
        self._cap_joint_id = self._find_ordered_joints(self.screwdriver, (SCREWDRIVER_CAP_JOINT_NAME,))[0]
        self._screwdriver_body_ids = self._find_ordered_bodies(self.screwdriver, SCREWDRIVER_BODY_NAMES)
        self._shaft_body_id, self._handle_body_id, self._cap_body_id = self._screwdriver_body_ids

        self._fingertip_body_ids = self._find_ordered_bodies(
            self.hand, tuple(spec.fingertip_body_names[f] for f in self.controlled_fingers)
        )
        self._thumb_tip_index = (
            self.controlled_fingers.index(spec.thumb_name) if spec.thumb_name in self.controlled_fingers else None
        )
        self._non_thumb_tip_indices = [
            i for i, f in enumerate(self.controlled_fingers) if f != spec.thumb_name
        ]

        # ---- pregrasp + joint limits ----
        def _pregrasp_vec(fingers: Sequence[str]) -> torch.Tensor:
            vals = [v for f in fingers for v in spec.pregrasp[f]]
            return torch.tensor(vals, dtype=torch.float32, device=self.device)

        self._pregrasp_controlled = _pregrasp_vec(self.controlled_fingers).repeat(self.num_envs, 1)
        self._pregrasp_parked = (
            _pregrasp_vec(spec.parked_fingers).repeat(self.num_envs, 1)
            if spec.parked_fingers
            else torch.zeros((self.num_envs, 0), device=self.device)
        )

        limits = self.hand.data.soft_joint_pos_limits[:, self._controlled_joint_ids]
        margin = max(0.0, float(self.cfg.joint_target_margin))
        self._joint_lower = limits[..., 0] + margin
        self._joint_upper = limits[..., 1] - margin
        bad = self._joint_lower > self._joint_upper
        if torch.any(bad):
            mid = 0.5 * (limits[..., 0] + limits[..., 1])
            self._joint_lower = torch.where(bad, mid, self._joint_lower)
            self._joint_upper = torch.where(bad, mid, self._joint_upper)

        # ---- state buffers ----
        N = self.num_envs
        self.actions = torch.zeros((N, self.num_finger_dofs), device=self.device)
        self._prev_actions = torch.zeros_like(self.actions)
        self._cur_targets = self._pregrasp_controlled.clone()
        self._prev_z = torch.zeros(N, device=self.device)
        self._prev_tilt_xy = torch.zeros(N, 2, device=self.device)
        self._prev_shaft_quat = torch.zeros(N, 4, device=self.device)
        self._prev_shaft_quat[:, 0] = 1.0
        self._net_turn = torch.zeros(N, device=self.device)
        self._total_turn = torch.zeros(N, device=self.device)
        self._prev_milestone_count = torch.zeros(N, device=self.device)
        self._lost_contact_steps = torch.zeros(N, dtype=torch.long, device=self.device)
        self._z_history = torch.zeros(N, int(self.cfg.stagnation_window), device=self.device)

        self._history_dim = int(self.cfg.history_obs_dim)
        self._proprio_hist_buf = torch.zeros(
            (N, int(self.cfg.prop_hist_len), self._history_dim), device=self.device
        )

        self._policy_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)

        # ---- domain randomization ----
        self._randomizer = DomainRandomizer(
            self.cfg.dr,
            hand=self.hand,
            screwdriver=self.screwdriver,
            controlled_joint_ids=self._controlled_joint_ids,
            screwdriver_z_joint_id=self._z_joint_id,
            screwdriver_body_ids=self._screwdriver_body_ids,
            num_envs=N,
            device=self.device,
        )

        # Stagger episode starts: without this every env times out on the same
        # step and cumulative metrics show a sawtooth artifact.
        self.episode_length_buf = torch.randint(
            0, self.max_episode_length, (N,), device=self.device, dtype=self.episode_length_buf.dtype
        )

        self._validate_spaces()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.screwdriver = Articulation(self.cfg.screwdriver_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        self.scene.articulations["screwdriver"] = self.screwdriver

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.dr.action_noise_std > 0.0:
            actions = actions + self.cfg.dr.action_noise_std * torch.randn_like(actions)
        clip = float(self.cfg.action_clip)
        if clip > 0.0:
            actions = torch.clamp(actions, -clip, clip)
        self.actions = actions.clone()

        # HORA-style integrated delta targets: action=0 holds position.
        targets = self._cur_targets + float(self.cfg.action_delta_scale) * actions
        self._cur_targets = torch.clamp(targets, self._joint_lower, self._joint_upper)

        self._update_proprio_hist()

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self._cur_targets, joint_ids=self._controlled_joint_ids)
        if self._parked_joint_ids:
            self.hand.set_joint_position_target(self._pregrasp_parked, joint_ids=self._parked_joint_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        finger_q = self.hand.data.joint_pos[:, self._controlled_joint_ids]
        if self.cfg.dr.obs_noise_std > 0.0:
            finger_q = finger_q + self.cfg.dr.obs_noise_std * torch.randn_like(finger_q)
        parts = [finger_q, self._cur_targets]
        if self.cfg.include_object_in_policy_obs:
            parts.append(self.screwdriver.data.joint_pos[:, self._euler_joint_ids])
        obs = torch.cat(parts, dim=-1)

        result = {"policy": obs}
        if self.cfg.asymmetric_obs:
            result["critic"] = self._compute_privileged_obs()
            result["proprio_hist"] = self._proprio_hist_buf.clone()
        return result

    def _compute_privileged_obs(self) -> torch.Tensor:
        euler = self.screwdriver.data.joint_pos[:, self._euler_joint_ids]
        tilt_xy = euler[:, :2]
        z = euler[:, 2]
        joint_vel = self.screwdriver.data.joint_vel[:, self._euler_joint_ids]
        cap_vel = self.screwdriver.data.joint_vel[:, self._cap_joint_id].unsqueeze(-1)
        tip_dist = self._fingertip_screwdriver_distances()
        rel_pos = self.screwdriver.data.root_pos_w - self.hand.data.root_pos_w
        mean_torque = (
            self.hand.data.applied_torque[:, self._controlled_joint_ids].abs().mean(dim=-1, keepdim=True)
        )
        return torch.cat(
            [
                tilt_xy,  # 2
                torch.sin(z).unsqueeze(-1),  # 1
                torch.cos(z).unsqueeze(-1),  # 1
                joint_vel,  # 3
                cap_vel,  # 1
                tip_dist,  # num controlled fingers
                rel_pos,  # 3
                self._randomizer.priv_features(),  # 7
                mean_torque,  # 1
            ],
            dim=-1,
        )

    def _update_proprio_hist(self) -> None:
        finger_q = self.hand.data.joint_pos[:, self._controlled_joint_ids]
        frame = torch.cat([finger_q, self._cur_targets], dim=-1)
        self._proprio_hist_buf = torch.roll(self._proprio_hist_buf, shifts=-1, dims=1)
        self._proprio_hist_buf[:, -1, :] = frame

    # ------------------------------------------------------------------
    # Contact proxy
    # ------------------------------------------------------------------

    def _fingertip_screwdriver_distances(self) -> torch.Tensor:
        """(N, num_controlled_fingers) fingertip distance to the handle axis.

        Distance to the segment handle-origin -> cap-origin: physically
        interpretable (handle radius 0.02 m => pad contact ~0.03 m) regardless
        of grip height.
        """
        tips = self.hand.data.body_pos_w[:, self._fingertip_body_ids]
        handle_base = self.screwdriver.data.body_pos_w[:, self._handle_body_id]
        handle_top = self.screwdriver.data.body_pos_w[:, self._cap_body_id]
        return R.point_segment_distance(tips, handle_base, handle_top)

    def _fingertip_speeds(self) -> torch.Tensor:
        vel = self.hand.data.body_lin_vel_w[:, self._fingertip_body_ids]
        return torch.linalg.norm(vel, dim=-1)

    def _turn_reward_gate(self, tip_dist: torch.Tensor) -> torch.Tensor:
        threshold = float(self.cfg.turn_reward_contact_distance)
        if threshold <= 0.0:
            return torch.ones(self.num_envs, device=self.device)
        contact_mask = tip_dist <= threshold
        contact_count = contact_mask.sum(dim=-1)
        min_contacts = max(1, int(self.cfg.turn_reward_min_contact_fingers))
        contact_gate = (contact_count >= min_contacts).float()

        speeds = self._fingertip_speeds()
        weights = contact_mask.float()
        active = torch.clamp(weights.sum(dim=-1), min=1.0)
        contact_speed = (speeds * weights).sum(dim=-1) / active
        m_gate = R.motion_gate(
            contact_speed,
            float(self.cfg.turn_reward_min_fingertip_speed),
            float(self.cfg.turn_reward_full_fingertip_speed),
        )
        self.extras["eval_turn_contact_count"] = contact_count.detach()
        self.extras["eval_turn_contact_gate"] = contact_gate.detach()
        self.extras["eval_turn_motion_gate"] = m_gate.detach()
        return contact_gate * m_gate

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        euler = self.screwdriver.data.joint_pos[:, self._euler_joint_ids]
        tilt_xy = euler[:, :2]
        z_curr = euler[:, 2]
        upright_norm = torch.linalg.norm(tilt_xy, dim=-1)

        # --- spin progress ---
        euler_delta = R.wrap_to_pi(cfg.turn_direction * (z_curr - self._prev_z))
        self._prev_z = z_curr.detach().clone()
        if cfg.use_shaft_spin_measure:
            shaft_quat = self.screwdriver.data.body_quat_w[:, self._shaft_body_id]
            delta_z = R.shaft_spin_delta(shaft_quat, self._prev_shaft_quat, cfg.turn_direction)
            self._prev_shaft_quat = shaft_quat.detach().clone()
        else:
            delta_z = euler_delta

        turn_velocity, fwd_vel, rev_vel = R.turn_velocities(
            delta_z, self._policy_dt, float(cfg.turn_velocity_clip)
        )

        # --- gates ---
        tip_dist = self._fingertip_screwdriver_distances()
        contact_motion_gate = self._turn_reward_gate(tip_dist)
        u_gate = R.upright_gate(upright_norm, float(cfg.turn_upright_gate_std))
        # Both spin terms carry the SAME gates: gating only the positive side
        # makes the expected value of contact negative and the optimum becomes
        # "open the fingers and never touch".
        gate = contact_motion_gate * u_gate

        turn_reward = cfg.reward_turn_weight * fwd_vel * gate
        reverse_cost = cfg.reward_reverse_weight * rev_vel * gate

        # --- accumulated progress + milestones ---
        self._total_turn += torch.clamp(delta_z, min=0.0).detach()
        self._net_turn += delta_z.detach()
        milestone, self._prev_milestone_count = R.milestone_reward(
            self._net_turn, self._prev_milestone_count, float(cfg.milestone_angle), float(cfg.milestone_bonus)
        )
        milestone = milestone * gate

        # --- uprightness costs ---
        upright_cost = cfg.reward_upright_weight * torch.sum(tilt_xy**2, dim=-1)
        tilt_vel = (tilt_xy - self._prev_tilt_xy) / self._policy_dt
        self._prev_tilt_xy = tilt_xy.detach().clone()
        tilt_vel_cost = cfg.reward_tilt_velocity_weight * torch.linalg.vector_norm(tilt_vel, ord=1, dim=-1)

        # --- regularization ---
        finger_q = self.hand.data.joint_pos[:, self._controlled_joint_ids]
        finger_vel = self.hand.data.joint_vel[:, self._controlled_joint_ids]
        torque = self.hand.data.applied_torque[:, self._controlled_joint_ids]
        action_cost = cfg.reward_action_weight * torch.sum(self.actions**2, dim=-1)
        action_rate_cost = cfg.reward_action_rate_weight * torch.mean(
            (self.actions - self._prev_actions) ** 2, dim=-1
        )
        self._prev_actions = self.actions.detach().clone()
        finger_pose_cost = cfg.reward_finger_pose_weight * torch.sum(
            (finger_q - self._pregrasp_controlled) ** 2, dim=-1
        )
        finger_vel_cost = cfg.reward_finger_velocity_weight * torch.mean(finger_vel**2, dim=-1)
        limit_cost = cfg.reward_joint_limit_weight * R.joint_limit_barrier(
            finger_q, self._joint_lower, self._joint_upper, float(cfg.joint_limit_margin)
        )
        work_cost = cfg.reward_work_weight * torch.sum(torch.abs(torque * finger_vel), dim=-1)

        # --- near-contact shaping (discovery; decays to 0 in curriculum) ---
        near = torch.exp(-tip_dist / max(float(cfg.near_reward_std), 1.0e-6))
        near_score = R.near_contact_score(
            near, self._thumb_tip_index, self._non_thumb_tip_indices, int(cfg.near_reward_top_k)
        )
        near_reward = cfg.near_reward_weight * near_score

        reward = (
            turn_reward
            + milestone
            + near_reward
            - reverse_cost
            - upright_cost
            - tilt_vel_cost
            - action_cost
            - action_rate_cost
            - finger_pose_cost
            - finger_vel_cost
            - limit_cost
            - work_cost
        )

        # --- metrics (consumed by the trainer console/tensorboard) ---
        ex = self.extras
        ex["eval_turn_delta"] = delta_z.detach()
        ex["eval_euler_turn_delta"] = euler_delta.detach()
        ex["eval_turn_velocity"] = turn_velocity.detach()
        ex["eval_forward_turn_velocity"] = fwd_vel.detach()
        ex["eval_reverse_turn_velocity"] = rev_vel.detach()
        ex["eval_total_turns"] = (self._total_turn / (2.0 * math.pi)).detach().clone()
        ex["eval_net_turns"] = (self._net_turn / (2.0 * math.pi)).detach().clone()
        ex["eval_screwdriver_upright_norm"] = upright_norm.detach()
        ex["eval_turn_upright_gate"] = u_gate.detach()
        ex["eval_turn_gate"] = gate.detach()
        ex["eval_turn_reward"] = turn_reward.detach()
        ex["eval_reverse_cost"] = reverse_cost.detach()
        ex["eval_milestone_reward"] = milestone.detach()
        ex["eval_upright_cost"] = upright_cost.detach()
        ex["eval_tilt_velocity_cost"] = tilt_vel_cost.detach()
        ex["eval_action_cost"] = action_cost.detach()
        ex["eval_action_rate_cost"] = action_rate_cost.detach()
        ex["eval_finger_pose_cost"] = finger_pose_cost.detach()
        ex["eval_finger_velocity_cost"] = finger_vel_cost.detach()
        ex["eval_joint_limit_cost"] = limit_cost.detach()
        ex["eval_work_cost"] = work_cost.detach()
        ex["eval_near_reward"] = near_reward.detach()
        ex["eval_near_score"] = near_score.detach()
        ex["eval_mean_fingertip_dist"] = tip_dist.mean(dim=-1).detach()
        ex["eval_min_fingertip_dist"] = tip_dist.min(dim=-1).values.detach()

        return torch.nan_to_num(reward, nan=-1.0e6)

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        euler = self.screwdriver.data.joint_pos[:, self._euler_joint_ids]
        upright_norm = torch.linalg.norm(euler[:, :2], dim=-1)
        if cfg.upright_termination_threshold > 0.0:
            tilted = upright_norm > cfg.upright_termination_threshold
            terminated |= tilted
            self.extras["eval_upright_terminated"] = tilted.detach()

        # Lost contact: consecutive policy steps with too few fingertips near
        # the handle.
        if cfg.lost_contact_termination_distance > 0.0:
            tip_dist = self._fingertip_screwdriver_distances()
            contact_count = (tip_dist <= cfg.lost_contact_termination_distance).sum(dim=-1)
            out = contact_count < max(1, int(cfg.lost_contact_min_fingers))
            self._lost_contact_steps = torch.where(
                out, self._lost_contact_steps + 1, torch.zeros_like(self._lost_contact_steps)
            )
            lost = self._lost_contact_steps >= max(1, int(cfg.lost_contact_grace_steps))
            terminated |= lost
            self.extras["eval_lost_contact_terminated"] = lost.detach()

        # Stagnation: z angle barely moved over the window (dexscrew-style).
        self._z_history = torch.roll(self._z_history, shifts=-1, dims=1)
        self._z_history[:, -1] = euler[:, 2]
        if cfg.stagnation_variance_eps > 0.0:
            warm = self.episode_length_buf >= int(cfg.stagnation_grace_steps)
            stagnant = (self._z_history.var(dim=1) < cfg.stagnation_variance_eps) & warm
            terminated |= stagnant
            self.extras["eval_stagnation_terminated"] = stagnant.detach()

        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)
        n = len(env_ids)
        dr = self.cfg.dr

        # ---- hand ----
        root_state = self.hand.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.hand.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.hand.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        for finger, jids in self._joint_ids_by_finger.items():
            joint_pos[:, jids] = torch.tensor(
                self._hand_spec.pregrasp[finger], dtype=torch.float32, device=self.device
            )
        if dr.randomize_init_state and dr.pregrasp_noise > 0.0:
            noise = (2.0 * torch.rand(n, len(self._controlled_joint_ids), device=self.device) - 1.0)
            joint_pos[:, self._controlled_joint_ids] += dr.pregrasp_noise * noise
        joint_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        self.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # ---- screwdriver ----
        screw_root = self.screwdriver.data.default_root_state[env_ids].clone()
        screw_root[:, :3] += self.scene.env_origins[env_ids]
        self.screwdriver.write_root_pose_to_sim(screw_root[:, :7], env_ids=env_ids)
        self.screwdriver.write_root_velocity_to_sim(screw_root[:, 7:], env_ids=env_ids)

        screw_pos = torch.zeros_like(self.screwdriver.data.default_joint_pos[env_ids])
        screw_vel = torch.zeros_like(self.screwdriver.data.default_joint_vel[env_ids])
        if dr.randomize_init_state:
            lo, hi = dr.init_z_angle_range
            screw_pos[:, self._z_joint_id] = lo + (hi - lo) * torch.rand(n, device=self.device)
            if dr.init_tilt_max > 0.0:
                tilt = dr.init_tilt_max * (2.0 * torch.rand(n, 2, device=self.device) - 1.0)
                screw_pos[:, self._euler_joint_ids[0]] = tilt[:, 0]
                screw_pos[:, self._euler_joint_ids[1]] = tilt[:, 1]
        self.screwdriver.write_joint_state_to_sim(screw_pos, screw_vel, env_ids=env_ids)

        # ---- domain randomization (also feeds privileged obs) ----
        self._randomizer.reset(env_ids)

        # ---- reset internal buffers ----
        self._cur_targets[env_ids] = joint_pos[:, self._controlled_joint_ids]
        self.actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._net_turn[env_ids] = 0.0
        self._total_turn[env_ids] = 0.0
        self._prev_milestone_count[env_ids] = 0.0
        self._lost_contact_steps[env_ids] = 0

        self._settle_contacts()

        # Sync deltas to the post-settle state so the first reward step does
        # not see a fake jump.
        euler = self.screwdriver.data.joint_pos[env_ids][:, self._euler_joint_ids]
        self._prev_z[env_ids] = euler[:, 2].detach()
        self._prev_tilt_xy[env_ids] = euler[:, :2].detach()
        self._prev_shaft_quat[env_ids] = self.screwdriver.data.body_quat_w[env_ids, self._shaft_body_id].detach()
        # Fill the stagnation window with values spread wider than the
        # variance threshold so fresh episodes cannot trigger it.
        self._z_history[env_ids] = euler[:, 2:3] + torch.linspace(
            -1.0, 1.0, self._z_history.shape[1], device=self.device
        ).unsqueeze(0)

        finger_q = self.hand.data.joint_pos[env_ids][:, self._controlled_joint_ids]
        frame = torch.cat([finger_q, self._cur_targets[env_ids]], dim=-1)
        self._proprio_hist_buf[env_ids] = frame.unsqueeze(1).repeat(1, self._proprio_hist_buf.shape[1], 1)

    def _settle_contacts(self) -> None:
        if self.cfg.reset_contact_steps <= 0:
            return
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        for _ in range(int(self.cfg.reset_contact_steps)):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_ordered_joints(self, articulation: Articulation, names: Sequence[str]) -> list[int]:
        patterns = [f"^{re.escape(name)}$" for name in names]
        ids, found = articulation.find_joints(patterns, preserve_order=True)
        if len(ids) != len(names):
            raise RuntimeError(
                f"Could not resolve joints {tuple(names)} on {articulation.cfg.prim_path}; found {tuple(found)}."
            )
        return ids

    def _find_ordered_bodies(self, articulation: Articulation, names: Sequence[str]) -> list[int]:
        patterns = [f"^{re.escape(name)}$" for name in names]
        ids, found = articulation.find_bodies(patterns, preserve_order=True)
        if len(ids) != len(names):
            raise RuntimeError(
                f"Could not resolve bodies {tuple(names)} on {articulation.cfg.prim_path}; found {tuple(found)}."
            )
        return ids

    def _validate_spaces(self) -> None:
        expected_obs = 2 * self.num_finger_dofs + (3 if self.cfg.include_object_in_policy_obs else 0)
        obs_space = self.single_observation_space
        obs_shape = obs_space["policy"].shape if hasattr(obs_space, "spaces") else obs_space.shape
        act_shape = self.single_action_space.shape
        if act_shape != (self.num_finger_dofs,):
            raise ValueError(f"action space {act_shape} != expected {(self.num_finger_dofs,)}")
        if obs_shape != (expected_obs,):
            raise ValueError(f"policy obs space {obs_shape} != expected {(expected_obs,)}")
        if self._history_dim != 2 * self.num_finger_dofs:
            raise ValueError(
                f"history_obs_dim {self._history_dim} != 2 * num_finger_dofs {2 * self.num_finger_dofs}"
            )
