"""Linker G20/L20 64 mm free-cube in-hand rotation environment."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from screwdriver_rl.core import rewards
from screwdriver_rl.deploy.codecs import ProprioCodec, inhand_linker_g20_codec_spec
from screwdriver_rl.utils.inhand_grasp_cache_manifest import (
    CACHE_ROW_WIDTH,
    load_and_validate_manifest,
    manifest_path,
    new_manifest,
)

from .inhand_rotation_env_cfg import (
    INHAND_CUBE_EDGE_M,
    LinkerL20InhandRotationEnvCfg,
    grasp_cache_filename,
)


def _spawn_local_ground() -> None:
    ground_cfg = sim_utils.CuboidCfg(
        size=(200.0, 200.0, 0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.01))


class LinkerL20InhandRotationEnv(DirectRLEnv):
    """HORA-style free-cube rotation for the current left Linker G20 mapping."""

    cfg: LinkerL20InhandRotationEnvCfg

    FINGERTIP_BODY_NAMES = {
        "index": "index_distal",
        "middle": "middle_distal",
        "ring": "ring_distal",
        "pinky": "pinky_distal",
        "thumb": "thumb_distal",
    }

    FINGER_JOINT_NAMES = {
        "index": ("index_mcp_roll", "index_mcp_pitch", "index_pip"),
        "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
        "ring": ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
        "pinky": ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
        "thumb": ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
    }

    COUPLED_JOINTS = {
        "index_dip": ("index_pip", 0.8917, 0.0),
        "middle_dip": ("middle_pip", 0.8917, 0.0),
        "ring_dip": ("ring_pip", 0.8917, 0.0),
        "pinky_dip": ("pinky_pip", 0.8917, 0.0),
        "thumb_ip": ("thumb_mcp", 1.1619, 0.0),
    }

    # Every hand link that is NOT a fingertip (palm, metacarpals, proximal and
    # middle phalanges).  The grasp-gen fingertip-only filter rejects any state
    # where the object touches one of these.  One single-body ContactSensor is
    # created per link: a multi-body regex sensor breaks PhysX filtered-force
    # views (it expects one filter entry per matched body).
    NONTIP_BODY_NAMES = (
        "hand_base_link",
        "index_metacarpals", "index_proximal", "index_middle",
        "middle_metacarpals", "middle_proximal", "middle_middle",
        "ring_metacarpals", "ring_proximal", "ring_middle",
        "pinky_metacarpals", "pinky_proximal", "pinky_middle",
        "thumb_metacarpals_base1", "thumb_metacarpals_base2",
        "thumb_metacarpals", "thumb_proximal",
    )
    # Deployment rejects support on the physical palm.  Contacts on finger
    # sides are allowed; they remain separately instrumented for diagnostics
    # and cache generation can still impose the stronger fingertip-only rule.
    PALM_BODY_NAMES = ("hand_base_link",)

    SELF_COLLISION_FILTER_PAIRS = [
        ("hand_base_link", "index_proximal"),
        ("hand_base_link", "middle_proximal"),
        ("hand_base_link", "ring_proximal"),
        ("hand_base_link", "pinky_proximal"),
        ("hand_base_link", "thumb_metacarpals_base1"),
        ("hand_base_link", "thumb_metacarpals"),
        ("hand_base_link", "thumb_proximal"),
        ("index_metacarpals", "middle_metacarpals"),
        ("middle_metacarpals", "ring_metacarpals"),
        ("ring_metacarpals", "pinky_metacarpals"),
        ("thumb_metacarpals_base2", "index_metacarpals"),
        ("thumb_metacarpals_base2", "thumb_metacarpals"),
        ("thumb_metacarpals_base2", "thumb_proximal"),
        ("thumb_metacarpals_base2", "thumb_distal"),
        ("thumb_metacarpals_base1", "thumb_proximal"),
        ("thumb_metacarpals_base1", "thumb_distal"),
        ("thumb_metacarpals", "thumb_distal"),
    ]

    def __init__(
        self,
        cfg: LinkerL20InhandRotationEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._curriculum_phase = cfg.curriculum_phases[0]
        self._global_steps: int = 0

        super().__init__(cfg, render_mode, **kwargs)

        self.fingers: tuple[str, ...] = tuple(cfg.fingers)
        self._finger_joint_ids_by_name = self._resolve_finger_joints()
        self._finger_joint_ids = [
            jid
            for finger in self.fingers
            for jid in self._finger_joint_ids_by_name[finger]
        ]
        self.num_finger_dofs = len(self._finger_joint_ids)
        self._resolve_coupled_joints()
        self._fingertip_body_ids = self._resolve_bodies(
            self.hand, [self.FINGERTIP_BODY_NAMES[f] for f in self.fingers]
        )

        finger_limits = self.hand.data.soft_joint_pos_limits[:, self._finger_joint_ids]
        margin = float(cfg.joint_target_margin)
        self._finger_lower = finger_limits[..., 0] + margin
        self._finger_upper = finger_limits[..., 1] - margin
        self._default_finger_pos = self._make_default_finger_pos()
        self._cur_targets = self._default_finger_pos.clone()
        self._home_targets = self._default_finger_pos.clone()
        self._init_pose_buf = self._default_finger_pos.clone()
        self._steps_since_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reset_action_scale = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        self._policy_dt = float(cfg.decimation) * float(cfg.sim.dt)
        self._proprio_codec = ProprioCodec(
            inhand_linker_g20_codec_spec(
                self._finger_lower[0].detach().cpu().tolist(),
                self._finger_upper[0].detach().cpu().tolist(),
                cfg.prop_hist_len,
            )
        )
        if cfg.num_obs_frames != self._proprio_codec.spec.actor_frame_count:
            raise ValueError("in-hand actor frame count does not match its ProprioCodec")
        if round(self._policy_dt * 1_000_000_000) != self._proprio_codec.spec.control_period_ns:
            raise ValueError("in-hand policy rate does not match its ProprioCodec")
        self._rot_axis = torch.tensor(cfg.rot_axis, dtype=torch.float32, device=self.device)
        self._rot_axis = self._rot_axis / torch.linalg.norm(self._rot_axis).clamp(min=1e-6)

        # Each env is assigned object prototype ``env_id % n_assets`` deterministically
        # (MultiAssetSpawnerCfg random_choice=False), so its scale (and hence which
        # per-scale grasp cache to sample) is known from the cfg's per-asset scale
        # index — independent of the shape mix (cylinder/cuboid/sphere).
        asset_scale_idx = torch.tensor(
            cfg.object_asset_scale_idx, dtype=torch.long, device=self.device
        )
        asset_shape_idx = torch.tensor(
            cfg.object_asset_shape_idx, dtype=torch.long, device=self.device
        )
        asset_proto_idx = torch.tensor(
            cfg.object_asset_proto_idx, dtype=torch.long, device=self.device
        )
        n_assets = max(len(asset_scale_idx), 1)
        asset_idx = torch.arange(self.num_envs, device=self.device) % n_assets
        self._env_scale_idx = asset_scale_idx[asset_idx]
        self._env_shape_idx = asset_shape_idx[asset_idx]
        self._env_proto_idx = asset_proto_idx[asset_idx]
        scale_values = torch.tensor(cfg.object_scales, dtype=torch.float32, device=self.device)
        self._env_scale = scale_values[self._env_scale_idx]

        self._grasp_cache = self._load_grasp_caches()
        # Validation-only hook used by the cache replay certifier.  Production
        # reset sampling leaves this as None and remains uniformly random.
        self._forced_cache_row_indices: torch.Tensor | None = None
        self._canonical_obj_pos = torch.tensor(
            cfg.grasp_gen_obj_init_pos, dtype=torch.float32, device=self.device
        )
        self._canonical_obj_quat = torch.tensor(
            cfg.grasp_gen_obj_init_rot, dtype=torch.float32, device=self.device
        )

        self._obs_hist = torch.zeros(
            (self.num_envs, cfg.prop_hist_len, cfg.history_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._hist_reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._rb_forces = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._obj_pos_prev = self.object.data.root_pos_w.detach().clone()
        self._obj_quat_prev = self.object.data.root_quat_w.detach().clone()
        self._tilt_prev = self._object_vertical_tilt(self._obj_quat_prev)
        self._episode_forward_turns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_reverse_turns = torch.zeros_like(
            self._episode_forward_turns
        )

        self._env_mass = self._default_object_mass()
        self._env_friction = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self._env_com = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._startup_randomise_dynamics()

        self.episode_length_buf = torch.randint(
            0,
            self.max_episode_length,
            (self.num_envs,),
            device=self.device,
            dtype=self.episode_length_buf.dtype,
        )

        # Completed-episode stats (count-weighted EMA over ~20k episodes), fed to
        # the training logger: mean length at done and the fraction of episodes
        # that reached timeout still holding the object.  The first reset batch
        # is skipped — episode_length_buf starts randomized, not at real lengths.
        self._ep_len_ema: float = float("nan")
        self._hold_frac_ema: float = float("nan")
        self._ep_stats_started: bool = False

        self._current_epoch: int = 0
        self._log_stage: int = 1
        self._stage2_loss: float = float("nan")
        from screwdriver_rl.utils.logging import RotationTrainingLogger

        # Keep terminal cadence in policy steps rather than aggregate samples.
        self._logger = RotationTrainingLogger(
            log_interval_steps=max(2000, 1200 * self.num_envs)
        )

    @property
    def _prop_hist_buf(self) -> torch.Tensor:
        return self._obs_hist

    def _setup_scene(self) -> None:
        self.hand = Articulation(self.cfg.robot_cfg)
        self.allegro = self.hand
        self.object = RigidObject(self.cfg.object_cfg)

        self._finger_sensors: list[ContactSensor] = []
        if self.cfg.enable_fingertip_sensors:
            obj_prim = self.cfg.object_cfg.prim_path
            for finger in self.cfg.fingers:
                distal = self.FINGERTIP_BODY_NAMES[finger]
                sensor = ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"{self.cfg.robot_cfg.prim_path}/{distal}",
                        history_length=0,
                        update_period=0.0,
                        track_air_time=False,
                        filter_prim_paths_expr=[obj_prim],
                    )
                )
                self.scene.sensors[f"contact_{finger}"] = sensor
                self._finger_sensors.append(sensor)

        self._nontip_sensors: list[ContactSensor] = []
        self._nontip_sensor_names: list[str] = []
        if self.cfg.enable_nontip_sensors or self.cfg.enable_palm_sensor:
            sensor_bodies = (
                self.NONTIP_BODY_NAMES
                if self.cfg.enable_nontip_sensors
                else self.PALM_BODY_NAMES
            )
            for body in sensor_bodies:
                sensor = ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"{self.cfg.robot_cfg.prim_path}/{body}",
                        history_length=0,
                        update_period=0.0,
                        track_air_time=False,
                        filter_prim_paths_expr=[self.cfg.object_cfg.prim_path],
                    )
                )
                self.scene.sensors[f"contact_nontip_{body}"] = sensor
                self._nontip_sensors.append(sensor)
                self._nontip_sensor_names.append(body)

        _spawn_local_ground()
        self._finalize_scene()
        self.scene.articulations["allegro"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._global_steps += self.num_envs
        self._update_curriculum()

        self._obj_pos_prev = self.object.data.root_pos_w.detach().clone()
        self._obj_quat_prev = self.object.data.root_quat_w.detach().clone()

        if self.cfg.action_clip > 0.0:
            actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        hold = max(int(self.cfg.reset_action_hold_steps), 0)
        ramp = max(int(self.cfg.reset_action_ramp_steps), 0)
        if ramp > 0:
            action_scale = (
                (self._steps_since_reset - hold + 1).to(actions.dtype)
                / float(ramp)
            ).clamp(0.0, 1.0)
        else:
            action_scale = (self._steps_since_reset >= hold).to(actions.dtype)
        self._reset_action_scale = action_scale
        self.actions = actions * action_scale.unsqueeze(-1)
        motion_range = float(self.cfg.joint_motion_range)
        if motion_range > 0.0:
            lower = torch.maximum(
                self._finger_lower, self._home_targets - motion_range
            )
            upper = torch.minimum(
                self._finger_upper, self._home_targets + motion_range
            )
        else:
            lower, upper = self._finger_lower, self._finger_upper
        if self.cfg.absolute_action_targets:
            target = self._home_targets + motion_range * self.actions
        else:
            target = (
                self._cur_targets
                + float(self.cfg.action_delta_scale) * self.actions
            )
        self._cur_targets = torch.clamp(target, lower, upper)
        self._steps_since_reset += 1
        self._update_random_forces()

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self._cur_targets, joint_ids=self._finger_joint_ids)
        self._apply_coupled_joint_targets()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        frame = self._make_obs_frame()
        self._obs_hist = torch.roll(self._obs_hist, shifts=-1, dims=1)
        self._obs_hist[:, -1] = frame
        if torch.any(self._hist_reset_mask):
            ids = torch.nonzero(self._hist_reset_mask, as_tuple=False).squeeze(-1)
            self._obs_hist[ids] = frame[ids].unsqueeze(1).expand(-1, self.cfg.prop_hist_len, -1)
            self._hist_reset_mask[ids] = False

        proprio = self._proprio_codec.assemble_actor_input(self._obs_hist)
        priv = self._compute_privileged_obs()
        actor_extrinsics = (
            self._compute_actor_extrinsics()
            if self.cfg.slow_extrinsics_only
            else priv
        )
        result = {"policy": torch.cat([proprio, actor_extrinsics], dim=-1)}
        if self.cfg.asymmetric_obs:
            result["critic"] = priv
            result["proprio_hist"] = self._obs_hist.clone()
        return result

    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self.object.data.root_pos_w
        obj_quat = self.object.data.root_quat_w
        delta_q = rewards.quat_mul(obj_quat, rewards.quat_conjugate(self._obj_quat_prev))
        rotvec = rewards.axis_angle_from_quat(delta_q)
        obj_angvel = rotvec / max(self._policy_dt, 1e-6)
        obj_linvel = (obj_pos - self._obj_pos_prev) / max(self._policy_dt, 1e-6)

        raw_rotate = torch.sum(obj_angvel * self._rot_axis, dim=-1)
        forward_vel = torch.clamp(raw_rotate, min=0.0)
        reverse_vel = torch.clamp(-raw_rotate, min=0.0)
        radians_to_turns = self._policy_dt / (2.0 * torch.pi)
        self._episode_forward_turns += forward_vel * radians_to_turns
        self._episode_reverse_turns += reverse_vel * radians_to_turns
        net_turns = self._episode_forward_turns - self._episode_reverse_turns
        total_turns = self._episode_forward_turns + self._episode_reverse_turns
        osc_ratio = self._episode_reverse_turns / self._episode_forward_turns.clamp(
            min=1.0e-6
        )
        # Mean(per-env ratio) is dominated by freshly-reset envs with almost
        # zero forward progress.  Broadcast the batch aggregate separately for
        # stable training/Stage-2 logging while retaining the per-env ratio for
        # detailed evaluation.
        osc_ratio_aggregate = self._episode_reverse_turns.sum() / (
            self._episode_forward_turns.sum().clamp(min=1.0e-6)
        )
        linvel_cost = torch.linalg.norm(obj_linvel, ord=1, dim=-1)
        finger_q = self.hand.data.joint_pos[:, self._finger_joint_ids]
        pose_cost = torch.sum((finger_q - self._init_pose_buf) ** 2, dim=-1)
        motion_range = float(self.cfg.joint_motion_range)
        if motion_range > 0.0:
            target_edge_fraction = (
                (self._cur_targets - self._home_targets).abs() / motion_range
            )
            target_bound_cost = float(self.cfg.w_target_bound) * torch.mean(
                torch.relu(target_edge_fraction - 0.8).square(), dim=-1
            )
        else:
            target_bound_cost = torch.zeros_like(pose_cost)
        tau = self.hand.data.applied_torque[:, self._finger_joint_ids]
        qdot = self.hand.data.joint_vel[:, self._finger_joint_ids]
        torque_cost = torch.sum(tau**2, dim=-1)
        work_cost = torch.sum(tau * qdot, dim=-1) ** 2

        obj_z = obj_pos[:, 2] - self.scene.env_origins[:, 2]
        fall = obj_z < float(self.cfg.reset_height_threshold)
        drop_margin = max(float(self.cfg.drop_margin_m), 1.0e-6)
        drop_margin_cost = torch.clamp(
            (
                float(self.cfg.reset_height_threshold)
                + drop_margin
                - obj_z
            )
            / drop_margin,
            min=0.0,
            max=1.0,
        )
        downward_velocity_cost = torch.clamp(-obj_linvel[:, 2], min=0.0)
        tilt_norm = self._object_vertical_tilt(obj_quat)
        upright_tilt_cost = tilt_norm.square()
        tilt_velocity_cost = torch.abs(tilt_norm - self._tilt_prev) / max(
            self._policy_dt, 1.0e-6
        )
        self._tilt_prev = tilt_norm.detach().clone()

        tip_force = self._read_tip_object_forces()
        nontip_force = self._read_nontip_object_forces()
        palm_force = self._read_named_nontip_object_forces(self.PALM_BODY_NAMES)
        if self._finger_sensors:
            # Evaluation and grasp generation use exact filtered object force.
            tip_contact = tip_force > float(self.cfg.nontip_force_eps)
        else:
            # Training-scale contact authority: distal origins are ~0.10 m from
            # the cube centre at pad contact.  This is the same conservative
            # geometry sanity bound used to certify cache candidates.
            tip_pos = self.hand.data.body_state_w[:, self._fingertip_body_ids, :3]
            tip_dist = torch.linalg.norm(
                tip_pos - obj_pos.unsqueeze(1), dim=-1
            )
            tip_contact = tip_dist <= float(self.cfg.tip_dist_max)
        # A rotating cube may transfer load from a distal pad to the same
        # finger's middle/proximal/side surface.  Those contacts are explicitly
        # legal; only the physical palm is forbidden.  Group all exact filtered
        # forces by finger and require two independently contacting fingers so
        # free-flight/coasting remains unauthorized without imposing the old
        # thumb+two-TIP cage throughout a regrasp gait.
        finger_force = tip_force.clone()
        for finger_index, finger in enumerate(self.fingers):
            side_bodies = tuple(
                body
                for body in self.NONTIP_BODY_NAMES
                if body.startswith(f"{finger}_")
            )
            finger_force[:, finger_index] += self._read_named_nontip_object_forces(
                side_bodies
            )
        finger_contact = finger_force > float(self.cfg.nontip_force_eps)
        contact_count = finger_contact.float().sum(dim=-1)
        contact_gate = contact_count >= 2.0
        contact_quality = torch.clamp(contact_count / 2.0, max=1.0)
        palm_support = palm_force > float(self.cfg.palm_support_force_threshold)
        hold_gate = contact_gate & ~palm_support & ~fall
        turn_upright_gate = torch.exp(
            -0.5
            * (
                tilt_norm
                / max(float(self.cfg.turn_upright_gate_std), 1.0e-6)
            ).square()
        )
        turn_height_gate = torch.clamp(
            (obj_z - float(self.cfg.reset_height_threshold))
            / max(float(self.cfg.turn_height_margin_m), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        stability_gate = turn_upright_gate * turn_height_gate
        turn_reward_gate = hold_gate.to(raw_rotate.dtype) * stability_gate
        rotation_credit_gate = (
            turn_reward_gate
            if bool(self.cfg.gate_rotation_reward)
            else torch.ones_like(turn_reward_gate)
        )
        tip_state = self.hand.data.body_state_w[:, self._fingertip_body_ids]
        tip_rel = tip_state[..., :3] - obj_pos.unsqueeze(1)
        desired_axis = self._rot_axis.view(1, 1, 3).expand_as(tip_rel)
        desired_tangent = torch.linalg.cross(desired_axis, tip_rel, dim=-1)
        desired_tangent = desired_tangent / torch.linalg.norm(
            desired_tangent, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        drive_velocity = torch.sum(
            tip_state[..., 7:10] * desired_tangent, dim=-1
        )
        # Credit an in-contact push in the requested direction, but do not
        # penalize the opposite fingertip motion here.  A finger must briefly
        # return while contact unloads before it can regrasp; pricing that
        # return created a local optimum at the target-window edge.  Actual
        # reverse object rotation remains penalized by ``rotate_signal``.
        drive_signal = torch.clamp(
            drive_velocity / max(float(self.cfg.drive_speed_ref), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        drive_reward = (
            float(self.cfg.drive_reward_scale)
            * (
                drive_signal * finger_contact.to(drive_signal.dtype)
            ).sum(dim=-1)
            / contact_count.clamp_min(2.0)
            * turn_reward_gate
        )
        turn_weight = float(self._curriculum_phase.reward_turn_weight)
        clipped_forward = torch.clamp(
            raw_rotate, min=0.0, max=float(self.cfg.angvel_clip[1])
        )
        clipped_reverse = torch.clamp(
            -raw_rotate, min=0.0, max=-float(self.cfg.angvel_clip[0])
        )
        rotate_signal = (
            clipped_forward
            - float(self.cfg.reverse_reward_ratio) * clipped_reverse
        )
        rotate_reward_scale = (
            float(self.cfg.rotate_reward_scale_final)
            if turn_weight >= 1.0
            else float(self.cfg.rotate_reward_scale)
        )
        rotate_reward = (
            turn_weight
            * rotate_reward_scale
            * rotate_signal
            * rotation_credit_gate
        )
        strict_contact_bonus_scale = float(
            self._curriculum_phase.strict_contact_bonus
        )
        partial_contact_bonus_scale = float(
            self._curriculum_phase.partial_contact_bonus
        )
        contact_authority_reward = (
            strict_contact_bonus_scale * contact_gate.float()
            + partial_contact_bonus_scale * contact_quality
        ) * (~palm_support & ~fall).float()
        palm_support_cost = palm_support.float()

        reward = (
            rotate_reward
            + drive_reward
            + float(self.cfg.linvel_penalty_scale) * linvel_cost
            + float(self.cfg.pose_penalty_scale) * pose_cost
            - target_bound_cost
            + float(self.cfg.torque_penalty_scale) * torque_cost
            + float(self.cfg.work_penalty_scale) * work_cost
            + float(self.cfg.fall_penalty) * fall.float()
            + contact_authority_reward
            + float(self.cfg.palm_support_penalty_scale) * palm_support_cost
            + float(self.cfg.drop_margin_penalty_scale) * drop_margin_cost
            + float(self.cfg.downward_velocity_penalty_scale)
            * downward_velocity_cost
            + float(self.cfg.upright_tilt_penalty_scale) * upright_tilt_cost
            + float(self.cfg.tilt_velocity_penalty_scale) * tilt_velocity_cost
        )
        self.extras.update(
            {
                "eval_rotate_reward": rotate_reward.detach(),
                "eval_turn_reward": rotate_reward.detach(),
                "eval_drive_reward": drive_reward.detach(),
                "eval_drive_velocity": drive_velocity.mean(dim=-1).detach(),
                "eval_fwd_vel": forward_vel.detach(),
                "eval_rev_vel": reverse_vel.detach(),
                "eval_net_turns": net_turns.detach().clone(),
                "eval_total_turns": total_turns.detach().clone(),
                "eval_raw_net_turns": net_turns.detach().clone(),
                "eval_raw_total_turns": total_turns.detach().clone(),
                "eval_osc_ratio": osc_ratio.detach(),
                "eval_osc_ratio_aggregate": torch.full_like(
                    osc_ratio, osc_ratio_aggregate
                ).detach(),
                "eval_obj_z": obj_z.detach(),
                "eval_fall_frac": fall.float().detach(),
                "eval_tilt_norm": tilt_norm.detach(),
                "eval_upright_gate": (tilt_norm < 0.20).float().detach(),
                "eval_contact_gate": contact_gate.float().detach(),
                "eval_binary_gate": contact_gate.float().detach(),
                "eval_turn_reward_gate": turn_reward_gate.detach(),
                "eval_rotation_credit_gate": rotation_credit_gate.detach(),
                "eval_turn_upright_gate": turn_upright_gate.detach(),
                "eval_turn_height_gate": turn_height_gate.detach(),
                "eval_stability_gate": stability_gate.detach(),
                "eval_contact_count": contact_count.detach(),
                "eval_contact_force": finger_force.sum(dim=-1).detach(),
                "eval_tip_contact_count": tip_contact.float().sum(dim=-1).detach(),
                "eval_contact_quality": contact_quality.detach(),
                "eval_contact_authority_reward": contact_authority_reward.detach(),
                # WrongSurf is the deployment no-palm guard.  Keep the full
                # non-tip aggregate under its own diagnostic key so finger-side
                # contact is visible without being misclassified as palm use.
                "eval_wrong_surface_force": palm_force.detach(),
                "eval_wrong_surface_object_force": palm_force.detach(),
                "eval_palm_support_force": palm_force.detach(),
                "eval_palm_support_present": palm_support_cost.detach(),
                "eval_palm_support_cost": palm_support_cost.detach(),
                "eval_nontip_surface_force": nontip_force.detach(),
                "eval_drop_margin_cost": drop_margin_cost.detach(),
                "eval_downward_velocity_cost": downward_velocity_cost.detach(),
                "eval_upright_tilt_cost": upright_tilt_cost.detach(),
                "eval_tilt_velocity_cost": tilt_velocity_cost.detach(),
                "eval_linvel_cost": linvel_cost.detach(),
                "eval_pose_cost": pose_cost.detach(),
                "eval_target_bound_cost": target_bound_cost.detach(),
                "eval_torque_cost": torque_cost.detach(),
                "eval_work_cost": work_cost.detach(),
                "eval_total_reward": reward.detach(),
                "eval_ep_len": torch.full(
                    (self.num_envs,), self._ep_len_ema, device=self.device
                ),
                "eval_hold_frac": torch.full(
                    (self.num_envs,), self._hold_frac_ema, device=self.device
                ),
                "eval_curriculum_phase": torch.full(
                    (self.num_envs,),
                    float(self.cfg.curriculum_phases.index(self._curriculum_phase) + 1),
                    device=self.device,
                ),
                "eval_num_phases": torch.full(
                    (self.num_envs,), float(len(self.cfg.curriculum_phases)), device=self.device
                ),
            }
        )
        if self._log_stage == 1:
            self._logger.log(self._global_steps, self.extras, epoch=self._current_epoch)
        return torch.nan_to_num(reward, nan=-1.0e6)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_z = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        terminated = obj_z < float(self.cfg.reset_height_threshold)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["eval_fall_frac"] = terminated.float().detach()
        return terminated, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)

        self._update_episode_stats(env_ids)

        super()._reset_idx(env_ids)

        finger_q, finger_target, obj_pos_local, obj_quat = self._sample_reset_rows(env_ids)

        root = self.hand.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.hand.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.hand.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

        jpos = self.hand.data.default_joint_pos[env_ids].clone()
        jvel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        jpos[:, self._finger_joint_ids] = finger_q
        if self._coupled_mult is not None:
            masters = jpos[:, self._coupled_master_joint_ids]
            jpos[:, self._coupled_follower_ids] = masters * self._coupled_mult + self._coupled_offset
        # Command the cached PD TARGETS (not the measured pose): the
        # target-position gap re-applies the squeeze that holds the grasp.
        jtarget = jpos.clone()
        jtarget[:, self._finger_joint_ids] = finger_target
        if self._coupled_mult is not None:
            masters_t = jtarget[:, self._coupled_master_joint_ids]
            jtarget[:, self._coupled_follower_ids] = masters_t * self._coupled_mult + self._coupled_offset
        self.hand.set_joint_position_target(jtarget, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)

        obj_pose = torch.zeros((len(env_ids), 7), dtype=torch.float32, device=self.device)
        obj_pose[:, :3] = self.scene.env_origins[env_ids] + obj_pos_local
        obj_pose[:, 3:7] = obj_quat
        self.object.write_root_pose_to_sim(obj_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device),
            env_ids=env_ids,
        )

        self._cur_targets[env_ids] = finger_target
        self._home_targets[env_ids] = finger_target
        self._init_pose_buf[env_ids] = finger_q
        self._steps_since_reset[env_ids] = 0
        self._reset_action_scale[env_ids] = 0.0
        self._episode_forward_turns[env_ids] = 0.0
        self._episode_reverse_turns[env_ids] = 0.0
        self._obj_pos_prev[env_ids] = obj_pose[:, :3].detach()
        self._obj_quat_prev[env_ids] = obj_pose[:, 3:7].detach()
        self._tilt_prev[env_ids] = self._object_vertical_tilt(
            obj_pose[:, 3:7]
        ).detach()
        self._rb_forces[env_ids] = 0.0
        self.object.set_external_force_and_torque(
            self._rb_forces[env_ids].unsqueeze(1),
            torch.zeros((len(env_ids), 1, 3), dtype=torch.float32, device=self.device),
            env_ids=env_ids,
            is_global=True,
        )
        self._randomise_pd_gains(env_ids)
        self._hist_reset_mask[env_ids] = True

    def _update_episode_stats(self, env_ids: torch.Tensor) -> None:
        """Fold the lengths of the episodes ending now into the logger EMAs.

        Must run BEFORE ``super()._reset_idx`` zeroes ``episode_length_buf``.
        """
        if not self._ep_stats_started:
            self._ep_stats_started = True  # first batch = randomized buf, skip
            return
        n = int(env_ids.numel())
        if n == 0:
            return
        lengths = self.episode_length_buf[env_ids].float()
        held = (self.episode_length_buf[env_ids] >= self.max_episode_length - 1).float()
        batch_len = float(lengths.mean().item())
        batch_held = float(held.mean().item())
        if self._ep_len_ema != self._ep_len_ema:  # NaN -> first real batch
            self._ep_len_ema = batch_len
            self._hold_frac_ema = batch_held
            return
        alpha = min(1.0, n / 20000.0)
        self._ep_len_ema += alpha * (batch_len - self._ep_len_ema)
        self._hold_frac_ema += alpha * (batch_held - self._hold_frac_ema)

    def _update_curriculum(self) -> None:
        """Select the curriculum phase from the global step count (same pattern
        as the screwdriver task): the latest phase whose ``step_start`` has been
        reached is active; transitions print a banner."""
        phases = self.cfg.curriculum_phases
        active = phases[0]
        for phase in phases:
            if self._global_steps >= phase.step_start:
                active = phase
        if active is not self._curriculum_phase:
            print(
                f"\n{'=' * 60}\n"
                f"  CURRICULUM TRANSITION @ {self._global_steps:,} steps\n"
                f"  turn_weight : {self._curriculum_phase.reward_turn_weight}"
                f"  ->  {active.reward_turn_weight}\n"
                f"  strict_contact_bonus : "
                f"{self._curriculum_phase.strict_contact_bonus}"
                f"  ->  {active.strict_contact_bonus}\n"
                f"{'=' * 60}\n",
                flush=True,
            )
            self._curriculum_phase = active
            self.cfg.episode_length_s = active.episode_length_s

    def _make_obs_frame(self) -> torch.Tensor:
        finger_q = self.hand.data.joint_pos[:, self._finger_joint_ids]
        frame = self._proprio_codec.encode_frame(finger_q, self._cur_targets)
        noise = float(self.cfg.domain_rand.joint_noise_scale) if self.cfg.domain_rand.enabled else 0.0
        if noise > 0.0:
            measured = frame[:, : self.num_finger_dofs]
            measured = measured + (2.0 * torch.rand_like(measured) - 1.0) * noise
            frame = torch.cat([measured, frame[:, self.num_finger_dofs :]], dim=-1)
        return frame

    def _compute_privileged_obs(self) -> torch.Tensor:
        obj_pos_local = self.object.data.root_pos_w - self.scene.env_origins
        return torch.cat(
            [
                obj_pos_local,
                self.object.data.root_quat_w,
                self.object.data.root_lin_vel_w,
                self.object.data.root_ang_vel_w,
                self._env_scale.unsqueeze(-1),
                self._env_mass.unsqueeze(-1),
                self._env_friction.unsqueeze(-1),
                self._env_com,
            ],
            dim=-1,
        )

    def _compute_actor_extrinsics(self) -> torch.Tensor:
        """Original-HORA 9-D teacher state encoded by the Stage-1 actor latent.

        HORA includes object position xyz alongside the six slow physical
        properties, but does not expose object orientation or linear/angular
        velocity to the actor.  Stage 2 must infer the resulting 8-D latent from
        proprioceptive history; deployment still requires no object sensor.
        """
        obj_pos_local = self.object.data.root_pos_w - self.scene.env_origins
        return torch.cat(
            [
                obj_pos_local,
                self._env_scale.unsqueeze(-1),
                self._env_mass.unsqueeze(-1),
                self._env_friction.unsqueeze(-1),
                self._env_com,
            ],
            dim=-1,
        )

    def _read_tip_object_forces(self) -> torch.Tensor:
        forces = torch.zeros(
            self.num_envs,
            len(self.fingers),
            dtype=torch.float32,
            device=self.device,
        )
        for i, sensor in enumerate(getattr(self, "_finger_sensors", [])):
            fmat = sensor.data.force_matrix_w
            if fmat is not None:
                forces[:, i] = torch.linalg.norm(fmat, dim=-1).sum(dim=(1, 2))
        return forces

    def _read_nontip_object_forces(self) -> torch.Tensor:
        return self._read_named_nontip_object_forces(self.NONTIP_BODY_NAMES)

    def _read_named_nontip_object_forces(
        self, body_names: tuple[str, ...]
    ) -> torch.Tensor:
        forces = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        selected = set(body_names)
        for body, sensor in zip(
            getattr(self, "_nontip_sensor_names", []),
            getattr(self, "_nontip_sensors", []),
        ):
            if body not in selected:
                continue
            fmat = sensor.data.force_matrix_w
            if fmat is not None:
                forces += torch.linalg.norm(fmat, dim=-1).sum(dim=(1, 2))
        return forces

    @staticmethod
    def _object_vertical_tilt(obj_quat: torch.Tensor) -> torch.Tensor:
        """Unsigned tilt of the cube's local Z axis from the gravity axis."""
        w, x, y, z = obj_quat.unbind(dim=-1)
        local_z_world_z = 1.0 - 2.0 * (x.square() + y.square())
        # A cube is invariant under a 180-degree flip, so use the closest of
        # local +Z/-Z rather than labeling an inverted cube as a failure.
        return torch.acos(local_z_world_z.abs().clamp(max=1.0))

    def _load_grasp_caches(self) -> dict[tuple[int, int, int], torch.Tensor]:
        """Caches keyed by (scale_idx, shape_idx, proto_idx) — one file per
        object prototype present in the grid (fingertip cages do not transfer
        between prototypes)."""
        caches: dict[tuple[int, int, int], torch.Tensor] = {}
        if not self.cfg.load_grasp_cache:
            return caches
        from .inhand_rotation_env_cfg import INHAND_CACHE_SHAPES

        root = Path(self.cfg.grasp_cache_dir)
        repo_root = Path(__file__).resolve().parents[3]
        shape_protos = sorted(
            set(zip(self.cfg.object_asset_shape_idx, self.cfg.object_asset_proto_idx))
        )
        required: list[tuple[tuple[int, int, int], float, str, int, Path]] = []
        for i, scale in enumerate(self.cfg.object_scales):
            for si, pi in shape_protos:
                required.append(
                    (
                        (i, si, pi),
                        float(scale),
                        INHAND_CACHE_SHAPES[si],
                        int(pi),
                        root
                        / grasp_cache_filename(
                            scale,
                            self.cfg.grasp_cache_name,
                            INHAND_CACHE_SHAPES[si],
                            pi,
                        ),
                    )
                )

        if self.cfg.require_complete_grasp_cache:
            expected_manifest = new_manifest(
                cache_name=self.cfg.grasp_cache_name,
                orientation=self.cfg.grasp_orientation,
                object_scales=self.cfg.grasp_manifest_scales,
                cube_edge_m=INHAND_CUBE_EDGE_M,
                repo_root=repo_root,
                certification=self.cfg.grasp_cache_certification,
            )
            expected_manifest.pop("entries")
            manifest = load_and_validate_manifest(
                path=manifest_path(root, self.cfg.grasp_cache_name),
                expected=expected_manifest,
                required_files=[item[-1] for item in required],
            )
            for _, scale, shape, proto, cache_path in required:
                entry = manifest["entries"][cache_path.name]
                expected_entry = {
                    "scale": scale,
                    "shape": shape,
                    "prototype": proto,
                }
                for field, expected_value in expected_entry.items():
                    if entry.get(field) != expected_value:
                        raise ValueError(
                            f"grasp-cache manifest entry {cache_path.name} has "
                            f"{field}={entry.get(field)!r}, expected {expected_value!r}"
                        )

        missing: list[tuple[tuple[int, int, int], Path]] = []
        for key, _, _, _, path in required:
            if not path.exists():
                missing.append((key, path))
                continue
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[1] != CACHE_ROW_WIDTH:
                raise ValueError(
                    f"Invalid grasp cache shape {arr.shape} in {path}; expected "
                    f"(N, {CACHE_ROW_WIDTH}) = [q(16), pd_targets(16), obj pose(7)] "
                    "— regenerate stale caches with tools/gen_inhand_grasp_cache.py"
                )
            caches[key] = torch.as_tensor(
                arr, dtype=torch.float32, device=self.device
            )
        if missing and caches:
            lines = "\n".join(f"  (scale,shape,proto)={k}: {p}" for k, p in missing)
            raise FileNotFoundError(
                "Missing LinkerL20 in-hand grasp cache file(s):\n"
                f"{lines}\nGenerate them with tools/gen_inhand_grasp_cache.py."
            )
        if missing:
            if self.cfg.require_complete_grasp_cache:
                lines = "\n".join(
                    f"  (scale,shape,proto)={k}: {p}" for k, p in missing
                )
                raise FileNotFoundError(
                    "Complete grasp caches are required for this task, but no "
                    "cache files were found:\n"
                    f"{lines}\nGenerate them with tools/gen_inhand_grasp_cache.py."
                )
            print(
                "[inhand-reset] No grasp caches found; using canonical pose fallback. "
                "Generate caches with tools/gen_inhand_grasp_cache.py before training.",
                flush=True,
            )
        return caches

    def deployment_grasp_targets(self) -> dict[str, Any]:
        """Return one deterministic, central nominal-cache posture for hardware.

        Cache rows contain the settled measured joints followed by the PD target
        that produced their squeeze.  The former is the collision-safe approach
        posture and the latter is the policy home.  A robust central row avoids
        making the exported bundle depend on environment/reset RNG state.
        """
        from .inhand_rotation_env_cfg import INHAND_CACHE_SHAPES

        nominal = [
            index
            for index, scale in enumerate(self.cfg.object_scales)
            if abs(float(scale) - 1.0) <= 1.0e-9
        ]
        if len(nominal) != 1:
            raise ValueError(
                "deployment requires exactly one nominal (scale=1.0) cube bucket"
            )
        shape_index = INHAND_CACHE_SHAPES.index("cuboid")
        key = (nominal[0], shape_index, 0)
        rows = self._grasp_cache.get(key)
        if rows is None or rows.shape[0] < 1:
            raise ValueError(f"deployment cache has no nominal cube rows for key {key}")
        features = rows[:, : 2 * self.num_finger_dofs]
        centre = features.median(dim=0).values
        robust_scale = (features - centre).abs().median(dim=0).values.clamp(min=1.0e-3)
        score = ((features - centre) / robust_scale).square().mean(dim=-1)
        row_index = int(torch.argmin(score).item())
        row = rows[row_index]
        cache_file = grasp_cache_filename(
            1.0,
            self.cfg.grasp_cache_name,
            "cuboid",
            0,
        )
        return {
            "startup_reset_targets": row[: self.num_finger_dofs]
            .detach()
            .cpu()
            .tolist(),
            "home_targets": row[
                self.num_finger_dofs : 2 * self.num_finger_dofs
            ]
            .detach()
            .cpu()
            .tolist(),
            "cache_file": cache_file,
            "cache_row": row_index,
            "cache_key": list(key),
        }

    def _sample_reset_rows(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (finger_q, finger_target, obj_pos, obj_quat).  ``finger_target``
        is the cached PD command that produced the grasp's squeeze; the canonical
        fallback uses the pose itself (geometrically supportive cage)."""
        finger_q = self._default_finger_pos[env_ids].clone()
        finger_target = finger_q.clone()
        obj_pos = self._canonical_obj_pos.unsqueeze(0).expand(len(env_ids), -1).clone()
        obj_quat = self._canonical_obj_quat.unsqueeze(0).expand(len(env_ids), -1).clone()
        if not self._grasp_cache:
            return finger_q, finger_target, obj_pos, obj_quat

        env_scale = self._env_scale_idx[env_ids]
        env_shape = self._env_shape_idx[env_ids]
        env_proto = self._env_proto_idx[env_ids]
        key_code = (env_scale * 100 + env_shape) * 100 + env_proto
        for code_t in torch.unique(key_code):
            code = int(code_t.item())
            cache = self._grasp_cache.get((code // 10000, (code // 100) % 100, code % 100))
            if cache is None:
                continue
            mask = key_code == code
            local = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if self._forced_cache_row_indices is None:
                rows = torch.randint(
                    0, cache.shape[0], (len(local),), device=self.device
                )
            else:
                rows = self._forced_cache_row_indices[env_ids[local]].remainder(
                    cache.shape[0]
                )
            sample = cache[rows]
            nd = self.num_finger_dofs
            finger_q[local] = sample[:, :nd]
            finger_target[local] = sample[:, nd : 2 * nd]
            obj_pos[local] = sample[:, 2 * nd : 2 * nd + 3]
            obj_quat[local] = sample[:, 2 * nd + 3 : 2 * nd + 7]
        return finger_q, finger_target, obj_pos, obj_quat

    def _default_object_mass(self) -> torch.Tensor:
        mass = getattr(self.object.data, "default_mass", None)
        if mass is None:
            return torch.full((self.num_envs,), 0.05, dtype=torch.float32, device=self.device)
        return mass.reshape(self.num_envs, -1)[:, 0].to(self.device, dtype=torch.float32)

    def _startup_randomise_dynamics(self) -> None:
        dr = self.cfg.domain_rand
        env_ids_cpu = torch.arange(self.num_envs, dtype=torch.int, device="cpu")
        if dr.enabled:
            mass = torch.empty(self.num_envs, device=self.device).uniform_(*dr.mass_range)
            com = torch.empty(self.num_envs, 3, device=self.device).uniform_(*dr.com_range)
            friction = torch.empty(self.num_envs, device=self.device).uniform_(*dr.friction_range)
        else:
            mass = self._default_object_mass()
            com = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
            friction = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        masses = self.object.root_physx_view.get_masses()
        masses[:, 0] = mass.detach().to("cpu")
        self.object.root_physx_view.set_masses(masses, env_ids_cpu)

        coms = self.object.root_physx_view.get_coms().clone()
        if coms.ndim == 2:
            coms[:, :3] += com.detach().to("cpu")
        else:
            coms[:, 0, :3] += com.detach().to("cpu")
        self.object.root_physx_view.set_coms(coms, env_ids_cpu)

        for asset in (self.object, self.hand):
            mats = asset.root_physx_view.get_material_properties()
            friction_cpu = friction.detach().to("cpu")
            if mats.ndim == 2:
                mats[:, 0] = friction_cpu
                mats[:, 1] = friction_cpu
            else:
                mats[..., 0] = friction_cpu.view(-1, 1)
                mats[..., 1] = friction_cpu.view(-1, 1)
            asset.root_physx_view.set_material_properties(mats, env_ids_cpu)

        self._env_mass = mass
        self._env_com = com
        self._env_friction = friction

    def _randomise_pd_gains(self, env_ids: torch.Tensor) -> None:
        dr = self.cfg.domain_rand
        lo, hi = dr.pd_gain_range
        if (not dr.enabled) or (lo == 1.0 and hi == 1.0):
            return
        n = len(env_ids)
        n_j = len(self._finger_joint_ids)
        scale = torch.empty(n, 1, device=self.device).uniform_(lo, hi)
        actuator = self.cfg.robot_cfg.actuators["fingers"]
        self.hand.write_joint_stiffness_to_sim(
            (float(actuator.stiffness) * scale).expand(-1, n_j),
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )
        self.hand.write_joint_damping_to_sim(
            (float(actuator.damping) * scale).expand(-1, n_j),
            joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
        )

    def _update_random_forces(self) -> None:
        dr = self.cfg.domain_rand
        if not dr.enabled or dr.force_scale <= 0.0:
            return
        decay = float(dr.force_decay) ** (self._policy_dt / max(float(dr.force_decay_interval), 1e-6))
        self._rb_forces *= decay
        mask = torch.rand(self.num_envs, device=self.device) < float(dr.random_force_prob)
        if torch.any(mask):
            direction = torch.randn(int(mask.sum().item()), 3, device=self.device)
            direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
            self._rb_forces[mask] = direction * (float(dr.force_scale) * self._env_mass[mask]).unsqueeze(-1)
        self.object.set_external_force_and_torque(
            self._rb_forces.unsqueeze(1),
            torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device),
            is_global=True,
        )

    def _make_default_finger_pos(self) -> torch.Tensor:
        pos = [v for finger in self.fingers for v in self.cfg.pregrasp_positions[finger]]
        return torch.tensor(pos, dtype=torch.float32, device=self.device).expand(self.num_envs, -1).clone()

    def _find_joints(self, articulation: Articulation, names: Sequence[str]) -> list[int]:
        patterns = [f"^{re.escape(n)}$" for n in names]
        ids, found = articulation.find_joints(patterns, preserve_order=True)
        if len(ids) != len(names):
            raise RuntimeError(
                f"Could not find joints {names} on {articulation.cfg.prim_path}. Found: {found}"
            )
        return ids

    def _resolve_finger_joints(self) -> dict[str, list[int]]:
        unknown = set(self.fingers) - set(self.FINGER_JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unknown finger names: {sorted(unknown)}")
        return {
            finger: self._find_joints(self.hand, self.FINGER_JOINT_NAMES[finger])
            for finger in self.FINGER_JOINT_NAMES
        }

    def _resolve_coupled_joints(self) -> None:
        self._coupled_follower_ids: list[int] = []
        self._coupled_master_joint_ids: list[int] = []
        self._coupled_master_cols_t: torch.Tensor | None = None
        self._coupled_mult: torch.Tensor | None = None
        self._coupled_offset: torch.Tensor | None = None
        if not self.COUPLED_JOINTS:
            return

        name_to_col: dict[str, int] = {}
        col = 0
        for finger in self.fingers:
            for joint_name in self.FINGER_JOINT_NAMES[finger]:
                name_to_col[joint_name] = col
                col += 1

        master_cols: list[int] = []
        mults: list[float] = []
        offsets: list[float] = []
        for follower, (master, mult, offset) in self.COUPLED_JOINTS.items():
            if master not in name_to_col:
                continue
            found_ids, _ = self.hand.find_joints([f"^{re.escape(follower)}$"], preserve_order=True)
            if len(found_ids) != 1:
                continue
            self._coupled_follower_ids.append(found_ids[0])
            c = name_to_col[master]
            master_cols.append(c)
            self._coupled_master_joint_ids.append(self._finger_joint_ids[c])
            mults.append(float(mult))
            offsets.append(float(offset))

        if self._coupled_follower_ids:
            self._coupled_master_cols_t = torch.tensor(master_cols, dtype=torch.long, device=self.device)
            self._coupled_mult = torch.tensor(mults, dtype=torch.float32, device=self.device).view(1, -1)
            self._coupled_offset = torch.tensor(offsets, dtype=torch.float32, device=self.device).view(1, -1)

    def _apply_coupled_joint_targets(self) -> None:
        if self._coupled_mult is None:
            return
        masters = self._cur_targets.index_select(1, self._coupled_master_cols_t)
        follower_targets = masters * self._coupled_mult + self._coupled_offset
        self.hand.set_joint_position_target(follower_targets, joint_ids=self._coupled_follower_ids)

    def _resolve_bodies(self, articulation: Articulation, names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for name in names:
            found_ids, found_names = articulation.find_bodies([f"^{re.escape(name)}$"], preserve_order=True)
            if len(found_ids) != 1:
                raise RuntimeError(
                    f"Expected exactly one body named {name!r} on "
                    f"{articulation.cfg.prim_path}. Found: {found_names}"
                )
            ids.append(found_ids[0])
        return ids

    def _finalize_scene(self) -> None:
        if self.scene.cfg.replicate_physics:
            self._apply_self_collision_filters(env_indices=(0,))
            self.scene.clone_environments(copy_from_source=False)
        else:
            self._apply_self_collision_filters(env_indices=range(self.scene.num_envs))
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _apply_self_collision_filters(self, env_indices=(0,)) -> None:
        if not self.SELF_COLLISION_FILTER_PAIRS:
            return
        import omni.usd
        from pxr import Sdf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        env_indices = list(env_indices)
        applied = 0
        for i in env_indices:
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
            f"[self-collision-filter] applied {applied} filtered pairs over {len(env_indices)} env(s)",
            flush=True,
        )
