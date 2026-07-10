"""Grasp-cache generation environment for LinkerL20 in-hand rotation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .inhand_rotation_env import LinkerL20InhandRotationEnv
from .inhand_rotation_env_cfg import LinkerL20InhandGraspGenEnvCfg


class LinkerL20InhandGraspGenEnv(LinkerL20InhandRotationEnv):
    cfg: LinkerL20InhandGraspGenEnvCfg

    def __init__(self, cfg: LinkerL20InhandGraspGenEnvCfg, render_mode=None, **kwargs):
        # Harvested rows bucketed per object prototype (fingertip cages do not
        # transfer between prototypes, so each gets its own cache file).
        self._harvest_rows: dict[int, list[torch.Tensor]] = {}
        self._last_acceptance: torch.Tensor | None = None
        self._last_timed_out: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        acceptance = self._compute_acceptance()
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        self._last_acceptance = acceptance.detach().clone()
        self._last_timed_out = timed_out.detach().clone()
        # Grace window: the dropped object may transiently violate the strict
        # criteria (e.g. graze a non-distal link) while settling into the cage.
        in_grace = self.episode_length_buf < int(self.cfg.grasp_gen_grace_steps)
        terminated = ~acceptance & ~in_grace
        self.extras["eval_grasp_acceptance"] = acceptance.float().detach()
        return terminated, timed_out

    def _reset_idx(self, env_ids) -> None:
        if env_ids is not None and self._last_acceptance is not None and self._last_timed_out is not None:
            if not isinstance(env_ids, torch.Tensor):
                env_ids_t = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            else:
                env_ids_t = env_ids.to(dtype=torch.long, device=self.device)
            survived = self._last_acceptance[env_ids_t] & self._last_timed_out[env_ids_t]
            harvest_ids = env_ids_t[survived]
            if len(harvest_ids) > 0:
                rows = self._cache_rows(harvest_ids).detach().cpu()
                protos = self._env_proto_idx[harvest_ids].detach().cpu()
                for proto_t in torch.unique(protos):
                    proto = int(proto_t.item())
                    self._harvest_rows.setdefault(proto, []).append(rows[protos == proto_t])

        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids_t = self.hand._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_t = env_ids.to(dtype=torch.long, device=self.device)

        noise = float(self.cfg.grasp_gen_pose_noise)
        finger_q = self._default_finger_pos[env_ids_t].clone()
        if noise > 0.0:
            finger_q = finger_q + noise * (2.0 * torch.rand_like(finger_q) - 1.0)
            finger_q = torch.clamp(finger_q, self._finger_lower[env_ids_t], self._finger_upper[env_ids_t])
        self._cur_targets[env_ids_t] = finger_q
        self._init_pose_buf[env_ids_t] = finger_q

        jpos = self.hand.data.joint_pos[env_ids_t].clone()
        jvel = torch.zeros_like(self.hand.data.joint_vel[env_ids_t])
        jpos[:, self._finger_joint_ids] = finger_q
        if self._coupled_mult is not None:
            masters = jpos[:, self._coupled_master_joint_ids]
            jpos[:, self._coupled_follower_ids] = masters * self._coupled_mult + self._coupled_offset
        self.hand.set_joint_position_target(jpos, env_ids=env_ids_t)
        self.hand.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids_t)

        obj_pose = torch.zeros((len(env_ids_t), 7), dtype=torch.float32, device=self.device)
        obj_pose[:, :3] = self.scene.env_origins[env_ids_t] + torch.tensor(
            self.cfg.grasp_gen_obj_init_pos, dtype=torch.float32, device=self.device
        )
        obj_pose[:, 3] = 1.0
        self.object.write_root_pose_to_sim(obj_pose, env_ids=env_ids_t)
        self.object.write_root_velocity_to_sim(
            torch.zeros((len(env_ids_t), 6), dtype=torch.float32, device=self.device),
            env_ids=env_ids_t,
        )
        self._obj_pos_prev[env_ids_t] = obj_pose[:, :3].detach()
        self._obj_quat_prev[env_ids_t] = obj_pose[:, 3:7].detach()
        self._hist_reset_mask[env_ids_t] = True

    def _compute_acceptance(self) -> torch.Tensor:
        obj_center = self.object.data.root_pos_w
        tip_pos = self.hand.data.body_state_w[:, self._fingertip_body_ids, :3]
        tip_dist = torch.linalg.norm(tip_pos - obj_center.unsqueeze(1), dim=-1)
        close = torch.all(tip_dist <= float(self.cfg.tip_dist_max), dim=-1)

        tip_contact = self._read_tip_object_forces() > 0.0
        thumb_col = self.fingers.index("thumb")
        other_cols = [i for i in range(len(self.fingers)) if i != thumb_col]
        thumb_contact = tip_contact[:, thumb_col]
        other_count = tip_contact[:, other_cols].float().sum(dim=-1)
        grip = other_count >= int(self.cfg.min_other_finger_contacts)
        if self.cfg.require_thumb_contact:
            grip = grip & thumb_contact

        nontip_force = self._read_nontip_object_forces()
        tips_only = nontip_force <= float(self.cfg.nontip_force_eps)

        obj_z = obj_center[:, 2] - self.scene.env_origins[:, 2]
        high = obj_z >= float(self.cfg.reset_height_threshold) + float(
            self.cfg.grasp_gen_accept_z_margin
        )
        self.extras.update(
            {
                "eval_tip_close": close.float().detach(),
                "eval_contact_fingers": tip_contact.float().sum(dim=-1).detach(),
                "eval_thumb_contact": thumb_contact.float().detach(),
                "eval_other_contacts": other_count.detach(),
                "eval_nontip_force": nontip_force.detach(),
                "eval_tips_only": tips_only.float().detach(),
                "eval_obj_high": high.float().detach(),
            }
        )
        acceptance = close & grip & high
        if self.cfg.forbid_nontip_contact:
            acceptance = acceptance & tips_only
        return acceptance

    def _read_tip_object_forces(self) -> torch.Tensor:
        forces = torch.zeros(self.num_envs, len(self.fingers), dtype=torch.float32, device=self.device)
        for i, sensor in enumerate(self._finger_sensors):
            fmat = sensor.data.force_matrix_w
            if fmat is None:
                continue
            forces[:, i] = torch.linalg.norm(fmat, dim=-1).sum(dim=(1, 2))
        return forces

    def _read_nontip_object_forces(self) -> torch.Tensor:
        """Total object-contact force magnitude over all non-distal hand links."""
        forces = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for sensor in getattr(self, "_nontip_sensors", []):
            fmat = sensor.data.force_matrix_w
            if fmat is None:
                continue
            forces += torch.linalg.norm(fmat, dim=-1).sum(dim=(1, 2))
        return forces

    def _cache_rows(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Row = [q(16), pd_targets(16), obj_pos(3), obj_quat(4)] (39 cols).

        The PD targets are what hold the grasp: during generation the fingers
        are commanded toward the perturbed pose while the object blocks them,
        and that target-position gap IS the squeeze.  Replaying only ``q`` with
        ``target = q`` (HORA's 23-col format) zeroes the grip force and drops
        most objects at reset.
        """
        finger_q = self.hand.data.joint_pos[env_ids][:, self._finger_joint_ids]
        finger_target = self._cur_targets[env_ids]
        obj_pos = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        obj_quat = self.object.data.root_quat_w[env_ids]
        return torch.cat([finger_q, finger_target, obj_pos, obj_quat], dim=-1)

    def harvest_counts(self) -> dict[int, int]:
        """Rows harvested so far, per object prototype."""
        n_protos = int(max(self.cfg.object_asset_proto_idx)) + 1
        return {
            p: sum(t.shape[0] for t in self._harvest_rows.get(p, []))
            for p in range(n_protos)
        }

    def save_if_full(self, path_for_proto, n: int = 12500) -> bool:
        """Save one cache file per prototype once EVERY prototype has ``n``
        accepted rows.  ``path_for_proto(proto_idx)`` returns the output path."""
        counts = self.harvest_counts()
        if not counts or min(counts.values()) < n:
            return False
        for proto, lists in sorted(self._harvest_rows.items()):
            rows = torch.cat(lists, dim=0)
            path = Path(path_for_proto(proto))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, rows[:n].numpy())
        return True
