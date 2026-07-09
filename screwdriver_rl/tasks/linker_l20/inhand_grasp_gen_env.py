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
        self._harvest_rows: list[torch.Tensor] = []
        self._last_acceptance: torch.Tensor | None = None
        self._last_timed_out: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        acceptance = self._compute_acceptance()
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        self._last_acceptance = acceptance.detach().clone()
        self._last_timed_out = timed_out.detach().clone()
        terminated = ~acceptance
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
                self._harvest_rows.append(self._cache_rows(harvest_ids).detach().cpu())

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
        contact_count = (self._read_tip_object_forces() > 0.0).float().sum(dim=-1)
        obj_z = obj_center[:, 2] - self.scene.env_origins[:, 2]
        high = obj_z >= float(self.cfg.reset_height_threshold)
        self.extras.update(
            {
                "eval_tip_close": close.float().detach(),
                "eval_contact_fingers": contact_count.detach(),
                "eval_obj_high": high.float().detach(),
            }
        )
        return close & (contact_count >= int(self.cfg.min_contact_fingers)) & high

    def _read_tip_object_forces(self) -> torch.Tensor:
        forces = torch.zeros(self.num_envs, len(self.fingers), dtype=torch.float32, device=self.device)
        for i, sensor in enumerate(self._finger_sensors):
            fmat = sensor.data.force_matrix_w
            if fmat is None:
                continue
            forces[:, i] = torch.linalg.norm(fmat, dim=-1).sum(dim=(1, 2))
        return forces

    def _cache_rows(self, env_ids: torch.Tensor) -> torch.Tensor:
        finger_q = self.hand.data.joint_pos[env_ids][:, self._finger_joint_ids]
        obj_pos = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        obj_quat = self.object.data.root_quat_w[env_ids]
        return torch.cat([finger_q, obj_pos, obj_quat], dim=-1)

    def save_if_full(self, path: str | Path, n: int = 50000) -> bool:
        if self._harvest_rows:
            rows = torch.cat(self._harvest_rows, dim=0)
        else:
            rows = torch.empty(0, 23)
        if rows.shape[0] < n:
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, rows[:n].numpy())
        return True
