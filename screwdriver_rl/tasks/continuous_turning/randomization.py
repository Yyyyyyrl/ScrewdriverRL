"""Manual per-env domain randomization with privileged-obs exposure.

We do not use Isaac Lab's Events manager because the teacher / adaptation
module must *observe* the sampled values: every sample is stored in a per-env
tensor and concatenated into the privileged observation by the env
(:meth:`priv_features`). Version-specific physx-view APIs are isolated here.

All physx-view buffer setters (materials/masses/coms) operate on CPU tensors
with CPU index tensors; joint-gain writes go through the Articulation API and
accept device tensors.
"""

from __future__ import annotations

import torch

from .env_cfg import DomainRandCfg


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


class DomainRandomizer:
    """Samples and applies per-env physics randomization on reset.

    Exposed privileged features (7): friction scale, screwdriver mass scale,
    screwdriver COM offset xy, hand PD stiffness scale, hand PD damping scale,
    screwdriver z-joint damping.
    """

    NUM_PRIV_FEATURES = 7

    def __init__(
        self,
        cfg: DomainRandCfg,
        hand,
        screwdriver,
        controlled_joint_ids: list[int],
        screwdriver_z_joint_id: int,
        screwdriver_body_ids: list[int],
        num_envs: int,
        device: str,
    ):
        self.cfg = cfg
        self.hand = hand
        self.screwdriver = screwdriver
        self.controlled_joint_ids = controlled_joint_ids
        self.z_joint_id = screwdriver_z_joint_id
        self.screwdriver_body_ids = screwdriver_body_ids
        self.num_envs = num_envs
        self.device = device

        # Per-env sampled values (identity defaults).
        self.friction_scale = torch.ones(num_envs, device=device)
        self.mass_scale = torch.ones(num_envs, device=device)
        self.com_offset = torch.zeros(num_envs, 2, device=device)
        self.stiffness_scale = torch.ones(num_envs, device=device)
        self.damping_scale = torch.ones(num_envs, device=device)
        default_z_damping = float(self.screwdriver.data.default_joint_damping[0, self.z_joint_id].item())
        self.z_damping = torch.full((num_envs,), default_z_damping, device=device)
        self._default_z_damping = default_z_damping

        # Baseline buffers captured once so repeated randomization composes
        # with the defaults, not with previous samples. CPU (physx-view space).
        self._hand_materials0 = self.hand.root_physx_view.get_material_properties().clone()
        self._screw_materials0 = self.screwdriver.root_physx_view.get_material_properties().clone()
        self._screw_coms0 = self.screwdriver.root_physx_view.get_coms().clone()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample new physics for the given envs and write them to the sim."""
        n = len(env_ids)
        cpu_ids = env_ids.cpu()

        if self.cfg.randomize_friction:
            self.friction_scale[env_ids] = _uniform(*self.cfg.friction_range, n, self.device)
            scale_cpu = self.friction_scale[env_ids].cpu()
            for view_materials0, asset in (
                (self._hand_materials0, self.hand),
                (self._screw_materials0, self.screwdriver),
            ):
                materials = asset.root_physx_view.get_material_properties()
                base = view_materials0[cpu_ids]
                materials[cpu_ids, :, 0] = base[:, :, 0] * scale_cpu.view(-1, 1)
                materials[cpu_ids, :, 1] = base[:, :, 1] * scale_cpu.view(-1, 1)
                asset.root_physx_view.set_material_properties(materials, cpu_ids)
        else:
            self.friction_scale[env_ids] = 1.0

        if self.cfg.randomize_mass:
            self.mass_scale[env_ids] = _uniform(*self.cfg.mass_scale_range, n, self.device)
            masses = self.screwdriver.root_physx_view.get_masses()
            default_mass = self.screwdriver.data.default_mass  # (num_envs, num_bodies), CPU
            body_ids = torch.tensor(self.screwdriver_body_ids, dtype=torch.long)
            masses[cpu_ids[:, None], body_ids] = (
                default_mass[cpu_ids[:, None], body_ids] * self.mass_scale[env_ids].cpu().view(-1, 1)
            )
            self.screwdriver.root_physx_view.set_masses(masses, cpu_ids)
        else:
            self.mass_scale[env_ids] = 1.0

        if self.cfg.randomize_com:
            self.com_offset[env_ids] = (
                2.0 * torch.rand(n, 2, device=self.device) - 1.0
            ) * self.cfg.com_offset_max
            coms = self.screwdriver.root_physx_view.get_coms()
            offset_cpu = self.com_offset[env_ids].cpu()
            if coms.ndim == 3:  # (num_envs, num_bodies, 7) articulation view
                body_ids = torch.tensor(self.screwdriver_body_ids, dtype=torch.long)
                base = self._screw_coms0[cpu_ids[:, None], body_ids, :2]
                coms[cpu_ids[:, None], body_ids, :2] = base + offset_cpu.unsqueeze(1)
            else:  # (num_envs, 7) rigid-object view fallback
                coms[cpu_ids, :2] = self._screw_coms0[cpu_ids, :2] + offset_cpu
            self.screwdriver.root_physx_view.set_coms(coms, cpu_ids)
        else:
            self.com_offset[env_ids] = 0.0

        if self.cfg.randomize_gains:
            self.stiffness_scale[env_ids] = _uniform(*self.cfg.stiffness_scale_range, n, self.device)
            self.damping_scale[env_ids] = _uniform(*self.cfg.damping_scale_range, n, self.device)
            joint_ids = self.controlled_joint_ids
            default_stiffness = self.hand.data.default_joint_stiffness[env_ids][:, joint_ids]
            default_damping = self.hand.data.default_joint_damping[env_ids][:, joint_ids]
            self.hand.write_joint_stiffness_to_sim(
                default_stiffness * self.stiffness_scale[env_ids].view(-1, 1),
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
            self.hand.write_joint_damping_to_sim(
                default_damping * self.damping_scale[env_ids].view(-1, 1),
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        else:
            self.stiffness_scale[env_ids] = 1.0
            self.damping_scale[env_ids] = 1.0

        if self.cfg.randomize_z_damping:
            self.z_damping[env_ids] = _uniform(*self.cfg.z_damping_range, n, self.device)
            self.screwdriver.write_joint_damping_to_sim(
                self.z_damping[env_ids].view(-1, 1),
                joint_ids=[self.z_joint_id],
                env_ids=env_ids,
            )
        else:
            self.z_damping[env_ids] = self._default_z_damping

    def priv_features(self) -> torch.Tensor:
        """(num_envs, 7) tensor of the currently active DR values."""
        return torch.cat(
            [
                self.friction_scale.unsqueeze(-1),
                self.mass_scale.unsqueeze(-1),
                self.com_offset,
                self.stiffness_scale.unsqueeze(-1),
                self.damping_scale.unsqueeze(-1),
                self.z_damping.unsqueeze(-1),
            ],
            dim=-1,
        )
