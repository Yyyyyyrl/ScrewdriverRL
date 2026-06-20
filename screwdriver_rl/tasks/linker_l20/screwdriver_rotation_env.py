"""Linker Hand L20 (Left) continuous screwdriver rotation environment.

Reworked from scratch.  The hand-agnostic plumbing (joint/body resolution, mimic
coupling, reset, domain randomisation, proprioceptive history, shaft-spin measure)
is still inherited from :class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnv`,
but this class overrides the scene, reward, contact, curriculum and privileged-obs
logic so the LinkerL20 task can use a completely new design without disturbing the
Allegro task.

Key differences from the base/Allegro design
---------------------------------------------
* **Per-fingertip ContactSensors** (one per ``*_distal`` body, filtered against the
  screwdriver's stick/body/cap) measure exactly how hard each finger presses the
  screwdriver — and the cap specifically.  Contact is judged purely from this
  force through a trapezoidal "good pressure" window; there is no distance gate and
  no pad-facing gate.  A separate unfiltered sensor over the non-fingertip links
  flags wrong-surface (back/knuckle/palm) contact.
* **Full Coulomb load from step 0** (the curriculum pins ``screwdriver_load_scale``
  to 1.0) plus strong rotation/tilt damping ⇒ the handle never free-spins.
* **Prescribed-lite finger roles**: the index holds the cap down, the thumb +
  middle/ring/pinky drive the rotation, an anti-idle term keeps every finger used.
* **Joint-range restriction**: each finger DOF is hard-clamped to a small window
  around its home (pregrasp) value and a soft deviation penalty keeps motions small.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from screwdriver_rl.core import rewards
from screwdriver_rl.tasks.base.screwdriver_rotation_env import ScrewdriverRotationEnv

from .screwdriver_rotation_env_cfg import LinkerL20ScrewdriverRotationEnvCfg


# Screwdriver collision bodies the per-finger sensors filter against, in the order
# they appear as columns of ``force_matrix_w`` (= the filter list order below).
_SCREWDRIVER_FILTER_BODIES = ("screwdriver_stick", "screwdriver_body", "screwdriver_cap")
_SD_STICK, _SD_BODY, _SD_CAP = 0, 1, 2


class LinkerL20ScrewdriverRotationEnv(ScrewdriverRotationEnv):
    """Continuous screwdriver rotation with the Linker Hand L20 (left)."""

    cfg: LinkerL20ScrewdriverRotationEnvCfg

    # Fingertip (distal pad) bodies — only these should touch the handle.
    FINGERTIP_BODY_NAMES = {
        "index":  "index_distal",
        "middle": "middle_distal",
        "ring":   "ring_distal",
        "pinky":  "pinky_distal",
        "thumb":  "thumb_distal",
    }

    # Non-fingertip links to penalise when they register contact force with the
    # screwdriver (palm, metacarpals, proximal and medial phalanges).  Everything
    # BEHIND the distal pads.
    PROXIMAL_BODY_PATTERNS = [
        r"^hand_base_link$",                          # palm
        r"^(index|middle|ring|pinky)_metacarpals$",   # knuckle bases
        r"^(index|middle|ring|pinky)_proximal$",      # proximal phalanges
        r"^(index|middle|ring|pinky)_middle$",        # medial phalanges
        r"^thumb_metacarpals_base[12]$",              # thumb CMC staging
        r"^thumb_metacarpals$",
        r"^thumb_proximal$",
    ]

    # Per-finger INDEPENDENT joint names (semantic order).  Mimic distal joints
    # (*_dip, thumb_ip) are NOT listed here — they are driven via COUPLED_JOINTS.
    FINGER_JOINT_NAMES = {
        "index":  ("index_mcp_roll", "index_mcp_pitch", "index_pip"),
        "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
        "ring":   ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
        "pinky":  ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
        "thumb":  ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
    }

    # Mimic followers: follower -> (master, multiplier, offset).  Multipliers
    # taken verbatim from the URDF <mimic> tags.
    COUPLED_JOINTS = {
        "index_dip":  ("index_pip", 0.8917, 0.0),
        "middle_dip": ("middle_pip", 0.8917, 0.0),
        "ring_dip":   ("ring_pip", 0.8917, 0.0),
        "pinky_dip":  ("pinky_pip", 0.8917, 0.0),
        "thumb_ip":   ("thumb_mcp", 1.1619, 0.0),
    }

    # Self-collision pair filters: physically-impossible overlaps created by the
    # inflated collision hulls near the palm.  These links are rigidly clustered
    # at the palm and cannot touch on the real hand, so filtering them is
    # sim-to-real-safe.  The deployment-critical collisions (fingertip<->fingertip,
    # a finger crossing into a neighbour's middle/distal) are NOT filtered.
    SELF_COLLISION_FILTER_PAIRS = [
        # palm <-> each finger's proximal phalanx (can't fold back into the palm)
        ("hand_base_link", "index_proximal"),
        ("hand_base_link", "middle_proximal"),
        ("hand_base_link", "ring_proximal"),
        ("hand_base_link", "pinky_proximal"),
        # palm <-> thumb's non-adjacent CMC chain + proximal
        ("hand_base_link", "thumb_metacarpals_base1"),
        ("hand_base_link", "thumb_metacarpals"),
        ("hand_base_link", "thumb_proximal"),
        # adjacent knuckle bases (rigidly packed at the palm)
        ("index_metacarpals", "middle_metacarpals"),
        ("middle_metacarpals", "ring_metacarpals"),
        ("ring_metacarpals", "pinky_metacarpals"),
        ("thumb_metacarpals_base2", "index_metacarpals"),
        # thumb's nested 3-stage CMC chain: non-adjacent internal segments whose
        # convex hulls overlap (base2->base1->metacarpals->proximal->distal).
        ("thumb_metacarpals_base2", "thumb_metacarpals"),
        ("thumb_metacarpals_base2", "thumb_proximal"),
        ("thumb_metacarpals_base2", "thumb_distal"),
        ("thumb_metacarpals_base1", "thumb_proximal"),
        ("thumb_metacarpals_base1", "thumb_distal"),
        ("thumb_metacarpals", "thumb_distal"),
    ]

    # -----------------------------------------------------------------------
    # Init (Linker-specific bookkeeping after the base sets everything up)
    # -----------------------------------------------------------------------

    def __init__(
        self,
        cfg: LinkerL20ScrewdriverRotationEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        # Finger-role indices (index = cap stabiliser; the rest = drive fingers).
        self._index_tip_idx: int = self.fingers.index("index")
        self._drive_tip_idxs: list[int] = [
            i for i, f in enumerate(self.fingers) if f != "index"
        ]
        self._drive_tip_idxs_t = torch.tensor(
            self._drive_tip_idxs, dtype=torch.long, device=self.device
        )

        # Home (pregrasp) targets per finger DOF, and the per-DOF motion window.
        self._home_targets: torch.Tensor = self._default_finger_pos.clone()  # (N, D)
        range_t = self._build_joint_range_tensor()  # (1, D)
        # Tighten the base target-clamp bounds to home +/- range so base
        # _pre_physics_step physically restricts each DOF to a small window.
        self._finger_lower = torch.maximum(self._finger_lower, self._home_targets - range_t)
        self._finger_upper = torch.minimum(self._finger_upper, self._home_targets + range_t)
        # Keep the reset pose inside the (now tighter) window.
        self._cur_targets = torch.clamp(self._cur_targets, self._finger_lower, self._finger_upper)

    def _build_joint_range_tensor(self) -> torch.Tensor:
        """Per-DOF motion half-width (rad) around home, shape ``(1, num_finger_dofs)``.

        Defaults to ``cfg.joint_motion_range`` for every joint, with optional
        per-joint overrides from ``cfg.joint_motion_range_overrides``.  Joint order
        matches ``self._finger_joint_ids`` (fingers x their FINGER_JOINT_NAMES).
        """
        names = [jn for f in self.fingers for jn in self.FINGER_JOINT_NAMES[f]]
        r = torch.full(
            (len(names),), float(self.cfg.joint_motion_range), device=self.device
        )
        for i, name in enumerate(names):
            if name in self.cfg.joint_motion_range_overrides:
                r[i] = float(self.cfg.joint_motion_range_overrides[name])
        return r.view(1, -1)

    # -----------------------------------------------------------------------
    # Scene (per-finger filtered sensors + wrong-surface sensor)
    # -----------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self.allegro = Articulation(self.cfg.robot_cfg)
        self.screwdriver = Articulation(self.cfg.screwdriver_cfg)

        # --- Per-fingertip filtered contact sensors ---
        # Isaac Lab only reports filtered (force_matrix_w) contacts one-to-many, so
        # each fingertip needs its own single-body sensor.  Filtering against the
        # three screwdriver bodies gives, per finger, the force on [stick, body,
        # cap] — the cap column is what tells us the index is pressing the cap.
        sd_prim = self.cfg.screwdriver_cfg.prim_path
        filters = [f"{sd_prim}/{b}" for b in _SCREWDRIVER_FILTER_BODIES]
        self._finger_sensors: list[ContactSensor] = []
        for finger in self.cfg.fingers:
            distal = self.FINGERTIP_BODY_NAMES[finger]
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"{self.cfg.robot_cfg.prim_path}/{distal}",
                    history_length=0,
                    update_period=0.0,
                    track_air_time=False,
                    filter_prim_paths_expr=list(filters),
                )
            )
            self.scene.sensors[f"contact_{finger}"] = sensor
            self._finger_sensors.append(sensor)

        # The base post-init builds a single-sensor body-order map only when this
        # attribute is set; we use our own per-finger sensors instead.
        self._fingertip_contact_sensor = None

        # --- Wrong-surface sensor (one unfiltered sensor over all non-tip links) ---
        prox_regex = "(" + "|".join(p.strip("^$") for p in self.PROXIMAL_BODY_PATTERNS) + ")"
        self._proximal_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=f"{self.cfg.robot_cfg.prim_path}/{prox_regex}",
                history_length=0,
                update_period=0.0,
                track_air_time=False,
            )
        )
        self.scene.sensors["contact_proximal"] = self._proximal_sensor

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.cfg.friction_coefficient,
                    dynamic_friction=self.cfg.friction_coefficient,
                )
            ),
        )
        # Apply self-collision pair filters on the source env BEFORE cloning.
        self._apply_self_collision_filters()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["allegro"] = self.allegro
        self.scene.articulations["screwdriver"] = self.screwdriver
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -----------------------------------------------------------------------
    # Contact-force reading
    # -----------------------------------------------------------------------

    def _read_contact_forces(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-finger contact-force magnitudes against the screwdriver.

        Returns ``(F_total, F_body, F_cap, wrong_surface_force)`` where the first
        three are ``(N, n_fingers)`` (in ``self.fingers`` order) and the last is
        ``(N,)`` — the total contact force on all non-fingertip links.
        """
        n, nf = self.num_envs, len(self.fingers)
        F_total = torch.zeros(n, nf, device=self.device)
        F_body = torch.zeros(n, nf, device=self.device)
        F_cap = torch.zeros(n, nf, device=self.device)
        for i, sensor in enumerate(self._finger_sensors):
            fmat = sensor.data.force_matrix_w  # (N, 1, 3, 3) or None
            if fmat is None:
                continue
            mag = torch.linalg.norm(fmat, dim=-1)[:, 0, :]  # (N, 3) over [stick, body, cap]
            F_total[:, i] = mag.sum(dim=-1)
            F_body[:, i] = mag[:, _SD_BODY]
            F_cap[:, i] = mag[:, _SD_CAP]

        wrong = torch.zeros(n, device=self.device)
        if self._proximal_sensor is not None:
            net = self._proximal_sensor.data.net_forces_w  # (N, n_prox, 3) or None
            if net is not None:
                wrong = torch.linalg.norm(net, dim=-1).sum(dim=-1)
        return F_total, F_body, F_cap, wrong

    def _compute_fingertip_tangential_speed(self) -> torch.Tensor:
        """Per-finger fingertip speed tangential to the handle axis, ``(N, n_fingers)``."""
        tip_pos = self.allegro.data.body_state_w[:, self._fingertip_body_ids, :3]
        tip_vel = self.allegro.data.body_state_w[:, self._fingertip_body_ids, 7:10]
        base = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_base_idx], :3]
        top = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_cap_idx], :3]
        return rewards.tangential_speed(tip_pos, tip_vel, base, top)

    # -----------------------------------------------------------------------
    # Curriculum (same selection as base; prints the new phase fields)
    # -----------------------------------------------------------------------

    def _update_curriculum(self) -> None:
        phases = self.cfg.curriculum_phases
        active = phases[0]
        for phase in phases:
            if self._global_steps >= phase.step_start:
                active = phase
        if active is not self._curriculum_phase:
            print(
                f"\n{'='*60}\n"
                f"  CURRICULUM TRANSITION: Phase @{self._curriculum_phase.step_start:,}"
                f"  ->  Phase @{active.step_start:,}\n"
                f"  Global steps    : {self._global_steps:,}\n"
                f"  turn_weight     : {self._curriculum_phase.reward_turn_weight}"
                f"  ->  {active.reward_turn_weight}\n"
                f"  min_drive_fing. : {self._curriculum_phase.min_drive_fingers}"
                f"  ->  {active.min_drive_fingers}\n"
                f"  term_threshold  : {self._curriculum_phase.upright_termination_threshold}"
                f"  ->  {active.upright_termination_threshold} rad\n"
                f"  episode_s       : {self._curriculum_phase.episode_length_s}"
                f"  ->  {active.episode_length_s}\n"
                f"{'='*60}\n",
                flush=True,
            )
            self._curriculum_phase = active
            self.cfg.episode_length_s = active.episode_length_s

    # -----------------------------------------------------------------------
    # Rewards (force-based; prescribed-lite finger roles; stay-home)
    # -----------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        phase = self._curriculum_phase
        cfg = self.cfg
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        z_curr = euler[:, 2]

        # ---- Rotation delta (prefer true shaft-axis spin over Euler-z) ----
        raw_delta_z = cfg.turn_direction * (z_curr - self._prev_z)
        delta_z = rewards.wrap_to_pi(raw_delta_z)
        self._prev_z = z_curr.detach().clone()
        if cfg.use_shaft_spin_measure:
            shaft_delta = self._compute_shaft_spin_delta()
            if shaft_delta is not None:
                delta_z = shaft_delta
        turn_vel, fwd_vel, rev_vel = rewards.turn_velocities(
            delta_z, self._policy_dt, cfg.turn_velocity_clip
        )

        # ---- Upright gate ----
        tilt_xy = euler[:, :2]
        tilt_norm = torch.linalg.norm(tilt_xy, dim=-1)
        upright_gate = rewards.upright_gate(tilt_norm, cfg.turn_upright_gate_std)

        # ---- Contact forces -> per-finger "good pressure" scores ----
        F_total, F_body, F_cap, wrong_force = self._read_contact_forces()
        fw = lambda f: rewards.force_window(  # noqa: E731
            f, cfg.contact_f_min, cfg.contact_f_lo, cfg.contact_f_hi, cfg.contact_f_max
        )
        engage = fw(F_total)        # (N, nf) total touch quality per finger
        body_engage = fw(F_body)    # (N, nf) touch quality on the handle body
        cap_engage = fw(F_cap)      # (N, nf) touch quality on the cap

        # Turn gate: enough DRIVE fingers (non-index) pressing the body.
        drive_body_engage = body_engage.index_select(1, self._drive_tip_idxs_t)  # (N, n_drive)
        turn_gate = rewards.soft_count_gate(drive_body_engage, phase.min_drive_fingers)
        combined_gate = turn_gate * upright_gate

        # ---- Positive terms ----
        turn_reward = phase.reward_turn_weight * fwd_vel * combined_gate

        index_cap_reward = phase.w_index_cap * cap_engage[:, self._index_tip_idx] * upright_gate

        tang = self._compute_fingertip_tangential_speed()  # (N, nf)
        tang_factor = (tang / max(cfg.drive_full_tangential_speed, 1e-6)).clamp(0.0, 1.0)
        drive_terms = drive_body_engage * tang_factor.index_select(1, self._drive_tip_idxs_t)
        drive_reward = phase.w_drive * drive_terms.mean(dim=-1) * upright_gate

        grip_reward = phase.w_grip * engage.mean(dim=-1)

        # ---- Progress tracking + milestone ----
        self._total_turn += torch.clamp(delta_z, min=0.0).detach()
        self._net_turn += delta_z.detach()
        milestone_reward = self._compute_milestone_reward(gate=combined_gate)

        # ---- Negative terms ----
        reverse_cost = cfg.reward_reverse_weight * rev_vel * combined_gate
        upright_cost = cfg.reward_upright_weight * torch.sum(tilt_xy ** 2, dim=-1)
        tilt_vel = (tilt_xy - self._prev_tilt_xy) / self._policy_dt
        self._prev_tilt_xy = tilt_xy.detach().clone()
        tilt_vel_cost = cfg.reward_tilt_velocity_weight * torch.linalg.norm(tilt_vel, ord=1, dim=-1)

        excess_cost = phase.w_excess * rewards.excess_force(F_total, cfg.contact_f_max).sum(dim=-1)
        wrong_surface_cost = phase.w_wrong * wrong_force
        idle_cost = phase.w_idle * (F_total < cfg.contact_f_min).float().sum(dim=-1)

        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        home_dev_cost = cfg.w_home_dev * rewards.home_deviation(
            finger_q, self._home_targets, cfg.home_deviation_deadband
        )

        action_cost = cfg.reward_action_weight * torch.sum(self.actions ** 2, dim=-1)
        action_rate_cost = cfg.reward_action_rate_weight * torch.mean(
            (self.actions - self._prev_actions) ** 2, dim=-1
        )
        self._prev_actions = self.actions.detach().clone()
        finger_vel = self.allegro.data.joint_vel[:, self._finger_joint_ids]
        finger_vel_cost = cfg.reward_finger_velocity_weight * torch.mean(finger_vel ** 2, dim=-1)

        reward = (
            turn_reward
            + index_cap_reward
            + drive_reward
            + grip_reward
            + milestone_reward
            - reverse_cost
            - upright_cost
            - tilt_vel_cost
            - excess_cost
            - wrong_surface_cost
            - idle_cost
            - home_dev_cost
            - action_cost
            - action_rate_cost
            - finger_vel_cost
        )

        # ---- Logging extras ----
        osc_ratio = (self._total_turn - self._net_turn.clamp(min=0.0)) / (self._total_turn + 1e-6)
        drive_count = (drive_body_engage > 0.5).float().sum(dim=-1)
        binary_gate = (drive_count >= phase.min_drive_fingers).float()
        idle_count = (F_total < cfg.contact_f_min).float().sum(dim=-1)
        max_joint_dev = (finger_q - self._home_targets).abs().max(dim=-1).values
        self.extras.update({
            # Progress
            "eval_total_turns":      (self._total_turn / (2.0 * math.pi)).detach(),
            "eval_net_turns":        (self._net_turn / (2.0 * math.pi)).detach(),
            "eval_osc_ratio":        osc_ratio.detach(),
            "eval_turn_vel":         turn_vel.detach(),
            "eval_fwd_vel":          fwd_vel.detach(),
            "eval_rev_vel":          rev_vel.detach(),
            # Object / upright
            "eval_tilt_norm":        tilt_norm.detach(),
            "eval_upright_gate":     upright_gate.detach(),
            "eval_upright_cost":     upright_cost.detach(),
            "eval_tilt_vel_cost":    tilt_vel_cost.detach(),
            # Contact (force-based)
            "eval_contact_gate":     turn_gate.detach(),
            "eval_binary_gate":      binary_gate.detach(),
            "eval_drive_count":      drive_count.detach(),
            "eval_in_window":        engage.mean(dim=-1).detach(),
            "eval_contact_force":    F_total.mean(dim=-1).detach(),
            "eval_contact_force_max": F_total.max(dim=-1).values.detach(),
            "eval_index_cap_force":  F_cap[:, self._index_tip_idx].detach(),
            "eval_idle_count":       idle_count.detach(),
            "eval_wrong_surface_force": wrong_force.detach(),
            "eval_max_joint_dev":    max_joint_dev.detach(),
            # Reward breakdown
            "eval_turn_reward":      turn_reward.detach(),
            "eval_index_cap_reward": index_cap_reward.detach(),
            "eval_drive_reward":     drive_reward.detach(),
            "eval_grip_reward":      grip_reward.detach(),
            "eval_milestone":        milestone_reward.detach(),
            "eval_reverse_cost":     reverse_cost.detach(),
            "eval_excess_cost":      excess_cost.detach(),
            "eval_wrong_surface_cost": wrong_surface_cost.detach(),
            "eval_idle_cost":        idle_cost.detach(),
            "eval_home_dev_cost":    home_dev_cost.detach(),
            "eval_action_cost":      action_cost.detach(),
            "eval_action_rate":      action_rate_cost.detach(),
            "eval_finger_vel_cost":  finger_vel_cost.detach(),
            "eval_total_reward":     reward.detach(),
            # Curriculum
            "eval_curriculum_phase": torch.full(
                (self.num_envs,),
                float(self.cfg.curriculum_phases.index(self._curriculum_phase) + 1),
                device=self.device,
            ),
            "eval_num_phases": torch.full(
                (self.num_envs,), float(len(self.cfg.curriculum_phases)), device=self.device
            ),
        })

        self._logger.log(self._global_steps, self.extras, epoch=self._current_epoch)
        return torch.nan_to_num(reward, nan=-1.0e6)

    # -----------------------------------------------------------------------
    # Privileged observations (per-finger force replaces fingertip distances)
    # -----------------------------------------------------------------------

    def _compute_privileged_obs(self) -> torch.Tensor:
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        angvel = self.screwdriver.data.joint_vel[:, self._screwdriver_euler_ids]
        rel_pos = self.screwdriver.data.root_pos_w - self.allegro.data.root_pos_w
        quat = self.screwdriver.data.root_quat_w
        if self._base_load_torque > 0.0:
            friction = (self._env_load_torque / self._base_load_torque).unsqueeze(-1)
        else:
            friction = (self._env_rotation_damping / self._base_rotation_damping).unsqueeze(-1)
        F_total, _, _, _ = self._read_contact_forces()  # (N, n_fingers)
        return torch.cat([euler, angvel, rel_pos, quat, friction, F_total], dim=-1)
