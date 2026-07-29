"""Linker Hand L20 (Left) continuous screwdriver rotation environment.

Reworked from scratch.  The hand-agnostic plumbing (joint/body resolution, mimic
coupling, reset, domain randomisation, proprioceptive history, shaft-spin measure)
is still inherited from :class:`screwdriver_rl.tasks.base.ScrewdriverRotationEnv`,
but this class overrides the scene, reward, contact, curriculum and privileged-obs
logic so the LinkerL20 task can use a completely new design without disturbing the
Allegro task.

Key differences from the base/Allegro design
---------------------------------------------
* **Kinematic fingertip contact** uses signed surface clearance and fixed
  per-finger margins.  ContactSensor forces remain diagnostic-only; they do not
  gate progress, scale rewards, or enter policy/privileged observations.  A
  separate unfiltered sensor over non-fingertip links supplies the binary
  wrong-surface safety predicate.
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

        contact_margins = [
            float(
                cfg.contact_d_margin_by_finger.get(
                    finger, cfg.contact_d_margin
                )
            )
            for finger in self.fingers
        ]
        self._contact_d_margin = torch.tensor(
            contact_margins, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._contact_d_ramp_width = float(
            cfg.contact_d_far_margin - cfg.contact_d_margin
        )

        # Home (pregrasp) targets per finger DOF, and the per-DOF motion window.
        self._home_targets: torch.Tensor = self._default_finger_pos.clone()  # (N, D)
        range_t = self._build_joint_range_tensor()  # (1, D)
        self._joint_range = range_t
        # Tighten the base target-clamp bounds to home +/- range so base
        # _pre_physics_step physically restricts each DOF to a small window.
        self._finger_lower = torch.maximum(self._finger_lower, self._home_targets - range_t)
        self._finger_upper = torch.minimum(self._finger_upper, self._home_targets + range_t)
        # Keep the reset pose inside the (now tighter) window.
        self._cur_targets = torch.clamp(self._cur_targets, self._finger_lower, self._finger_upper)

        # Reward authorization state.  The base counters are repurposed below to
        # track only contact-authorized turn progress; raw shaft motion is kept in
        # separate diagnostics so coasting cannot satisfy evaluation success.
        self._drive_contact_streak = torch.zeros(self.num_envs, device=self.device)
        self._raw_total_turn = torch.zeros(self.num_envs, device=self.device)
        self._raw_net_turn = torch.zeros(self.num_envs, device=self.device)
        self._steps_since_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reset_action_scale = torch.zeros(self.num_envs, device=self.device)
        self._reset_contact_guard_retries = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reset_contact_guard_donor_repaired = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Cached, already-settled kinematic states form an empirical
        # contact-conditioned proposal bank.  They let large batches repair
        # rejection-sampling tails without advancing the entire PhysX scene by
        # another 32 hidden steps for every remaining outlier.
        self._reset_contact_cache_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reset_contact_cache_variant = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._reset_contact_cache_hand_root = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        self._reset_contact_cache_hand_q = torch.zeros_like(
            self.allegro.data.joint_pos
        )
        self._reset_contact_cache_hand_target = torch.zeros_like(
            self.allegro.data.joint_pos
        )
        self._reset_contact_cache_screw_root = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        self._reset_contact_cache_screw_q = torch.zeros_like(
            self.screwdriver.data.joint_pos
        )
        self._reset_contact_cache_cur_targets = torch.zeros_like(
            self._cur_targets
        )
        self._reset_contact_cache_root_pos_noise = torch.zeros_like(
            self._env_reset_root_pos_noise
        )
        self._reset_contact_cache_root_rpy_noise = torch.zeros_like(
            self._env_reset_root_rpy_noise
        )
        self._reset_contact_cache_screw_tilt_noise = torch.zeros_like(
            self._env_reset_screwdriver_tilt_noise
        )
        self._reset_contact_cache_joint_bias = torch.zeros_like(
            self._env_joint_position_bias
        )

    def _reset_idx(self, env_ids) -> None:
        """Reset Linker-only contact authorization and raw-motion diagnostics."""
        if env_ids is None:
            reset_ids = self.allegro._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            reset_ids = env_ids.to(dtype=torch.long, device=self.device)
        else:
            reset_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        # The base reset performs the normal compliant settle.  Top-down tasks
        # may opt into a hard distance-contact precondition: failed environment
        # rows are resampled, while a bounded exhaustion raises instead of
        # silently starting an invalid episode.
        if hasattr(self, "_reset_contact_guard_retries"):
            self._enforce_initial_contact_guard(reset_ids)

        # DirectRLEnv may dispatch here while ``super().__init__`` is still
        # constructing the subclass, before these Linker-only buffers exist.
        if hasattr(self, "_drive_contact_streak"):
            self._drive_contact_streak[reset_ids] = 0.0
            self._raw_total_turn[reset_ids] = 0.0
            self._raw_net_turn[reset_ids] = 0.0
            self._steps_since_reset[reset_ids] = 0
            self._reset_action_scale[reset_ids] = 0.0

    def _enforce_initial_contact_guard(self, reset_ids: torch.Tensor) -> None:
        """Guarantee the configured distance-contact count before policy step 0."""
        min_fingers = int(self.cfg.reset_contact_guard_min_fingers)
        max_resamples = int(self.cfg.reset_contact_guard_max_resamples)
        if min_fingers <= 0 or reset_ids.numel() == 0:
            return
        if min_fingers > len(self.fingers):
            raise ValueError(
                "reset_contact_guard_min_fingers exceeds resolved fingertips"
            )
        if max_resamples < 0:
            raise ValueError("reset_contact_guard_max_resamples must be non-negative")
        if self.cfg.reset_contact_steps <= 0:
            raise ValueError("reset contact guard requires reset_contact_steps > 0")

        self._reset_contact_guard_retries[reset_ids] = 0
        self._reset_contact_guard_donor_repaired[reset_ids] = False
        donor_attempted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for resample_round in range(max_resamples + 1):
            _, _, present = self._compute_distance_contact()
            contact_count = present[reset_ids].sum(dim=-1)
            passing = reset_ids[contact_count >= min_fingers]
            self._cache_reset_contact_states(passing)
            failed = reset_ids[contact_count < min_fingers]
            if failed.numel() == 0:
                retries = self._reset_contact_guard_retries[reset_ids].float()
                self.extras["reset_contact_guard_retry_mean"] = retries.mean()
                self.extras["reset_contact_guard_retry_max"] = retries.max()
                self.extras["reset_contact_guard_pass"] = torch.tensor(
                    True, device=self.device
                )
                self.extras["reset_contact_guard_donor_repair_fraction"] = (
                    self._reset_contact_guard_donor_repaired[reset_ids]
                    .float()
                    .mean()
                )
                return
            if resample_round >= max_resamples:
                self.extras["reset_contact_guard_pass"] = torch.tensor(
                    False, device=self.device
                )
                failed_preview = failed[:16].detach().cpu().tolist()
                raise RuntimeError(
                    "reset contact guard exhausted after "
                    f"{max_resamples} resamples; {failed.numel()} envs still "
                    f"have fewer than {min_fingers} contacts; ids={failed_preview}"
                )

            donor_candidates = failed[~donor_attempted[failed]]
            repaired = self._restore_contact_states_from_donors(
                donor_candidates
            )
            if repaired.numel() > 0:
                donor_attempted[repaired] = True
                self._reset_contact_guard_donor_repaired[repaired] = True
                self._reset_contact_guard_retries[repaired] += 1

            unresolved = failed[~torch.isin(failed, repaired)]
            if unresolved.numel() == 0:
                # One forward in the donor restore updated body poses; validate
                # the kinematic predicate at the top of the next loop.
                continue

            self._reset_contact_guard_retries[unresolved] += 1
            # No cached donor exists for this geometry (possible for a tiny
            # asynchronous reset subset).  Fall back to the original physical
            # rejection sample, with complement-state freezing in the base env.
            super()._reset_idx(unresolved)

    def _cache_reset_contact_states(self, env_ids: torch.Tensor) -> None:
        """Store settled initial states that already satisfy the contact guard."""
        if env_ids.numel() == 0:
            return
        variant = getattr(self, "_env_variant_idx", None)
        if variant is None:
            self._reset_contact_cache_variant[env_ids] = 0
        else:
            self._reset_contact_cache_variant[env_ids] = variant[env_ids]
        self._reset_contact_cache_hand_root[env_ids] = (
            self.allegro.data.root_state_w[env_ids, :7]
        )
        self._reset_contact_cache_hand_q[env_ids] = self.allegro.data.joint_pos[
            env_ids
        ]
        self._reset_contact_cache_hand_target[env_ids] = (
            self.allegro.data.joint_pos_target[env_ids]
        )
        self._reset_contact_cache_screw_root[env_ids] = (
            self.screwdriver.data.root_state_w[env_ids, :7]
        )
        self._reset_contact_cache_screw_q[env_ids] = (
            self.screwdriver.data.joint_pos[env_ids]
        )
        self._reset_contact_cache_cur_targets[env_ids] = self._cur_targets[
            env_ids
        ]
        self._reset_contact_cache_root_pos_noise[env_ids] = (
            self._env_reset_root_pos_noise[env_ids]
        )
        self._reset_contact_cache_root_rpy_noise[env_ids] = (
            self._env_reset_root_rpy_noise[env_ids]
        )
        self._reset_contact_cache_screw_tilt_noise[env_ids] = (
            self._env_reset_screwdriver_tilt_noise[env_ids]
        )
        self._reset_contact_cache_joint_bias[env_ids] = (
            self._env_joint_position_bias[env_ids]
        )
        self._reset_contact_cache_valid[env_ids] = True

    def _restore_contact_states_from_donors(
        self, failed: torch.Tensor
    ) -> torch.Tensor:
        """Repair failed rows from cached, same-geometry settled reset states."""
        if failed.numel() == 0:
            return failed
        variant = getattr(self, "_env_variant_idx", None)
        current_variant = (
            torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            if variant is None
            else variant
        )
        repaired_chunks: list[torch.Tensor] = []
        donor_chunks: list[torch.Tensor] = []
        for bucket in torch.unique(current_variant[failed]):
            bucket_failed = failed[current_variant[failed] == bucket]
            donor_pool = torch.nonzero(
                self._reset_contact_cache_valid
                & (self._reset_contact_cache_variant == bucket),
                as_tuple=False,
            ).squeeze(-1)
            if donor_pool.numel() == 0:
                continue
            donor_index = torch.arange(
                bucket_failed.numel(), device=self.device
            ) % donor_pool.numel()
            repaired_chunks.append(bucket_failed)
            donor_chunks.append(donor_pool[donor_index])
        if not repaired_chunks:
            return failed[:0]

        repaired = torch.cat(repaired_chunks)
        donors = torch.cat(donor_chunks)
        donor_hand_root = self._reset_contact_cache_hand_root[donors].clone()
        donor_hand_root[:, :3] += (
            self.scene.env_origins[repaired] - self.scene.env_origins[donors]
        )
        donor_screw_root = self._reset_contact_cache_screw_root[donors].clone()
        donor_screw_root[:, :3] += (
            self.scene.env_origins[repaired] - self.scene.env_origins[donors]
        )

        self.allegro.write_root_pose_to_sim(donor_hand_root, env_ids=repaired)
        self.allegro.write_root_velocity_to_sim(
            torch.zeros((repaired.numel(), 6), device=self.device),
            env_ids=repaired,
        )
        hand_q = self._reset_contact_cache_hand_q[donors].clone()
        self.allegro.set_joint_position_target(
            self._reset_contact_cache_hand_target[donors], env_ids=repaired
        )
        self.allegro.write_joint_state_to_sim(
            hand_q, torch.zeros_like(hand_q), env_ids=repaired
        )
        self.screwdriver.write_root_pose_to_sim(
            donor_screw_root, env_ids=repaired
        )
        self.screwdriver.write_root_velocity_to_sim(
            torch.zeros((repaired.numel(), 6), device=self.device),
            env_ids=repaired,
        )
        screw_q = self._reset_contact_cache_screw_q[donors].clone()
        self.screwdriver.write_joint_state_to_sim(
            screw_q, torch.zeros_like(screw_q), env_ids=repaired
        )

        self._cur_targets[repaired] = self._reset_contact_cache_cur_targets[
            donors
        ]
        self._env_reset_root_pos_noise[repaired] = (
            self._reset_contact_cache_root_pos_noise[donors]
        )
        self._env_reset_root_rpy_noise[repaired] = (
            self._reset_contact_cache_root_rpy_noise[donors]
        )
        self._env_reset_screwdriver_tilt_noise[repaired] = (
            self._reset_contact_cache_screw_tilt_noise[donors]
        )
        self._env_joint_position_bias[repaired] = (
            self._reset_contact_cache_joint_bias[donors]
        )
        self._prev_z[repaired] = screw_q[:, self._screwdriver_z_id]
        self._prev_tilt_xy[repaired] = screw_q[
            :, self._screwdriver_euler_ids[:2]
        ]
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        if self._prev_shaft_quat is not None:
            shaft_quat = self._get_shaft_quat()
            if shaft_quat is not None:
                self._prev_shaft_quat[repaired] = shaft_quat[repaired].detach()
        if self.cfg.asymmetric_obs:
            finger_q = hand_q[:, self._finger_joint_ids]
            observed_finger_q = self._observed_finger_q(finger_q, repaired)
            frame = self._proprio_codec.encode_frame(
                observed_finger_q, self._cur_targets[repaired]
            )
            self._prop_hist_buf[repaired] = frame.unsqueeze(1).expand(
                -1, self.cfg.prop_hist_len, -1
            )
        return repaired

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Preserve the validated reset grasp before blending in policy actions."""
        hold = max(int(self.cfg.reset_action_hold_steps), 0)
        ramp = max(int(self.cfg.reset_action_ramp_steps), 0)
        age = self._steps_since_reset
        if ramp > 0:
            scale = ((age - hold + 1).float() / float(ramp)).clamp(0.0, 1.0)
        else:
            scale = (age >= hold).to(dtype=actions.dtype)
        self._reset_action_scale = scale
        phase = getattr(self, "_curriculum_phase", None)
        phase_action_scale = float(
            getattr(phase, "action_scale_multiplier", 1.0)
        )
        super()._pre_physics_step(
            actions * scale.unsqueeze(-1) * phase_action_scale
        )
        if self.cfg.absolute_action_targets:
            target = self._home_targets + self._joint_range * self.actions
            self._cur_targets = torch.clamp(
                target, self._finger_lower, self._finger_upper
            )
        self._steps_since_reset += 1

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
        # Clone envs + collision filtering (replicate-physics vs per-env-geometry
        # paths differ; see base._finalize_scene).
        self._finalize_scene()
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

    def _compute_distance_contact(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return surface clearance, soft contact score, and binary contact."""
        clearance = self.compute_surface_clearance()
        if clearance.shape[1] == 0:
            return clearance, clearance, clearance.to(dtype=torch.bool)
        margin = self._contact_d_margin
        if margin.shape[1] != clearance.shape[1]:
            raise RuntimeError(
                "contact distance margin width does not match resolved fingertips"
            )
        far = margin + self._contact_d_ramp_width
        score = rewards.distance_window(clearance, margin, far)
        present = rewards.contact_present_dist(clearance, margin)
        return clearance, score, present

    def _compute_fingertip_tangential_speed(self) -> torch.Tensor:
        """Per-finger fingertip speed tangential to the handle axis, ``(N, n_fingers)``."""
        tip_pos = self.allegro.data.body_state_w[:, self._fingertip_body_ids, :3]
        tip_vel = self.allegro.data.body_state_w[:, self._fingertip_body_ids, 7:10]
        base = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_base_idx], :3]
        top = self.screwdriver.data.body_state_w[:, self._handle_body_ids[self._handle_cap_idx], :3]
        return rewards.tangential_speed(tip_pos, tip_vel, base, top)

    def _compute_surface_co_motion(self) -> torch.Tensor:
        """Per-finger co-motion score with the handle surface, ``(N, n_fingers)``.

        Measured from the handle body's world twist, so precession/tilt do not
        alias into the score; see ``rewards.surface_co_motion``.
        """
        tip = self.allegro.data.body_state_w[:, self._fingertip_body_ids]
        handle = self.screwdriver.data.body_state_w[
            :, self._handle_body_ids[self._handle_base_idx]
        ]
        return rewards.surface_co_motion(
            tip[..., :3],
            tip[..., 7:10],
            handle[..., :3],
            handle[..., 7:10],
            handle[..., 10:13],
            self.cfg.turn_motion_surface_speed_floor,
        )

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
    # Rewards (distance-contact; prescribed-lite finger roles; stay-home)
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

        # ---- Kinematic fingertip contact; force remains diagnostic only ----
        clearance, distance_score, distance_present = (
            self._compute_distance_contact()
        )
        diagnostic_tip_force, _, diagnostic_cap_force, wrong_force = (
            self._read_contact_forces()
        )
        index_contact_binary = distance_present[:, self._index_tip_idx]
        if cfg.role_neutral_fingertip_contact:
            # Palm-down top-down grasp: any three fingertips may authorize
            # rotation, so
            # fingers can release and re-contact without a five-finger hard gate.
            drive_score = distance_score
            drive_present = distance_present
            required_contacts = float(cfg.role_neutral_min_contact_fingers)
            drive_count = drive_present.float().sum(dim=-1)
            valid_contact = drive_count >= required_contacts
        else:
            # Historical lateral grasp keeps its drive/index roles, but both are
            # judged by geometry rather than body/cap force labels.
            drive_score = distance_score.index_select(1, self._drive_tip_idxs_t)
            drive_present = distance_present.index_select(
                1, self._drive_tip_idxs_t
            )
            required_contacts = float(phase.min_drive_fingers)
            drive_count = drive_present.float().sum(dim=-1)
            valid_contact = drive_count >= required_contacts
            if cfg.turn_require_index_cap:
                valid_contact = valid_contact & index_contact_binary
        sustained_gate, self._drive_contact_streak = rewards.sustained_binary_gate(
            valid_contact,
            self._drive_contact_streak,
            cfg.turn_contact_hold_steps,
        )
        contact_quality = rewards.soft_count_gate(
            drive_score, required_contacts
        )
        turn_gate = sustained_gate * contact_quality
        combined_gate = turn_gate * upright_gate

        # ---- Co-motion authorization (anti-coasting / anti-creep) ----
        # Distance contact is satisfied by a motionless resting grasp, so it
        # cannot tell an active drive from solver creep.  Progress only pays
        # while >= turn_motion_min_fingers contacting tips move with the
        # handle surface.
        if getattr(cfg, "turn_motion_authorized", False):
            co_motion = self._compute_surface_co_motion()
            motion_auth = rewards.soft_count_gate(
                co_motion * distance_present.to(co_motion.dtype),
                cfg.turn_motion_min_fingers,
            )
        else:
            co_motion = None
            motion_auth = torch.ones_like(upright_gate)

        # ---- Positive terms ----
        turn_reward_gate = (
            upright_gate
            if getattr(cfg, "reward_physical_progress_outside_contact", False)
            else combined_gate
        ) * motion_auth
        speed_power = float(cfg.turn_reward_power)
        fwd_signal = fwd_vel.pow(speed_power)
        rev_signal = rev_vel.pow(speed_power)
        turn_reward = phase.reward_turn_weight * fwd_signal * turn_reward_gate
        contact_authority_reward = phase.w_contact_authority * combined_gate
        if cfg.role_neutral_fingertip_contact:
            index_cap_reward = torch.zeros_like(turn_reward)
        else:
            index_cap_reward = (
                phase.w_index_cap
                * distance_score[:, self._index_tip_idx]
                * upright_gate
            )

        tang = self._compute_fingertip_tangential_speed()  # (N, nf)
        drive_speed_ref: float | torch.Tensor = max(
            cfg.drive_full_tangential_speed, 1.0e-6
        )
        if (
            getattr(cfg, "scale_drive_speed_with_geometry", False)
            and self._env_geom_scale is not None
        ):
            drive_speed_ref = drive_speed_ref * self._env_geom_scale[:, :1]
        tang_factor = (tang / drive_speed_ref).clamp(0.0, 1.0)
        if cfg.role_neutral_fingertip_contact:
            drive_terms = drive_score * tang_factor
        else:
            drive_terms = drive_score * tang_factor.index_select(
                1, self._drive_tip_idxs_t
            )
        drive_reward = phase.w_drive * drive_terms.mean(dim=-1) * upright_gate

        grip_reward = phase.w_grip * distance_score.mean(dim=-1)

        # ---- Progress tracking + milestone ----
        # Existing public metrics are authorization-qualified (contact AND
        # co-motion) and therefore safe for checkpoint selection/promotion:
        # creep-rotation under motionless fingers accrues nothing.  Raw shaft
        # motion remains visible under explicit ``eval_raw_*`` diagnostics.
        qualified_delta_z = delta_z * sustained_gate * motion_auth
        self._total_turn += torch.clamp(qualified_delta_z, min=0.0).detach()
        self._net_turn += qualified_delta_z.detach()
        self._raw_total_turn += torch.clamp(delta_z, min=0.0).detach()
        self._raw_net_turn += delta_z.detach()
        milestone_reward = self._compute_milestone_reward(
            gate=combined_gate * motion_auth
        )

        # ---- Negative terms ----
        # Forward credit remains contact-authorized, but reverse shaft motion is
        # a real loss even while the three-tip gate is temporarily closed.  Do
        # not let the policy hide back-drive by deliberately dropping contact.
        reverse_gate = (
            upright_gate
            if getattr(cfg, "penalize_reverse_outside_contact", False)
            else combined_gate
        )
        reverse_weight = (
            phase.reward_turn_weight * cfg.reverse_to_turn_ratio
            if getattr(cfg, "match_reverse_weight_to_turn_weight", False)
            else cfg.reward_reverse_weight
        )
        reverse_cost = reverse_weight * rev_signal * reverse_gate
        upright_cost = cfg.reward_upright_weight * torch.sum(tilt_xy ** 2, dim=-1)
        fall_cost = phase.reward_fall_weight * (
            tilt_norm > phase.upright_termination_threshold
        ).to(dtype=tilt_norm.dtype)
        tilt_vel = (tilt_xy - self._prev_tilt_xy) / self._policy_dt
        self._prev_tilt_xy = tilt_xy.detach().clone()
        tilt_vel_cost = cfg.reward_tilt_velocity_weight * torch.linalg.norm(tilt_vel, ord=1, dim=-1)

        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        excess_cost = (
            phase.w_excess
            * cfg.target_penetration_scale
            * rewards.target_penetration(
                self._cur_targets, finger_q, cfg.pen_deadband
            )
        )
        wrong_surface_present = (
            wrong_force > cfg.wrong_surface_force_threshold
        ).to(dtype=wrong_force.dtype)
        wrong_surface_cost = phase.w_wrong * wrong_surface_present
        idle_count = torch.relu(
            torch.as_tensor(required_contacts, device=self.device) - drive_count
        )
        idle_cost = phase.w_idle * idle_count

        home_dev_cost = cfg.w_home_dev * rewards.home_deviation(
            finger_q, self._home_targets, cfg.home_deviation_deadband
        )
        target_span = (self._finger_upper - self._finger_lower).clamp_min(1.0e-6)
        target_center = 0.5 * (self._finger_upper + self._finger_lower)
        target_edge_fraction = (
            2.0 * (self._cur_targets - target_center).abs() / target_span
        )
        target_bound_cost = cfg.w_target_bound * torch.mean(
            torch.relu(target_edge_fraction - 0.8) ** 2, dim=-1
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
            + contact_authority_reward
            + index_cap_reward
            + drive_reward
            + grip_reward
            + milestone_reward
            - reverse_cost
            - upright_cost
            - fall_cost
            - tilt_vel_cost
            - excess_cost
            - wrong_surface_cost
            - idle_cost
            - home_dev_cost
            - target_bound_cost
            - action_cost
            - action_rate_cost
            - finger_vel_cost
        )

        # ---- Logging extras ----
        osc_ratio = (self._total_turn - self._net_turn.clamp(min=0.0)) / (self._total_turn + 1e-6)
        max_joint_dev = (finger_q - self._home_targets).abs().max(dim=-1).values
        self.extras.update({
            # Progress
            "eval_total_turns":      (self._total_turn / (2.0 * math.pi)).detach(),
            "eval_net_turns":        (self._net_turn / (2.0 * math.pi)).detach(),
            "eval_raw_total_turns":  (self._raw_total_turn / (2.0 * math.pi)).detach(),
            "eval_raw_net_turns":    (self._raw_net_turn / (2.0 * math.pi)).detach(),
            "eval_osc_ratio":        osc_ratio.detach(),
            "eval_turn_vel":         turn_vel.detach(),
            "eval_fwd_vel":          fwd_vel.detach(),
            "eval_rev_vel":          rev_vel.detach(),
            # Object / upright
            "eval_tilt_norm":        tilt_norm.detach(),
            "eval_upright_gate":     upright_gate.detach(),
            "eval_upright_cost":     upright_cost.detach(),
            "eval_fall_cost":        fall_cost.detach(),
            "eval_tilt_vel_cost":    tilt_vel_cost.detach(),
            "eval_reset_action_scale": self._reset_action_scale.detach(),
            # Contact (distance-based; force values below are diagnostics only)
            "eval_contact_gate":     turn_gate.detach(),
            "eval_binary_gate":      sustained_gate.detach(),
            "eval_instant_contact_gate": valid_contact.float().detach(),
            "eval_motion_auth":      motion_auth.detach(),
            "eval_co_motion":        (
                co_motion.mean(dim=-1).detach()
                if co_motion is not None
                else torch.ones_like(motion_auth)
            ),
            "eval_contact_streak":   self._drive_contact_streak.detach(),
            "eval_index_cap_binary": index_contact_binary.float().detach(),
            "eval_drive_count":      drive_count.detach(),
            "eval_contact_count":    distance_present.float().sum(dim=-1).detach(),
            "eval_in_window":        distance_score.mean(dim=-1).detach(),
            "eval_distance_score":   distance_score.mean(dim=-1).detach(),
            "eval_surface_clearance": clearance.mean(dim=-1).detach(),
            "eval_contact_force":    diagnostic_tip_force.mean(dim=-1).detach(),
            "eval_contact_force_max": diagnostic_tip_force.max(dim=-1).values.detach(),
            "eval_index_cap_force":  diagnostic_cap_force[
                :, self._index_tip_idx
            ].detach(),
            "eval_idle_count":       idle_count.detach(),
            "eval_wrong_surface_force": wrong_force.detach(),
            "eval_wrong_surface_present": wrong_surface_present.detach(),
            "eval_max_joint_dev":    max_joint_dev.detach(),
            # Reward breakdown
            "eval_turn_reward":      turn_reward.detach(),
            "eval_contact_authority_reward": contact_authority_reward.detach(),
            "eval_index_cap_reward": index_cap_reward.detach(),
            "eval_drive_reward":     drive_reward.detach(),
            "eval_grip_reward":      grip_reward.detach(),
            "eval_milestone":        milestone_reward.detach(),
            "eval_reverse_cost":     reverse_cost.detach(),
            "eval_excess_cost":      excess_cost.detach(),
            "eval_wrong_surface_cost": wrong_surface_cost.detach(),
            "eval_idle_cost":        idle_cost.detach(),
            "eval_home_dev_cost":    home_dev_cost.detach(),
            "eval_target_bound_cost": target_bound_cost.detach(),
            "eval_target_edge_fraction": target_edge_fraction.mean(dim=-1).detach(),
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

        # Stage 2 suppresses the per-step log (the adaptation trainer drives the
        # logger once per iter); see base env _get_rewards.
        if self._log_stage == 1:
            self._logger.log(self._global_steps, self.extras, epoch=self._current_epoch)
        return torch.nan_to_num(reward, nan=-1.0e6)

    # -----------------------------------------------------------------------
    # Privileged observations (force-free per-finger distance contact scores)
    # -----------------------------------------------------------------------

    def _compute_privileged_obs(self) -> torch.Tensor:
        euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_ids]
        angvel = self.screwdriver.data.joint_vel[:, self._screwdriver_euler_ids]
        rel_pos = self.screwdriver.data.root_pos_w - self.allegro.data.root_pos_w
        quat = self.screwdriver.data.root_quat_w
        # Keep load/bearing resistance and real contact friction as independent
        # channels: enabling friction DR must not silently erase the load proxy.
        if self._base_load_torque > 0.0:
            load_proxy = (
                self._env_load_torque / self._base_load_torque
            ).unsqueeze(-1)
        else:
            load_proxy = (
                self._env_rotation_damping / self._base_rotation_damping
            ).unsqueeze(-1)
        contact_friction = (
            self._env_friction / self._base_friction
        ).unsqueeze(-1)
        _, distance_score, _ = self._compute_distance_contact()
        parts = [
            euler, angvel, rel_pos, quat, load_proxy, contact_friction,
            distance_score,
        ]
        # +2 geometry channels (diameter, length scale) when geometry DR is on;
        # privileged_obs_dim is bumped 20→22 to match (see cfg __post_init__).
        if self._env_geom_scale is not None:
            parts.append(self._env_geom_scale)  # (N, 2)
        return torch.cat(parts, dim=-1)
