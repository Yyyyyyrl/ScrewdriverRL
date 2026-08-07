#!/usr/bin/env python3
"""Batch Isaac-physics search around the statically safe Linker L20 top-down pose.

Each cloned environment receives a different small root/joint perturbation.  The
search ranks candidates from real filtered contact sensors after gravity settling;
it does not replace the full-mesh validator, which must be rerun on shortlisted
candidates before any posture is accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import traceback

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/linker_l20_screwdriver_topdown/physics_posture_search.json"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument(
    "--replicas_per_candidate",
    type=int,
    default=1,
    help="Evaluate each unique candidate in this many independent environments.",
)
parser.add_argument(
    "--replica_layout",
    choices=("contiguous", "interleaved"),
    default="contiguous",
    help=(
        "Map replicas to env IDs contiguously (legacy) or interleave candidates "
        "across the full env-ID range to expose GPU solver/index sensitivity."
    ),
)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument(
    "--enable-domain-rand",
    action="store_true",
    help="sample the task dynamics DR independently in every candidate replica",
)
parser.add_argument(
    "--geometry_variant_index",
    type=int,
    choices=(0, 1, 2),
    default=None,
    help="Pin every search environment to one top-down diameter asset.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=0.0,
    help="Place isolated cloned environments at the same world coordinates for origin-stable comparison.",
)
parser.add_argument(
    "--root_span_scale",
    type=float,
    default=1.0,
    help="Multiplier for root xyz perturbations; use 0 to retain a statically validated wrist.",
)
parser.add_argument(
    "--root_yaw_offset",
    type=float,
    default=0.0,
    help="World-Z yaw offset in radians; palm normal remains exactly world -Z.",
)
parser.add_argument(
    "--reset_physics_steps",
    type=int,
    default=32,
    help="Raw no-action physics steps matching cfg.reset_contact_steps before policy stepping.",
)
parser.add_argument(
    "--ramp_reset_target",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Linearly close from the safe reset state to each candidate during raw reset steps.",
)
parser.add_argument(
    "--reset_target_blend",
    type=float,
    default=0.0,
    help="Blend the pinned bucket reset toward each PD target before ramping (0..1).",
)
parser.add_argument(
    "--release_settle_steps",
    type=int,
    default=0,
    help="Raw unpinned physics steps after fixture release and before episode stepping.",
)
parser.add_argument(
    "--reset_screwdriver_tilt_x", type=float, default=0.0,
    help="Fixture/reset screwdriver X-tilt joint in radians.",
)
parser.add_argument(
    "--reset_screwdriver_tilt_y", type=float, default=0.0,
    help="Fixture/reset screwdriver Y-tilt joint in radians.",
)
parser.add_argument("--settle_steps", type=int, default=96)
parser.add_argument("--measure_steps", type=int, default=48)
parser.add_argument(
    "--max_zero_action_drift_rad_s",
    type=float,
    default=0.005,
    help="Maximum absolute raw shaft drift allowed by the physics posture gate.",
)
parser.add_argument(
    "--drift_cost_weight",
    type=float,
    default=20.0,
    help="Ranking weight for squared zero-action shaft drift relative to the gate.",
)
parser.add_argument("--top_k", type=int, default=64)
parser.add_argument(
    "--focus_finger",
    choices=("index", "middle", "ring", "pinky", "thumb"),
    default=None,
    help="Vary only one finger joints (root stays fixed) for a wide local search.",
)
parser.add_argument(
    "--joint_span_scale",
    type=float,
    default=1.0,
    help="Multiplier for joint perturbation spans; guarded URDF limits still apply.",
)
parser.add_argument(
    "--focus_root_span_scale",
    type=float,
    default=0.0,
    help="When focusing one finger, also vary root xyz by this span multiplier.",
)
parser.add_argument("--initial_search", type=Path, default=None)
parser.add_argument(
    "--reset_search",
    type=Path,
    default=None,
    help="Optional collision-safe state at the same root when candidates are PD targets.",
)
parser.add_argument(
    "--allow_reset_root_retarget",
    action="store_true",
    help=(
        "Reuse reset-search joint angles at each candidate root.  This is only "
        "allowed within --max_reset_root_retarget_m and is intended for bounded "
        "palmward root searches."
    ),
)
parser.add_argument(
    "--max_reset_root_retarget_m",
    type=float,
    default=0.010,
    help="Maximum reset-search to candidate-root translation when retargeting.",
)
parser.add_argument(
    "--allow_reset_root_yaw_retarget",
    action="store_true",
    help=(
        "Allow a pure world-Z yaw difference between reset and target roots. "
        "The angle remains bounded by --max_reset_root_yaw_retarget_rad."
    ),
)
parser.add_argument(
    "--max_reset_root_yaw_retarget_rad",
    type=float,
    default=0.020,
    help="Maximum reset-search to candidate-root world-Z yaw when retargeting.",
)
parser.add_argument(
    "--reset_candidate_index", type=int, default=None,
    help="Candidate index to select when --reset_search is a full search artifact.",
)
parser.add_argument(
    "--reset_use_settled_candidate", action="store_true",
    help="Use selected --reset_search record settled joints as the reset state.",
)
parser.add_argument(
    "--reset_use_initial_settled", action="store_true",
    help="Use the selected initial record settled joints as the reset state while retaining its PD target.",
)
parser.add_argument("--initial_candidate_index", type=int, default=None)
parser.add_argument(
    "--use_settled_candidate",
    action="store_true",
    help="Use the selected record's settled joints instead of its PD target.",
)
parser.add_argument("--candidate_bank", type=Path, default=None)
parser.add_argument(
    "--repeat_initial",
    action="store_true",
    help="Replicate the selected initial candidate across every env to measure solver reproducibility.",
)
parser.add_argument(
    "--candidate_as_target",
    action="store_true",
    help="Keep the configured collision-safe reset state and apply candidates only as PD targets.",
)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument(
    "--audit_ring_middle_contacts",
    action="store_true",
    help="Record filtered ring_middle forces against hand links and screwdriver bodies.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import screwdriver_rl.tasks  # noqa: E402,F401
from screwdriver_rl.utils.linker_topdown_contact_gate import (  # noqa: E402
    CRITICAL_ROLE_NAMES,
    FUNCTIONAL_ACTIVE_ROLE_FRACTION,
    FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT,
    FUNCTIONAL_MIN_CRITICAL_FRACTION,
    functional_physics_gate as evaluate_functional_physics_gate,
)
from screwdriver_rl.utils.linker_topdown_diameter_postures import (  # noqa: E402
    TOPDOWN_HANDLE_RADII_M,
    TOPDOWN_PREGRASP_POSITIONS_BUCKETS,
    TOPDOWN_RESET_POSITIONS_BUCKETS,
    TOPDOWN_ROOT_POS_OFFSETS_BUCKETS,
    TOPDOWN_ROOT_POS_W,
    TOPDOWN_ROOT_QUAT_WXYZ,
)
from screwdriver_rl.utils.linker_topdown_geometry import UrdfGeometry  # noqa: E402

try:  # noqa: E402
    from isaaclab_tasks.utils import parse_env_cfg
except ImportError:  # pragma: no cover - compatibility with older Isaac Lab
    try:
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    except ImportError:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


FINGERS = ("index", "middle", "ring", "pinky", "thumb")
JOINT_NAMES = (
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
    "thumb_cmc_yaw",
    "thumb_cmc_roll",
    "thumb_cmc_pitch",
    "thumb_mcp",
)

# Root xyz in metres followed by the 16 independent joints in radians.  The
# statically validated pose has >=0.11 rad joint/mimic margin; these local spans
# intentionally stay far below the task's +/-0.35 rad policy working window.
SEARCH_HALF_SPAN = (
    0.0025,
    0.0025,
    0.0020,
    0.020,
    0.050,
    0.080,
    0.025,
    0.080,
    0.100,
    0.025,
    0.080,
    0.100,
    0.020,
    0.080,
    0.100,
    0.060,
    0.060,
    0.060,
    0.060,
)


def _search_root_quat_wxyz() -> tuple[float, float, float, float]:
    """Apply a world-Z yaw while preserving the strict world -Z palm normal."""

    half = 0.5 * args.root_yaw_offset
    c, s = math.cos(half), math.sin(half)
    bw, bx, by, bz = _initial_posture().get("root_quat_wxyz", TOPDOWN_ROOT_QUAT_WXYZ)
    return (
        c * bw - s * bz,
        c * bx - s * by,
        c * by + s * bx,
        c * bz + s * bw,
    )


def _initial_posture() -> dict:
    if args.initial_search is None:
        bucket = 1 if args.geometry_variant_index is None else args.geometry_variant_index
        positions = TOPDOWN_PREGRASP_POSITIONS_BUCKETS[bucket]
        values = [value for finger in FINGERS for value in positions[finger]]
        root = [
            float(base + offset)
            for base, offset in zip(
                TOPDOWN_ROOT_POS_W,
                TOPDOWN_ROOT_POS_OFFSETS_BUCKETS[bucket],
                strict=True,
            )
        ]
        return {
            "root_pos_w": root,
            "joint_positions_independent": dict(zip(JOINT_NAMES, values)),
            "label": f"diameter_bucket_{bucket}_seed",
        }
    data = json.loads(args.initial_search.read_text())
    if "top_candidates" in data:
        if args.initial_candidate_index is None:
            candidate = data["top_candidates"][0]
        else:
            candidate = next(
                row for row in data["top_candidates"]
                if row["candidate_index"] == args.initial_candidate_index
            )
    else:
        candidate = data
    if args.use_settled_candidate:
        candidate = dict(candidate)
        candidate["joint_positions_independent"] = dict(
            candidate["settled_joint_positions_independent"]
        )
        candidate["label"] = f"settled:{candidate.get('candidate_index', 'unknown')}"
    missing = set(JOINT_NAMES).difference(candidate["joint_positions_independent"])
    if missing:
        raise ValueError(f"initial posture missing joints: {sorted(missing)}")
    return candidate


def _guarded_joint_bounds(margin: float = 0.105) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent bounds narrowed so every mimic follower keeps ``margin``."""

    model = UrdfGeometry(REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf")
    limits = model.joint_limits(margin)
    lo = torch.tensor([limits[name][0] for name in JOINT_NAMES], dtype=torch.float64)
    hi = torch.tensor([limits[name][1] for name in JOINT_NAMES], dtype=torch.float64)
    index = {name: i for i, name in enumerate(JOINT_NAMES)}
    for follower in model.joints.values():
        if follower.mimic is None or follower.mimic.source not in index:
            continue
        if follower.lower is None or follower.upper is None:
            continue
        mult = follower.mimic.multiplier
        offset = follower.mimic.offset
        guarded = (
            (follower.lower + margin - offset) / mult,
            (follower.upper - margin - offset) / mult,
        )
        i = index[follower.mimic.source]
        lo[i] = max(float(lo[i]), min(guarded))
        hi[i] = min(float(hi[i]), max(guarded))
    return lo, hi


def _make_candidates(
    count: int,
    seed: int,
    focus_finger: str | None = None,
    joint_span_scale: float = 1.0,
    focus_root_span_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return root positions, independent joints and provenance labels on CPU."""

    if count < 1:
        raise ValueError("--num_envs must be positive")
    if args.candidate_bank is not None:
        bank = json.loads(args.candidate_bank.read_text())
        rows = bank.get("candidates", [])
        if len(rows) != count:
            raise ValueError(
                f"candidate bank has {len(rows)} rows but --num_envs is {count}"
            )
        root = torch.tensor([row["root_pos_w"] for row in rows], dtype=torch.float64)
        q = torch.tensor(
            [
                [row["joint_positions_independent"][name] for name in JOINT_NAMES]
                for row in rows
            ],
            dtype=torch.float64,
        )
        lo, hi = _guarded_joint_bounds()
        q = torch.maximum(torch.minimum(q, hi), lo)
        return root, q, [row.get("label", "candidate_bank") for row in rows]
    dimension = 3 + len(JOINT_NAMES)
    initial = _initial_posture()
    base_root = torch.tensor(initial["root_pos_w"], dtype=torch.float64)
    base_q = torch.tensor(
        [initial["joint_positions_independent"][name] for name in JOINT_NAMES],
        dtype=torch.float64,
    )
    span = torch.tensor(SEARCH_HALF_SPAN, dtype=torch.float64)
    # focus_root_span_scale is an independent opt-in root search.  Do not
    # compound it with root_span_scale: focused searches commonly set the
    # global root span to zero to freeze it unless this explicit opt-in is used.
    root_span_scale = (
        focus_root_span_scale if focus_finger is not None else args.root_span_scale
    )
    span[:3] *= root_span_scale
    span[3:] *= joint_span_scale
    root = base_root.expand(count, -1).clone()
    q = base_q.expand(count, -1).clone()
    labels = ["sobol_local"] * count
    if args.repeat_initial:
        return root, q, ["initial_repeat"] * count

    # Random candidates are biased 1 mm downward because the first real-physics
    # gate showed lost drive-finger preload, while still covering [-3,+1] mm.
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    delta = (2.0 * engine.draw(count).to(torch.float64) - 1.0) * span
    if focus_finger is None:
        if args.root_span_scale > 0.0:
            delta[:, 2] -= 0.001
        probe_dimensions = tuple(range(dimension))
    else:
        finger_start = {
            "index": 0,
            "middle": 3,
            "ring": 6,
            "pinky": 9,
            "thumb": 12,
        }[focus_finger]
        finger_width = 4 if focus_finger == "thumb" else 3
        finger_dimensions = tuple(range(3 + finger_start, 3 + finger_start + finger_width))
        keep = ((0, 1, 2) + finger_dimensions) if focus_root_span_scale > 0.0 else finger_dimensions
        mask = torch.zeros(dimension, dtype=torch.bool)
        mask[list(keep)] = True
        delta[:, ~mask] = 0.0
        probe_dimensions = keep
    root += delta[:, :3]
    q += delta[:, 3:]

    # Candidate zero is the exact statically validated pose.  Then reserve pairs
    # for +/- one-dimensional probes so the result also exposes local sensitivity.
    root[0] = base_root
    q[0] = base_q
    labels[0] = "initial_seed:{}".format(initial.get("candidate_index", "static"))
    cursor = 1
    for dimension_index in probe_dimensions:
        for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
            if cursor >= count:
                break
            root[cursor] = base_root
            q[cursor] = base_q
            if dimension_index < 3:
                root[cursor, dimension_index] += sign * span[dimension_index]
                name = ("root_x", "root_y", "root_z")[dimension_index]
            else:
                q_index = dimension_index - 3
                q[cursor, q_index] += sign * span[dimension_index]
                name = JOINT_NAMES[q_index]
            labels[cursor] = f"probe_{name}_{suffix}"
            cursor += 1

    lo, hi = _guarded_joint_bounds()
    q = torch.maximum(torch.minimum(q, hi), lo)
    return root, q, labels


def _write_candidate_state(
    base, root_cpu: torch.Tensor, q_cpu: torch.Tensor
) -> tuple[float | None, float | None]:
    device = base.device
    env_ids = torch.arange(base.num_envs, dtype=torch.long, device=device)
    root_values = root_cpu.to(device=device, dtype=torch.float32)
    q = q_cpu.to(device=device, dtype=torch.float32)
    reset_q = None
    if args.candidate_as_target:
        base_reset_q = None
        if args.reset_use_initial_settled:
            initial = _initial_posture()
            settled = initial.get("settled_joint_positions_independent")
            if settled is None:
                raise ValueError("selected initial record has no settled joint state")
            base_reset_q = torch.tensor(
                [settled[name] for name in JOINT_NAMES],
                device=device, dtype=torch.float32,
            )
            raw_lo, raw_hi = _guarded_joint_bounds(margin=0.0)
            base_reset_q = torch.maximum(
                torch.minimum(base_reset_q, raw_hi.to(device=device, dtype=base_reset_q.dtype)),
                raw_lo.to(device=device, dtype=base_reset_q.dtype)
            )
        elif args.reset_search is not None:
            reset_data = json.loads(args.reset_search.read_text())
            if "top_candidates" in reset_data:
                if args.reset_candidate_index is None:
                    reset_data = reset_data["top_candidates"][0]
                else:
                    reset_data = next(
                        row for row in reset_data["top_candidates"]
                        if row["candidate_index"] == args.reset_candidate_index
                    )
            reset_root_cpu = torch.tensor(reset_data["root_pos_w"], dtype=torch.float64)
            reset_root_delta_m = torch.linalg.vector_norm(
                root_cpu - reset_root_cpu.expand_as(root_cpu), dim=1
            )
            max_reset_root_delta_m = float(reset_root_delta_m.max())
            if max_reset_root_delta_m > 1.0e-9:
                if not args.allow_reset_root_retarget:
                    raise ValueError(
                        "reset and target roots must match exactly unless "
                        "--allow_reset_root_retarget is set"
                    )
                if max_reset_root_delta_m > args.max_reset_root_retarget_m:
                    raise ValueError(
                        "reset root retarget exceeds "
                        f"--max_reset_root_retarget_m: "
                        f"{max_reset_root_delta_m:.9f} > "
                        f"{args.max_reset_root_retarget_m:.9f} m"
                    )
            reset_quat = torch.tensor(
                reset_data["root_quat_wxyz"], dtype=torch.float64
            )
            target_quat = torch.tensor(_search_root_quat_wxyz(), dtype=torch.float64)
            reset_quat = reset_quat / torch.linalg.vector_norm(reset_quat)
            target_quat = target_quat / torch.linalg.vector_norm(target_quat)
            max_reset_root_yaw_delta_rad = 0.0
            if not torch.allclose(reset_quat, target_quat, atol=1.0e-9, rtol=0.0):
                rw, rx, ry, rz = (float(value) for value in reset_quat)
                tw, tx, ty, tz = (float(value) for value in target_quat)
                relative = [
                    tw * rw + tx * rx + ty * ry + tz * rz,
                    -tw * rx + tx * rw - ty * rz + tz * ry,
                    -tw * ry + tx * rz + ty * rw - tz * rx,
                    -tw * rz - tx * ry + ty * rx + tz * rw,
                ]
                if relative[0] < 0.0:
                    relative = [-value for value in relative]
                if abs(relative[1]) > 1.0e-8 or abs(relative[2]) > 1.0e-8:
                    raise ValueError(
                        "reset root rotation retarget must be a pure world-Z yaw"
                    )
                max_reset_root_yaw_delta_rad = 2.0 * math.atan2(
                    abs(relative[3]), max(relative[0], 0.0)
                )
                if not args.allow_reset_root_yaw_retarget:
                    raise ValueError(
                        "reset and target root quaternions must match exactly unless "
                        "--allow_reset_root_yaw_retarget is set"
                    )
                if (
                    max_reset_root_yaw_delta_rad
                    > args.max_reset_root_yaw_retarget_rad
                ):
                    raise ValueError(
                        "reset root yaw retarget exceeds "
                        f"--max_reset_root_yaw_retarget_rad: "
                        f"{max_reset_root_yaw_delta_rad:.9f} > "
                        f"{args.max_reset_root_yaw_retarget_rad:.9f} rad"
                    )
            reset_joint_key = (
                "settled_joint_positions_independent"
                if args.reset_use_settled_candidate
                else "joint_positions_independent"
            )
            base_reset_q = torch.tensor(
                [reset_data[reset_joint_key][name] for name in JOINT_NAMES],
                device=device, dtype=torch.float32,
            )
            raw_lo, raw_hi = _guarded_joint_bounds(margin=0.0)
            base_reset_q = torch.maximum(
                torch.minimum(
                    base_reset_q,
                    raw_hi.to(device=device, dtype=base_reset_q.dtype),
                ),
                raw_lo.to(device=device, dtype=base_reset_q.dtype),
            )
        elif args.geometry_variant_index is not None:
            positions = TOPDOWN_RESET_POSITIONS_BUCKETS[args.geometry_variant_index]
            base_reset_q = torch.tensor(
                [value for finger in FINGERS for value in positions[finger]],
                device=device,
                dtype=torch.float32,
            )
        if base_reset_q is not None:
            reset_q = torch.lerp(
                base_reset_q.expand(base.num_envs, -1),
                q,
                float(args.reset_target_blend),
            )
    elif args.reset_search is not None:
        raise ValueError("--reset_search requires --candidate_as_target")

    hand_root = base.allegro.data.default_root_state.clone()
    hand_root[:, :3] = root_values + base.scene.env_origins
    hand_root[:, 3:7] = torch.tensor(
        _search_root_quat_wxyz(), dtype=torch.float32, device=device
    )
    hand_root[:, 7:] = 0.0
    base.allegro.write_root_pose_to_sim(hand_root[:, :7], env_ids=env_ids)
    base.allegro.write_root_velocity_to_sim(hand_root[:, 7:], env_ids=env_ids)

    # Make repeated searches independent of any contact settling performed by
    # env.reset() before this controlled reset/target ramp.
    screwdriver_root = base.screwdriver.data.default_root_state.clone()
    screwdriver_root[:, :3] += base.scene.env_origins
    base.screwdriver.write_root_pose_to_sim(screwdriver_root[:, :7], env_ids=env_ids)
    base.screwdriver.write_root_velocity_to_sim(screwdriver_root[:, 7:], env_ids=env_ids)
    screwdriver_jpos = torch.zeros_like(base.screwdriver.data.default_joint_pos)
    screwdriver_jpos[:, base._screwdriver_euler_ids[0]] = args.reset_screwdriver_tilt_x
    screwdriver_jpos[:, base._screwdriver_euler_ids[1]] = args.reset_screwdriver_tilt_y
    screwdriver_jvel = torch.zeros_like(base.screwdriver.data.default_joint_vel)
    base.screwdriver.write_joint_state_to_sim(
        screwdriver_jpos, screwdriver_jvel, env_ids=env_ids
    )

    target_jpos = base.allegro.data.default_joint_pos.clone()
    target_jpos[:, base._finger_joint_ids] = q
    if base._coupled_mult is not None:
        masters = q.index_select(1, base._coupled_master_cols_t)
        followers = masters * base._coupled_mult + base._coupled_offset
        target_jpos[:, base._coupled_follower_ids] = followers
    state_jpos = target_jpos
    if reset_q is not None:
        state_jpos = base.allegro.data.default_joint_pos.clone()
        state_jpos[:, base._finger_joint_ids] = reset_q
        if base._coupled_mult is not None:
            reset_masters = reset_q.index_select(1, base._coupled_master_cols_t)
            state_jpos[:, base._coupled_follower_ids] = (
                reset_masters * base._coupled_mult + base._coupled_offset
            )
    initial_target_jpos = (
        state_jpos
        if reset_q is not None and args.ramp_reset_target and args.reset_physics_steps > 0
        else target_jpos
    )
    base.allegro.set_joint_position_target(initial_target_jpos, env_ids=env_ids)
    if reset_q is not None or not args.candidate_as_target:
        jvel = torch.zeros_like(base.allegro.data.default_joint_vel)
        base.allegro.write_joint_state_to_sim(state_jpos, jvel, env_ids=env_ids)
    base._cur_targets = q.clone()
    base._home_targets = q.clone()
    base._default_finger_pos = q.clone()
    # Candidates are URDF-guarded; keep the home-relative clamp from replacing them.
    base._finger_lower = torch.minimum(base._finger_lower, q)
    base._finger_upper = torch.maximum(base._finger_upper, q)
    base._prev_actions.zero_()
    base.episode_length_buf.zero_()

    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(dt=base.physics_dt)
    return (
        max_reset_root_delta_m if args.reset_search is not None else None,
        max_reset_root_yaw_delta_rad if args.reset_search is not None else None,
    )


def _role_forces(base):
    total, body, cap, wrong = base._read_contact_forces()
    if base.cfg.role_neutral_fingertip_contact:
        role = total
    else:
        role = torch.stack(
            (cap[:, 0], body[:, 1], body[:, 2], body[:, 3], body[:, 4]),
            dim=-1,
        )
    tilt = torch.linalg.norm(
        base.screwdriver.data.joint_pos[:, base._screwdriver_euler_ids[:2]], dim=-1
    )
    proximal = torch.zeros((base.num_envs, 0), device=base.device)
    if base._proximal_sensor is not None:
        net = base._proximal_sensor.data.net_forces_w
        if net is not None:
            proximal = torch.linalg.norm(net, dim=-1)
    return role, total, wrong, tilt, proximal


def _candidate_record(
    index: int,
    root: torch.Tensor,
    q: torch.Tensor,
    settled_q: torch.Tensor,
    settled_screwdriver_q: torch.Tensor,
    screwdriver_joint_names: list[str],
    label: str,
    role_mean: torch.Tensor,
    role_max: torch.Tensor,
    total_force_max: torch.Tensor,
    proximal_force_max: torch.Tensor,
    proximal_names: list[str],
    fraction: torch.Tensor,
    wrong_max: torch.Tensor,
    tilt_max: torch.Tensor,
    done: torch.Tensor,
    terminated_any: torch.Tensor,
    truncated_any: torch.Tensor,
    first_done_step: torch.Tensor,
    raw_drift_rate: torch.Tensor,
    qualified_drift_rate: torch.Tensor,
    cost: torch.Tensor,
) -> dict:
    joint_dict = {name: float(q[i]) for i, name in enumerate(JOINT_NAMES)}
    settled_joint_dict = {
        name: float(settled_q[i]) for i, name in enumerate(JOINT_NAMES)
    }
    record = {
        "candidate_index": index,
        "label": label,
        "cost": float(cost[index]),
        "root_pos_w": [float(v) for v in root],
        "root_quat_wxyz": [float(v) for v in _search_root_quat_wxyz()],
        "joint_positions_independent": joint_dict,
        "settled_joint_positions_independent": settled_joint_dict,
        "settled_screwdriver_joint_positions": {
            name: float(settled_screwdriver_q[index, i])
            for i, name in enumerate(screwdriver_joint_names)
        },
        "settled_target_error_max_rad": max(
            abs(settled_joint_dict[name] - joint_dict[name]) for name in JOINT_NAMES
        ),
        "role_force_mean_n": {
            finger: float(role_mean[index, i]) for i, finger in enumerate(FINGERS)
        },
        "role_force_max_n": {
            finger: float(role_max[index, i]) for i, finger in enumerate(FINGERS)
        },
        "fingertip_total_force_max_n": {
            finger: float(total_force_max[index, i]) for i, finger in enumerate(FINGERS)
        },
        "proximal_force_max_n": {
            name: float(proximal_force_max[index, i])
            for i, name in enumerate(proximal_names)
        },
        "role_contact_fraction": {
            finger: float(fraction[index, i]) for i, finger in enumerate(FINGERS)
        },
        "wrong_surface_force_max_n": float(wrong_max[index]),
        "tilt_max_rad": float(tilt_max[index]),
        "terminated_or_truncated": bool(done[index]),
        "terminated": bool(terminated_any[index]),
        "truncated": bool(truncated_any[index]),
        "first_done_step": int(first_done_step[index]),
        "zero_action_raw_shaft_drift_rad_s": float(raw_drift_rate[index]),
        "zero_action_qualified_shaft_drift_rad_s": float(
            qualified_drift_rate[index]
        ),
    }
    strict_gate = bool(
        (fraction[index] >= 0.95).all()
        and (role_mean[index] >= 0.10).all()
        and (role_mean[index] <= 8.0).all()
        and (total_force_max[index] <= 8.0).all()
        and wrong_max[index] <= 0.05
        and tilt_max[index] <= 0.35
        and abs(float(raw_drift_rate[index]))
        <= args.max_zero_action_drift_rad_s
        and not done[index]
    )
    functional_gate = evaluate_functional_physics_gate(
        record,
        max_zero_action_drift_rad_s=args.max_zero_action_drift_rad_s,
    )
    # Preserve the historical field for reproducibility while making the
    # functional release criterion explicit and independently auditable.
    record["physics_contact_gate"] = strict_gate
    record["strict_all_finger_physics_gate"] = strict_gate
    record["functional_physics_gate"] = bool(functional_gate["pass"])
    record["functional_contact_topology"] = functional_gate["topology"]
    return record


def run_search() -> dict:
    if not 0.0 <= args.reset_target_blend <= 1.0:
        raise ValueError("--reset_target_blend must be in [0, 1]")
    if args.max_zero_action_drift_rad_s <= 0.0:
        raise ValueError("--max_zero_action_drift_rad_s must be positive")
    if args.drift_cost_weight < 0.0:
        raise ValueError("--drift_cost_weight must be non-negative")
    if args.max_reset_root_retarget_m <= 0.0:
        raise ValueError("--max_reset_root_retarget_m must be positive")
    if args.max_reset_root_yaw_retarget_rad <= 0.0:
        raise ValueError("--max_reset_root_yaw_retarget_rad must be positive")
    replicas = int(args.replicas_per_candidate)
    if replicas < 1:
        raise ValueError("--replicas_per_candidate must be positive")
    if args.num_envs % replicas != 0:
        raise ValueError("--num_envs must be divisible by --replicas_per_candidate")
    if args.release_settle_steps < 0:
        raise ValueError("--release_settle_steps must be non-negative")
    unique_candidate_count = args.num_envs // replicas
    unique_root, unique_q, unique_labels = _make_candidates(
        unique_candidate_count,
        args.seed,
        focus_finger=args.focus_finger,
        joint_span_scale=args.joint_span_scale,
        focus_root_span_scale=args.focus_root_span_scale,
    )
    if args.replica_layout == "interleaved":
        root_cpu = unique_root.repeat(replicas, 1)
        q_cpu = unique_q.repeat(replicas, 1)
        labels = [
            f"{unique_labels[group]}:group={group}:replica={replica}"
            for replica in range(replicas)
            for group in range(unique_candidate_count)
        ]
    else:
        root_cpu = unique_root.repeat_interleave(replicas, dim=0)
        q_cpu = unique_q.repeat_interleave(replicas, dim=0)
        labels = [
            f"{unique_labels[group]}:group={group}:replica={replica}"
            for group in range(unique_candidate_count)
            for replica in range(replicas)
        ]
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # Seed config defaults from a guarded candidate before gym.make().  This is
    # required when posture search is regenerating a task after tighter URDF
    # limits make the previously configured reset invalid at articulation init.
    seed_joints = {
        name: float(q_cpu[0, index]) for index, name in enumerate(JOINT_NAMES)
    }
    seed_model = UrdfGeometry(
        REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
    )
    env_cfg.robot_cfg.init_state.pos = tuple(float(value) for value in root_cpu[0])
    env_cfg.robot_cfg.init_state.rot = _search_root_quat_wxyz()
    env_cfg.robot_cfg.init_state.joint_pos = seed_model.expanded_positions(seed_joints)
    seed_per_finger = {}
    cursor = 0
    for finger in FINGERS:
        width = 4 if finger == "thumb" else 3
        seed_per_finger[finger] = tuple(
            float(value) for value in q_cpu[0, cursor : cursor + width]
        )
        cursor += width
    env_cfg.pregrasp_positions = dict(seed_per_finger)
    env_cfg.reset_joint_positions = dict(seed_per_finger)
    for attribute in (
        "pregrasp_positions_buckets", "reset_joint_positions_buckets"
    ):
        table = getattr(env_cfg, attribute, None)
        if table is not None:
            setattr(
                env_cfg, attribute,
                [dict(seed_per_finger) for _ in range(len(table))],
            )
    env_cfg.seed = args.seed
    env_cfg.scene.env_spacing = args.env_spacing
    env_cfg.randomize_obj_start = False
    env_cfg.domain_rand.enabled = bool(args.enable_domain_rand)
    env_cfg.reset_contact_steps = 0
    # The reset contact guard resamples environments that fail a distance-contact
    # precondition, which exists so training never starts an invalid episode.  It
    # requires reset_contact_steps > 0 and would resample exactly the postures the
    # search is here to measure, so it is disabled alongside the settle steps this
    # tool replaces with its own loop.
    env_cfg.reset_contact_guard_min_fingers = 0
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 30.0)
    # A large posture sweep creates more broadphase aggregate pairs than the
    # training default.  This is search-tool-local and leaves the task config
    # unchanged.
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = max(
        int(env_cfg.sim.physx.gpu_found_lost_pairs_capacity), 2**25
    )
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = max(
        int(env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity), 2**27
    )
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = max(
        int(env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity), 2**23
    )
    if args.geometry_variant_index is not None:
        spawn = env_cfg.screwdriver_cfg.spawn
        if not hasattr(spawn, "assets_cfg"):
            raise TypeError("pinned diameter search requires a MultiAsset spawn config")
        env_cfg.screwdriver_cfg.spawn = spawn.assets_cfg[args.geometry_variant_index]
        env_cfg.domain_rand.randomize_geometry = False
        env_cfg.screwdriver_handle_radius = TOPDOWN_HANDLE_RADII_M[
            args.geometry_variant_index
        ]

    env = None
    try:
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        base._log_stage = 2
        ring_contact_labels = []
        ring_contact_view = None
        ring_contact_max = None
        if args.audit_ring_middle_contacts:
            if base.num_envs != 1:
                raise ValueError("--audit_ring_middle_contacts requires --num_envs 1")
            robot_glob = base.cfg.robot_cfg.prim_path.replace(".*", "*")
            screwdriver_glob = base.cfg.screwdriver_cfg.prim_path.replace(".*", "*")
            ring_contact_filters = []
            for body_name in base.allegro.body_names:
                if body_name == "ring_middle":
                    continue
                ring_contact_labels.append(f"hand:{body_name}")
                ring_contact_filters.append(f"{robot_glob}/{body_name}")
            for body_name in base.screwdriver.body_names:
                ring_contact_labels.append(f"screwdriver:{body_name}")
                ring_contact_filters.append(f"{screwdriver_glob}/{body_name}")
            ring_contact_view = base._proximal_sensor._physics_sim_view.create_rigid_contact_view(
                f"{robot_glob}/ring_middle",
                filter_patterns=ring_contact_filters,
                max_contact_data_count=256 * len(ring_contact_filters),
            )
            ring_contact_max = torch.zeros(len(ring_contact_labels), device=base.device)
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdPhysics

        collision_offsets = {}
        named_prim_audit = {}
        collision_prim_paths = []
        stage = omni.usd.get_context().get_stage()
        audit_fragments = (
            "ring_middle",
            "ring_distal",
            "screwdriver_body",
            "screwdriver_cap",
        )
        audit_prims = [stage.GetPrimAtPath("/World/envs/env_0")]
        while audit_prims:
            prim = audit_prims.pop(0)
            audit_prims += prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
            path = str(prim.GetPath())
            is_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            if not path.startswith("/World/envs/env_0"):
                continue
            if is_collision:
                collision_prim_paths.append(path)
            if not any(fragment in path for fragment in audit_fragments):
                continue
            named_prim_audit[path] = {
                "prim_type": prim.GetTypeName(),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "is_collision": is_collision,
            }
            if not is_collision:
                continue
            api = PhysxSchema.PhysxCollisionAPI(prim)
            collision_offsets[path] = {
                "prim_type": prim.GetTypeName(),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "physx_collision_api_applied": prim.HasAPI(PhysxSchema.PhysxCollisionAPI),
                "contact_offset_m": api.GetContactOffsetAttr().Get(),
                "rest_offset_m": api.GetRestOffsetAttr().Get(),
            }
        env.reset(seed=args.seed)
        (
            reset_root_retarget_max_m,
            reset_root_yaw_retarget_max_rad,
        ) = _write_candidate_state(base, root_cpu, q_cpu)
        runtime_initial_poses = {}
        hand_body_state = getattr(base.allegro.data, "body_link_state_w", base.allegro.data.body_state_w)
        screwdriver_body_state = getattr(
            base.screwdriver.data, "body_link_state_w", base.screwdriver.data.body_state_w
        )
        for label, asset, state, body_name in (
            ("ring_middle", base.allegro, hand_body_state, "ring_middle"),
            ("screwdriver_body", base.screwdriver, screwdriver_body_state, "screwdriver_body"),
            ("screwdriver_cap", base.screwdriver, screwdriver_body_state, "screwdriver_cap"),
        ):
            body_ids, _ = asset.find_bodies([f"^{body_name}$"], preserve_order=True)
            runtime_initial_poses[label] = [
                float(value) for value in state[0, body_ids[0], :7].detach().cpu()
            ]
        runtime_initial_poses["hand_root"] = [
            float(value) for value in base.allegro.data.root_link_state_w[0, :7].detach().cpu()
        ]
        pin_screwdriver = bool(base.cfg.reset_pin_screwdriver_upright)
        pinned_screwdriver_jpos = base.screwdriver.data.joint_pos.clone()
        pinned_screwdriver_jvel = torch.zeros_like(
            base.screwdriver.data.default_joint_vel
        )
        if args.ramp_reset_target and args.reset_physics_steps > 0:
            reset_jpos = base.allegro.data.joint_pos.clone()
            reset_q = reset_jpos[:, base._finger_joint_ids].clone()
            for step in range(args.reset_physics_steps):
                alpha = float(step + 1) / float(args.reset_physics_steps)
                ramp_q = torch.lerp(reset_q, q_cpu.to(base.device, dtype=torch.float32), alpha)
                ramp_target = reset_jpos.clone()
                ramp_target[:, base._finger_joint_ids] = ramp_q
                if base._coupled_mult is not None:
                    ramp_masters = ramp_q.index_select(1, base._coupled_master_cols_t)
                    ramp_target[:, base._coupled_follower_ids] = (
                        ramp_masters * base._coupled_mult + base._coupled_offset
                    )
                base.allegro.set_joint_position_target(ramp_target)
                if pin_screwdriver:
                    base.screwdriver.write_joint_state_to_sim(
                        pinned_screwdriver_jpos, pinned_screwdriver_jvel
                    )
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(dt=base.physics_dt)
        else:
            for _ in range(args.reset_physics_steps):
                if pin_screwdriver:
                    base.screwdriver.write_joint_state_to_sim(
                        pinned_screwdriver_jpos, pinned_screwdriver_jvel
                    )
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(dt=base.physics_dt)
        if pin_screwdriver and args.reset_physics_steps > 0:
            # Match the environment reset boundary: release the fixture only
            # after clearing ramp-induced hand velocity.
            settled_hand_jpos = base.allegro.data.joint_pos.clone()
            settled_hand_jvel = torch.zeros_like(
                base.allegro.data.default_joint_vel
            )
            base.allegro.write_joint_state_to_sim(
                settled_hand_jpos, settled_hand_jvel
            )
            base.screwdriver.write_joint_state_to_sim(
                pinned_screwdriver_jpos, pinned_screwdriver_jvel
            )
            base.scene.write_data_to_sim()
            base.sim.forward()
            base.scene.update(dt=base.physics_dt)
        for _ in range(args.release_settle_steps):
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(dt=base.physics_dt)
        if args.release_settle_steps > 0:
            # Hand the episode a settled state, not residual release velocity.
            settled_hand_jpos = base.allegro.data.joint_pos.clone()
            settled_hand_jvel = torch.zeros_like(
                base.allegro.data.default_joint_vel
            )
            settled_screwdriver_jpos = base.screwdriver.data.joint_pos.clone()
            settled_screwdriver_jvel = torch.zeros_like(
                base.screwdriver.data.default_joint_vel
            )
            base.allegro.write_joint_state_to_sim(
                settled_hand_jpos, settled_hand_jvel
            )
            base.screwdriver.write_joint_state_to_sim(
                settled_screwdriver_jpos, settled_screwdriver_jvel
            )
            base.scene.write_data_to_sim()
            base.sim.forward()
            base.scene.update(dt=base.physics_dt)

        action_dim = int(base.cfg.action_space.shape[0])
        zero = torch.zeros((base.num_envs, action_dim), dtype=torch.float32, device=base.device)
        role_sum = torch.zeros((base.num_envs, 5), device=base.device)
        role_max = torch.zeros_like(role_sum)
        total_force_max = torch.zeros_like(role_sum)
        proximal_names = (
            list(base._proximal_sensor.body_names)
            if base._proximal_sensor is not None
            else []
        )
        proximal_force_max = torch.zeros(
            (base.num_envs, len(proximal_names)), device=base.device
        )
        contact_count = torch.zeros_like(role_sum)
        wrong_max = torch.zeros(base.num_envs, device=base.device)
        tilt_max = torch.zeros(base.num_envs, device=base.device)
        done_any = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        terminated_any = torch.zeros_like(done_any)
        truncated_any = torch.zeros_like(done_any)
        first_done_step = torch.full(
            (base.num_envs,), -1, dtype=torch.long, device=base.device
        )
        raw_drift_sum = torch.zeros(base.num_envs, device=base.device)
        qualified_drift_sum = torch.zeros_like(raw_drift_sum)

        total_steps = args.settle_steps + args.measure_steps
        for step in range(total_steps):
            _, _, terminated, truncated, _ = env.step(zero)
            terminated_now = terminated.to(dtype=torch.bool)
            truncated_now = truncated.to(dtype=torch.bool)
            done_now = terminated_now | truncated_now
            newly_done = (first_done_step < 0) & done_now
            first_done_step[newly_done] = step
            terminated_any |= terminated_now
            truncated_any |= truncated_now
            done_any |= done_now
            if step < args.settle_steps:
                continue
            if ring_contact_view is not None:
                ring_matrix = ring_contact_view.get_contact_force_matrix(dt=base.physics_dt)
                ring_matrix = ring_matrix.view(-1, ring_contact_view.filter_count, 3)
                ring_force = torch.linalg.vector_norm(ring_matrix, dim=-1).amax(dim=0)
                ring_contact_max = torch.maximum(ring_contact_max, ring_force)
            role, total_force, wrong, tilt, proximal = _role_forces(base)
            role_sum += role
            role_max = torch.maximum(role_max, role)
            total_force_max = torch.maximum(total_force_max, total_force)
            proximal_force_max = torch.maximum(proximal_force_max, proximal)
            contact_count += (role >= 0.10).to(role.dtype)
            wrong_max = torch.maximum(wrong_max, wrong)
            tilt_max = torch.maximum(tilt_max, tilt)
            signed_velocity = (
                base.extras["eval_fwd_vel"] - base.extras["eval_rev_vel"]
            )
            raw_drift_sum += signed_velocity
            qualified_drift_sum += (
                signed_velocity * base.extras["eval_binary_gate"]
            )

        settled_q = base.allegro.data.joint_pos[:, base._finger_joint_ids].detach().clone()
        settled_screwdriver_q = base.screwdriver.data.joint_pos.detach().clone()
        screwdriver_joint_names = list(base.screwdriver.joint_names)
        role_mean = role_sum / float(args.measure_steps)
        fraction = contact_count / float(args.measure_steps)
        raw_drift_rate = raw_drift_sum / float(args.measure_steps)
        qualified_drift_rate = qualified_drift_sum / float(args.measure_steps)
        drift_scale = max(float(args.max_zero_action_drift_rad_s), 1.0e-6)
        drift_cost = args.drift_cost_weight * (
            raw_drift_rate.abs() / drift_scale
        ).square()

        # The task authorizes any three fingertips. Rank the best three contact
        # channels for contact availability, while retaining force ceilings for
        # every fingertip and the unchanged global safety penalties.
        low = torch.relu(0.50 - role_mean) / 0.50
        high = torch.relu(role_mean - 4.0) / 4.0
        crushing = torch.relu(total_force_max - 8.0) / 8.0
        persistence = torch.relu(0.95 - fraction) / 0.95
        contact_availability = 4.0 * low.square() + 8.0 * persistence.square()
        required_contact_cost = torch.topk(
            contact_availability,
            k=FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT,
            dim=-1,
            largest=False,
        ).values.sum(dim=-1)
        cost = (
            required_contact_cost
            + 2.0 * high.square().sum(dim=-1)
            + 2.0 * crushing.square().sum(dim=-1)
            + 4.0 * (wrong_max / 0.05).square()
            + (tilt_max / 0.35).square()
            + drift_cost
            + 1000.0 * done_any.to(torch.float32)
        )
        if args.focus_finger is not None:
            focus_index = FINGERS.index(args.focus_finger)
            cost = (
                4.0 * low[:, focus_index].square()
                + 2.0 * high[:, focus_index].square()
                + 2.0 * crushing[:, focus_index].square()
                + 8.0 * persistence[:, focus_index].square()
                + 4.0 * (wrong_max / 0.05).square()
                + (tilt_max / 0.35).square()
                + drift_cost
                + 1000.0 * done_any.to(torch.float32)
            )

        ranking = torch.argsort(cost).detach().cpu().tolist()
        keep = ranking[: min(args.top_k, len(ranking))]
        records = [
            _candidate_record(
                i,
                root_cpu[i],
                q_cpu[i],
                settled_q[i],
                settled_screwdriver_q,
                screwdriver_joint_names,
                labels[i],
                role_mean,
                role_max,
                total_force_max,
                proximal_force_max,
                proximal_names,
                fraction,
                wrong_max,
                tilt_max,
                done_any,
                terminated_any,
                truncated_any,
                first_done_step,
                raw_drift_rate,
                qualified_drift_rate,
                cost,
            )
            for i in keep
        ]
        for record in records:
            env_index = int(record["candidate_index"])
            if args.replica_layout == "interleaved":
                record["candidate_group_index"] = env_index % unique_candidate_count
                record["replica_index"] = env_index // unique_candidate_count
            else:
                record["candidate_group_index"] = env_index // replicas
                record["replica_index"] = env_index % replicas
        base_record = _candidate_record(
            0,
            root_cpu[0],
            q_cpu[0],
            settled_q[0],
            settled_screwdriver_q,
            screwdriver_joint_names,
            labels[0],
            role_mean,
            role_max,
            total_force_max,
            proximal_force_max,
            proximal_names,
            fraction,
            wrong_max,
            tilt_max,
            done_any,
            terminated_any,
            truncated_any,
            first_done_step,
            raw_drift_rate,
            qualified_drift_rate,
            cost,
        )
        base_record["candidate_group_index"] = 0
        base_record["replica_index"] = 0
        return {
            "task": args.task,
            "device": str(base.device),
            "num_candidates": unique_candidate_count,
            "num_environments": args.num_envs,
            "replicas_per_candidate": replicas,
            "replica_layout": args.replica_layout,
            "seed": args.seed,
            "domain_randomization_enabled": bool(args.enable_domain_rand),
            "geometry_variant_index": args.geometry_variant_index,
            "env_spacing_m": args.env_spacing,
            "root_span_scale": args.root_span_scale,
            "root_yaw_offset_rad": args.root_yaw_offset,
            "reset_physics_steps": args.reset_physics_steps,
            "release_settle_steps": args.release_settle_steps,
            "reset_screwdriver_tilt_xy_rad": [
                args.reset_screwdriver_tilt_x, args.reset_screwdriver_tilt_y
            ],
            "reset_target_blend": args.reset_target_blend,
            "ramp_reset_target": args.ramp_reset_target,
            "focus_finger": args.focus_finger,
            "joint_span_scale": args.joint_span_scale,
            "focus_root_span_scale": args.focus_root_span_scale,
            "initial_search": None if args.initial_search is None else str(args.initial_search),
            "reset_search": None if args.reset_search is None else str(args.reset_search),
            "allow_reset_root_retarget": args.allow_reset_root_retarget,
            "max_reset_root_retarget_m": args.max_reset_root_retarget_m,
            "reset_root_retarget_max_m": reset_root_retarget_max_m,
            "allow_reset_root_yaw_retarget": (
                args.allow_reset_root_yaw_retarget
            ),
            "max_reset_root_yaw_retarget_rad": (
                args.max_reset_root_yaw_retarget_rad
            ),
            "reset_root_yaw_retarget_max_rad": (
                reset_root_yaw_retarget_max_rad
            ),
            "reset_candidate_index": args.reset_candidate_index,
            "reset_use_settled_candidate": args.reset_use_settled_candidate,
            "reset_use_initial_settled": args.reset_use_initial_settled,
            "initial_candidate_index": args.initial_candidate_index,
            "use_settled_candidate": args.use_settled_candidate,
            "candidate_bank": None if args.candidate_bank is None else str(args.candidate_bank),
            "repeat_initial": args.repeat_initial,
            "candidate_as_target": args.candidate_as_target,
            "settle_steps": args.settle_steps,
            "measure_steps": args.measure_steps,
            "max_zero_action_drift_rad_s": args.max_zero_action_drift_rad_s,
            "drift_cost_weight": args.drift_cost_weight,
            "collision_offsets": collision_offsets,
            "runtime_initial_link_poses_xyzw_or_wxyz_per_isaac_tensor": runtime_initial_poses,
            "named_prim_audit": named_prim_audit,
            "collision_prim_paths": collision_prim_paths,
            "ring_middle_contact_audit_filter_count": (
                None if ring_contact_view is None else ring_contact_view.filter_count
            ),
            "ring_middle_contact_audit_max_n": (
                None
                if ring_contact_max is None
                else {
                    label: float(ring_contact_max[i].item())
                    for i, label in enumerate(ring_contact_labels)
                }
            ),
            "force_roles": (
                {finger: "any_screwdriver_part" for finger in FINGERS}
                if base.cfg.role_neutral_fingertip_contact
                else {
                    "index": "screwdriver_cap",
                    "middle": "screwdriver_body",
                    "ring": "screwdriver_body",
                    "pinky": "screwdriver_body",
                    "thumb": "screwdriver_body",
                }
            ),
            "functional_contact_gate_spec": {
                "critical_roles": list(CRITICAL_ROLE_NAMES),
                "minimum_critical_role_fraction": FUNCTIONAL_MIN_CRITICAL_FRACTION,
                "active_role_fraction": FUNCTIONAL_ACTIVE_ROLE_FRACTION,
                "minimum_active_role_count": FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT,
                "mean_role_fraction_is_diagnostic_only": True,
                "safety_limits_unchanged_from_strict_gate": True,
            },
            "release_physics_gate_field": "functional_physics_gate",
            "search_half_span_root_m_then_joints_rad": list(SEARCH_HALF_SPAN),
            "base_candidate": base_record,
            "passing_candidate_count_in_top_k": sum(
                int(record["physics_contact_gate"]) for record in records
            ),
            "functional_passing_candidate_count_in_top_k": sum(
                int(record["functional_physics_gate"]) for record in records
            ),
            "top_candidates": records,
        }
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        result = run_search()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("search complete; ranked candidates:", len(result["top_candidates"]), flush=True)
        print(f"wrote {args.output}", flush=True)
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        watchdog = threading.Timer(60.0, lambda: os._exit(exit_code))
        watchdog.daemon = True
        watchdog.start()
        simulation_app.close()
        os._exit(exit_code)
