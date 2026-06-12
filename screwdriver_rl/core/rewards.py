"""Pure-torch reward / measurement primitives for continuous screwdriver turning.

Everything here is a stateless function of tensors so it can be unit-tested
without Isaac Sim (see tests/test_rewards.py). The env composes these into the
full reward; weights live in the env config and are scheduled by the
curriculum. Design rationale for each term is in docs/reward_design.md.

Quaternion convention: wxyz (Isaac Lab native).
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Angle / quaternion helpers
# ---------------------------------------------------------------------------


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]. Robust to revolute-coordinate wrapping."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by quaternions q (both (..., 3/4))."""
    qvec = q[..., 1:]
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qvec, t, dim=-1)


def axis_angle_from_quat(q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert quaternions to axis-angle vectors (angle * axis)."""
    # Resolve double cover so small rotations stay small.
    q = torch.where(q[..., :1] < 0.0, -q, q)
    sin_half = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., :1])
    axis = q[..., 1:] / torch.clamp(sin_half, min=eps)
    return torch.where(sin_half > eps, angle * axis, 2.0 * q[..., 1:])


def shaft_spin_delta(
    shaft_quat: torch.Tensor, prev_shaft_quat: torch.Tensor, direction: float
) -> torch.Tensor:
    """Signed per-step rotation about the screwdriver's own shaft axis.

    The Euler-z gimbal coordinate of the mount also moves under precession of
    a tilted shaft, so rewarding it directly lets wobble-scraping count as
    turning. Instead we take the delta-quaternion between policy steps,
    convert to axis-angle, and project it onto the current shaft axis (body z),
    so only true spin about the shaft is measured (HORA-style).
    """
    delta_quat = quat_mul(shaft_quat, quat_conjugate(prev_shaft_quat))
    rotvec = axis_angle_from_quat(delta_quat)
    shaft_axis = torch.zeros_like(shaft_quat[..., 1:])
    shaft_axis[..., 2] = 1.0
    shaft_axis = quat_apply(shaft_quat, shaft_axis)
    return direction * torch.sum(rotvec * shaft_axis, dim=-1)


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def turn_velocities(
    delta_z: torch.Tensor, dt: float, velocity_clip: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split per-step turn progress into clipped forward/reverse velocities.

    Clipping (rather than an explicit over-speed penalty) caps the reward a
    single flick can earn, so steady turning beats impulsive spinning.
    """
    turn_velocity = delta_z / dt
    forward = torch.clamp(turn_velocity, min=0.0, max=velocity_clip)
    reverse = torch.clamp(-turn_velocity, min=0.0, max=velocity_clip)
    return turn_velocity, forward, reverse


def upright_gate(tilt_norm: torch.Tensor, gate_std: float) -> torch.Tensor:
    """Multiplicative gaussian gate exp(-(tilt/std)^2) on the turn reward.

    Tilting directly forfeits the dominant positive term instead of racing an
    additive penalty (which a strong turn reward always wins — the
    tilt-and-scrape exploit). gate_std <= 0 disables (returns ones).
    """
    if gate_std <= 0.0:
        return torch.ones_like(tilt_norm)
    return torch.exp(-((tilt_norm / gate_std) ** 2))


def milestone_reward(
    net_turn: torch.Tensor,
    prev_milestone_count: torch.Tensor,
    milestone_angle: float,
    milestone_bonus: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse bonus each time *net* accumulated turn crosses a new multiple of
    ``milestone_angle``.

    Counting net (signed) progress — not total forward motion — and never
    decreasing the paid-out count makes forward-backward flicking unable to
    farm the bonus: backing up and re-crossing an already-paid milestone pays
    nothing.

    Returns (reward, updated_milestone_count).
    """
    if milestone_angle <= 0.0 or milestone_bonus <= 0.0:
        zeros = torch.zeros_like(net_turn)
        return zeros, prev_milestone_count
    positive_net = torch.clamp(net_turn, min=0.0)
    count = torch.floor(positive_net / milestone_angle)
    new_milestones = torch.clamp(count - prev_milestone_count, min=0.0)
    updated = torch.maximum(prev_milestone_count, count)
    return milestone_bonus * new_milestones, updated


def near_contact_score(
    near: torch.Tensor,
    thumb_index: int | None,
    non_thumb_indices: list[int],
    top_k: int,
) -> torch.Tensor:
    """Thumb-weighted proximity score in [0, 1].

    ``near`` is exp(-tip_dist/std) per controlled fingertip. The thumb gets
    half the weight because an opposed thumb is necessary for any stable turn;
    the other half is the mean of the best ``top_k`` non-thumb fingertips so a
    far-away spare finger does not dilute the gradient.
    """
    if near.shape[-1] == 0:
        return torch.zeros(near.shape[0], dtype=near.dtype, device=near.device)
    non_thumb_score = None
    if non_thumb_indices:
        non_thumb = near[:, non_thumb_indices]
        k = max(1, min(int(top_k), non_thumb.shape[1]))
        non_thumb_score = torch.topk(non_thumb, k=k, dim=-1).values.mean(dim=-1)
    thumb_score = near[:, thumb_index] if thumb_index is not None else None
    if thumb_score is not None and non_thumb_score is not None:
        return 0.5 * (thumb_score + non_thumb_score)
    if thumb_score is not None:
        return thumb_score
    if non_thumb_score is not None:
        return non_thumb_score
    return near.mean(dim=-1)


def joint_limit_barrier(
    joint_pos: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Linear ReLU barrier that activates within ``margin`` of a soft limit.

    Under delta actions the integrated target can park joints against their
    limits; this keeps fingers off the hard stops (and out of self-collision
    postures) with zero cost in the interior of the range.
    """
    if margin <= 0.0:
        return torch.zeros(joint_pos.shape[0], dtype=joint_pos.dtype, device=joint_pos.device)
    low_violation = torch.clamp((lower + margin) - joint_pos, min=0.0)
    high_violation = torch.clamp(joint_pos - (upper - margin), min=0.0)
    return torch.sum(low_violation + high_violation, dim=-1) / margin


def point_segment_distance(
    points: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor
) -> torch.Tensor:
    """Distance from points (N, K, 3) to segments (N, 3) -> (N, K).

    Used as the fingertip contact proxy: distance to the handle *axis* is
    physically interpretable (handle radius 0.02 m => pad contact at ~0.03 m
    axis distance) regardless of grip height, unlike body-origin distances.
    """
    axis = seg_end - seg_start
    axis_len_sq = torch.sum(axis**2, dim=-1, keepdim=True).clamp_min(1.0e-9)
    rel = points - seg_start.unsqueeze(1)
    t = torch.sum(rel * axis.unsqueeze(1), dim=-1, keepdim=True) / axis_len_sq.unsqueeze(1)
    t = t.clamp(0.0, 1.0)
    closest = seg_start.unsqueeze(1) + t * axis.unsqueeze(1)
    return torch.linalg.norm(points - closest, dim=-1)


def motion_gate(
    contact_speed: torch.Tensor, min_speed: float, full_speed: float
) -> torch.Tensor:
    """Linear ramp on mean in-contact fingertip speed, in [0, 1].

    Together with the contact gate this stops a flicked screwdriver from
    coasting for reward after the fingertips leave or stop moving.
    """
    full = max(full_speed, min_speed + 1.0e-6)
    return torch.clamp((contact_speed - min_speed) / (full - min_speed), 0.0, 1.0)
