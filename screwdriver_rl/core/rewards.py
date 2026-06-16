"""Pure-torch reward and geometry primitives for the screwdriver rotation task.

These functions hold the mathematical core of the reward in one place so it can
be unit-tested on CPU without Isaac Sim.  ``AllegroScrewdriverRotationEnv`` calls
into them; keeping them here (rather than inlined in the env) means the wrapping,
gating, milestone, distance and quaternion logic all have direct tests in
``tests/test_rewards.py``.

Quaternions follow the Isaac Lab convention: ``(w, x, y, z)`` with the scalar
component first.  The quaternion helpers are re-implemented here (instead of
importing ``isaaclab.utils.math``) so this module stays free of the USD/``pxr``
dependency that the rest of Isaac Lab pulls in.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Rotation bookkeeping
# ---------------------------------------------------------------------------


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``(-pi, pi]`` so coordinate resets don't create huge deltas."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def turn_velocities(
    delta: torch.Tensor, dt: float, velocity_clip: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Signed turn velocity plus its clipped forward / reverse components.

    Returns ``(turn_vel, fwd_vel, rev_vel)`` where the forward and reverse
    components are each clamped into ``[0, velocity_clip]`` so a single forceful
    flick cannot produce an outsized reward spike.
    """
    turn_vel = delta / dt
    fwd_vel = torch.clamp(turn_vel, 0.0, velocity_clip)
    rev_vel = torch.clamp(-turn_vel, 0.0, velocity_clip)
    return turn_vel, fwd_vel, rev_vel


def milestone_reward(
    net_turn: torch.Tensor,
    prev_count: torch.Tensor,
    milestone_angle: float,
    bonus: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse bonus for each new milestone of *net forward* progress.

    The bonus is paid once per milestone crossed in the forward direction; backing
    up and re-crossing the same milestone pays nothing (the running ``prev_count``
    only ever increases).  Returns ``(reward, updated_count)``.  The caller is
    responsible for applying any contact/upright gate to ``reward``.
    """
    if milestone_angle <= 0.0 or bonus <= 0.0:
        return torch.zeros_like(net_turn), prev_count
    net_fwd = net_turn.clamp(min=0.0)
    count = torch.floor(net_fwd / milestone_angle)
    new = (count - prev_count).clamp(min=0.0)
    updated_count = torch.maximum(prev_count, count)
    return bonus * new, updated_count


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def upright_gate(tilt_norm: torch.Tensor, gate_std: float) -> torch.Tensor:
    """Multiplicative Gaussian gate ``exp(-(tilt_norm / std)^2)``.

    ``gate_std <= 0`` disables the gate (returns all ones).  At ``tilt_norm == std``
    the gate is ``e^-1 ~ 0.37``; the turn reward is therefore almost fully
    suppressed by moderate tilt, which an additive penalty cannot achieve.
    """
    if gate_std <= 0.0:
        return torch.ones_like(tilt_norm)
    return torch.exp(-((tilt_norm / gate_std) ** 2))


def motion_gate(
    speed: torch.Tensor, min_speed: float, full_speed: float
) -> torch.Tensor:
    """Linear ramp gate on fingertip speed, clamped to ``[0, 1]``.

    Zero below ``min_speed`` (static pressure earns no turn reward) and one at and
    above ``full_speed``.
    """
    full_speed = max(full_speed, min_speed + 1e-6)
    return ((speed - min_speed) / (full_speed - min_speed)).clamp(0.0, 1.0)


def joint_limit_barrier(
    q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Per-row soft barrier cost for joints inside ``margin`` of a limit.

    The cost is normalised by ``margin`` (so it is ``1`` per joint exactly at a
    limit) and summed across joints.  ``margin <= 0`` disables the barrier.
    """
    if margin <= 0.0:
        return torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    lower_violation = (lower + margin - q).clamp(min=0.0)
    upper_violation = (q - (upper - margin)).clamp(min=0.0)
    return ((lower_violation + upper_violation) / margin).sum(dim=-1)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def point_segment_distance(
    points: torch.Tensor, seg_a: torch.Tensor, seg_b: torch.Tensor
) -> torch.Tensor:
    """Distance from each point to the line segment ``seg_a -> seg_b``.

    Args:
        points: ``(N, K, 3)`` query points (e.g. fingertips per env).
        seg_a:  ``(N, 3)`` segment start (handle base origin).
        seg_b:  ``(N, 3)`` segment end (handle cap origin).

    Returns:
        ``(N, K)`` closest-point distances, clamped to the segment endpoints.
    """
    axis = seg_b - seg_a                                              # (N, 3)
    axis_len_sq = (axis ** 2).sum(-1, keepdim=True).clamp(min=1e-9)   # (N, 1)
    rel = points - seg_a.unsqueeze(1)                                 # (N, K, 3)
    t = (rel * axis.unsqueeze(1)).sum(-1, keepdim=True) / axis_len_sq.unsqueeze(1)
    t = t.clamp(0.0, 1.0)
    closest = seg_a.unsqueeze(1) + t * axis.unsqueeze(1)
    return torch.linalg.norm(points - closest, dim=-1)


def tangential_speed(
    points: torch.Tensor,
    velocities: torch.Tensor,
    seg_a: torch.Tensor,
    seg_b: torch.Tensor,
) -> torch.Tensor:
    """Speed of each point *tangential* to the axis ``seg_a -> seg_b``.

    Only motion that circles the axis (genuine rolling of the handle) is
    counted; the radial (approach/retreat) and axial (sliding along the shaft)
    components are rejected.  The tangential direction is ``t = a_hat x r_hat``
    where ``a_hat`` is the unit axis direction and ``r_hat`` is the unit radial
    direction from the axis to the point.

    Args:
        points:     ``(N, K, 3)`` query points (fingertip positions).
        velocities: ``(N, K, 3)`` matching point velocities (world frame).
        seg_a:      ``(N, 3)`` axis start (handle base origin).
        seg_b:      ``(N, 3)`` axis end (handle cap origin).

    Returns:
        ``(N, K)`` non-negative tangential speed magnitudes.  A pure tremor
        (mostly radial / jitter) or an axial slide returns ~0.
    """
    axis = seg_b - seg_a                                              # (N, 3)
    a_hat = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-9)
    a_hat_k = a_hat.unsqueeze(1)                                      # (N, 1, 3)
    rel = points - seg_a.unsqueeze(1)                                 # (N, K, 3)
    axial_len = (rel * a_hat_k).sum(-1, keepdim=True)                 # (N, K, 1)
    radial = rel - axial_len * a_hat_k                                # (N, K, 3)
    r_hat = radial / torch.linalg.norm(radial, dim=-1, keepdim=True).clamp(min=1e-9)
    t_hat = torch.linalg.cross(a_hat_k.expand_as(r_hat), r_hat, dim=-1)
    return (velocities * t_hat).sum(-1).abs()                         # (N, K)


def signed_tangential_speed(
    points: torch.Tensor,
    velocities: torch.Tensor,
    seg_a: torch.Tensor,
    seg_b: torch.Tensor,
    direction: float = 1.0,
) -> torch.Tensor:
    """Signed tangential speed of each point about the axis ``seg_a -> seg_b``.

    Identical geometry to :func:`tangential_speed`, but the result keeps its sign
    and is multiplied by ``direction``: it is **positive** when the point moves in
    the sense that would drive the handle forward (i.e. consistent with a forward
    ``shaft_spin_delta`` for the same ``direction``), negative when it drives the
    handle backward, and ~0 for a static squeeze or a purely radial/axial motion.

    The sign convention: a surface point rotating with positive angular velocity
    ``Ω`` about ``a_hat`` moves at ``Ω * r * t_hat`` with ``t_hat = a_hat x r_hat``.
    The forward turn reward uses ``fwd_vel = clamp(direction * shaft_spin, 0)``, so
    ``direction * (v . t_hat)`` is positive exactly when the fingertip is rolling
    the handle in the rewarded direction.

    Args:
        points:     ``(N, K, 3)`` query points (fingertip positions).
        velocities: ``(N, K, 3)`` matching point velocities (world frame).
        seg_a:      ``(N, 3)`` axis start (handle base origin).
        seg_b:      ``(N, 3)`` axis end (handle cap origin).
        direction:  sign of the desired rotation (matches ``cfg.turn_direction``).

    Returns:
        ``(N, K)`` signed tangential speeds (m/s).
    """
    axis = seg_b - seg_a                                              # (N, 3)
    a_hat = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-9)
    a_hat_k = a_hat.unsqueeze(1)                                      # (N, 1, 3)
    rel = points - seg_a.unsqueeze(1)                                 # (N, K, 3)
    axial_len = (rel * a_hat_k).sum(-1, keepdim=True)                 # (N, K, 1)
    radial = rel - axial_len * a_hat_k                                # (N, K, 3)
    r_hat = radial / torch.linalg.norm(radial, dim=-1, keepdim=True).clamp(min=1e-9)
    t_hat = torch.linalg.cross(a_hat_k.expand_as(r_hat), r_hat, dim=-1)
    return direction * (velocities * t_hat).sum(-1)                   # (N, K)


def rolling_consistency(
    handle_fwd_omega: torch.Tensor,
    finger_fwd_speed: torch.Tensor,
    ref_radius: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Fraction of the handle's forward spin that the fingertips are *driving*.

    Returns a ``[0, 1]`` factor comparing the forward fingertip tangential speed
    ``finger_fwd_speed`` (m/s) to the handle surface speed ``handle_fwd_omega *
    ref_radius`` (m/s).  Because a no-slip fingertip orbits at **>=** the handle
    surface speed, genuine rolling saturates the ratio to 1, while a static
    squeeze (``finger_fwd_speed ~ 0``) drives it to 0 *regardless of how fast a
    damping-driven handle is spinning* — this is what removes the "handle spins
    by itself under a standing torque" reward exploit.

    When the handle is essentially still (``handle_fwd_omega ~ 0``) the surface
    speed floors at ``eps`` and the ratio saturates to 1, but the caller's turn
    reward is ``~ fwd_vel * factor`` which is ~0 there anyway, so no reward leaks.

    Args:
        handle_fwd_omega: ``(N,)`` forward angular speed of the handle (rad/s, >=0).
        finger_fwd_speed: ``(N,)`` forward tangential speed of the in-contact
            fingertips (m/s); clamp negatives to 0 before calling so back-driving
            fingers earn no credit.
        ref_radius:       representative fingertip orbit radius (m).

    Returns:
        ``(N,)`` rolling-consistency factor in ``[0, 1]``.
    """
    handle_surface_speed = (handle_fwd_omega * ref_radius).clamp(min=eps)
    return (finger_fwd_speed / handle_surface_speed).clamp(0.0, 1.0)


def over_force_penalty(force: torch.Tensor, target: float) -> torch.Tensor:
    """Summed per-fingertip contact force exceeding ``target`` (N).

    Returns ``(N,)`` = ``sum_k clamp(force_k - target, min=0)``.  Zero while every
    fingertip grips at or below ``target``; grows linearly with any excess.  Used to
    penalise *crushing* the handle: a hard squeeze (e.g. ~26 N) lets sub-mm finger
    motions transfer large torque — the "spins by itself" / high-force micro-gait
    exploit — and is also unrealistic for real hardware (fingertips grip at ~1-5 N).
    """
    return (force - target).clamp(min=0.0).sum(dim=-1)


def contact_engagement(
    force: torch.Tensor, near_mask: torch.Tensor, target: float
) -> torch.Tensor:
    """Dense per-fingertip contact reward, summed over fingers.

    Each near fingertip contributes ``clamp(force / target, 0, 1)``: it rises with
    contact force up to ``target`` and is then **flat** (no incentive to press
    harder — that is the grip-force penalty's job).  ``near_mask`` (0/1 per finger,
    typically "within the contact distance") localises the reward to genuine handle
    contact so finger-finger forces don't count.

    This bridges the exploration gap between "hover near the handle" (which the
    distance-based near-reward already maxes out) and "turn the handle" (which needs
    a hard simultaneous press+roll): it gives a smooth gradient that rewards simply
    establishing a firm-but-gentle grip, from which the turn reward is reachable.
    """
    engaged = (force / max(target, 1e-6)).clamp(0.0, 1.0)
    return (engaged * near_mask).sum(dim=-1)


def near_contact_score(
    near: torch.Tensor,
    thumb_index: int | None,
    non_thumb_indices: list[int] | None,
    top_k: int,
) -> torch.Tensor:
    """Combine per-fingertip proximity scores with a thumb / non-thumb split.

    ``near`` is the already-decayed proximity (e.g. ``exp(-dist / std)``) of shape
    ``(N, num_fingers)``.  The thumb opposes the other fingers, so it is scored
    separately and averaged 50/50 with the top-``k`` closest non-thumb fingertips
    (top-k prevents every finger clustering on one side of the handle).
    """
    non_thumb_score = None
    if non_thumb_indices:
        nt = near[:, non_thumb_indices]
        k = min(top_k, nt.shape[1])
        non_thumb_score = torch.topk(nt, k=k, dim=-1).values.mean(dim=-1)

    thumb_score = near[:, thumb_index] if thumb_index is not None else None

    if thumb_score is not None and non_thumb_score is not None:
        return 0.5 * (thumb_score + non_thumb_score)
    if thumb_score is not None:
        return thumb_score
    if non_thumb_score is not None:
        return non_thumb_score
    return near.mean(dim=-1)


# ---------------------------------------------------------------------------
# Quaternion helpers (Isaac Lab convention: w, x, y, z)
# ---------------------------------------------------------------------------


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two ``(w, x, y, z)`` quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) ``v`` (``..., 3``) by quaternion(s) ``q`` (``..., 4``)."""
    qw = q[..., 0:1]
    qvec = q[..., 1:]
    t = 2.0 * torch.linalg.cross(qvec, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qvec, t, dim=-1)


def axis_angle_from_quat(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rotation vector (axis * angle) from a ``(w, x, y, z)`` quaternion.

    Resolves the double cover (``-q`` and ``q`` map to the same rotation) by
    flipping to the hemisphere with non-negative ``w`` (the shortest rotation).
    """
    q = torch.where(q[..., 0:1] < 0.0, -q, q)
    xyz = q[..., 1:]
    mag = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    half_angle = torch.atan2(mag.squeeze(-1), q[..., 0])
    angle = (2.0 * half_angle).unsqueeze(-1)
    axis = xyz / mag.clamp(min=eps)
    rotvec = angle * axis
    return torch.where(mag < eps, torch.zeros_like(rotvec), rotvec)


def shaft_spin_delta(
    curr_quat: torch.Tensor, prev_quat: torch.Tensor, direction: float = 1.0
) -> torch.Tensor:
    """Signed rotation about a body's own shaft (local +z) axis between two frames.

    Projects the inter-frame rotation vector onto the *current* shaft axis, so
    precession of a tilted shaft (which advances Euler-z coordinates) does not
    count as genuine spin.  ``direction`` flips the sign of the desired rotation.
    """
    delta_q = quat_mul(curr_quat, quat_conjugate(prev_quat))
    # Resolve quaternion double-cover so small rotations stay small.
    delta_q = torch.where(delta_q[..., 0:1] < 0.0, -delta_q, delta_q)
    rotvec = axis_angle_from_quat(delta_q)

    z_local = torch.zeros_like(rotvec)
    z_local[..., 2] = 1.0
    shaft_axis = quat_apply(curr_quat, z_local)
    spin = (rotvec * shaft_axis).sum(dim=-1)
    return direction * spin
