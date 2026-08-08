"""Unit tests for the pure-torch reward primitives. No Isaac Sim required.

Run:  python -m pytest tests/ -q   (or python tests/test_rewards.py)
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.core import rewards as R  # noqa: E402


def test_wrap_to_pi_across_boundary():
    # A small forward step across the +pi/-pi seam must stay small.
    z_prev = torch.tensor([math.pi - 0.05])
    z_curr = torch.tensor([-math.pi + 0.05])
    delta = R.wrap_to_pi(z_curr - z_prev)
    assert torch.allclose(delta, torch.tensor([0.1]), atol=1e-5)
    # And backwards.
    delta_back = R.wrap_to_pi(z_prev - z_curr)
    assert torch.allclose(delta_back, torch.tensor([-0.1]), atol=1e-5)


def test_turn_velocities_clipping():
    delta = torch.tensor([0.2, -0.2, 0.0])
    vel, fwd, rev = R.turn_velocities(delta, dt=0.1, velocity_clip=1.0)
    assert torch.allclose(vel, torch.tensor([2.0, -2.0, 0.0]))
    assert torch.allclose(fwd, torch.tensor([1.0, 0.0, 0.0]))  # clipped at 1
    assert torch.allclose(rev, torch.tensor([0.0, 1.0, 0.0]))


def test_milestone_no_double_pay_on_oscillation():
    angle = 0.5 * math.pi
    count = torch.zeros(1)
    # Forward past the first milestone -> pays once.
    net = torch.tensor([angle + 0.01])
    rew1, count = R.milestone_reward(net, count, angle, 1.0)
    assert rew1.item() == 1.0
    # Back up below the milestone, then re-cross: pays nothing.
    net = torch.tensor([angle - 0.3])
    rew2, count = R.milestone_reward(net, count, angle, 1.0)
    net = torch.tensor([angle + 0.01])
    rew3, count = R.milestone_reward(net, count, angle, 1.0)
    assert rew2.item() == 0.0 and rew3.item() == 0.0
    # Continue to the second milestone: pays once more.
    net = torch.tensor([2 * angle + 0.01])
    rew4, count = R.milestone_reward(net, count, angle, 1.0)
    assert rew4.item() == 1.0


def test_milestone_disabled():
    rew, count = R.milestone_reward(torch.tensor([10.0]), torch.zeros(1), 0.0, 1.0)
    assert rew.item() == 0.0


def test_upright_gate():
    tilt = torch.tensor([0.0, 0.15, 10.0])
    gate = R.upright_gate(tilt, gate_std=0.15)
    assert gate[0].item() == 1.0
    assert abs(gate[1].item() - math.exp(-1.0)) < 1e-5
    assert gate[2].item() < 1e-6
    # disabled
    assert torch.all(R.upright_gate(tilt, 0.0) == 1.0)


def test_motion_gate_ramp():
    speed = torch.tensor([0.0, 0.003, 0.009, 0.015, 1.0])
    gate = R.motion_gate(speed, min_speed=0.003, full_speed=0.015)
    assert gate[0].item() == 0.0
    assert gate[1].item() == 0.0
    assert abs(gate[2].item() - 0.5) < 1e-5
    assert gate[3].item() == 1.0
    assert gate[4].item() == 1.0


def test_joint_limit_barrier():
    lower = torch.tensor([[0.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])
    # Mid-range: zero cost.
    q = torch.tensor([[0.5, 0.5]])
    assert R.joint_limit_barrier(q, lower, upper, 0.05).item() == 0.0
    # At the limit: cost = 1 per violating joint (margin-normalized).
    q = torch.tensor([[0.0, 0.5]])
    assert abs(R.joint_limit_barrier(q, lower, upper, 0.05).item() - 1.0) < 1e-5
    # Disabled margin.
    assert R.joint_limit_barrier(q, lower, upper, 0.0).item() == 0.0


def test_point_segment_distance():
    # Vertical segment from (0,0,0) to (0,0,1); point at (0.02, 0, 0.5).
    pts = torch.tensor([[[0.02, 0.0, 0.5], [0.0, 0.0, 2.0]]])
    a = torch.tensor([[0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 1.0]])
    d = R.point_segment_distance(pts, a, b)
    assert abs(d[0, 0].item() - 0.02) < 1e-6
    # Beyond the segment end: distance to the endpoint.
    assert abs(d[0, 1].item() - 1.0) < 1e-6


def test_shaft_spin_delta_pure_z_rotation():
    # Rotation of +0.1 rad about z between steps -> spin = direction * 0.1.
    half = 0.05
    prev = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    curr = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
    spin = R.shaft_spin_delta(curr, prev, direction=1.0)
    assert abs(spin.item() - 0.1) < 1e-5
    spin_neg = R.shaft_spin_delta(curr, prev, direction=-1.0)
    assert abs(spin_neg.item() + 0.1) < 1e-5


def test_shaft_spin_delta_ignores_precession():
    # Tilt the shaft (rotation about x) without spinning about its own axis:
    # the projected spin must be ~0 even though Euler-z style coordinates move.
    half = 0.1
    prev = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    curr = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])  # x tilt
    spin = R.shaft_spin_delta(curr, prev, direction=1.0)
    assert abs(spin.item()) < 1e-5


def test_force_window_trapezoid():
    f = torch.tensor([0.0, 0.1, 0.3, 0.5, 2.0, 4.0, 6.0, 8.0, 10.0])
    w = R.force_window(f, f_min=0.1, f_lo=0.5, f_hi=4.0, f_max=8.0)
    assert w[0].item() == 0.0            # below f_min
    assert w[1].item() == 0.0            # at f_min
    assert abs(w[2].item() - 0.5) < 1e-6  # halfway up the rise
    assert w[3].item() == 1.0            # at f_lo
    assert w[4].item() == 1.0            # inside the flat top
    assert w[5].item() == 1.0            # at f_hi
    assert abs(w[6].item() - 0.5) < 1e-6  # halfway down the fall
    assert w[7].item() == 0.0            # at f_max
    assert w[8].item() == 0.0            # above f_max


def test_distance_window_endpoints_and_ramp():
    distance = torch.tensor([0.025, 0.030, 0.035, 0.040, 0.045])
    score = R.distance_window(distance, d_contact=0.030, d_far=0.040)
    assert torch.allclose(
        score, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]), atol=1.0e-6
    )
    present = R.contact_present_dist(distance, d_contact=0.030)
    assert present.tolist() == [True, True, False, False, False]


def test_distance_window_broadcasts_per_environment_thresholds():
    distance = torch.tensor(
        [
            [0.030, 0.035, 0.040],
            [0.034, 0.040, 0.046],
        ]
    )
    d_contact = torch.tensor([[0.030], [0.034]])
    d_far = torch.tensor([[0.040], [0.046]])
    score = R.distance_window(distance, d_contact=d_contact, d_far=d_far)
    assert torch.allclose(
        score,
        torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]),
        atol=1.0e-6,
    )
    assert R.contact_present_dist(distance, d_contact).tolist() == [
        [True, False, False],
        [True, False, False],
    ]


def test_contact_present_uses_configured_physical_contact_floor():
    force = torch.tensor([0.0, 0.099, 0.1, 0.2, 8.0])
    present = R.contact_present(force, min_force=0.1)
    assert present.tolist() == [False, False, True, True, True]


def test_excess_force():
    f = torch.tensor([0.0, 4.0, 8.0, 10.0])
    e = R.excess_force(f, f_max=8.0)
    assert torch.allclose(e, torch.tensor([0.0, 0.0, 0.0, 2.0]))


def test_soft_count_gate():
    scores = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])
    g = R.soft_count_gate(scores, target=2.0)
    assert abs(g[0].item() - 1.0) < 1e-6   # 2 fingers -> fully open
    assert abs(g[1].item() - 0.25) < 1e-6  # half a finger -> 0.25
    # Saturates at 1 even when more than the target engage.
    g2 = R.soft_count_gate(torch.tensor([[1.0, 1.0, 1.0]]), target=2.0)
    assert g2[0].item() == 1.0


def test_sustained_binary_gate_requires_consecutive_steps():
    streak = torch.zeros(3)

    gate, streak = R.sustained_binary_gate(
        torch.tensor([True, True, False]), streak, hold_steps=3
    )
    assert gate.tolist() == [0.0, 0.0, 0.0]
    assert streak.tolist() == [1.0, 1.0, 0.0]

    gate, streak = R.sustained_binary_gate(
        torch.tensor([True, False, True]), streak, hold_steps=3
    )
    assert gate.tolist() == [0.0, 0.0, 0.0]
    assert streak.tolist() == [2.0, 0.0, 1.0]

    gate, streak = R.sustained_binary_gate(
        torch.tensor([True, True, True]), streak, hold_steps=3
    )
    assert gate.tolist() == [1.0, 0.0, 0.0]
    assert streak.tolist() == [3.0, 1.0, 2.0]

    gate, streak = R.sustained_binary_gate(
        torch.tensor([False, True, True]), streak, hold_steps=3
    )
    assert gate.tolist() == [0.0, 0.0, 1.0]
    assert streak.tolist() == [0.0, 2.0, 3.0]


def test_sustained_binary_gate_hold_one_opens_immediately():
    gate, streak = R.sustained_binary_gate(
        torch.tensor([False, True]), torch.zeros(2), hold_steps=1
    )
    assert gate.tolist() == [0.0, 1.0]
    assert streak.tolist() == [0.0, 1.0]


def test_home_deviation_deadband():
    q = torch.tensor([[0.0, 0.5, -0.5]])
    home = torch.zeros(1, 3)
    # |dev|-0.1 clamped -> [0, 0.4, 0.4]; squared sum = 0.16 + 0.16 = 0.32
    c = R.home_deviation(q, home, deadband=0.1)
    assert abs(c.item() - 0.32) < 1e-6
    # Motion within the deadband is free.
    c0 = R.home_deviation(torch.tensor([[0.05, -0.05]]), torch.zeros(1, 2), deadband=0.1)
    assert c0.item() == 0.0


def test_target_penetration_deadband_and_row_reduction():
    targets = torch.tensor([[0.05, -0.10, 0.20], [0.0, 0.0, 0.0]])
    q = torch.zeros_like(targets)
    cost = R.target_penetration(targets, q, deadband=0.10)
    assert torch.allclose(cost, torch.tensor([0.01, 0.0]), atol=1.0e-7)


def test_target_penetration_supports_per_joint_deadband():
    targets = torch.tensor([[0.10, 0.20]])
    q = torch.zeros_like(targets)
    deadband = torch.tensor([[0.05, 0.20]])
    cost = R.target_penetration(targets, q, deadband)
    assert torch.allclose(cost, torch.tensor([0.0025]), atol=1.0e-7)


def test_near_contact_score_thumb_weighting():
    near = torch.tensor([[1.0, 0.0, 0.5]])  # index, middle, thumb
    score = R.near_contact_score(near, thumb_index=2, non_thumb_indices=[0, 1], top_k=1)
    # 0.5 * thumb(0.5) + 0.5 * top1 non-thumb(1.0) = 0.75
    assert abs(score.item() - 0.75) < 1e-6
    score_k2 = R.near_contact_score(near, thumb_index=2, non_thumb_indices=[0, 1], top_k=2)
    assert abs(score_k2.item() - (0.5 * 0.5 + 0.5 * 0.5)) < 1e-6


def _spinning_handle(omega_z: float):
    """A handle at the origin spinning about +z, with two tips on the surface.

    Tips sit at (+r, 0, 0) and (-r, 0, 0); the surface velocity there is
    (0, +omega*r, 0) and (0, -omega*r, 0) respectively.
    """
    r = 0.032
    tip_pos = torch.tensor([[[r, 0.0, 0.0], [-r, 0.0, 0.0]]])
    body_pos = torch.zeros(1, 3)
    body_lin = torch.zeros(1, 3)
    body_ang = torch.tensor([[0.0, 0.0, omega_z]])
    v_surf = torch.tensor(
        [[[0.0, omega_z * r, 0.0], [0.0, -omega_z * r, 0.0]]]
    )
    return tip_pos, body_pos, body_lin, body_ang, v_surf


def test_surface_co_motion_static_fingers_score_zero():
    # The 20260728 creep exploit: handle spins, fingertips motionless -> no
    # co-motion credit for either tip.
    tip_pos, body_pos, body_lin, body_ang, _ = _spinning_handle(0.3)
    tip_vel = torch.zeros_like(tip_pos)
    score = R.surface_co_motion(tip_pos, tip_vel, body_pos, body_lin, body_ang, 0.002)
    assert torch.allclose(score, torch.zeros(1, 2), atol=1e-6)


def test_surface_co_motion_driving_and_partial_and_opposing():
    tip_pos, body_pos, body_lin, body_ang, v_surf = _spinning_handle(0.3)
    # Moving exactly with the surface -> 1; half speed -> 0.5; opposing -> 0.
    full = R.surface_co_motion(tip_pos, v_surf, body_pos, body_lin, body_ang, 0.002)
    assert torch.allclose(full, torch.ones(1, 2), atol=1e-5)
    half = R.surface_co_motion(tip_pos, 0.5 * v_surf, body_pos, body_lin, body_ang, 0.002)
    assert torch.allclose(half, torch.full((1, 2), 0.5), atol=1e-5)
    opposing = R.surface_co_motion(tip_pos, -v_surf, body_pos, body_lin, body_ang, 0.002)
    assert torch.allclose(opposing, torch.zeros(1, 2), atol=1e-6)
    # Faster than the surface (over-rolling) is clamped, not over-credited.
    over = R.surface_co_motion(tip_pos, 2.0 * v_surf, body_pos, body_lin, body_ang, 0.002)
    assert torch.allclose(over, torch.ones(1, 2), atol=1e-5)


def test_surface_co_motion_static_handle_never_vetoes():
    # Below the surface-speed floor the gate must stay open so the policy can
    # start a stationary handle.
    tip_pos, body_pos, body_lin, _, _ = _spinning_handle(0.0)
    body_ang = torch.zeros(1, 3)
    tip_vel = torch.zeros_like(tip_pos)
    score = R.surface_co_motion(tip_pos, tip_vel, body_pos, body_lin, body_ang, 0.002)
    assert torch.allclose(score, torch.ones(1, 2), atol=1e-6)


def test_surface_co_motion_exact_under_tilt_precession():
    # A tilted, translating handle: surface velocity includes the linear term,
    # and a tip riding the full rigid velocity still scores 1.
    tip_pos = torch.tensor([[[0.03, 0.01, 0.02]]])
    body_pos = torch.tensor([[0.005, -0.002, 0.01]])
    body_lin = torch.tensor([[0.003, -0.001, 0.002]])
    body_ang = torch.tensor([[0.05, -0.02, 0.4]])
    rel = tip_pos - body_pos.unsqueeze(1)
    v_surf = body_lin.unsqueeze(1) + torch.linalg.cross(
        body_ang.unsqueeze(1).expand_as(rel), rel, dim=-1
    )
    score = R.surface_co_motion(tip_pos, v_surf, body_pos, body_lin, body_ang, 0.002)
    assert torch.allclose(score, torch.ones(1, 1), atol=1e-5)


def test_creep_exploit_regression_motion_auth_gate_closes():
    # End-to-end gate composition for the frozen-finger creep run: four tips in
    # distance contact but motionless while the handle spins -> the soft-count
    # motion authorization is zero, so turn reward and qualified progress pay 0.
    tip_pos, body_pos, body_lin, body_ang, _ = _spinning_handle(0.28)
    tips4 = torch.cat([tip_pos, tip_pos], dim=1)  # (1, 4, 3)
    tip_vel = torch.zeros_like(tips4)
    co = R.surface_co_motion(tips4, tip_vel, body_pos, body_lin, body_ang, 0.002)
    present = torch.ones(1, 4)
    motion_auth = R.soft_count_gate(co * present, 2.0)
    assert motion_auth.item() == 0.0
    # And with two of four tips genuinely driving, authorization is full.
    _, _, _, _, v_surf = _spinning_handle(0.28)
    tip_vel[:, :2] = v_surf
    co = R.surface_co_motion(tips4, tip_vel, body_pos, body_lin, body_ang, 0.002)
    motion_auth = R.soft_count_gate(co * present, 2.0)
    assert abs(motion_auth.item() - 1.0) < 1e-5


def test_quat_roundtrip_axis_angle():
    # axis_angle_from_quat(quat) recovers the rotation vector.
    angle = 0.3
    axis = torch.tensor([0.0, 1.0, 0.0])
    q = torch.cat([torch.tensor([math.cos(angle / 2)]), math.sin(angle / 2) * axis]).unsqueeze(0)
    rotvec = R.axis_angle_from_quat(q)
    assert torch.allclose(rotvec, (angle * axis).unsqueeze(0), atol=1e-5)
    # Double-cover: -q is the same rotation.
    assert torch.allclose(R.axis_angle_from_quat(-q), rotvec, atol=1e-5)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    raise SystemExit(1 if failures else 0)
