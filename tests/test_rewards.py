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


def test_near_contact_score_thumb_weighting():
    near = torch.tensor([[1.0, 0.0, 0.5]])  # index, middle, thumb
    score = R.near_contact_score(near, thumb_index=2, non_thumb_indices=[0, 1], top_k=1)
    # 0.5 * thumb(0.5) + 0.5 * top1 non-thumb(1.0) = 0.75
    assert abs(score.item() - 0.75) < 1e-6
    score_k2 = R.near_contact_score(near, thumb_index=2, non_thumb_indices=[0, 1], top_k=2)
    assert abs(score_k2.item() - (0.5 * 0.5 + 0.5 * 0.5)) < 1e-6


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
