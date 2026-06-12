"""Sim-free task math (pure PyTorch): reward primitives, quaternion helpers.

Nothing in this package may import isaaclab - it is unit-tested without
Isaac Sim (tests/test_rewards.py).
"""

from . import rewards

__all__ = ["rewards"]
