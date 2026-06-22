"""Alternate top-down initial grasp for the Linker L20 screwdriver task.

This configuration inherits all dynamics, rewards, observations, curriculum,
and articulation settings from :class:`LinkerL20ScrewdriverRotationEnvCfg`.
Only the initial hand transform and pregrasp joint positions differ.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .screwdriver_rotation_env_cfg import LinkerL20ScrewdriverRotationEnvCfg


# Top-down five-fingertip wrap inspired by a conventional power grasp.
_TOP_GRASP_POS = (0.00749555, 0.07053220, 1.52299065)
_TOP_GRASP_ROT = (-0.20776033, -0.57429015, -0.72821611, 0.31101088)
_TOP_GRASP_JOINT_POS = {
    "index_mcp_roll": 0.130000,
    "index_mcp_pitch": 0.888000,
    "index_pip": 1.522023,
    "index_dip": 1.357188,
    "middle_mcp_roll": 0.030740,
    "middle_mcp_pitch": 0.932999,
    "middle_pip": 0.558307,
    "middle_dip": 0.497842,
    "ring_mcp_roll": 0.002060,
    "ring_mcp_pitch": 0.495695,
    "ring_pip": 0.989701,
    "ring_dip": 0.882516,
    "pinky_mcp_roll": 0.115255,
    "pinky_mcp_pitch": 0.580380,
    "pinky_pip": 1.078316,
    "pinky_dip": 0.961534,
    "thumb_cmc_yaw": 1.056250,
    "thumb_cmc_roll": 0.557376,
    "thumb_cmc_pitch": 0.213078,
    "thumb_mcp": 0.402924,
    "thumb_ip": 0.468157,
}
_TOP_GRASP_PREGRASP = {
    "index": (0.130000, 0.888000, 1.522023),
    "middle": (0.030740, 0.932999, 0.558307),
    "ring": (0.002060, 0.495695, 0.989701),
    "pinky": (0.115255, 0.580380, 1.078316),
    "thumb": (1.056250, 0.557376, 0.213078, 0.402924),
}


@configclass
class LinkerL20ScrewdriverRotationTopGraspEnvCfg(
    LinkerL20ScrewdriverRotationEnvCfg
):
    """Top-down five-fingertip wrap inspired by a conventional power grasp."""

    def __post_init__(self) -> None:
        # ``@configclass`` materialises ``robot_cfg`` as a per-instance field via
        # ``default_factory``, so the base posture is only available on ``self``
        # (the class attribute is gone).  Override the initial transform and
        # pregrasp posture here, after the parent has finished wiring.
        super().__post_init__()

        self.robot_cfg.init_state.pos = _TOP_GRASP_POS
        self.robot_cfg.init_state.rot = _TOP_GRASP_ROT
        self.robot_cfg.init_state.joint_pos = dict(_TOP_GRASP_JOINT_POS)
        self.pregrasp_positions = {
            finger: tuple(pos) for finger, pos in _TOP_GRASP_PREGRASP.items()
        }
