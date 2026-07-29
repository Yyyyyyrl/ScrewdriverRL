"""Validated reset state and compliant target for the 64 mm top-down task.

``TOPDOWN_RESET_*`` is the only pose written directly into simulation. It passed
the full URDF-mesh, convex-hull, non-distal, self-collision and joint-margin
validator. ``TOPDOWN_TARGET_*`` is a position-controller preload target applied
during the inherited reset settling steps; PhysX contact constraints determine
the actual settled pose. This module has no Isaac imports so CPU tests can audit
both states.
"""

from __future__ import annotations


TOPDOWN_HANDLE_RADIUS_M = 0.032
TOPDOWN_PHYSICS_CANDIDATE_INDEX = 117
TOPDOWN_ROOT_YAW_RAD = -0.02679684443708501

# Isaac and URDF quaternions are scalar-first (w, x, y, z). The palm local +X
# normal maps exactly to world -Z.
# The registered diameter task adds its per-bucket offset to this shared base.
TOPDOWN_ROOT_POS_W = (-0.0006267097531600672, 0.1912800647761465, 1.4598869497013336)
TOPDOWN_ROOT_QUAT_WXYZ = (
    0.4932561105686752,
    0.5066541319151191,
    0.4932561105686752,
    -0.5066541319151191,
)
TOPDOWN_SCREWDRIVER_TILT_XY = (
    -0.1333760768175125,
    0.01544977817684412,
)

# Collision-resolved state from the 64 mm equilibrium search, written directly.
TOPDOWN_RESET_JOINT_POSITIONS = {
    "index_mcp_roll": -0.04091014713048935,
    "index_mcp_pitch": 0.8236588835716248,
    "index_pip": 0.5197473168373108,
    "index_dip": 0.46345868242383004,
    "middle_mcp_roll": -0.043827660381793976,
    "middle_mcp_pitch": 0.6052597761154175,
    "middle_pip": 0.79217129945755,
    "middle_dip": 0.7063791477262974,
    "ring_mcp_roll": -0.06000000000000001,
    "ring_mcp_pitch": 0.7350769639015198,
    "ring_pip": 0.6084202527999878,
    "ring_dip": 0.5425283394217492,
    "pinky_mcp_roll": 0.040318358689546585,
    "pinky_mcp_pitch": 0.7814591526985168,
    "pinky_pip": 0.7680953145027161,
    "pinky_dip": 0.6849105919420719,
    "thumb_cmc_yaw": 0.7977368235588074,
    "thumb_cmc_roll": 1.0845370292663574,
    "thumb_cmc_pitch": 0.11,
    "thumb_mcp": 0.3222019672393799,
    "thumb_ip": 0.37436646573543547,
}

TOPDOWN_RESET_POSITIONS = {
    "index": (-0.04091014713048935, 0.8236588835716248, 0.5197473168373108),
    "middle": (-0.043827660381793976, 0.6052597761154175, 0.79217129945755),
    "ring": (-0.06000000000000001, 0.7350769639015198, 0.6084202527999878),
    "pinky": (0.040318358689546585, 0.7814591526985168, 0.7680953145027161),
    "thumb": (0.7977368235588074, 1.0845370292663574, 0.11, 0.3222019672393799),
}


# Candidate 117 passed the functional contact release gate in two independent
# 128-environment DR runs. It is a compliant Phase-0 target, not a teleported state.
TOPDOWN_TARGET_JOINT_POSITIONS = {
    "index_mcp_roll": -0.042032288312911996,
    "index_mcp_pitch": 0.7203785375536257,
    "index_pip": 0.5280907168623586,
    "index_dip": 0.47089849222616526,
    "middle_mcp_roll": -0.03899578297327875,
    "middle_mcp_pitch": 0.548135936465427,
    "middle_pip": 0.7630190464548652,
    "middle_dip": 0.6803840837238033,
    "ring_mcp_roll": -0.0481909389793873,
    "ring_mcp_pitch": 0.6385637202016128,
    "ring_pip": 0.5874077435541302,
    "ring_dip": 0.5237914849272179,
    "pinky_mcp_roll": 0.03948873943645571,
    "pinky_mcp_pitch": 0.7440400604986218,
    "pinky_pip": 0.7794977181891592,
    "pinky_dip": 0.6950781153092733,
    "thumb_cmc_yaw": 1.2406667465104224,
    "thumb_cmc_roll": 0.8572340336349435,
    "thumb_cmc_pitch": 0.105,
    "thumb_mcp": 0.105,
    "thumb_ip": 0.12199949999999998,
}

TOPDOWN_TARGET_POSITIONS = {
    "index": (-0.042032288312911996, 0.7203785375536257, 0.5280907168623586),
    "middle": (-0.03899578297327875, 0.548135936465427, 0.7630190464548652),
    "ring": (-0.0481909389793873, 0.6385637202016128, 0.5874077435541302),
    "pinky": (0.03948873943645571, 0.7440400604986218, 0.7794977181891592),
    "thumb": (1.2406667465104224, 0.8572340336349435, 0.105, 0.105),
}


TOPDOWN_PREGRASP_POSITIONS = {
    finger: tuple(values) for finger, values in TOPDOWN_TARGET_POSITIONS.items()
}

# Backward-compatible name used by CPU tests and render tooling.
TOPDOWN_JOINT_POSITIONS = TOPDOWN_RESET_JOINT_POSITIONS
