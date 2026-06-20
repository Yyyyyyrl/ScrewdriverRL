"""Four-finger Allegro continuous screwdriver rotation task.

This variant keeps the original three-finger posture
(:class:`AllegroScrewdriverRotationEnvCfg`) — index pressing the cap from the
top, thumb opposing from one side — and adds the **ring** finger alongside the
middle finger so the handle is grasped by all four fingers:

    index  -> presses the cap (top)
    middle -> one side of the handle
    ring   -> same side, alongside middle   (newly added)
    thumb  -> the opposing side

Everything else (curriculum, domain randomisation, rewards, screwdriver asset,
simulation) is inherited unchanged.  The grasp closes on the SHARED 20 mm
screwdriver, so no enlarged handle / new URDF is required.

The pregrasp joint angles were validated geometrically (contact, clearance,
joint-limit margins) with ``tools/render_allegro_4f_posture.py`` — see
``artifacts/allegro_4f_initial_posture/``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import field

import gymnasium as gym
import numpy as np

from isaaclab.utils import configclass

from .screwdriver_rotation_env_cfg import AllegroScrewdriverRotationEnvCfg


@configclass
class AllegroScrewdriverRotation4FEnvCfg(AllegroScrewdriverRotationEnvCfg):
    """Allegro screwdriver rotation with all four fingers engaged.

    Observation space (35-D): [finger_q(16), cur_targets(16), euler(3)]
    Action space (16-D): HORA-style delta targets for index+middle+ring+thumb (4 each)
    Privileged obs (18-D): euler(3)+angvel(3)+rel_pos(3)+quat(4)+friction(1)+tip_dist(4)
    """

    # ---- Gym spaces (4 fingers x 4 joints) ----
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
    state_space = 0

    # ---- Active fingers: ring is now driven by the policy ----
    fingers: tuple[str, ...] = ("index", "middle", "ring", "thumb")

    # ---- RMA dims grow with the extra finger ----
    privileged_obs_dim: int = 18
    """3 euler + 3 angvel + 3 rel-pos + 4 quat + 1 friction + 4 fingertip-dist."""
    history_obs_dim: int = 32
    """[finger_q(16), cur_targets(16)] per frame."""

    # ---- Robot: original transform, original index/thumb, +ring, lightly
    #      re-curled middle so all four fingertips contact the 20 mm handle. ----
    robot_cfg = deepcopy(AllegroScrewdriverRotationEnvCfg.robot_cfg)
    robot_cfg.init_state.joint_pos = {
        # index (hitosashi) — unchanged; drapes over and presses the cap-top
        "allegro_hand_hitosashi_finger_finger_joint_0": 0.10,
        "allegro_hand_hitosashi_finger_finger_joint_1": 0.60,
        "allegro_hand_hitosashi_finger_finger_joint_2": 0.60,
        "allegro_hand_hitosashi_finger_finger_joint_3": 0.60,
        # middle (naka) — near-original, slightly more curl to seat on the side
        "allegro_hand_naka_finger_finger_joint_4": -0.07,
        "allegro_hand_naka_finger_finger_joint_5": 0.53,
        "allegro_hand_naka_finger_finger_joint_6": 0.93,
        "allegro_hand_naka_finger_finger_joint_7": 0.92,
        # ring (kusuri) — added alongside middle on the same side
        "allegro_hand_kusuri_finger_finger_joint_8": 0.24,
        "allegro_hand_kusuri_finger_finger_joint_9": 0.89,
        "allegro_hand_kusuri_finger_finger_joint_10": 0.70,
        "allegro_hand_kusuri_finger_finger_joint_11": 0.98,
        # thumb (oya) — near-original (+0.01); opposes from the other side
        "allegro_hand_oya_finger_joint_12": 1.21,
        "allegro_hand_oya_finger_joint_13": 0.30,
        "allegro_hand_oya_finger_joint_14": 0.30,
        "allegro_hand_oya_finger_joint_15": 1.21,
    }

    # ---- Pregrasp joint positions (per finger, 4 joints each) ----
    # MUST match robot_cfg.init_state.joint_pos above so the PD targets equal the
    # spawn pose (no reset transient).  index/middle/ring share one handle side
    # from above; the thumb opposes from the side.
    pregrasp_positions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "index":  (0.10, 0.60, 0.60, 0.60),
            "middle": (-0.07, 0.53, 0.93, 0.92),
            "ring":   (0.24, 0.89, 0.70, 0.98),
            "thumb":  (1.21, 0.30, 0.30, 1.21),
        }
    )
