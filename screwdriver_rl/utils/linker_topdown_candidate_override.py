"""Apply an audited top-down posture-search candidate to an env config."""

from __future__ import annotations

import json
from pathlib import Path


JOINT_NAMES = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
)
FINGER_WIDTHS = (("index", 3), ("middle", 3), ("ring", 3), ("pinky", 3), ("thumb", 4))


def load_candidate(path: str | Path, candidate_index: int | None) -> dict:
    data = json.loads(Path(path).read_text())
    if "top_candidates" not in data:
        return data
    rows = data["top_candidates"]
    if candidate_index is None:
        return rows[0]
    matches = [row for row in rows if int(row["candidate_index"]) == candidate_index]
    if len(matches) != 1:
        raise ValueError(f"expected one candidate_index={candidate_index}, found {len(matches)}")
    return matches[0]


def apply_candidate(env_cfg, candidate: dict, *, fixed_64mm: bool = False) -> None:
    joints = candidate["joint_positions_independent"]
    target = {}
    cursor = 0
    for finger, width in FINGER_WIDTHS:
        names = JOINT_NAMES[cursor : cursor + width]
        target[finger] = tuple(float(joints[name]) for name in names)
        cursor += width

    root = tuple(float(value) for value in candidate["root_pos_w"])
    quat = tuple(float(value) for value in candidate["root_quat_wxyz"])
    env_cfg.robot_cfg.init_state.pos = root
    env_cfg.robot_cfg.init_state.rot = quat
    env_cfg.pregrasp_positions = target

    # A checkpoint posture snapshot may also carry the collision-resolved reset
    # state. Older search JSONs only describe the controller target, so keep
    # this extension optional and backwards compatible.
    reset = None
    reset_joints = candidate.get("reset_joint_positions_independent")
    if reset_joints is not None:
        reset = {}
        cursor = 0
        for finger, width in FINGER_WIDTHS:
            names = JOINT_NAMES[cursor : cursor + width]
            reset[finger] = tuple(float(reset_joints[name]) for name in names)
            cursor += width
        env_cfg.reset_joint_positions = reset

    reset_expanded = candidate.get("reset_joint_positions_expanded")
    if reset_expanded is not None:
        env_cfg.robot_cfg.init_state.joint_pos = {
            str(name): float(value) for name, value in reset_expanded.items()
        }

    reset_tilt = candidate.get("reset_screwdriver_tilt_xy")
    if reset_tilt is not None:
        env_cfg.reset_screwdriver_tilt_xy = tuple(float(value) for value in reset_tilt)

    reset_zero_tension = candidate.get("reset_zero_tension_targets")
    if reset_zero_tension is not None:
        env_cfg.reset_zero_tension_targets = bool(reset_zero_tension)

    contact_margins = candidate.get("contact_d_margin_by_finger")
    if contact_margins is not None:
        env_cfg.contact_d_margin_by_finger = {
            str(name): float(value) for name, value in contact_margins.items()
        }

    if hasattr(env_cfg, "pregrasp_positions_buckets") and hasattr(
        env_cfg, "reset_joint_positions_buckets"
    ):
        # Preserve an explicitly supplied checkpoint-era reset. Previously this
        # block silently replaced it with the current D64 bucket after applying
        # the candidate, creating a mixed old-target/new-reset configuration.
        reset64 = reset if reset is not None else env_cfg.reset_joint_positions_buckets[1]
        tilt64 = (
            tuple(float(value) for value in reset_tilt)
            if reset_tilt is not None
            else env_cfg.reset_screwdriver_tilt_xy_buckets[1]
        )
        env_cfg.pregrasp_positions_buckets = [target, target, target]
        env_cfg.reset_joint_positions_buckets = [reset64, reset64, reset64]
        env_cfg.pregrasp_root_pos_offsets_buckets = [(0.0, 0.0, 0.0)] * 3
        env_cfg.pregrasp_root_quats_buckets = [quat] * 3
        env_cfg.reset_screwdriver_tilt_xy_buckets = [tilt64] * 3
        env_cfg.reset_joint_positions = reset64
        env_cfg.reset_screwdriver_tilt_xy = tilt64

    if fixed_64mm:
        assets_cfg = list(env_cfg.screwdriver_cfg.spawn.assets_cfg)
        if len(assets_cfg) != 3:
            raise ValueError("fixed 64 mm override expects the 60/64/68 asset bank")
        # A single geometry can use Isaacs shared cooked collider. Keeping a
        # one-element MultiAssetSpawner would leave replicate_physics=False and
        # costs roughly 5x throughput at 8192 environments.
        env_cfg.screwdriver_cfg.spawn = assets_cfg[1]
        env_cfg.scene.replicate_physics = True
        repo_root = Path(__file__).resolve().parents[2]
        env_cfg.screwdriver_variants_dir = str(
            repo_root / "assets/screwdriver/topdown_variants_fixed64"
        )
