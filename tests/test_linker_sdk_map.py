"""Tests for the 16-joint ↔ LinkerHand L20/G20 SDK-slot mapping.

No hardware, no SDK import at runtime: the vendored calibration tables are the
source of truth, and ``test_tables_match_sdk`` re-reads the SDK checkout (when
present) to catch drift against the live SDK's ``LinkerHand/utils/mapping.py``.

Run:  python -m pytest tests/test_linker_sdk_map.py -q
"""

import ast
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screwdriver_rl.deploy import linker_sdk_map as sdkmap  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_calibration():
    yield
    sdkmap.reset_calibration()


def _sdk_mapping_path():
    root = os.environ.get("LINKERHAND_SDK_ROOT", "/home/user/linkerhand-ros-sdk")
    return Path(root) / "linker_hand_sdk_ros" / "scripts" / "LinkerHand" / "utils" / "mapping.py"


# --------------------------------------------------------------------------- #
# Table parity with the live SDK (drift guard).
# --------------------------------------------------------------------------- #

def test_tables_match_sdk():
    path = _sdk_mapping_path()
    if not path.exists():
        pytest.skip(f"LinkerHand SDK checkout not found at {path}")
    spec = importlib.util.spec_from_file_location("_lh_mapping", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module has no imports — safe to exec
    assert list(mod.l20_l_min) == list(sdkmap.L20_L_MIN)
    assert list(mod.l20_l_max) == list(sdkmap.L20_L_MAX)
    assert list(mod.l20_l_derict) == list(sdkmap.L20_L_DIRECT)


def test_stale_range_to_arc_module_is_not_used():
    """The SDK's standalone range_to_arc tables are stale (inverted thumb slots
    0/10, different endpoints).  Guard against anyone re-vendoring them."""
    # Authoritative table: every active slot is direction -1.
    active = [i for i in range(20) if i not in (11, 12, 13, 14)]
    assert all(sdkmap.L20_L_DIRECT[i] == -1 for i in active)
    # The stale table had slot 0 min at -1.57; the live one starts at 0.
    assert sdkmap.L20_L_MIN[0] == 0


# --------------------------------------------------------------------------- #
# Direction + monotonicity (regression test for the stale-table thumb bug).
# --------------------------------------------------------------------------- #

def test_direction_lo_is_open_hi_is_flexed():
    lo16 = [js.lo for js in sdkmap.DEFAULT_JOINTS]
    hi16 = [js.hi for js in sdkmap.DEFAULT_JOINTS]
    cmd_lo = sdkmap.joints16_to_sdk_range(lo16)
    cmd_hi = sdkmap.joints16_to_sdk_range(hi16)
    for js in sdkmap.DEFAULT_JOINTS:
        # 255 = open/extended (arc min), 0 = flexed (arc max) on every slot —
        # with the stale tables this failed on slots 0 and 10 (thumb).
        assert cmd_lo[js.slot] == 255, f"{js.name}: lo should map to 255, got {cmd_lo[js.slot]}"
        assert cmd_hi[js.slot] == 0, f"{js.name}: hi should map to 0, got {cmd_hi[js.slot]}"


def test_per_joint_monotonicity():
    base = list(sdkmap.PREGRASP_16)
    for idx, js in enumerate(sdkmap.DEFAULT_JOINTS):
        prev = None
        for k in range(21):
            t16 = list(base)
            t16[idx] = js.lo + (js.hi - js.lo) * k / 20.0
            cmd = sdkmap.joints16_to_sdk_range(t16)[js.slot]
            if prev is not None:
                assert cmd <= prev, f"{js.name}: command not monotone decreasing"
            prev = cmd


# --------------------------------------------------------------------------- #
# Round-trip + reserved slots.
# --------------------------------------------------------------------------- #

def test_roundtrip_identity():
    rng = random.Random(0)
    for _ in range(50):
        j = [js.lo + rng.random() * (js.hi - js.lo) for js in sdkmap.DEFAULT_JOINTS]
        j2 = sdkmap.sdk_range_to_joints16(sdkmap.joints16_to_sdk_range(j))
        err = max(abs(a - b) for a, b in zip(j, j2))
        assert err < 0.02, f"round-trip error {err:.4f} rad"


def test_reserved_slots():
    cmd = sdkmap.joints16_to_sdk_range(sdkmap.PREGRASP_16)
    assert [cmd[i] for i in (11, 12, 13, 14)] == [0, 0, 0, 0]
    # Inverse ignores whatever sits in the reserved slots.
    state = [float(v) for v in cmd]
    for i in (11, 12, 13, 14):
        state[i] = 77.0
    assert sdkmap.sdk_range_to_joints16(state) == sdkmap.sdk_range_to_joints16(cmd)


def test_input_length_validation():
    with pytest.raises(ValueError):
        sdkmap.joints16_to_sdk_range([0.0] * 15)
    with pytest.raises(ValueError):
        sdkmap.sdk_range_to_joints16([0.0] * 19)


# --------------------------------------------------------------------------- #
# Calibration overlay.
# --------------------------------------------------------------------------- #

def test_overlay_flip_mirrors_command():
    lo16 = [js.lo for js in sdkmap.DEFAULT_JOINTS]
    plain = sdkmap.joints16_to_sdk_range(lo16)
    sdkmap.apply_calibration({"version": 1, "joints": {"index_mcp_roll": {"flip": True}}})
    flipped = sdkmap.joints16_to_sdk_range(lo16)
    assert plain[6] == 255 and flipped[6] == 0  # mirrored
    others = [s for s in range(20) if s != 6]
    assert [plain[s] for s in others] == [flipped[s] for s in others]
    # Read-back uses the same flip: round-trip still identity.
    j2 = sdkmap.sdk_range_to_joints16(sdkmap.joints16_to_sdk_range(sdkmap.PREGRASP_16))
    err = max(abs(a - b) for a, b in zip(sdkmap.PREGRASP_16, j2))
    assert err < 0.02


def test_overlay_slot_swap():
    q = list(sdkmap.PREGRASP_16)
    before = sdkmap.joints16_to_sdk_range(q)
    sdkmap.apply_calibration(
        {"joints": {"index_pip": {"slot": 17}, "middle_pip": {"slot": 16}}})
    after = sdkmap.joints16_to_sdk_range(q)
    assert after[17] == before[16] and after[16] == before[17]


def test_overlay_file_roundtrip(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"version": 1, "note": "test",
                                "joints": {"pinky_mcp_roll": {"flip": True}}}))
    table = sdkmap.apply_calibration(str(path))
    spec = {js.name: js for js in table}["pinky_mcp_roll"]
    assert spec.flip is True
    sdkmap.reset_calibration()
    assert sdkmap.active_joints() == sdkmap.DEFAULT_JOINTS


@pytest.mark.parametrize("overlay", [
    {"joints": {"nope": {"flip": True}}},                     # unknown joint
    {"joints": {"index_pip": {"flap": True}}},                # unknown key
    {"joints": {"index_pip": {"slot": 12}}},                  # reserved slot
    {"joints": {"index_pip": {"slot": 20}}},                  # out of range
    {"joints": {"index_pip": {"flip": 1}}},                   # non-bool flip
    {"joints": {"index_pip": {"slot": 17}}},                  # duplicate (17 in use)
    {"bogus": {}},                                            # unknown top-level key
])
def test_overlay_invalid_raises(overlay):
    with pytest.raises(ValueError):
        sdkmap.build_joint_table(overlay)


# --------------------------------------------------------------------------- #
# Reference poses.
# --------------------------------------------------------------------------- #

def test_poses_within_bounds():
    for pose in (sdkmap.PREGRASP_16, sdkmap.open_pose_16()):
        assert len(pose) == 16
        for v, js in zip(pose, sdkmap.DEFAULT_JOINTS):
            assert js.lo - 1e-9 <= v <= js.hi + 1e-9, f"{js.name}: {v} outside [{js.lo}, {js.hi}]"


def test_pregrasp_matches_env_cfg():
    """AST drift-guard: PREGRASP_16 must equal the env cfg's pregrasp dict."""
    cfg_path = (Path(__file__).resolve().parents[1] / "screwdriver_rl" / "tasks"
                / "linker_l20" / "screwdriver_rotation_env_cfg.py")
    tree = ast.parse(cfg_path.read_text())
    fingers = ("index", "middle", "ring", "pinky", "thumb")
    found = None
    for node in ast.walk(tree):
        # Anchor on the `pregrasp_positions: ... = field(default_factory=...)`
        # assignment, then take the dict literal inside it.
        target = getattr(node, "target", None)
        if not (isinstance(node, ast.AnnAssign) and getattr(target, "id", None) == "pregrasp_positions"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict):
                try:
                    keys = [ast.literal_eval(k) for k in sub.keys]
                    vals = {k: ast.literal_eval(v) for k, v in zip(keys, sub.values)}
                except Exception:
                    continue
                if set(keys) == set(fingers):
                    found = vals
                    break
        break
    if found is None:
        pytest.skip("pregrasp_positions dict literal not found in env cfg")
    flat = [x for f in fingers for x in found[f]]
    assert flat == pytest.approx(sdkmap.PREGRASP_16, abs=1e-9), (
        "PREGRASP_16 drifted from the env cfg pregrasp_positions — update linker_sdk_map.py")
