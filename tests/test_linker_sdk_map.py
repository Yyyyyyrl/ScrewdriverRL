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
import xml.etree.ElementTree as ET

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


# --------------------------------------------------------------------------- #
# Calibration overlay lo/hi (absolute-angle mapping) support.
# --------------------------------------------------------------------------- #

def test_overlay_lo_hi_absolute_identity():
    """lo/hi == the slot's SDK arc range makes the map absolute (arc == clamp(v))."""
    overlay = {"joints": {"index_pip": {
        "lo": sdkmap.L20_L_MIN[16], "hi": sdkmap.L20_L_MAX[16]}}}
    sdkmap.apply_calibration(overlay)
    for v in (0.0, 0.5, 1.0, 1.08):
        arc = sdkmap.joints16_to_sdk_arc([v if js.name == "index_pip" else js.lo
                                          for js in sdkmap.active_joints()])
        assert arc[16] == pytest.approx(v, abs=1e-9)
    # beyond the SDK range the command clamps to the physical limit
    arc = sdkmap.joints16_to_sdk_arc([1.57 if js.name == "index_pip" else js.lo
                                      for js in sdkmap.active_joints()])
    assert arc[16] == pytest.approx(sdkmap.L20_L_MAX[16], abs=1e-9)


def test_overlay_lo_hi_roundtrip_inverse():
    overlay = {"joints": {"middle_pip": {"lo": 0.0, "hi": 1.08},
                          "index_mcp_roll": {"lo": -0.26, "hi": 0.26}}}
    sdkmap.apply_calibration(overlay)
    t16 = [0.5 * (js.lo + js.hi) for js in sdkmap.active_joints()]
    rt = sdkmap.sdk_range_to_joints16(sdkmap.joints16_to_sdk_range(t16))
    assert rt == pytest.approx(t16, abs=0.01)


def test_overlay_lo_hi_validation():
    with pytest.raises(ValueError, match="must be a number"):
        sdkmap.build_joint_table({"joints": {"index_pip": {"lo": "x"}}})
    with pytest.raises(ValueError, match="must be a number"):
        sdkmap.build_joint_table({"joints": {"index_pip": {"hi": True}}})
    with pytest.raises(ValueError, match="hi > lo"):
        sdkmap.build_joint_table({"joints": {"index_pip": {"lo": 1.0, "hi": 0.5}}})


# --------------------------------------------------------------------------- #
# Measured physical raw<->radian LUT support.
# --------------------------------------------------------------------------- #

def test_index_pip_candidate_lut_uses_one_table_for_command_and_readback():
    path = (
        Path(__file__).resolve().parents[1]
        / "linker_calib_index_pip_lut_candidate_20260802.json"
    )
    sdkmap.apply_calibration(str(path))
    specs = sdkmap.active_joints()
    index = next(i for i, joint in enumerate(specs) if joint.name == "index_pip")
    joint = specs[index]
    assert joint.physical_raw_knots[0] == 20.0
    assert joint.physical_raw_knots[-1] == 255.0
    assert joint.physical_rad_knots[0] == pytest.approx(1.5387503110826093)
    assert joint.physical_rad_knots[-1] == 0.0

    expected_commands = {
        0.0: 255,
        0.3: 218,
        0.5280907168623586: 184,
        0.97: 117,
        1.08: 101,
    }
    for target, expected_raw in expected_commands.items():
        values = [spec.lo for spec in specs]
        values[index] = target
        command = sdkmap.joints16_to_sdk_range(values)
        assert command[16] == pytest.approx(expected_raw, abs=1)
        readback = sdkmap.sdk_range_to_joints16(command)
        assert readback[index] == pytest.approx(target, abs=0.004)

    # The policy's full 0..1.08-rad range stays far above the fault-prone raw end.
    assert min(expected_commands.values()) >= 100
    # Readback reports measured physical radians, including outside the policy
    # range if an external load back-drives the joint beyond its commanded rail.
    raw20 = [255.0] * 20
    raw20[16] = 20.0
    assert sdkmap.sdk_range_to_joints16(raw20)[index] == pytest.approx(
        1.5387503110826093
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "2026-08-04: same revert as the PIP endpoint contract — the runtime URDF "
        "carries the vendor (OG) mcp_pitch 1.4 rather than the measured 1.20-1.25, "
        "so the candidate LUT and the runtime asset no longer agree. The measured "
        "asset is archived at assets/linker_hand_l20_calibfit/. Note OG is the "
        "LOOSER bound here (1.4 vs 1.25), but the +-0.35 rad home box caps the "
        "commanded mcp_pitch at 1.094 rad, below both — so the discrepancy is not "
        "reachable by the policy. Flip this marker off once the asset is adopted."
    ),
)
def test_index_mcp_pitch_candidate_lut_and_runtime_urdf_use_measured_range():
    root = Path(__file__).resolve().parents[1]
    path = root / "linker_calib_index_pip_pitch_lut_candidate_20260802.json"
    sdkmap.apply_calibration(str(path))
    specs = sdkmap.active_joints()
    index = next(
        i for i, joint in enumerate(specs)
        if joint.name == "index_mcp_pitch"
    )
    joint = specs[index]
    measured_upper = 1.2504421299201645
    assert joint.lo == 0.0
    assert joint.hi == pytest.approx(measured_upper)
    assert joint.physical_raw_knots[0] == 0.0
    assert joint.physical_raw_knots[-1] == 255.0
    assert joint.physical_rad_knots[0] == pytest.approx(measured_upper)
    assert joint.physical_rad_knots[-1] == 0.0

    expected_commands = {
        0.0: 255,
        0.140535: 230,
        0.5: 158,
        1.0: 53,
        1.2: 11,
        measured_upper: 0,
    }
    for target, expected_raw in expected_commands.items():
        values = [spec.lo for spec in specs]
        values[index] = target
        command = sdkmap.joints16_to_sdk_range(values)
        assert command[1] == pytest.approx(expected_raw, abs=1)
        readback = sdkmap.sdk_range_to_joints16(command)
        assert readback[index] == pytest.approx(target, abs=0.003)

    # Values from an older 1.4-rad policy clamp to the measured physical rail
    # and read back the real angle rather than a fictitious 1.4 rad.
    values = [spec.lo for spec in specs]
    values[index] = 1.4
    command = sdkmap.joints16_to_sdk_range(values)
    assert command[1] == 0
    assert sdkmap.sdk_range_to_joints16(command)[index] == pytest.approx(
        measured_upper
    )

    urdf = ET.parse(
        root / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
    )
    limit = next(
        joint_node.find("limit")
        for joint_node in urdf.getroot().findall("joint")
        if joint_node.get("name") == "index_mcp_pitch"
    )
    assert float(limit.get("lower")) == 0.0
    assert float(limit.get("upper")) == pytest.approx(measured_upper)
    default = next(
        spec
        for spec in sdkmap.DEFAULT_JOINTS
        if spec.name == "index_mcp_pitch"
    )
    assert default.hi == pytest.approx(measured_upper)


def test_thumb_mcp_pitch_candidate_luts_are_physical_and_candidate_only():
    root = Path(__file__).resolve().parents[1]
    path = root / "linker_calib_index_thumb_mcp_pitch_lut_candidate_20260802.json"
    sdkmap.apply_calibration(str(path))
    specs = sdkmap.active_joints()
    by_name = {joint.name: (index, joint) for index, joint in enumerate(specs)}
    measured = {
        "thumb_cmc_pitch": (0, 0.8100219654624176),
        "thumb_mcp": (15, 1.252264477797083),
    }
    for name, (slot, upper) in measured.items():
        index, joint = by_name[name]
        assert joint.lo == 0.0
        assert joint.hi == pytest.approx(upper)
        assert joint.physical_raw_knots[0] == 0.0
        assert joint.physical_raw_knots[-1] == 255.0
        assert joint.physical_rad_knots[0] == pytest.approx(upper)
        assert joint.physical_rad_knots[-1] == 0.0
        for target in (0.0, 0.25 * upper, 0.5 * upper, upper):
            values = [spec.lo for spec in specs]
            values[index] = target
            command = sdkmap.joints16_to_sdk_range(values)
            readback = sdkmap.sdk_range_to_joints16(command)
            assert readback[index] == pytest.approx(target, abs=0.004)
            if target == 0.0:
                assert command[slot] == 255
            elif target == upper:
                assert command[slot] == 0

    # Production defaults are the reviewed OG-local-q range intersected with
    # unchanged OG geometry.  Full physical LUT knots remain in the deployment
    # overlay, while these signed rails are shared by training and commands.
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "assets/calibrations/linker_g20_left_semantic_schema_v1.json"
        ).read_text()
    )
    signed = {entry["name"]: entry["position_limit"] for entry in schema["ordered_joints"]}
    defaults = {joint.name: joint for joint in sdkmap.DEFAULT_JOINTS}
    for name, joint in defaults.items():
        assert joint.lo == pytest.approx(signed[name][0], abs=1.0e-12), name
        assert joint.hi == pytest.approx(signed[name][1], abs=1.0e-12), name
    assert defaults["thumb_cmc_pitch"].hi == pytest.approx(0.79)
    assert defaults["thumb_mcp"].hi == pytest.approx(1.05)
    assert defaults["thumb_cmc_roll"].lo == pytest.approx(0.42)


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            {"physical_lut": {"raw": [20, 20, 255], "rad": [1.5, 1.0, 0.0]}},
            "raw must be strictly increasing",
        ),
        (
            {"physical_lut": {"raw": [20, 100, 255], "rad": [1.5, 0.5, 0.6]}},
            "rad must be strictly monotonic",
        ),
        (
            {"physical_lut": {"raw": [20, 255], "rad": [1.0, 0.0]}},
            "does not span semantic range",
        ),
        (
            {
                "flip": True,
                "physical_lut": {"raw": [20, 255], "rad": [1.6, 0.0]},
            },
            "cannot be combined with flip",
        ),
    ],
)
def test_physical_lut_validation(entry, message):
    with pytest.raises(ValueError, match=message):
        sdkmap.build_joint_table({"joints": {"index_pip": entry}})


def test_physical_lut_supports_increasing_urdf_local_q():
    """MCP-roll-style local coordinates increase with SDK raw."""
    overlay = {
        "joints": {
            "index_mcp_roll": {
                "lo": -0.25,
                "hi": 0.25,
                "physical_lut": {
                    "raw": [0, 128, 255],
                    "rad": [-0.25, 0.0, 0.25],
                },
            }
        }
    }
    sdkmap.apply_calibration(overlay)
    specs = sdkmap.active_joints()
    index = next(
        i for i, joint in enumerate(specs) if joint.name == "index_mcp_roll"
    )
    for target, expected_raw in ((-0.25, 0), (0.0, 128), (0.25, 255)):
        values = [spec.lo for spec in specs]
        values[index] = target
        command = sdkmap.joints16_to_sdk_range(values)
        assert command[6] == pytest.approx(expected_raw, abs=1)
        readback = sdkmap.sdk_range_to_joints16(command)
        assert readback[index] == pytest.approx(target, abs=0.003)


def test_thumb_yaw_phase_c_v2_candidate_lut_roundtrips_and_is_candidate_only():
    root = Path(__file__).resolve().parents[1]
    path = root / "linker_calib_phase_c_thumb_yaw_candidate_20260803_v2.json"
    sdkmap.apply_calibration(str(path))
    specs = sdkmap.active_joints()
    index = next(
        i for i, joint in enumerate(specs) if joint.name == "thumb_cmc_yaw"
    )
    joint = specs[index]
    assert joint.slot == 10
    assert joint.physical_raw_knots[0] == 17.0
    assert joint.physical_raw_knots[-1] == 251.0
    assert joint.physical_rad_knots[0] == pytest.approx(1.3153079831555723)
    assert joint.physical_rad_knots[-1] == pytest.approx(0.00041670737127360094)

    for target in (joint.lo, 0.0, 0.3, 0.7, 1.2, joint.hi):
        values = [spec.lo for spec in specs]
        values[index] = target
        command = sdkmap.joints16_to_sdk_range(values)
        assert 17 <= command[10] <= 251
        readback = sdkmap.sdk_range_to_joints16(command)
        assert readback[index] == pytest.approx(target, abs=0.004)

    # Loading this candidate does not alter the production defaults.
    default = next(
        spec for spec in sdkmap.DEFAULT_JOINTS
        if spec.name == "thumb_cmc_yaw"
    )
    assert default.physical_raw_knots is None
