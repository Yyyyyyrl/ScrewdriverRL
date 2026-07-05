"""LinkerDeployer session-logic tests: ramp, watchdog, startup continuity, and
the offline dry-run — all with fake read/send functions, no SDK/ROS/hardware.

Run:  python -m pytest tests/test_deploy_linker.py -q
"""

import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_deploy_policy_bundle import make_bundle  # noqa: E402
from screwdriver_rl.deploy.deploy_linker import LinkerDeployer  # noqa: E402
from screwdriver_rl.deploy import hw_utils, linker_sdk_map as sdkmap  # noqa: E402


def _deployer(**kw) -> LinkerDeployer:
    kw.setdefault("dry_run", True)
    kw.setdefault("ramp_s", 0.1)
    kw.setdefault("hz", 100.0)
    return LinkerDeployer(make_bundle(), **kw)


# --------------------------------------------------------------------------- #
# hw_utils
# --------------------------------------------------------------------------- #

def test_ramp_frames_endpoint_and_monotone():
    start = [0.0] * 16
    end = list(sdkmap.PREGRASP_16)
    frames = hw_utils.ramp_frames(start, end, duration_s=1.0, rate_hz=20.0)
    assert len(frames) == 20
    assert frames[-1] == [float(v) for v in end]  # exact endpoint, no residue
    for j in range(16):
        vals = [start[j]] + [f[j] for f in frames]
        diffs = [b - a for a, b in zip(vals, vals[1:])]
        assert all(d >= -1e-12 for d in diffs) or all(d <= 1e-12 for d in diffs)


def test_ramp_frames_degenerate():
    end = [1.0] * 16
    assert hw_utils.ramp_frames([0.0] * 16, end, duration_s=0.0) == [end]
    assert hw_utils.ramp_frames([0.0] * 16, end, duration_s=-1.0) == [end]
    with pytest.raises(ValueError):
        hw_utils.ramp_frames([0.0] * 16, end, 1.0, rate_hz=0.0)


@pytest.mark.parametrize("bad", [
    None,
    [0.0] * 19,
    [0.0] * 21,
    [-1.0] * 20,                       # G20 CAN parse-failure vector
    [""] * 20,                         # cache-not-filled placeholders
    [True] * 20,                       # bools are not positions
    [300.0] + [0.0] * 19,              # out of 0..255
    [float("nan")] + [0.0] * 19,
    42,
])
def test_validate_state20_rejects(bad):
    assert hw_utils.validate_state20(bad) is None


def test_validate_state20_accepts():
    assert hw_utils.validate_state20(list(range(20))) == [float(v) for v in range(20)]


def test_parse_five():
    assert hw_utils.parse_five("120") == [120] * 5
    assert hw_utils.parse_five("1,2,3,4,5") == [1, 2, 3, 4, 5]
    for bad in ("1,2", "300", "a", "1,2,3,4,5,6"):
        with pytest.raises(ValueError):
            hw_utils.parse_five(bad)


# --------------------------------------------------------------------------- #
# Deployer session logic
# --------------------------------------------------------------------------- #

def test_step_contract():
    d = _deployer()
    state = sdkmap.joints16_to_sdk_range(sdkmap.PREGRASP_16)
    cmd = d._step(state)
    assert len(cmd) == 20 and all(isinstance(v, int) and 0 <= v <= 255 for v in cmd)
    assert set(d.last_tick) == {"finger_q", "targets", "action", "cmd", "state"}
    assert len(d.last_tick["finger_q"]) == 16 and len(d.last_tick["action"]) == 16


def test_startup_continuity():
    d = _deployer()
    echo = [float(v) for v in sdkmap.joints16_to_sdk_range(sdkmap.open_pose_16())]
    sent = []

    def read_fn():
        return list(echo)

    def send_fn(cmd):
        echo[:] = [float(c) for c in cmd]
        sent.append(list(cmd))

    d._startup(read_fn, send_fn)
    ramp_end = sent[-1]
    first_cmd = d._step(read_fn())
    # The contract: the first policy command sits within one action delta
    # (0.05 rad, per-joint fraction of range → counts) of the ramp's endpoint.
    diffs = [abs(a - b) for a, b in zip(first_cmd, ramp_end)]
    for js in sdkmap.DEFAULT_JOINTS:
        max_counts = math.ceil(0.05 / (js.hi - js.lo) * 255) + 1
        assert diffs[js.slot] <= max_counts, (
            f"{js.name}: first policy cmd jumped {diffs[js.slot]} counts (> {max_counts})")


def test_watchdog_holds_then_exits():
    d = _deployer(stale_limit=5, max_ticks=0)
    good = [float(v) for v in sdkmap.joints16_to_sdk_range(sdkmap.PREGRASP_16)]
    reads = {"n": 0}
    sent = []

    def read_fn():
        reads["n"] += 1
        return list(good) if reads["n"] <= 2 else None

    def send_fn(cmd):
        sent.append(list(cmd))

    d._last_cmd = sdkmap.joints16_to_sdk_range(sdkmap.PREGRASP_16)
    code = d._loop(read_fn, send_fn)
    assert code == 2
    # 2 policy sends + 5 held re-sends of the last command.
    assert len(sent) == 7
    assert sent[-1] == sent[-2] == d._last_cmd


def test_dry_run_writes_csv(tmp_path):
    log = tmp_path / "run.csv"
    d = _deployer(max_ticks=5, record=str(log), ramp_s=0.2)
    code = d.run("can")  # dry_run short-circuits before any SDK import
    assert code == 0
    rows = list(csv.reader(log.open()))
    header, body = rows[0], rows[1:]
    assert header[:4] == ["tick", "phase", "t_wall", "t_mono"]
    assert len(header) == 4 + 16 * 3 + 20 * 2
    phases = [r[1] for r in body]
    assert phases.count("policy") == 5
    assert phases.count("ramp") >= 1
    # Policy rows carry a full state/command snapshot.
    row = body[phases.index("policy")]
    assert all(cell != "" for cell in row[4:])


def test_calib_applied_in_ctor(tmp_path):
    import json
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"joints": {"index_mcp_roll": {"flip": True}}}))
    try:
        _deployer(calib=str(path))
        spec = {js.name: js for js in sdkmap.active_joints()}["index_mcp_roll"]
        assert spec.flip is True
    finally:
        sdkmap.reset_calibration()
