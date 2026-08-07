"""LinkerDeployer session-logic tests: ramp, watchdog, startup continuity, and
the offline dry-run — all with fake read/send functions, no SDK/ROS/hardware.

Run:  python -m pytest tests/test_deploy_linker.py -q
"""

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_deploy_policy_bundle import make_bundle  # noqa: E402
from screwdriver_rl.deploy.deploy_linker import (  # noqa: E402
    HardwareSafetyError,
    LinkerDeployer,
)
from screwdriver_rl.deploy import hw_utils, linker_sdk_map as sdkmap  # noqa: E402


def _deployer(**kw) -> LinkerDeployer:
    kw.setdefault("dry_run", True)
    kw.setdefault("ramp_s", 0.1)
    kw.setdefault("hz", None)
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


def test_startup_only_never_enables_policy(monkeypatch):
    d = _deployer(startup_only=True, max_ticks=5, ramp_s=0.0)
    echo = [float(v) for v in sdkmap.joints16_to_sdk_range(sdkmap.open_pose_16())]

    def read_fn():
        return list(echo)

    def send_fn(cmd):
        echo[:] = [float(value) for value in cmd]

    monkeypatch.setattr("screwdriver_rl.deploy.deploy_linker.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        d.policy,
        "act",
        lambda *_args, **_kwargs: pytest.fail("startup-only enabled the policy"),
    )
    code = d._run_session(
        read_fn,
        send_fn,
        fault_fn=lambda: [0] * 20,
        contact_fn=lambda: [0.0] * 5,
    )
    assert code == 0
    assert d._n_emitted == 0
    assert d._ticks == 0


def test_rail_lock_trips_when_targets_sit_pinned():
    """A saturated policy pins its targets and then cannot recover: frozen
    targets freeze the proprio history, hence the latent, hence the action."""
    d = _deployer(rail_lock_ticks=3, rail_lock_joints=4)
    pinned = d.policy.finger_upper[0].tolist()
    assert d._rail_locked(pinned) is False  # 1st tick
    assert d._rail_locked(pinned) is False  # 2nd
    assert d._rail_locked(pinned) is True  # 3rd consecutive tick -> trip


def test_rail_lock_resets_when_targets_move_off_the_limits():
    d = _deployer(rail_lock_ticks=3, rail_lock_joints=4)
    lo = d.policy.finger_lower[0].tolist()
    hi = d.policy.finger_upper[0].tolist()
    mid = [(a + b) / 2.0 for a, b in zip(lo, hi)]
    d._rail_locked(hi)
    d._rail_locked(hi)
    assert d._rail_locked(mid) is False  # moving off the rail clears the run
    assert d._rail_locked(hi) is False  # counter restarted


def test_rail_lock_disabled_by_zero_ticks():
    d = _deployer(rail_lock_ticks=0)
    pinned = d.policy.finger_upper[0].tolist()
    assert all(d._rail_locked(pinned) is False for _ in range(10))


def test_fault_watchdog_stops_on_undocumented_bit_64():
    d = _deployer(max_ticks=100, fault_limit=2, fault_poll_ticks=1)
    state = sdkmap.joints16_to_sdk_range(sdkmap.PREGRASP_16)
    sent = []
    faults = [0] * 20
    faults[17] = 64
    code = d._loop(lambda: list(state), lambda cmd: sent.append(list(cmd)), lambda: faults)
    assert code == 4
    assert d._n_emitted == 1


def test_fault_mask_requires_explicit_opt_in():
    d = _deployer(max_ticks=2, fault_limit=1, fault_poll_ticks=1, allowed_fault_mask=0x40)
    state = sdkmap.joints16_to_sdk_range(sdkmap.PREGRASP_16)
    faults = [0] * 20
    faults[17] = 64
    assert d._loop(lambda: list(state), lambda _cmd: None, lambda: faults) == 0


def test_topdown_requires_measured_lut_for_every_pitch_and_pip():
    bundle = make_bundle()
    bundle["config"]["task"] = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
    bundle["config"]["startup_reset_targets"] = list(bundle["config"]["home_targets"])
    bundle["config"]["deployment_geometry_scale"] = [1.0, 1.0]
    try:
        d = LinkerDeployer(bundle, dry_run=False, calib=None)
        with pytest.raises(HardwareSafetyError, match="physical-LUT"):
            d._validate_topdown_mapping()
        d = LinkerDeployer(bundle, dry_run=False, calib="linker_calib_deploy.json")
        d._validate_topdown_mapping()
        d = LinkerDeployer(
            bundle,
            dry_run=False,
            calib="linker_calib_index_pip_lut_candidate_20260802.json",
        )
        # These historical partial candidates were written while the four PIPs
        # still shared the guessed 1.08 rad endpoint, so they now fail the signed
        # check as well as the missing-LUT check.  Both rejections are wanted.
        with pytest.raises(HardwareSafetyError, match="middle_pip=") as excinfo:
            d._validate_topdown_mapping()
        assert "middle_mcp_pitch=missing-physical-lut" in str(excinfo.value)
        assert "middle_pip=[0,1.08]!=signed" in str(excinfo.value)

        d = LinkerDeployer(
            bundle,
            dry_run=False,
            calib="linker_calib_index_pip_pitch_lut_candidate_20260802.json",
        )
        with pytest.raises(
            HardwareSafetyError,
            match="middle_mcp_pitch=missing-physical-lut",
        ):
            d._validate_topdown_mapping()

        # The accepted overlay has to carry each finger's *own* signed endpoint.
        # A single shared 1.08 rad used to pass here; it is now rejected, which
        # is the point -- the four PIPs were measured separately.
        signed = {joint.name: joint for joint in sdkmap.DEFAULT_JOINTS}
        complete = {
            "joints": {
                **{
                    f"{finger}_mcp_pitch": {
                        "lo": 0.0,
                        "hi": signed[f"{finger}_mcp_pitch"].hi,
                        "physical_lut": {
                            "raw": [0, 255],
                            "rad": [signed[f"{finger}_mcp_pitch"].hi, 0.0],
                        },
                    }
                    for finger in ("index", "middle", "ring", "pinky")
                },
                **{
                        f"{finger}_pip": {
                            "lo": signed[f"{finger}_pip"].lo,
                            "hi": signed[f"{finger}_pip"].hi,
                            "physical_lut": {
                                "raw": [100, 255],
                                "rad": [
                                    signed[f"{finger}_pip"].hi,
                                    signed[f"{finger}_pip"].lo,
                                ],
                            },
                    }
                    for finger in ("index", "middle", "ring", "pinky")
                },
            }
        }
        d = LinkerDeployer(bundle, dry_run=False, calib=complete)
        d._validate_topdown_mapping()

        # A shared guessed endpoint must now fail closed.
        shared = json.loads(json.dumps(complete))
        for finger in ("index", "middle", "ring", "pinky"):
            shared["joints"][f"{finger}_pip"]["hi"] = 1.08
            shared["joints"][f"{finger}_pip"]["physical_lut"]["rad"] = [1.08, 0.0]
        d = LinkerDeployer(bundle, dry_run=False, calib=shared)
        with pytest.raises(HardwareSafetyError, match="!=signed"):
            d._validate_topdown_mapping()

        # The superseded phase-C candidate used the pre-promotion signed rails
        # and must now fail; linker_calib_deploy.json above is the production asset.
        d = LinkerDeployer(
            bundle,
            dry_run=False,
            calib="linker_calib_phase_c_thumb_yaw_roll_candidate_20260803.json",
        )
        with pytest.raises(HardwareSafetyError, match="!=signed"):
            d._validate_topdown_mapping()
    finally:
        sdkmap.reset_calibration()


def test_g20_identity_is_bound_to_serial_and_side():
    bundle = make_bundle()
    bundle["config"]["task"] = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
    bundle["config"]["startup_reset_targets"] = list(bundle["config"]["home_targets"])
    bundle["config"]["deployment_geometry_scale"] = [1.0, 1.0]
    d = LinkerDeployer(
        bundle,
        dry_run=False,
        hand_joint="G20",
        expected_serial="LHT20-010-415-L-B-1-D",
        calib="linker_calib_deploy.json",
    )

    class FakeApi:
        def get_serial_number(self):
            return "LHT20-010-415-L-B-1-D"

        def get_embedded_version(self):
            return [1, 0, 7]

        def get_touch_type(self):
            return 2

    try:
        assert d._validate_can_identity(FakeApi())[:2] == (
            "LHT20-010-415-L-B-1-D",
            [1, 0, 7],
        )
        d.expected_serial = "wrong"
        with pytest.raises(HardwareSafetyError, match="serial mismatch"):
            d._validate_can_identity(FakeApi())
    finally:
        sdkmap.reset_calibration()


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
