from __future__ import annotations

import pytest

from screwdriver_rl.deploy import deploy_linker as deploy_module
from screwdriver_rl.deploy import linker_sdk_map as sdkmap
from screwdriver_rl.deploy.deploy_linker import LinkerDeployer
from test_deploy_policy_bundle import make_bundle


def test_deployer_uses_package_rate_when_cli_rate_is_omitted() -> None:
    mounted = LinkerDeployer(make_bundle(), dry_run=True)
    free_object = LinkerDeployer(make_bundle(free_object=True), dry_run=True)
    assert mounted.hz == 10.0
    assert free_object.hz == 20.0


def test_deployer_rejects_cli_rate_that_conflicts_with_package() -> None:
    with pytest.raises(ValueError, match="package-declared"):
        LinkerDeployer(make_bundle(free_object=True), hz=10.0, dry_run=True)


def test_no_send_seeds_uncommanded_measured_target(monkeypatch) -> None:
    bundle = make_bundle()
    deployer = LinkerDeployer(bundle, dry_run=True, no_send=True, ramp_s=0.0)
    monkeypatch.setattr(deploy_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(deployer, "_ramp", lambda *args, **kwargs: None)
    initial_state = sdkmap.joints16_to_sdk_range(sdkmap.open_pose_16())
    q0 = sdkmap.sdk_range_to_joints16(initial_state)
    captured = {}

    def record_reset(measured, effective):
        captured["measured"] = list(measured)
        captured["effective"] = list(effective)

    monkeypatch.setattr(deployer.policy, "reset", record_reset)
    deployer._startup(lambda: list(initial_state), lambda _command: None)

    assert captured["measured"] == pytest.approx(q0)
    assert captured["effective"] == pytest.approx(q0)


def test_topdown_startup_approaches_reset_before_contact_home(monkeypatch) -> None:
    bundle = make_bundle()
    config = bundle["config"]
    config["task"] = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
    config["deployment_geometry_scale"] = [1.0, 1.0]
    home = list(config["home_targets"])
    reset = [value - 0.05 for value in home]
    config["startup_reset_targets"] = reset

    deployer = LinkerDeployer(
        bundle,
        dry_run=True,
        ramp_s=1.25,
        contact_ramp_s=2.5,
    )
    calls = []

    def record_ramp(send_fn, start, end, duration, phase="ramp"):
        calls.append((phase, list(start), list(end), float(duration)))

    deployer._ramp = record_ramp
    monkeypatch.setattr(deploy_module.time, "sleep", lambda _seconds: None)
    initial_state = sdkmap.joints16_to_sdk_range(sdkmap.open_pose_16())
    deployer._startup(lambda: list(initial_state), lambda _command: None)

    assert [call[0] for call in calls] == ["approach", "contact-ramp"]
    assert calls[0][2] == pytest.approx(reset)
    assert calls[0][3] == pytest.approx(1.25)
    assert calls[1][2] == pytest.approx(home)
    assert calls[1][3] == pytest.approx(2.5)
    assert deployer.policy.cur_targets[0].tolist() == pytest.approx(home, abs=0.01)
