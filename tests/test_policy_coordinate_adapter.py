"""Legacy-policy coordinate bridge tests using the released fixed64 bundle."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from screwdriver_rl.deploy import linker_sdk_map as sdkmap
from screwdriver_rl.deploy.deploy_linker import HardwareSafetyError, LinkerDeployer
from screwdriver_rl.deploy.policy import DeployPolicy
from screwdriver_rl.deploy.policy_coordinate_adapter import (
    load_policy_coordinate_adapter,
    policy_coordinate_adapter_digest,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "deliverables/linker_l20_topdown_pip108_d64_300m_20260724/checkpoints/deploy_fixed64_release.pth"
CALIBRATION = ROOT / "linker_calib_deploy.json"
ADAPTER = ROOT / "assets/calibrations/linker_g20_left_legacy_topdown_fixed64_policy_adapter_20260805.json"
THUMB_YAW = 12


def _load_adapted_policy():
    base = DeployPolicy(str(CHECKPOINT))
    adapted = load_policy_coordinate_adapter(
        base, ADAPTER, CHECKPOINT, CALIBRATION
    )
    specs = sdkmap.build_joint_table(json.loads(CALIBRATION.read_text()))
    adapted.verify_active_hardware_limits(
        [joint.name for joint in specs],
        [joint.lo for joint in specs],
        [joint.hi for joint in specs],
    )
    return adapted


def test_released_adapter_is_bidirectional_and_changes_only_thumb_yaw():
    adapted = _load_adapted_policy()
    virtual = adapted.base_policy.home_targets
    hardware = adapted.policy_to_hardware(virtual)
    assert torch.allclose(
        adapted.hardware_to_policy(hardware), virtual, atol=1.0e-7, rtol=0.0
    )
    for index in range(16):
        delta = float(hardware[0, index] - virtual[0, index])
        assert delta == pytest.approx(-0.26 if index == THUMB_YAW else 0.0, abs=1.0e-7)

    assert float(adapted.home_targets[0, THUMB_YAW]) == pytest.approx(
        0.9806667465104224, abs=1.0e-7
    )
    assert float(adapted.startup_reset_targets[0, THUMB_YAW]) == pytest.approx(
        0.5377368235588074, abs=1.0e-7
    )
    assert float(adapted.finger_upper[0, THUMB_YAW]) == pytest.approx(1.12, abs=1.0e-7)
    assert float(adapted.finger_lower[0, THUMB_YAW]) == pytest.approx(
        0.6306667465104224, abs=1.0e-7
    )


def test_adapter_preserves_actor_action_and_virtual_history_dynamics():
    reference = DeployPolicy(str(CHECKPOINT))
    adapted = _load_adapted_policy()
    virtual_q = reference.home_targets.clone()
    hardware_q = adapted.policy_to_hardware(virtual_q)
    reference.reset(virtual_q, virtual_q)
    adapted.reset(hardware_q, hardware_q)

    reference_target, reference_action = reference.act(virtual_q, return_action=True)
    hardware_target, adapted_action = adapted.act(hardware_q, return_action=True)
    assert torch.allclose(adapted_action, reference_action, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(
        hardware_target,
        adapted.policy_to_hardware(reference_target),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        adapted.base_policy.hist, reference.hist, atol=1.0e-7, rtol=0.0
    )


def test_adapter_rejects_digest_and_checkpoint_binding_tampering(tmp_path):
    raw = json.loads(ADAPTER.read_text())
    bad_digest = deepcopy(raw)
    bad_digest["transform"]["offset_rad"][THUMB_YAW] = -0.25
    bad_digest_path = tmp_path / "bad_digest.json"
    bad_digest_path.write_text(json.dumps(bad_digest))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_policy_coordinate_adapter(
            DeployPolicy(str(CHECKPOINT)), bad_digest_path, CHECKPOINT, CALIBRATION
        )

    bad_binding = deepcopy(raw)
    bad_binding["source_checkpoint"]["sha256"] = "0" * 64
    bad_binding["adapter_digest"] = policy_coordinate_adapter_digest(bad_binding)
    bad_binding_path = tmp_path / "bad_binding.json"
    bad_binding_path.write_text(json.dumps(bad_binding))
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        load_policy_coordinate_adapter(
            DeployPolicy(str(CHECKPOINT)), bad_binding_path, CHECKPOINT, CALIBRATION
        )


def test_candidate_adapter_allows_dry_run_but_blocks_live_policy():
    try:
        deployer = LinkerDeployer(
            str(CHECKPOINT),
            dry_run=True,
            max_ticks=2,
            ramp_s=0.0,
            contact_ramp_s=0.0,
            calib=str(CALIBRATION),
            policy_coordinate_adapter=str(ADAPTER),
            expected_serial="LHT20-010-415-L-B-1-D",
        )
        assert deployer.run("can") == 0

        deployer.dry_run = False
        with pytest.raises(HardwareSafetyError, match="not 'live_promoted'"):
            deployer._validate_policy_coordinate_adapter_status()

        deployer.candidate_adapter_gate = True
        deployer.max_ticks = 20
        deployer.record = "bounded-gate.csv"
        deployer.task_frame_confirmed = True
        deployer._validate_policy_coordinate_adapter_status("can")
        with pytest.raises(HardwareSafetyError, match="CAN transport"):
            deployer._validate_policy_coordinate_adapter_status("ros")
        deployer.max_ticks = 0
        with pytest.raises(HardwareSafetyError, match="max-ticks"):
            deployer._validate_policy_coordinate_adapter_status("can")
    finally:
        sdkmap.reset_calibration()
