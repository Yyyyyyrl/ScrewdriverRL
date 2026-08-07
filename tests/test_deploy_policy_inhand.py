from __future__ import annotations

from copy import deepcopy

import pytest

from screwdriver_rl.deploy import linker_sdk_map as sdkmap
from screwdriver_rl.deploy.policy import DeployPolicy
from test_deploy_policy_bundle import make_bundle


def test_free_object_deploy_actor_consumes_exact_96_value_codec_input() -> None:
    policy = DeployPolicy(make_bundle(free_object=True), device="cpu")
    measured = list(sdkmap.PREGRASP_16)
    policy.reset(measured, measured)
    target, action = policy.act(measured, return_action=True)
    assert policy.codec.spec.actor_frame_count == 3
    assert policy.codec.assemble_actor_input(policy.hist).shape == (1, 96)
    assert policy.actor.proprio_dim == 96
    assert target.shape == action.shape == (1, 16)


def test_deploy_bundle_without_explicit_codec_is_rejected() -> None:
    bundle = deepcopy(make_bundle())
    del bundle["config"]["proprio_codec"]
    with pytest.raises(RuntimeError, match="ProprioCodec"):
        DeployPolicy(bundle, device="cpu")
