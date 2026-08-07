"""Deterministic coordinate bridge for a calibration-bound legacy policy.

This is deliberately separate from the learned Stage-2 proprioceptive adapter.
The learned adapter predicts the actor latent; this module only converts joint
coordinates at the policy/runtime boundary::

    hardware q --inverse affine--> legacy policy q --actor/integrator-->
    legacy target --forward affine--> hardware q --physical LUT--> SDK raw

Both measured positions and acknowledged targets are converted, so the old
actor and history encoder continue to see the coordinate contract used during
training.  Specs are immutable, content-addressed, and bound to one checkpoint
file and one production calibration overlay.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch


ADAPTER_DIGEST_ALGORITHM = "sha256-canonical-json-excluding-adapter-digest"
LIVE_PROMOTED_STATUS = "live_promoted"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_coordinate_adapter_digest(value: Mapping[str, object]) -> str:
    content = deepcopy(dict(value))
    content.pop("adapter_digest", None)
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def _numeric_vector(value: object, width: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} values")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        result.append(number)
    return tuple(result)


class PolicyCoordinateAdapter:
    """Expose a legacy :class:`DeployPolicy` in promoted hardware coordinates."""

    def __init__(self, policy, spec: Mapping[str, object]) -> None:
        self.base_policy = policy
        self.spec = deepcopy(dict(spec))
        self.adapter_id = str(spec["adapter_id"])
        self.status = str(spec["status"])
        self.device = policy.device
        self.n_finger = policy.n_finger

        joint_order = spec["joint_order"]
        if not isinstance(joint_order, list) or len(joint_order) != self.n_finger:
            raise ValueError("adapter joint_order width does not match policy")
        self.joint_order = tuple(str(name) for name in joint_order)
        transform = spec["transform"]
        if not isinstance(transform, Mapping):
            raise ValueError("adapter transform must be an object")
        if transform.get("kind") != "per-joint-positive-affine-bidirectional":
            raise ValueError("unsupported policy coordinate transform")
        scales = _numeric_vector(transform.get("scale"), self.n_finger, "transform.scale")
        offsets = _numeric_vector(
            transform.get("offset_rad"), self.n_finger, "transform.offset_rad"
        )
        if any(scale <= 0.0 for scale in scales):
            raise ValueError("all adapter scales must be positive")
        self.scale = torch.tensor(scales, dtype=torch.float32, device=self.device).view(1, -1)
        self.offset = torch.tensor(offsets, dtype=torch.float32, device=self.device).view(1, -1)

        source = spec["source_policy_contract"]
        target = spec["target_hardware_contract"]
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise ValueError("adapter source and target contracts must be objects")
        self._verify_source_policy(source)
        hardware_lower = _numeric_vector(
            target.get("position_lower_rad"), self.n_finger, "target position lower"
        )
        hardware_upper = _numeric_vector(
            target.get("position_upper_rad"), self.n_finger, "target position upper"
        )
        if any(high <= low for low, high in zip(hardware_lower, hardware_upper)):
            raise ValueError("target hardware limits must have upper > lower")
        self.hardware_contract_lower = torch.tensor(
            hardware_lower, dtype=torch.float32, device=self.device
        ).view(1, -1)
        self.hardware_contract_upper = torch.tensor(
            hardware_upper, dtype=torch.float32, device=self.device
        ).view(1, -1)

        self.finger_lower = self.policy_to_hardware(policy.finger_lower)
        self.finger_upper = self.policy_to_hardware(policy.finger_upper)
        self.home_targets = self.policy_to_hardware(policy.home_targets)
        self.startup_reset_targets = self.policy_to_hardware(policy.startup_reset_targets)
        self._require_inside_hardware_contract(self.finger_lower, "adapted policy lower")
        self._require_inside_hardware_contract(self.finger_upper, "adapted policy upper")
        self._require_inside_hardware_contract(self.home_targets, "adapted home")
        self._require_inside_hardware_contract(self.startup_reset_targets, "adapted startup reset")

        self.codec = policy.codec
        self.cfg = dict(policy.cfg)
        self.cfg["policy_coordinate_adapter_id"] = self.adapter_id
        self.cur_targets = self.home_targets.clone()

    def _verify_source_policy(self, source: Mapping[str, object]) -> None:
        expected_task = source.get("task_id")
        if expected_task != self.base_policy.cfg.get("task"):
            raise ValueError("adapter task_id does not match checkpoint")
        expected_delta = float(source.get("action_delta_scale_rad"))
        if not math.isclose(
            expected_delta, self.base_policy.action_delta_scale, abs_tol=1.0e-12
        ):
            raise ValueError("adapter action delta scale does not match checkpoint")
        for key, actual in (
            ("position_lower_rad", self.base_policy.finger_lower),
            ("position_upper_rad", self.base_policy.finger_upper),
            ("home_target_rad", self.base_policy.home_targets),
            ("startup_reset_target_rad", self.base_policy.startup_reset_targets),
        ):
            expected = torch.tensor(
                _numeric_vector(source.get(key), self.n_finger, f"source {key}"),
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1)
            if not torch.allclose(expected, actual, atol=1.0e-7, rtol=0.0):
                raise ValueError(f"adapter source {key} does not match checkpoint")

    def _row(self, values) -> torch.Tensor:
        row = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(1, -1)
        if row.shape != (1, self.n_finger):
            raise ValueError(f"expected {self.n_finger} joint values")
        return row

    def _require_inside_hardware_contract(self, values: torch.Tensor, label: str) -> None:
        if bool(
            (values < self.hardware_contract_lower - 1.0e-6).any()
            or (values > self.hardware_contract_upper + 1.0e-6).any()
        ):
            raise ValueError(f"{label} lies outside promoted hardware limits")

    def policy_to_hardware(self, values) -> torch.Tensor:
        return self._row(values) * self.scale + self.offset

    def hardware_to_policy(self, values) -> torch.Tensor:
        return (self._row(values) - self.offset) / self.scale

    def verify_active_hardware_limits(
        self, names: Sequence[str], lower: Sequence[float], upper: Sequence[float]
    ) -> None:
        if tuple(names) != self.joint_order:
            raise ValueError("active SDK joint order does not match adapter")
        actual_lower = self._row(lower)
        actual_upper = self._row(upper)
        if not torch.allclose(
            actual_lower, self.hardware_contract_lower, atol=1.0e-7, rtol=0.0
        ) or not torch.allclose(
            actual_upper, self.hardware_contract_upper, atol=1.0e-7, rtol=0.0
        ):
            raise ValueError("active SDK limits do not match adapter target contract")

    def reset(self, finger_q=None, effective_target=None) -> None:
        virtual_q = None if finger_q is None else self.hardware_to_policy(finger_q)
        virtual_target = (
            None if effective_target is None else self.hardware_to_policy(effective_target)
        )
        self.base_policy.reset(virtual_q, virtual_target)
        self.cur_targets = self.policy_to_hardware(self.base_policy.cur_targets)

    @torch.no_grad()
    def act(self, finger_q, return_action: bool = False):
        virtual_q = self.hardware_to_policy(finger_q)
        virtual_target, action = self.base_policy.act(virtual_q, return_action=True)
        self.cur_targets = self.policy_to_hardware(virtual_target)
        self._require_inside_hardware_contract(self.cur_targets, "adapted runtime target")
        if return_action:
            return self.cur_targets, action
        return self.cur_targets


def load_policy_coordinate_adapter(
    policy,
    adapter_path: str | Path,
    checkpoint_path: str | Path,
    calibration_overlay_path: str | Path,
) -> PolicyCoordinateAdapter:
    """Load and verify one immutable adapter against its exact runtime inputs."""

    path = Path(adapter_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("policy coordinate adapter root must be an object")
    required = {
        "schema_version", "adapter_id", "status", "joint_order", "source_checkpoint",
        "target_calibration", "source_policy_contract", "target_hardware_contract",
        "transform", "validation", "adapter_digest_algorithm", "adapter_digest",
    }
    if set(raw) != required:
        raise ValueError(
            f"invalid policy coordinate adapter fields; missing={sorted(required-set(raw))}, "
            f"extra={sorted(set(raw)-required)}"
        )
    if raw["schema_version"] != 1:
        raise ValueError("unsupported policy coordinate adapter schema version")
    if raw["adapter_digest_algorithm"] != ADAPTER_DIGEST_ALGORITHM:
        raise ValueError("unsupported policy coordinate adapter digest algorithm")
    digest = policy_coordinate_adapter_digest(raw)
    if raw["adapter_digest"] != digest:
        raise ValueError("policy coordinate adapter digest mismatch")

    source = raw["source_checkpoint"]
    target = raw["target_calibration"]
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError("adapter checkpoint/calibration bindings must be objects")
    if source.get("sha256") != _sha256_file(checkpoint_path):
        raise ValueError("policy coordinate adapter checkpoint SHA256 mismatch")
    if target.get("overlay_sha256") != _sha256_file(calibration_overlay_path):
        raise ValueError("policy coordinate adapter calibration overlay SHA256 mismatch")
    return PolicyCoordinateAdapter(policy, raw)
