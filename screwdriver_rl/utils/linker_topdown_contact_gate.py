"""Functional contact criteria for Linker L20 top-down screwdriver postures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROLE_NAMES = ("index", "middle", "ring", "pinky", "thumb")
CRITICAL_ROLE_NAMES: tuple[str, ...] = ()
FUNCTIONAL_MIN_CRITICAL_FRACTION = 0.0
FUNCTIONAL_ACTIVE_ROLE_FRACTION = 0.80
FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT = 3


def functional_contact_topology(
    role_contact_fraction: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate role-neutral contact with any three sustained fingertips."""

    fractions = {
        role: float(role_contact_fraction[role]) for role in ROLE_NAMES
    }
    active_roles = [
        role
        for role in ROLE_NAMES
        if fractions[role] >= FUNCTIONAL_ACTIVE_ROLE_FRACTION
    ]
    mean_fraction = sum(fractions.values()) / len(ROLE_NAMES)
    critical_pass = True
    passed = len(active_roles) >= FUNCTIONAL_MIN_ACTIVE_ROLE_COUNT
    return {
        "pass": passed,
        "critical_roles_pass": critical_pass,
        "active_role_count": len(active_roles),
        "active_roles": active_roles,
        "mean_role_contact_fraction": mean_fraction,
    }


def functional_physics_gate(
    record: Mapping[str, Any],
    *,
    max_zero_action_drift_rad_s: float,
) -> dict[str, Any]:
    """Evaluate functional topology plus the unchanged physics safety limits."""

    topology = functional_contact_topology(record["role_contact_fraction"])
    active_roles = topology["active_roles"]
    role_force_mean = record["role_force_mean_n"]
    active_force_floor_pass = all(
        float(role_force_mean[role]) >= 0.10 for role in active_roles
    )
    force_ceiling_pass = all(
        float(role_force_mean[role]) <= 8.0 for role in ROLE_NAMES
    ) and all(
        float(record["fingertip_total_force_max_n"][role]) <= 8.0
        for role in ROLE_NAMES
    )
    wrong_surface_pass = float(record["wrong_surface_force_max_n"]) <= 0.05
    tilt_pass = float(record["tilt_max_rad"]) <= 0.35
    drift_pass = (
        abs(float(record["zero_action_raw_shaft_drift_rad_s"]))
        <= max_zero_action_drift_rad_s
    )
    episode_pass = not bool(record["terminated_or_truncated"])
    passed = (
        bool(topology["pass"])
        and active_force_floor_pass
        and force_ceiling_pass
        and wrong_surface_pass
        and tilt_pass
        and drift_pass
        and episode_pass
    )
    return {
        "pass": passed,
        "topology": topology,
        "active_force_floor_pass": active_force_floor_pass,
        "force_ceiling_pass": force_ceiling_pass,
        "wrong_surface_pass": wrong_surface_pass,
        "tilt_pass": tilt_pass,
        "drift_pass": drift_pass,
        "episode_pass": episode_pass,
    }
