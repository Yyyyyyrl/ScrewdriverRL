from screwdriver_rl.utils.linker_topdown_contact_gate import (
    functional_contact_topology,
    functional_physics_gate,
)


def _record(fractions: dict[str, float]) -> dict:
    roles = ("index", "middle", "ring", "pinky", "thumb")
    return {
        "role_contact_fraction": fractions,
        "role_force_mean_n": {role: 0.5 for role in roles},
        "fingertip_total_force_max_n": {role: 1.0 for role in roles},
        "wrong_surface_force_max_n": 0.0,
        "tilt_max_rad": 0.1,
        "zero_action_raw_shaft_drift_rad_s": 0.001,
        "terminated_or_truncated": False,
    }


def test_functional_topology_allows_one_unused_support_finger():
    result = functional_contact_topology(
        {
            "index": 1.0,
            "middle": 1.0,
            "ring": 0.0,
            "pinky": 1.0,
            "thumb": 1.0,
        }
    )
    assert result["pass"]
    assert result["active_role_count"] == 4
    assert result["mean_role_contact_fraction"] == 0.8


def test_functional_topology_has_no_named_critical_fingers():
    result = functional_contact_topology(
        {
            "index": 0.0,
            "middle": 1.0,
            "ring": 1.0,
            "pinky": 1.0,
            "thumb": 1.0,
        }
    )
    assert result["pass"]
    assert result["critical_roles_pass"]


def test_functional_topology_accepts_any_three_active_roles():
    result = functional_contact_topology(
        {
            "index": 1.0,
            "middle": 1.0,
            "ring": 1.0,
            "pinky": 0.75,
            "thumb": 0.75,
        }
    )
    assert result["mean_role_contact_fraction"] == 0.9
    assert result["active_role_count"] == 3
    assert result["pass"]


def test_functional_physics_gate_keeps_safety_limits_strict():
    record = _record(
        {
            "index": 1.0,
            "middle": 1.0,
            "ring": 0.5,
            "pinky": 1.0,
            "thumb": 1.0,
        }
    )
    assert functional_physics_gate(
        record, max_zero_action_drift_rad_s=0.005
    )["pass"]
    record["wrong_surface_force_max_n"] = 0.051
    result = functional_physics_gate(
        record, max_zero_action_drift_rad_s=0.005
    )
    assert not result["pass"]
    assert not result["wrong_surface_pass"]
