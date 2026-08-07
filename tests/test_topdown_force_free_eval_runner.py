"""Pure-Python contracts for the M4/M5 force-free evaluation matrix."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/evaluate_topdown_force_free_policy.py"
SPEC = importlib.util.spec_from_file_location("force_free_eval", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _report(p50=1.0, fall=0.0, osc=0.05):
    return {
        "episodes": {
            "count": 32,
            "net_turns_p50": p50,
            "net_turns_mean": p50,
            "fall_rate": fall,
        },
        "step_metrics": {"eval_osc_ratio": {"mean": osc}},
    }


def test_bias_matrix_is_symmetric_and_includes_m3_final_range():
    cases = TOOL._bias_cases((1.0, 2.0, 4.0, 5.0, 8.0), 1.5)

    assert len(cases) == 24
    labels = {case["label"] for case in cases}
    assert "pos_x_neg_1mm" in labels
    assert "pos_y_pos_8mm" in labels
    assert "tilt_roll_neg_1p5deg" in labels
    assert "tilt_pitch_pos_1p5deg" in labels


def test_summary_enforces_adapter_gap_biases_and_held_out_mse():
    cases = TOOL._bias_cases((1.0, 8.0), 1.5)
    oracle = {"baseline": _report(), "damping_x4": _report(p50=0.9)}
    adapter = {"baseline": _report(p50=0.95)}
    for case in cases:
        p50 = 0.85
        oracle[case["label"]] = _report(p50=p50)
        adapter[case["label"]] = _report(p50=p50)
    calibration = {
        "all_fingertip_force_n": {"mean": 0.5},
        "all_fingertip_force_window_distribution": {
            "fraction_in_0_5_to_4_n": 0.2
        },
    }
    old_calibration = {
        "all_fingertip_force_n": {"mean": 1.0},
        "all_fingertip_force_window_distribution": {
            "fraction_in_0_5_to_4_n": 0.6
        },
    }
    validation = {
        "adapter_latent_mse": 0.005,
        "screw_relative_position_mse": 0.002,
        "distance_contact_score_mse": 0.003,
    }

    result = TOOL._summarize(
        cases=cases,
        oracle_reports=oracle,
        adapter_reports=adapter,
        reference_report=_report(p50=1.0),
        current_calibration=calibration,
        reference_calibration=old_calibration,
        adaptation_validation=validation,
        old_force_channel_mse=0.01,
        trained_xy_range_mm=8.0,
        min_reference_retention=0.9,
        min_in_range_retention=0.8,
        min_damping_retention=0.8,
        min_adapter_oracle_retention=0.9,
        max_osc_ratio=0.15,
        max_in_range_fall_increase=0.05,
        max_ood_fall_rate=0.5,
    )

    assert result["promotion_pass"]
    assert not result["failed_gates"]
    assert not result["unavailable_gates"]


def test_summary_fails_closed_when_historical_evidence_is_missing():
    cases = TOOL._bias_cases((1.0,), 1.5)
    oracle = {"baseline": _report(), "damping_x4": _report()}
    for case in cases:
        oracle[case["label"]] = _report()

    result = TOOL._summarize(
        cases=cases,
        oracle_reports=oracle,
        adapter_reports=None,
        reference_report=None,
        current_calibration=None,
        reference_calibration=None,
        adaptation_validation=None,
        old_force_channel_mse=None,
        trained_xy_range_mm=8.0,
        min_reference_retention=0.9,
        min_in_range_retention=0.8,
        min_damping_retention=0.8,
        min_adapter_oracle_retention=0.9,
        max_osc_ratio=0.15,
        max_in_range_fall_increase=0.05,
        max_ood_fall_rate=0.5,
    )

    assert not result["promotion_pass"]
    assert "m4_reference_eval_available" in result["unavailable_gates"]
    assert "m4_force_distribution_vs_pre_refactor" in result["unavailable_gates"]
