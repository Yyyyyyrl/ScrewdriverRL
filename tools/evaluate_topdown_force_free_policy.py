#!/usr/bin/env python3
"""Run the M4/M5 force-free policy evaluation matrix and fail closed.

The tool launches ``eval.py`` in separate Isaac processes for the oracle and,
when supplied, adapter-latent policy.  It records the final-DR baseline,
rotation-damping probe, symmetric X/Y root-position biases, and roll/pitch
biases.  Existing reports in the output directory are reused so an interrupted
matrix can be resumed without repeating completed simulator runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
DEFAULT_REFERENCE_REPORT = (
    REPO_ROOT
    / "deliverables/linker_l20_topdown_pip108_d64_300m_20260724"
    / "evidence/gates/stage1_oracle_dr_turn01_192.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--adaptation-validation", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-turns", type=float, default=0.1)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=DEFAULT_REFERENCE_REPORT,
        help="Pre-refactor final-DR eval JSON used for M4 retention gates.",
    )
    parser.add_argument(
        "--reference-calibration",
        type=Path,
        default=None,
        help="Pre-refactor calibration JSON used for force-distribution comparison.",
    )
    parser.add_argument(
        "--old-force-channel-mse",
        type=float,
        default=None,
        help="Held-out raw-force-channel MSE from the pre-refactor adapter.",
    )
    parser.add_argument("--trained-xy-range-mm", type=float, default=8.0)
    parser.add_argument(
        "--bias-magnitudes-mm",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 4.0, 5.0, 8.0),
    )
    parser.add_argument("--tilt-bias-deg", type=float, default=1.5)
    parser.add_argument("--min-reference-retention", type=float, default=0.90)
    parser.add_argument("--min-in-range-retention", type=float, default=0.80)
    parser.add_argument("--min-damping-retention", type=float, default=0.80)
    parser.add_argument("--min-adapter-oracle-retention", type=float, default=0.90)
    parser.add_argument("--max-osc-ratio", type=float, default=0.15)
    parser.add_argument("--max-in-range-fall-increase", type=float, default=0.05)
    parser.add_argument("--max-ood-fall-rate", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write the report but return zero even when a gate fails/incomplete.",
    )
    return parser


def _bias_cases(
    magnitudes_mm: list[float] | tuple[float, ...], tilt_deg: float
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for magnitude in sorted(set(float(value) for value in magnitudes_mm)):
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError("bias magnitudes must be finite and positive")
        token = f"{magnitude:g}".replace(".", "p")
        for axis_index, axis_name in ((0, "x"), (1, "y")):
            for sign, sign_name in ((-1.0, "neg"), (1.0, "pos")):
                position = [0.0, 0.0, 0.0]
                position[axis_index] = sign * magnitude
                cases.append(
                    {
                        "label": f"pos_{axis_name}_{sign_name}_{token}mm",
                        "kind": "position",
                        "magnitude": magnitude,
                        "position_mm": position,
                        "rpy_deg": [0.0, 0.0, 0.0],
                    }
                )
    if not math.isfinite(tilt_deg) or tilt_deg <= 0.0:
        raise ValueError("tilt bias must be finite and positive")
    token = f"{tilt_deg:g}".replace(".", "p")
    for axis_index, axis_name in ((0, "roll"), (1, "pitch")):
        for sign, sign_name in ((-1.0, "neg"), (1.0, "pos")):
            rpy = [0.0, 0.0, 0.0]
            rpy[axis_index] = sign * tilt_deg
            cases.append(
                {
                    "label": f"tilt_{axis_name}_{sign_name}_{token}deg",
                    "kind": "tilt",
                    "magnitude": tilt_deg,
                    "position_mm": [0.0, 0.0, 0.0],
                    "rpy_deg": rpy,
                }
            )
    return cases


def _metrics(report: dict[str, Any]) -> dict[str, float | None]:
    episodes = report.get("episodes", {})
    step_metrics = report.get("step_metrics", {})
    oscillation = step_metrics.get("eval_osc_ratio", {}).get("mean")
    return {
        "episode_count": int(episodes.get("count", 0)),
        "median_net_turns": _finite_or_none(episodes.get("net_turns_p50")),
        "mean_net_turns": _finite_or_none(episodes.get("net_turns_mean")),
        "fall_rate": _finite_or_none(episodes.get("fall_rate")),
        "osc_ratio": _finite_or_none(oscillation),
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _retention(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0.0:
        return None
    return value / baseline


def _gate(
    gates: list[dict[str, Any]],
    name: str,
    passed: bool | None,
    observed: Any,
    requirement: str,
    *,
    note: str | None = None,
) -> None:
    row = {
        "name": name,
        "pass": passed,
        "observed": observed,
        "requirement": requirement,
    }
    if note is not None:
        row["note"] = note
    gates.append(row)


def _force_stats(calibration: dict[str, Any] | None) -> dict[str, float | None]:
    if calibration is None:
        return {"mean_n": None, "window_fraction": None}
    mean_n = calibration.get("all_fingertip_force_n", {}).get("mean")
    window_fraction = calibration.get(
        "all_fingertip_force_window_distribution", {}
    ).get("fraction_in_0_5_to_4_n")
    return {
        "mean_n": _finite_or_none(mean_n),
        "window_fraction": _finite_or_none(window_fraction),
    }


def _summarize(
    *,
    cases: list[dict[str, Any]],
    oracle_reports: dict[str, dict[str, Any]],
    adapter_reports: dict[str, dict[str, Any]] | None,
    reference_report: dict[str, Any] | None,
    current_calibration: dict[str, Any] | None,
    reference_calibration: dict[str, Any] | None,
    adaptation_validation: dict[str, Any] | None,
    old_force_channel_mse: float | None,
    trained_xy_range_mm: float,
    min_reference_retention: float,
    min_in_range_retention: float,
    min_damping_retention: float,
    min_adapter_oracle_retention: float,
    max_osc_ratio: float,
    max_in_range_fall_increase: float,
    max_ood_fall_rate: float,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    oracle_metrics = {key: _metrics(value) for key, value in oracle_reports.items()}
    baseline = oracle_metrics["baseline"]
    reference = _metrics(reference_report) if reference_report is not None else None

    if reference is None:
        _gate(gates, "m4_reference_eval_available", None, None, "required")
    else:
        reference_retention = _retention(
            baseline["median_net_turns"], reference["median_net_turns"]
        )
        _gate(
            gates,
            "m4_median_vs_pre_refactor",
            reference_retention is not None
            and reference_retention >= min_reference_retention,
            reference_retention,
            f">= {min_reference_retention:.3f}",
        )
        current_fall = baseline["fall_rate"]
        reference_fall = reference["fall_rate"]
        _gate(
            gates,
            "m4_fall_vs_pre_refactor",
            current_fall is not None
            and reference_fall is not None
            and current_fall <= reference_fall + 1.0e-12,
            {"current": current_fall, "reference": reference_fall},
            "current <= reference",
        )
        reference_osc = reference["osc_ratio"]
        current_osc = baseline["osc_ratio"]
        if reference_osc is None:
            _gate(
                gates,
                "m4_oscillation_vs_pre_refactor",
                None,
                {"current": current_osc, "reference": None},
                "current <= reference",
                note="pre-refactor eval did not persist eval_osc_ratio",
            )
        else:
            _gate(
                gates,
                "m4_oscillation_vs_pre_refactor",
                current_osc is not None and current_osc <= reference_osc + 1.0e-12,
                {"current": current_osc, "reference": reference_osc},
                "current <= reference",
            )

    _gate(
        gates,
        "m4_absolute_oscillation_guardrail",
        baseline["osc_ratio"] is not None and baseline["osc_ratio"] <= max_osc_ratio,
        baseline["osc_ratio"],
        f"<= {max_osc_ratio:.3f}",
    )
    damping = oracle_metrics["damping_x4"]
    damping_retention = _retention(
        damping["median_net_turns"], baseline["median_net_turns"]
    )
    _gate(
        gates,
        "m4_rotation_damping_x4_retention",
        damping_retention is not None and damping_retention >= min_damping_retention,
        damping_retention,
        f">= {min_damping_retention:.3f}",
    )
    _gate(
        gates,
        "m4_rotation_damping_x4_fall",
        damping["fall_rate"] is not None
        and baseline["fall_rate"] is not None
        and damping["fall_rate"]
        <= baseline["fall_rate"] + max_in_range_fall_increase + 1.0e-12,
        damping["fall_rate"],
        f"<= baseline + {max_in_range_fall_increase:.3f}",
    )

    def add_bias_gates(prefix: str, metrics: dict[str, dict[str, Any]]) -> None:
        mode_baseline = metrics["baseline"]
        for case in cases:
            row = metrics[case["label"]]
            retention = _retention(
                row["median_net_turns"], mode_baseline["median_net_turns"]
            )
            in_range = (
                case["kind"] == "position"
                and case["magnitude"] <= trained_xy_range_mm + 1.0e-12
            ) or case["kind"] == "tilt"
            if in_range:
                _gate(
                    gates,
                    f"{prefix}_{case['label']}_retention",
                    retention is not None and retention >= min_in_range_retention,
                    retention,
                    f">= {min_in_range_retention:.3f}",
                )
                _gate(
                    gates,
                    f"{prefix}_{case['label']}_fall",
                    row["fall_rate"] is not None
                    and mode_baseline["fall_rate"] is not None
                    and row["fall_rate"]
                    <= mode_baseline["fall_rate"]
                    + max_in_range_fall_increase
                    + 1.0e-12,
                    row["fall_rate"],
                    f"<= baseline + {max_in_range_fall_increase:.3f}",
                )
            if case["kind"] == "position" and abs(case["magnitude"] - 8.0) < 1e-12:
                _gate(
                    gates,
                    f"{prefix}_{case['label']}_ood_progress",
                    row["median_net_turns"] is not None
                    and row["median_net_turns"] > 0.0,
                    row["median_net_turns"],
                    "> 0 net turns",
                )
                _gate(
                    gates,
                    f"{prefix}_{case['label']}_ood_fall",
                    row["fall_rate"] is not None
                    and row["fall_rate"] <= max_ood_fall_rate,
                    row["fall_rate"],
                    f"<= {max_ood_fall_rate:.3f}",
                )

    add_bias_gates("m4_oracle", oracle_metrics)

    current_force = _force_stats(current_calibration)
    reference_force = _force_stats(reference_calibration)
    if current_force["mean_n"] is None or reference_force["mean_n"] is None:
        _gate(
            gates,
            "m4_force_distribution_vs_pre_refactor",
            None,
            {"current": current_force, "reference": reference_force},
            "mean force <= 90% reference or window concentration <= 90% reference",
        )
    else:
        mean_improved = current_force["mean_n"] <= 0.90 * reference_force["mean_n"]
        concentration_improved = (
            current_force["window_fraction"] is not None
            and reference_force["window_fraction"] is not None
            and current_force["window_fraction"]
            <= 0.90 * reference_force["window_fraction"]
        )
        _gate(
            gates,
            "m4_force_distribution_vs_pre_refactor",
            mean_improved or concentration_improved,
            {"current": current_force, "reference": reference_force},
            "mean force <= 90% reference or window concentration <= 90% reference",
        )

    adapter_metrics = None
    if adapter_reports is not None:
        adapter_metrics = {
            key: _metrics(value) for key, value in adapter_reports.items()
        }
        adapter_retention = _retention(
            adapter_metrics["baseline"]["median_net_turns"],
            baseline["median_net_turns"],
        )
        _gate(
            gates,
            "m5_adapter_vs_oracle_median",
            adapter_retention is not None
            and adapter_retention >= min_adapter_oracle_retention,
            adapter_retention,
            f">= {min_adapter_oracle_retention:.3f}",
        )
        add_bias_gates("m5_adapter", adapter_metrics)

        validation = adaptation_validation or {}
        latent_mse = _finite_or_none(validation.get("adapter_latent_mse"))
        _gate(
            gates,
            "m5_held_out_adapter_latent_mse",
            latent_mse is not None and latent_mse < 0.01,
            latent_mse,
            "< 0.01",
        )
        rel_pos_mse = _finite_or_none(
            validation.get("screw_relative_position_mse")
        )
        contact_mse = _finite_or_none(
            validation.get("distance_contact_score_mse")
        )
        _gate(
            gates,
            "m5_force_free_channel_metrics_reported",
            rel_pos_mse is not None and contact_mse is not None,
            {"relative_position_mse": rel_pos_mse, "contact_score_mse": contact_mse},
            "both 3-D rel_pos and 5-D contact-score MSE present",
        )
        if old_force_channel_mse is None:
            _gate(
                gates,
                "m5_force_free_channels_vs_old_force_channels",
                None,
                None,
                "both force-free channel MSE values < old force-channel MSE",
                note="old held-out force-channel MSE was not supplied",
            )
        else:
            _gate(
                gates,
                "m5_force_free_channels_vs_old_force_channels",
                rel_pos_mse is not None
                and contact_mse is not None
                and rel_pos_mse < old_force_channel_mse
                and contact_mse < old_force_channel_mse,
                {
                    "relative_position_mse": rel_pos_mse,
                    "contact_score_mse": contact_mse,
                    "old_force_channel_mse": old_force_channel_mse,
                },
                "both force-free channel MSE values < old force-channel MSE",
            )

    unavailable = [row["name"] for row in gates if row["pass"] is None]
    failed = [row["name"] for row in gates if row["pass"] is False]
    return {
        "schema_version": 1,
        "oracle_metrics": oracle_metrics,
        "adapter_metrics": adapter_metrics,
        "reference_metrics": reference,
        "adaptation_validation": adaptation_validation,
        "force_distribution": {
            "current": current_force,
            "reference": reference_force,
        },
        "gates": gates,
        "failed_gates": failed,
        "unavailable_gates": unavailable,
        "evidence_complete": not unavailable,
        "promotion_pass": not failed and not unavailable,
    }


def _eval_command(
    args: argparse.Namespace,
    *,
    report_path: Path,
    case: dict[str, Any] | None,
    adapter: bool,
    calibration_path: Path | None,
) -> list[str]:
    command = [
        args.python,
        "eval.py",
        "--task",
        TASK_ID,
        "--checkpoint",
        str(args.checkpoint),
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--success_turns",
        str(args.success_turns),
        "--json_output",
        str(report_path),
        "--headless",
    ]
    if adapter:
        command.extend(
            ["--deploy_eval", "--adapter_checkpoint", str(args.adapter_checkpoint)]
        )
    if case is not None:
        if case.get("rot_damping_scale") is not None:
            command.extend(
                ["--rot_damping_scale", str(case["rot_damping_scale"])]
            )
        command.extend(
            ["--root_pos_bias_mm", *(str(value) for value in case["position_mm"])]
        )
        command.extend(
            ["--root_rpy_bias_deg", *(str(value) for value in case["rpy_deg"])]
        )
    if calibration_path is not None:
        command.extend(["--calibration_output", str(calibration_path)])
    return command


def _run_eval(
    args: argparse.Namespace,
    *,
    mode: str,
    label: str,
    case: dict[str, Any] | None,
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    report_path = args.output / f"{mode}_{label}.json"
    log_path = args.output / f"{mode}_{label}.log"
    if report_path.is_file() and not args.overwrite:
        return json.loads(report_path.read_text())
    command = _eval_command(
        args,
        report_path=report_path,
        case=case,
        adapter=mode == "adapter",
        calibration_path=calibration_path,
    )
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_minutes * 60.0,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode}/{label} eval exited {completed.returncode}; see {log_path}"
        )
    if not report_path.is_file():
        raise RuntimeError(f"{mode}/{label} wrote no report: {report_path}")
    return json.loads(report_path.read_text())


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return None
    return json.loads(resolved.read_text())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.adapter_checkpoint is not None:
        args.adapter_checkpoint = args.adapter_checkpoint.expanduser().resolve()
        if not args.adapter_checkpoint.is_file():
            raise FileNotFoundError(args.adapter_checkpoint)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.trained_xy_range_mm <= 0.0:
        raise ValueError("--trained-xy-range-mm must be positive")

    cases = _bias_cases(args.bias_magnitudes_mm, args.tilt_bias_deg)
    damping_case = {
        "label": "damping_x4",
        "kind": "damping",
        "magnitude": 4.0,
        "position_mm": [0.0, 0.0, 0.0],
        "rpy_deg": [0.0, 0.0, 0.0],
        "rot_damping_scale": 4.0,
    }
    if args.plan_only:
        plan = {
            "oracle": ["baseline", "damping_x4", *(case["label"] for case in cases)],
            "adapter": (
                ["baseline", *(case["label"] for case in cases)]
                if args.adapter_checkpoint is not None
                else []
            ),
            "num_envs": args.num_envs,
            "trained_xy_range_mm": args.trained_xy_range_mm,
        }
        print(json.dumps(plan, indent=2))
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output / "oracle_force_calibration.json"
    oracle_reports = {
        "baseline": _run_eval(
            args,
            mode="oracle",
            label="baseline",
            case=None,
            calibration_path=calibration_path,
        ),
        "damping_x4": _run_eval(
            args,
            mode="oracle",
            label="damping_x4",
            case=damping_case,
        ),
    }
    for case in cases:
        oracle_reports[case["label"]] = _run_eval(
            args, mode="oracle", label=case["label"], case=case
        )

    adapter_reports = None
    adaptation_validation_path = args.adaptation_validation
    if args.adapter_checkpoint is not None:
        adapter_reports = {
            "baseline": _run_eval(
                args, mode="adapter", label="baseline", case=None
            )
        }
        for case in cases:
            adapter_reports[case["label"]] = _run_eval(
                args, mode="adapter", label=case["label"], case=case
            )
        if adaptation_validation_path is None:
            adaptation_validation_path = (
                args.adapter_checkpoint.parent / "adaptation_validation.json"
            )

    result = _summarize(
        cases=cases,
        oracle_reports=oracle_reports,
        adapter_reports=adapter_reports,
        reference_report=_load_optional(args.reference_report),
        current_calibration=_load_optional(calibration_path),
        reference_calibration=_load_optional(args.reference_calibration),
        adaptation_validation=_load_optional(adaptation_validation_path),
        old_force_channel_mse=args.old_force_channel_mse,
        trained_xy_range_mm=args.trained_xy_range_mm,
        min_reference_retention=args.min_reference_retention,
        min_in_range_retention=args.min_in_range_retention,
        min_damping_retention=args.min_damping_retention,
        min_adapter_oracle_retention=args.min_adapter_oracle_retention,
        max_osc_ratio=args.max_osc_ratio,
        max_in_range_fall_increase=args.max_in_range_fall_increase,
        max_ood_fall_rate=args.max_ood_fall_rate,
    )
    result.update(
        {
            "task": TASK_ID,
            "checkpoint": str(args.checkpoint),
            "adapter_checkpoint": (
                str(args.adapter_checkpoint)
                if args.adapter_checkpoint is not None
                else None
            ),
            "num_envs": args.num_envs,
            "seed": args.seed,
            "trained_xy_range_mm": args.trained_xy_range_mm,
            "bias_cases": cases,
        }
    )
    report_path = args.output / "force_free_policy_evaluation.json"
    _atomic_json(report_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nEvaluation report: {report_path}")
    if result["promotion_pass"] or args.report_only:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
