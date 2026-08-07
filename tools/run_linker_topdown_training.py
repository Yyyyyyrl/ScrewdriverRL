#!/usr/bin/env python3
"""Run the approved top-down Stage 1 + Stage 2 schedule with hard bounds.

This runner is intentionally separate from train.py: it does not change
training semantics. It adds unattended-run safety around the normal entrypoint:

* Stage 1 has both --max_epochs and a wall-clock watchdog.
* Stage 2 has both --adapt_iters and a wall-clock watchdog.
* log progress, checkpoint creation, stalls, NaN/OOM/fatal errors, and PIDs are
  persisted to training_status.json.
* Stage 2 starts from the canonical best PPO checkpoint, after Stage 1 proves it
  reached Phase 3 and the oracle policy passes DR and nominal simulation gates.
* the final deploy bundle must pass the same simulation gates and a bounded
  offline deployment load test before the run is marked complete.

The production defaults respect the measured multi-asset capacity knee:
8,192 environments x 32 horizon x 1,145 epochs = 300,154,880 Stage-1
samples, followed by the repository's full 500 x 512-step Stage-2 schedule.
The next tested point (16,384 environments) was killed by the host OOM
killer before PPO startup.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown"
STAGE1_HORIZON = 32
STAGE1_VALIDATED_MAX_ENVS = 8_192
STAGE1_DEFAULT_NUM_ENVS = 8_192
STAGE1_DEFAULT_MAX_EPOCHS = 1_145
STAGE1_PARALLELISM_EVIDENCE = (
    REPO_ROOT
    / (
        "artifacts/linker_l20_screwdriver_topdown/pip108_20260722/"
        "scale_sweep_role_neutral_20260723.json"
    )
)
FATAL_RE = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"Patch buffer overflow detected|Segmentation fault|\bnan\b",
    re.IGNORECASE,
)
EPOCH_RE = re.compile(r"epoch:\s*(\d+)/(\d+)", re.IGNORECASE)
PHASE_RE = re.compile(r"Curriculum Phase\s+(\d+)/(\d+)", re.IGNORECASE)
ADAPT_RE = re.compile(r"Stage 2 .*?Iter\s+(\d+)/(\d+)", re.IGNORECASE)
BEST_CHECKPOINT_NAME = "linker_l20_screwdriver_rotation.pth"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage1-num-envs",
        type=int,
        default=STAGE1_DEFAULT_NUM_ENVS,
    )
    parser.add_argument(
        "--stage1-max-epochs",
        type=int,
        default=STAGE1_DEFAULT_MAX_EPOCHS,
    )
    parser.add_argument(
        "--allow-unvalidated-stage1-parallelism",
        action="store_true",
        help=(
            "Explicitly bypass the measured 8,192-env production maximum. "
            "Unsafe for production promotion."
        ),
    )
    parser.add_argument("--stage1-init-global-steps", type=int, default=0)
    parser.add_argument(
        "--stage1-resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume Stage 1 from this RL-Games checkpoint. Pair with "
            "--stage1-resume-epoch and an exactly matching "
            "--stage1-init-global-steps value so curriculum state and the "
            "remaining sample contract stay auditable."
        ),
    )
    parser.add_argument(
        "--stage1-resume-epoch",
        type=int,
        default=0,
        help="Epoch stored in --stage1-resume-checkpoint (0 for a fresh run).",
    )
    parser.add_argument(
        "--stage1-ppo-log-std",
        type=float,
        default=math.log(0.30),
        help="Initial actor log standard deviation (default std=0.30).",
    )
    parser.add_argument(
        "--stage1-ppo-sigma-override",
        type=float,
        default=None,
        help=(
            "Optional actor log standard deviation applied after restoring a "
            "Stage-1 checkpoint. Intended for auditable low-noise fine-tuning."
        ),
    )
    parser.add_argument(
        "--stage1-ppo-entropy-coef",
        type=float,
        default=0.002,
        help="Actor entropy coefficient used to preserve gait exploration.",
    )
    parser.add_argument(
        "--stage1-score-to-win",
        type=float,
        default=1.0e12,
        help=(
            "RL-Games early-stop score. The fixed-sample runner defaults to an "
            "intentionally unreachable value so the requested sample contract wins."
        ),
    )
    parser.add_argument(
        "--stage1-ppo-learning-rate",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--stage1-ppo-lr-schedule",
        choices=("adaptive", "identity", "linear"),
        default="identity",
    )
    parser.add_argument("--stage1-ppo-mini-epochs", type=int, default=4)
    parser.add_argument(
        "--stage1-zero-mu-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the delta-action actor mean at zero.",
    )
    parser.add_argument("--stage1-timeout-hours", type=float, default=12.0)
    parser.add_argument("--stage2-num-envs", type=int, default=64)
    parser.add_argument("--stage2-iters", type=int, default=500)
    parser.add_argument("--stage2-rollout-steps", type=int, default=512)
    parser.add_argument("--stage2-continuous-rollouts", action="store_true")
    parser.add_argument("--stage2-timeout-hours", type=float, default=7.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stall-minutes", type=float, default=15.0)
    parser.add_argument("--checkpoint-grace-minutes", type=float, default=20.0)
    parser.add_argument("--quality-gate-num-envs", type=int, default=96)
    parser.add_argument("--quality-gate-timeout-minutes", type=float, default=20.0)
    parser.add_argument(
        "--gate-success-turns",
        type=float,
        default=3.0,
        help="minimum authorization-qualified turns for a successful episode",
    )
    parser.add_argument("--gate-min-net-turns", type=float, default=3.0)
    parser.add_argument("--gate-max-fall-rate", type=float, default=0.05)
    parser.add_argument("--gate-min-success-rate", type=float, default=0.85)
    parser.add_argument("--nominal-max-fall-rate", type=float, default=0.0)
    parser.add_argument("--nominal-min-success-rate", type=float, default=0.90)
    parser.add_argument(
        "--gate-max-wrong-surface-force",
        type=float,
        default=0.10,
        help="maximum allowed mean non-fingertip/wrong-surface force in eval",
    )
    parser.add_argument(
        "--gate-max-wrong-surface-fraction",
        type=float,
        default=0.015,
        help="maximum fraction of samples above 0.05 N wrong-surface force",
    )
    parser.add_argument(
        "--gate-max-wrong-surface-p99",
        type=float,
        default=1.5,
        help="maximum p99 wrong-surface force in N",
    )
    parser.add_argument(
        "--gate-max-wrong-surface-peak",
        type=float,
        default=30.0,
        help="maximum observed wrong-surface impulse peak in N",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the resolved bounded schedule without creating files or training.",
    )
    parser.add_argument(
        "--skip-deploy-dry-run",
        action="store_true",
        help="Skip the final 10-tick offline deploy-bundle load test.",
    )
    parser.add_argument(
        "--skip-quality-gates",
        action="store_true",
        help="Debug only: skip oracle/deploy DR and nominal promotion gates.",
    )
    return parser


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / TASK_ID / f"topdown_release_{stamp}"


def _validate_stage1_schedule(args: argparse.Namespace) -> None:
    if args.stage1_num_envs < 1:
        raise ValueError("--stage1-num-envs must be positive")
    if args.stage1_max_epochs < 1:
        raise ValueError("--stage1-max-epochs must be positive")
    if not math.isfinite(args.stage1_score_to_win) or args.stage1_score_to_win <= 0.0:
        raise ValueError("--stage1-score-to-win must be finite and positive")
    checkpoint = args.stage1_resume_checkpoint
    resume_epoch = int(args.stage1_resume_epoch)
    sigma_override = args.stage1_ppo_sigma_override
    if sigma_override is not None:
        if checkpoint is None:
            raise ValueError(
                "--stage1-ppo-sigma-override requires "
                "--stage1-resume-checkpoint"
            )
        if not math.isfinite(sigma_override):
            raise ValueError("--stage1-ppo-sigma-override must be finite")
    if checkpoint is None and resume_epoch != 0:
        raise ValueError(
            "--stage1-resume-epoch requires --stage1-resume-checkpoint"
        )
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Stage 1 resume checkpoint does not exist: {checkpoint}"
            )
        if resume_epoch <= 0:
            raise ValueError(
                "--stage1-resume-checkpoint requires a positive "
                "--stage1-resume-epoch"
            )
        if resume_epoch >= args.stage1_max_epochs:
            raise ValueError(
                "--stage1-resume-epoch must be less than "
                "--stage1-max-epochs"
            )
        expected_steps = (
            resume_epoch * args.stage1_num_envs * STAGE1_HORIZON
        )
        if args.stage1_init_global_steps != expected_steps:
            raise ValueError(
                "--stage1-init-global-steps must exactly equal "
                "resume_epoch * num_envs * horizon "
                f"({expected_steps:,})"
            )
    if (
        args.stage1_num_envs > STAGE1_VALIDATED_MAX_ENVS
        and not args.allow_unvalidated_stage1_parallelism
    ):
        raise ValueError(
            f"Stage 1 requested {args.stage1_num_envs} envs, but the measured "
            f"multi-asset solver ceiling is {STAGE1_VALIDATED_MAX_ENVS}; "
            "use --allow-unvalidated-stage1-parallelism only for explicit "
            "non-production experiments"
        )


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30.0)


def _checkpoints(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*.pth")
            if path.is_file() and path.stat().st_size > 0
        ),
        key=lambda path: path.stat().st_mtime,
    )


def _select_stage1_checkpoint(root: Path) -> tuple[Path, Path]:
    """Return ``(best, final_phase3)`` without promoting a regressed last epoch."""
    checkpoints = _checkpoints(root)
    phase3 = [path for path in checkpoints if path.name.endswith("_phase3.pth")]
    if not phase3:
        raise FileNotFoundError(
            "Stage 1 completed without a final Phase-3 checkpoint"
        )
    best = [path for path in checkpoints if path.name == BEST_CHECKPOINT_NAME]
    if not best:
        raise FileNotFoundError(
            f"Stage 1 completed without canonical best checkpoint {BEST_CHECKPOINT_NAME}"
        )
    selected = best[-1]
    final = phase3[-1]
    for label, path in (("best", selected), ("final Phase-3", final)):
        if path.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Stage 1 {label} checkpoint is unexpectedly small: {path}"
            )
    return selected, final


def _assert_quality_gate(
    report_path: Path,
    *,
    label: str,
    min_net_turns: float,
    max_fall_rate: float,
    min_success_rate: float,
    max_wrong_surface_force: float,
    max_wrong_surface_fraction: float = float("inf"),
    max_wrong_surface_p99: float = float("inf"),
    max_wrong_surface_peak: float = float("inf"),
) -> dict:
    """Validate aggregate and per-diameter episode results from ``eval.py``."""
    report = json.loads(report_path.read_text())
    episodes = report.get("episodes", {})
    if int(episodes.get("count", 0)) <= 0:
        raise RuntimeError(f"{label} completed no full episodes")

    failures: list[str] = []
    wrong_surface = (
        report.get("step_metrics", {})
        .get("eval_wrong_surface_force", {})
        .get("mean")
    )
    if wrong_surface is None:
        failures.append("wrong-surface force metric is missing")
    elif float(wrong_surface) > max_wrong_surface_force + 1.0e-12:
        failures.append(
            f"wrong-surface mean force {float(wrong_surface):.4f} N > "
            f"{max_wrong_surface_force:.4f} N"
        )
    wrong_distribution = report.get("wrong_surface_distribution", {})
    for key, limit, label_text, fmt in (
        (
            "fraction_above_0_05_n",
            max_wrong_surface_fraction,
            "wrong-surface fraction above 0.05 N",
            ".3%",
        ),
        ("p99_n", max_wrong_surface_p99, "wrong-surface p99", ".4f"),
        ("max_n", max_wrong_surface_peak, "wrong-surface peak", ".4f"),
    ):
        value = wrong_distribution.get(key)
        if value is None:
            failures.append(f"{label_text} metric is missing")
        elif float(value) > limit + 1.0e-12:
            suffix = "" if key == "fraction_above_0_05_n" else " N"
            failures.append(
                f"{label_text} {format(float(value), fmt)}{suffix} > "
                f"{format(limit, fmt)}{suffix}"
            )

    def check(row: dict, scope: str) -> None:
        net = float(row["net_turns_mean"])
        fall = float(row["fall_rate"])
        success = float(row["success_rate"])
        if net < min_net_turns:
            failures.append(
                f"{scope} net turns {net:.3f} < {min_net_turns:.3f}"
            )
        if fall > max_fall_rate + 1e-12:
            failures.append(
                f"{scope} fall rate {fall:.3%} > {max_fall_rate:.3%}"
            )
        if success + 1e-12 < min_success_rate:
            failures.append(
                f"{scope} success rate {success:.3%} < {min_success_rate:.3%}"
            )

    check(episodes, "aggregate")
    buckets = episodes.get("geometry_buckets", [])
    if not buckets:
        failures.append("no per-diameter episode buckets were reported")
    for bucket in buckets:
        diameter = bucket.get("diameter_mm")
        scope = f"{float(diameter):g} mm bucket" if diameter is not None else "unknown bucket"
        check(bucket, scope)
    if failures:
        raise RuntimeError(f"{label} quality gate failed: " + "; ".join(failures))
    return report


def _update_progress(status: dict, chunk: str) -> None:
    for key, regex in (
        ("stage1_epoch", EPOCH_RE),
        ("curriculum_phase", PHASE_RE),
        ("stage2_iteration", ADAPT_RE),
    ):
        matches = list(regex.finditer(chunk))
        if matches:
            status[key] = [int(value) for value in matches[-1].groups()]


def _run_logged(
    *,
    label: str,
    command: list[str],
    log_path: Path,
    status_path: Path,
    status: dict,
    timeout_seconds: float,
    poll_seconds: float,
    stall_seconds: float,
    checkpoint_dir: Path | None = None,
    checkpoint_grace_seconds: float = 0.0,
) -> None:
    start_wall = time.time()
    start_mono = time.monotonic()
    tail_offset = 0
    last_log_change = start_mono
    fatal_match: str | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("wb") as log:
        child_env = os.environ.copy()
        # Python switches stdout to block buffering when redirected to a file.
        # Long Isaac epochs can then make a healthy run look silent for longer
        # than the watchdog window.  Force line-level visibility for progress
        # parsing and fatal-pattern detection in every managed subprocess.
        child_env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status.update(
            {
                "active_stage": label,
                "active_pid": process.pid,
                "active_command": command,
                "active_log": str(log_path),
                "stage_started_unix": start_wall,
                "state": "running",
            }
        )
        _atomic_json(status_path, status)

        try:
            while True:
                now = time.monotonic()
                return_code = process.poll()
                size = log_path.stat().st_size if log_path.exists() else 0
                if size > tail_offset:
                    with log_path.open("rb") as reader:
                        reader.seek(tail_offset)
                        chunk = reader.read().decode("utf-8", errors="replace")
                    tail_offset = size
                    last_log_change = now
                    _update_progress(status, chunk)
                    match = FATAL_RE.search(chunk)
                    if match:
                        fatal_match = match.group(0)

                latest = None
                if checkpoint_dir is not None and checkpoint_dir.exists():
                    found = _checkpoints(checkpoint_dir)
                    if found:
                        latest = found[-1]
                        status["latest_checkpoint"] = str(latest)
                        status["latest_checkpoint_bytes"] = latest.stat().st_size

                elapsed = now - start_mono
                status.update(
                    {
                        "active_elapsed_seconds": round(elapsed, 1),
                        "active_log_bytes": size,
                        "last_log_change_age_seconds": round(
                            now - last_log_change, 1
                        ),
                        "last_poll_unix": time.time(),
                    }
                )
                _atomic_json(status_path, status)

                if fatal_match is not None:
                    status["fatal_pattern"] = fatal_match
                    _atomic_json(status_path, status)
                    if return_code is None:
                        _terminate_process_group(process)
                    raise RuntimeError(
                        f"{label} emitted fatal pattern: {fatal_match}"
                    )
                if return_code is not None:
                    if return_code != 0:
                        raise RuntimeError(
                            f"{label} exited with code {return_code}"
                        )
                    break
                if elapsed > timeout_seconds:
                    _terminate_process_group(process)
                    raise TimeoutError(
                        f"{label} exceeded wall-clock limit "
                        f"{timeout_seconds / 3600:.2f} h"
                    )
                if now - last_log_change > stall_seconds:
                    _terminate_process_group(process)
                    raise TimeoutError(
                        f"{label} log heartbeat stalled for "
                        f"{stall_seconds / 60:.1f} min"
                    )
                if (
                    checkpoint_dir is not None
                    and elapsed > checkpoint_grace_seconds
                    and latest is None
                ):
                    _terminate_process_group(process)
                    raise TimeoutError(
                        f"{label} wrote no checkpoint within "
                        f"{checkpoint_grace_seconds / 60:.1f} min"
                    )
                time.sleep(poll_seconds)
        except BaseException:
            _terminate_process_group(process)
            raise

    status[f"{label}_completed_unix"] = time.time()
    status[f"{label}_elapsed_seconds"] = round(
        time.monotonic() - start_mono, 1
    )
    status[f"{label}_return_code"] = 0
    _atomic_json(status_path, status)


def _stage1_command(args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        args.python,
        "train.py",
        "--task",
        TASK_ID,
        "--stage",
        "1",
        "--num_envs",
        str(args.stage1_num_envs),
        "--max_epochs",
        str(args.stage1_max_epochs),
        "--init_global_steps",
        str(args.stage1_init_global_steps),
        "--ppo_sigma_init",
        str(args.stage1_ppo_log_std),
        "--ppo_entropy_coef",
        str(args.stage1_ppo_entropy_coef),
        "--ppo_score_to_win",
        str(args.stage1_score_to_win),
        "--ppo_learning_rate",
        str(args.stage1_ppo_learning_rate),
        "--ppo_lr_schedule",
        args.stage1_ppo_lr_schedule,
        "--ppo_mini_epochs",
        str(args.stage1_ppo_mini_epochs),
        "--save_interval_steps",
        "2000000",
        "--output",
        str(output),
        "--seed",
        str(args.seed),
        "--headless",
    ]
    if args.stage1_zero_mu_init:
        command.append("--ppo_zero_mu_init")
    if args.stage1_resume_checkpoint is not None:
        command.extend(
            [
                "--checkpoint",
                str(args.stage1_resume_checkpoint.expanduser().resolve()),
            ]
        )
    if args.stage1_ppo_sigma_override is not None:
        command.extend(
            ["--ppo_sigma_override", str(args.stage1_ppo_sigma_override)]
        )
    return command


def _stage2_command(
    args: argparse.Namespace, output: Path, checkpoint: str
) -> list[str]:
    command = [
        args.python,
        "train.py",
        "--task",
        TASK_ID,
        "--stage",
        "2",
        "--num_envs",
        str(args.stage2_num_envs),
        "--checkpoint",
        checkpoint,
        "--output",
        str(output),
        "--adapt_iters",
        str(args.stage2_iters),
        "--adapt_rollout_steps",
        str(args.stage2_rollout_steps),
        "--adapt_save_interval",
        "20",
        "--stage2_phase",
        "final",
        "--seed",
        str(args.seed),
        "--headless",
    ]
    if args.stage2_continuous_rollouts:
        command.append("--adapt_continuous_rollouts")
    return command


def _deploy_command(
    args: argparse.Namespace, output: Path, deploy: Path
) -> list[str]:
    return [
        args.python,
        "-m",
        "screwdriver_rl.deploy.deploy_linker",
        "--checkpoint",
        str(deploy),
        "--hand-joint",
        "L20",
        "--calib",
        str(REPO_ROOT / "linker_calib_deploy.json"),
        "--dry-run",
        "--max-ticks",
        "10",
        "--record",
        str(output / "deploy_dry_run.csv"),
    ]


def _quality_gate_command(
    args: argparse.Namespace,
    checkpoint: Path,
    report_path: Path,
    *,
    adapter: Path | None,
    nominal: bool,
) -> list[str]:
    command = [
        args.python,
        "eval.py",
        "--task",
        TASK_ID,
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.quality_gate_num_envs),
        "--seed",
        str(args.seed),
        "--success_turns",
        str(args.gate_success_turns),
        "--json_output",
        str(report_path),
    ]
    if adapter is not None:
        command.extend(
            ["--deploy_eval", "--adapter_checkpoint", str(adapter)]
        )
    if nominal:
        command.extend(["--no_domain_rand", "--fixed_start"])
    return command


def _run_quality_gate(
    args: argparse.Namespace,
    output: Path,
    checkpoint: Path,
    status_path: Path,
    status: dict,
    *,
    prefix: str,
    adapter: Path | None,
    nominal: bool,
) -> dict:
    suffix = "nominal" if nominal else "dr"
    label = f"{prefix}_{suffix}"
    report_path = output / f"{label}.json"
    _run_logged(
        label=label,
        command=_quality_gate_command(
            args,
            checkpoint,
            report_path,
            adapter=adapter,
            nominal=nominal,
        ),
        log_path=output / f"{label}.log",
        status_path=status_path,
        status=status,
        timeout_seconds=args.quality_gate_timeout_minutes * 60.0,
        poll_seconds=min(args.poll_seconds, 5.0),
        stall_seconds=min(args.stall_minutes * 60.0, 300.0),
    )
    report = _assert_quality_gate(
        report_path,
        label=label,
        min_net_turns=args.gate_min_net_turns,
        max_fall_rate=(
            args.nominal_max_fall_rate
            if nominal
            else args.gate_max_fall_rate
        ),
        min_success_rate=(
            args.nominal_min_success_rate
            if nominal
            else args.gate_min_success_rate
        ),
        max_wrong_surface_force=args.gate_max_wrong_surface_force,
        max_wrong_surface_fraction=args.gate_max_wrong_surface_fraction,
        max_wrong_surface_p99=args.gate_max_wrong_surface_p99,
        max_wrong_surface_peak=args.gate_max_wrong_surface_peak,
    )
    status.setdefault("quality_gates", {})[label] = report["episodes"]
    _atomic_json(status_path, status)
    return report


def main() -> int:
    args = _parser().parse_args()
    _validate_stage1_schedule(args)
    output = (args.output or _default_output()).resolve()
    stage1_samples = (
        args.stage1_num_envs * STAGE1_HORIZON * args.stage1_max_epochs
    )
    stage1_remaining_samples = (
        args.stage1_num_envs
        * STAGE1_HORIZON
        * (args.stage1_max_epochs - args.stage1_resume_epoch)
    )
    plan = {
        "task": TASK_ID,
        "output": str(output),
        "stage1_samples": stage1_samples,
        "stage1_parallelism_validation": {
            "requested_num_envs": args.stage1_num_envs,
            "validated_max_num_envs": STAGE1_VALIDATED_MAX_ENVS,
            "within_validated_limit": (
                args.stage1_num_envs <= STAGE1_VALIDATED_MAX_ENVS
            ),
            "explicit_unvalidated_override": (
                args.allow_unvalidated_stage1_parallelism
            ),
            "evidence": str(STAGE1_PARALLELISM_EVIDENCE),
            "first_observed_failing_num_envs": 16_384,
            "first_observed_failure": "host_oom_kill_before_ppo_startup",
        },
        "stage1_init_global_steps": args.stage1_init_global_steps,
        "stage1_resume_checkpoint": (
            str(args.stage1_resume_checkpoint.expanduser().resolve())
            if args.stage1_resume_checkpoint is not None
            else None
        ),
        "stage1_resume_epoch": args.stage1_resume_epoch,
        "stage1_ppo_sigma_override": args.stage1_ppo_sigma_override,
        "stage1_remaining_samples": stage1_remaining_samples,
        "stage1_effective_target_samples": (
            args.stage1_init_global_steps + stage1_remaining_samples
        ),
        "stage1_command": _stage1_command(args, output),
        "stage1_wall_limit_hours": args.stage1_timeout_hours,
        "stage2_iterations": args.stage2_iters,
        "stage2_rollout_steps": args.stage2_rollout_steps,
        "stage2_continuous_rollouts": args.stage2_continuous_rollouts,
        "stage2_command_template": _stage2_command(
            args, output, "<stage1_best_checkpoint>"
        ),
        "stage2_wall_limit_hours": args.stage2_timeout_hours,
        "fatal_patterns": FATAL_RE.pattern,
        "stall_minutes": args.stall_minutes,
        "checkpoint_grace_minutes": args.checkpoint_grace_minutes,
        "quality_gate_config": {
            "enabled": not args.skip_quality_gates,
            "num_envs": args.quality_gate_num_envs,
            "min_net_turns": args.gate_min_net_turns,
            "success_turns": args.gate_success_turns,
            "dr_max_fall_rate": args.gate_max_fall_rate,
            "dr_min_success_rate": args.gate_min_success_rate,
            "nominal_max_fall_rate": args.nominal_max_fall_rate,
            "nominal_min_success_rate": args.nominal_min_success_rate,
            "max_wrong_surface_force_n": args.gate_max_wrong_surface_force,
            "max_wrong_surface_fraction_above_0_05_n": (
                args.gate_max_wrong_surface_fraction
            ),
            "max_wrong_surface_p99_n": args.gate_max_wrong_surface_p99,
            "max_wrong_surface_peak_n": args.gate_max_wrong_surface_peak,
        },
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0

    if output.exists():
        raise FileExistsError(
            f"refusing to reuse training output directory: {output}"
        )
    output.mkdir(parents=True)
    status_path = output / "training_status.json"
    status = {
        **plan,
        "state": "starting",
        "runner_pid": os.getpid(),
        "started_unix": time.time(),
    }
    _atomic_json(status_path, status)

    try:
        _run_logged(
            label="stage1",
            command=_stage1_command(args, output),
            log_path=output / "stage1.log",
            status_path=status_path,
            status=status,
            timeout_seconds=args.stage1_timeout_hours * 3600.0,
            poll_seconds=args.poll_seconds,
            stall_seconds=args.stall_minutes * 60.0,
            checkpoint_dir=output,
            checkpoint_grace_seconds=(
                args.checkpoint_grace_minutes * 60.0
            ),
        )
        stage1_checkpoint, stage1_final_checkpoint = (
            _select_stage1_checkpoint(output)
        )
        status["stage1_checkpoint"] = str(stage1_checkpoint)
        status["stage1_final_checkpoint"] = str(stage1_final_checkpoint)
        _atomic_json(status_path, status)

        if not args.skip_quality_gates:
            for nominal in (False, True):
                _run_quality_gate(
                    args,
                    output,
                    stage1_checkpoint,
                    status_path,
                    status,
                    prefix="stage1_oracle",
                    adapter=None,
                    nominal=nominal,
                )
        _run_logged(
            label="stage2",
            command=_stage2_command(
                args, output, str(stage1_checkpoint)
            ),
            log_path=output / "stage2.log",
            status_path=status_path,
            status=status,
            timeout_seconds=args.stage2_timeout_hours * 3600.0,
            poll_seconds=args.poll_seconds,
            stall_seconds=args.stall_minutes * 60.0,
            checkpoint_dir=output / "stage2_nn",
            checkpoint_grace_seconds=(
                args.checkpoint_grace_minutes * 60.0
            ),
        )
        adapter = output / "stage2_nn" / "proprio_adapt.pth"
        deploy = output / "stage2_nn" / "deploy.pth"
        if not adapter.is_file() or adapter.stat().st_size < 10_000:
            raise RuntimeError(
                f"missing or incomplete Stage 2 adapter: {adapter}"
            )
        if not deploy.is_file() or deploy.stat().st_size < 100_000:
            raise RuntimeError(
                f"missing or incomplete deploy bundle: {deploy}"
            )
        status["stage2_adapter"] = str(adapter)
        status["deploy_bundle"] = str(deploy)
        _atomic_json(status_path, status)

        if not args.skip_quality_gates:
            for nominal in (False, True):
                _run_quality_gate(
                    args,
                    output,
                    stage1_checkpoint,
                    status_path,
                    status,
                    prefix="stage2_deploy",
                    adapter=deploy,
                    nominal=nominal,
                )
        if not args.skip_deploy_dry_run:
            _run_logged(
                label="deploy_dry_run",
                command=_deploy_command(args, output, deploy),
                log_path=output / "deploy_dry_run.log",
                status_path=status_path,
                status=status,
                timeout_seconds=120.0,
                poll_seconds=min(args.poll_seconds, 5.0),
                stall_seconds=60.0,
            )

        status.update(
            {
                "state": "complete",
                "active_stage": None,
                "active_pid": None,
                "completed_unix": time.time(),
            }
        )
        _atomic_json(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        status.update(
            {
                "state": "failed",
                "active_stage": None,
                "active_pid": None,
                "failed_unix": time.time(),
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
