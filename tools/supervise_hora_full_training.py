#!/usr/bin/env python3
"""Supervise one HORA full-training stage and resume after an Isaac hang.

This process intentionally does not decide when Stage 1 has converged or when
Stage 2 has reached its MSE stopping criterion.  It only implements the
mechanical recovery contract from ``docs/handoff-full-training-prompt.md``:

* poll the stage log every 60 seconds;
* if it has not changed for more than 120 seconds while ``train.py`` is alive,
  kill the launched training process group;
* wait until no CUDA compute process remains;
* resume Stage 1 from the newest ``last_*_ep_*.pth`` with a checkpoint-aligned
  global step for bounded runs (or the last logged step otherwise), or resume
  Stage 2 from ``proprio_adapt_last.pth``;
* optionally bound Stage 1 by a target global-step count, recomputing the
  remaining local epochs after every hang/restart.

Non-zero exits and missing/corrupt recovery artifacts stop the supervisor for
human review instead of guessing at a fix.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora"
DEFAULT_PYTHON = Path("/home/user/miniconda3/envs/env_isaaclab/bin/python")
TRAIN_PY = REPO_ROOT / "train.py"
RUN_ROOT = REPO_ROOT / "runs" / TASK_ID
STEP_RE = re.compile(r"\bStep\s+([0-9][0-9,]*)\b")
CHECKPOINT_EPOCH_RE = re.compile(r"_ep_([0-9]+)(?:_|\.pth$)")
DEFAULT_HORIZON_LENGTH = 32


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stall-seconds", type=float, default=120.0)
    parser.add_argument(
        "--gpu-release-timeout-seconds",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Stage 1: optional initial resume checkpoint. "
            "Stage 2: required final Stage-1 checkpoint."
        ),
    )
    parser.add_argument(
        "--init-global-steps",
        type=int,
        default=None,
        help="Required with a Stage-1 --checkpoint; forbidden for Stage 2.",
    )
    parser.add_argument(
        "--target-global-steps",
        type=int,
        default=None,
        help=(
            "Stage 1 only: stop after the first local epoch at or beyond this "
            "global-step count. The supervisor recomputes --max_epochs after "
            "every recovery so resumed epoch counters cannot cause over-training."
        ),
    )
    parser.add_argument(
        "--horizon-length",
        type=int,
        default=DEFAULT_HORIZON_LENGTH,
        help=(
            "Stage-1 rollout horizon used to convert global steps to local "
            "epochs (task default: 32)."
        ),
    )
    parser.add_argument("--adapt-iters", type=int, default=150)
    parser.add_argument("--adapt-save-interval", type=int, default=20)
    parser.add_argument(
        "--adapt-resume-checkpoint",
        type=Path,
        default=None,
        help="Optional initial Stage-2 adapter resume checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and paths without creating files.",
    )
    return parser


def _resolved_log(stage: int, value: Path | None) -> Path:
    if value is None:
        return REPO_ROOT / f"stage{stage}.log"
    return value.expanduser().resolve()


def _validated_args(args: argparse.Namespace) -> argparse.Namespace:
    args.output = args.output.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.log = _resolved_log(args.stage, args.log)
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.adapt_resume_checkpoint is not None:
        args.adapt_resume_checkpoint = (
            args.adapt_resume_checkpoint.expanduser().resolve()
        )

    if not args.python.is_file():
        raise FileNotFoundError(f"Python interpreter does not exist: {args.python}")
    if not TRAIN_PY.is_file():
        raise FileNotFoundError(f"training entrypoint does not exist: {TRAIN_PY}")
    if not args.output.is_relative_to(RUN_ROOT):
        raise ValueError(f"--output must be inside {RUN_ROOT}: {args.output}")
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    if args.stall_seconds <= args.poll_seconds:
        raise ValueError("--stall-seconds must be greater than --poll-seconds")
    if args.gpu_release_timeout_seconds <= 0.0:
        raise ValueError("--gpu-release-timeout-seconds must be positive")

    if args.stage == 1:
        args.num_envs = args.num_envs or 4096
        if args.adapt_resume_checkpoint is not None:
            raise ValueError("--adapt-resume-checkpoint is only valid for Stage 2")
        if (args.checkpoint is None) != (args.init_global_steps is None):
            raise ValueError(
                "Stage 1 --checkpoint and --init-global-steps must be supplied together"
            )
        if args.horizon_length < 1:
            raise ValueError("--horizon-length must be positive")
        if args.target_global_steps is not None:
            initial_global_steps = args.init_global_steps or 0
            if args.target_global_steps <= initial_global_steps:
                raise ValueError(
                    "--target-global-steps must be greater than the initial "
                    "global-step count"
                )
    else:
        args.num_envs = args.num_envs or 2048
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise FileNotFoundError(
                "Stage 2 requires --checkpoint pointing to the final Stage-1 checkpoint"
            )
        if args.init_global_steps is not None:
            raise ValueError("--init-global-steps is only valid for Stage 1")
        if args.target_global_steps is not None:
            raise ValueError("--target-global-steps is only valid for Stage 1")
        if args.adapt_resume_checkpoint is not None:
            if not args.adapt_resume_checkpoint.is_file():
                raise FileNotFoundError(
                    "Stage-2 resume checkpoint does not exist: "
                    f"{args.adapt_resume_checkpoint}"
                )
        if args.adapt_iters < 1:
            raise ValueError("--adapt-iters must be positive")
        if args.adapt_save_interval < 1:
            raise ValueError("--adapt-save-interval must be positive")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    return args


def _build_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        str(TRAIN_PY),
        "--task",
        TASK_ID,
        "--stage",
        str(args.stage),
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--output",
        str(args.output),
        "--headless",
    ]
    if args.stage == 1:
        if args.checkpoint is not None:
            command.extend(
                [
                    "--checkpoint",
                    str(args.checkpoint),
                    "--init_global_steps",
                    str(args.init_global_steps),
                ]
            )
    else:
        command.extend(
            [
                "--checkpoint",
                str(args.checkpoint),
                "--adapt_iters",
                str(args.adapt_iters),
                "--adapt_save_interval",
                str(args.adapt_save_interval),
            ]
        )
        if args.adapt_resume_checkpoint is not None:
            command.extend(
                [
                    "--adapt_resume_checkpoint",
                    str(args.adapt_resume_checkpoint),
                ]
            )
    return command


def _replace_option(command: Sequence[str], option: str, value: str) -> list[str]:
    updated = list(command)
    if option in updated:
        index = updated.index(option)
        if index + 1 >= len(updated):
            raise ValueError(f"command option has no value: {option}")
        updated[index + 1] = value
    else:
        updated.extend([option, value])
    return updated


def _stage1_resume_command(
    command: Sequence[str], checkpoint: Path, global_steps: int
) -> list[str]:
    resumed = _replace_option(command, "--checkpoint", str(checkpoint))
    return _replace_option(
        resumed,
        "--init_global_steps",
        str(global_steps),
    )


def _bounded_stage1_command(
    command: Sequence[str],
    *,
    target_global_steps: int,
    epoch_step_offset: int,
    steps_per_epoch: int,
) -> tuple[list[str], int, int]:
    """Set the absolute RL-Games epoch that reaches a global-step target."""
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if epoch_step_offset < 0:
        raise ValueError("epoch_step_offset cannot be negative")
    target_steps_from_epoch_zero = target_global_steps - epoch_step_offset
    if target_steps_from_epoch_zero <= 0:
        raise ValueError("target_global_steps must exceed epoch_step_offset")
    target_max_epoch = (
        target_steps_from_epoch_zero + steps_per_epoch - 1
    ) // steps_per_epoch
    planned_final_global_steps = (
        epoch_step_offset + target_max_epoch * steps_per_epoch
    )
    bounded = _replace_option(command, "--max_epochs", str(target_max_epoch))
    return bounded, target_max_epoch, planned_final_global_steps


def _stage2_resume_command(
    command: Sequence[str], checkpoint: Path
) -> list[str]:
    return _replace_option(
        command,
        "--adapt_resume_checkpoint",
        str(checkpoint),
    )


def _latest_stage1_checkpoint(output: Path) -> Path:
    candidates = [
        path
        for path in output.glob("*/nn/last_*_ep_*.pth")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no resumable Stage-1 checkpoint under {output}/*/nn"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()


def _stage2_resume_checkpoint(output: Path) -> Path:
    checkpoint = output / "stage2_nn" / "proprio_adapt_last.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(
            f"no resumable Stage-2 checkpoint: {checkpoint}"
        )
    return checkpoint.resolve()


def _last_logged_steps(log_path: Path) -> int:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = STEP_RE.findall(text)
    if not matches:
        raise RuntimeError(f"no 'Step <N>' progress entry found in {log_path}")
    return int(matches[-1].replace(",", ""))


def _checkpoint_epoch(checkpoint: Path) -> int:
    match = CHECKPOINT_EPOCH_RE.search(checkpoint.name)
    if match is None:
        raise ValueError(f"cannot parse epoch from Stage-1 checkpoint: {checkpoint}")
    return int(match.group(1))


def _checkpoint_epoch_step_offset(
    checkpoint: Path | None,
    *,
    checkpoint_global_steps: int,
    steps_per_epoch: int,
) -> int:
    """Recover the constant offset between RL-Games epoch frames and global steps."""
    if checkpoint_global_steps < 0:
        raise ValueError("checkpoint_global_steps cannot be negative")
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    checkpoint_epoch = _checkpoint_epoch(checkpoint) if checkpoint is not None else 0
    offset = checkpoint_global_steps - checkpoint_epoch * steps_per_epoch
    if offset < 0:
        raise ValueError(
            "checkpoint global steps are smaller than its RL-Games epoch frames"
        )
    return offset


def _checkpoint_global_steps(
    checkpoint: Path,
    *,
    epoch_step_offset: int,
    steps_per_epoch: int,
) -> int:
    """Map an absolute checkpoint epoch to its exact global-step count."""
    if epoch_step_offset < 0:
        raise ValueError("epoch_step_offset cannot be negative")
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    return epoch_step_offset + _checkpoint_epoch(checkpoint) * steps_per_epoch


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _proc_commands() -> dict[int, list[str]]:
    commands: dict[int, list[str]] = {}
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            raw = (proc_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [
            token.decode("utf-8", errors="replace")
            for token in raw.split(b"\0")
            if token
        ]
        if argv:
            commands[int(proc_dir.name)] = argv
    return commands


def _active_training_or_eval_processes() -> dict[int, list[str]]:
    active: dict[int, list[str]] = {}
    for pid, argv in _proc_commands().items():
        if pid == os.getpid():
            continue
        joined = " ".join(argv)
        if (
            str(TRAIN_PY) in joined
            or re.search(r"(^|/)train\.py(?:\s|$)", joined)
            or re.search(r"(^|/)eval\.py(?:\s|$)", joined)
        ):
            active[pid] = argv
    return active


def _gpu_compute_processes() -> dict[int, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    processes: dict[int, str] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if not fields or not fields[0].isdigit():
            continue
        processes[int(fields[0])] = line.strip()
    return processes


def _wait_for_gpu_release(timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        processes = _gpu_compute_processes()
        if not processes:
            return
        if time.monotonic() >= deadline:
            details = "; ".join(
                f"pid={pid} {description}"
                for pid, description in sorted(processes.items())
            )
            raise TimeoutError(
                "GPU compute processes remain after the training process was killed: "
                + details
            )
        time.sleep(min(5.0, max(0.1, deadline - time.monotonic())))


def _hard_kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30.0)


def _log_has_training_progress(log_path: Path) -> bool:
    """True once rl_games has printed a throughput line for this stage.

    Before that the process is still building the scene (minutes at 4096 envs)
    and writes nothing, so an mtime-only stall check would kill every startup.
    Reads the tail only — the log grows to hundreds of MB.
    """
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 2_000_000))
            tail = stream.read()
    except OSError:
        return False
    return b"fps step" in tail or b"AdaptLoss" in tail


def _write_log_marker(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[hora-supervisor] {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run(args: argparse.Namespace) -> int:
    command = _build_command(args)
    stage1_steps_per_epoch = args.num_envs * args.horizon_length
    stage1_launch_global_steps = args.init_global_steps or 0
    stage1_epoch_step_offset = _checkpoint_epoch_step_offset(
        args.checkpoint if args.stage == 1 else None,
        checkpoint_global_steps=stage1_launch_global_steps,
        steps_per_epoch=stage1_steps_per_epoch,
    )
    target_max_epoch: int | None = None
    planned_final_global_steps: int | None = None
    if args.stage == 1 and args.target_global_steps is not None:
        (
            command,
            target_max_epoch,
            planned_final_global_steps,
        ) = _bounded_stage1_command(
            command,
            target_global_steps=args.target_global_steps,
            epoch_step_offset=stage1_epoch_step_offset,
            steps_per_epoch=stage1_steps_per_epoch,
        )
    state_path = args.output / f"stage{args.stage}_supervisor_state.json"
    lock_path = args.output / f".stage{args.stage}_supervisor.lock"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "log": str(args.log),
                    "output": str(args.output),
                    "state": str(state_path),
                    "target_global_steps": args.target_global_steps,
                    "steps_per_epoch": (
                        stage1_steps_per_epoch if args.stage == 1 else None
                    ),
                    "epoch_step_offset": (
                        stage1_epoch_step_offset if args.stage == 1 else None
                    ),
                    "target_max_epoch": target_max_epoch,
                    "planned_final_global_steps": planned_final_global_steps,
                },
                indent=2,
            )
        )
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"another Stage-{args.stage} supervisor owns {lock_path}"
        ) from error

    active = _active_training_or_eval_processes()
    if active:
        details = "; ".join(
            f"pid={pid} {' '.join(argv)}" for pid, argv in sorted(active.items())
        )
        raise RuntimeError(
            "refusing to start alongside an existing train.py/eval.py process: "
            + details
        )
    gpu_processes = _gpu_compute_processes()
    if gpu_processes:
        details = "; ".join(
            f"pid={pid} {description}"
            for pid, description in sorted(gpu_processes.items())
        )
        raise RuntimeError(
            "refusing to start while the GPU has a compute process: " + details
        )

    state: dict[str, Any] = {
        "schema_version": 1,
        "task": TASK_ID,
        "stage": args.stage,
        "output": str(args.output),
        "log": str(args.log),
        "status": "starting",
        "hang_count": 0,
        "resume_events": [],
        "started_unix": time.time(),
    }
    if args.stage == 1:
        state.update(
            {
                "launch_global_steps": stage1_launch_global_steps,
                "steps_per_epoch": stage1_steps_per_epoch,
                "epoch_step_offset": stage1_epoch_step_offset,
                "target_global_steps": args.target_global_steps,
                "target_max_epoch": target_max_epoch,
                "planned_final_global_steps": planned_final_global_steps,
            }
        )
    _atomic_json(state_path, state)

    stop_requested = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        state["stop_signal"] = signum

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while True:
        launch_unix = time.time()
        _write_log_marker(
            args.log,
            "launching: " + " ".join(command),
        )
        with args.log.open("ab", buffering=0) as log_stream:
            child_env = os.environ.copy()
            child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            child_env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=child_env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            state.update(
                {
                    "status": "running",
                    "pid": process.pid,
                    "command": command,
                    "launch_unix": launch_unix,
                }
            )
            if args.stage == 1:
                state.update(
                    {
                        "launch_global_steps": stage1_launch_global_steps,
                        "target_max_epoch": target_max_epoch,
                        "planned_final_global_steps": planned_final_global_steps,
                    }
                )
            _atomic_json(state_path, state)

            while True:
                if stop_requested:
                    state["status"] = "stopping"
                    _atomic_json(state_path, state)
                    _hard_kill_process_group(process)
                    _wait_for_gpu_release(args.gpu_release_timeout_seconds)
                    state.update(
                        {
                            "status": "stopped",
                            "stopped_unix": time.time(),
                        }
                    )
                    _atomic_json(state_path, state)
                    return 130

                return_code = process.poll()
                if return_code is not None:
                    state.update(
                        {
                            "return_code": return_code,
                            "finished_unix": time.time(),
                            "status": (
                                "completed" if return_code == 0 else "failed"
                            ),
                        }
                    )
                    if (
                        return_code == 0
                        and args.stage == 1
                        and args.target_global_steps is not None
                    ):
                        state["completed_global_steps"] = planned_final_global_steps
                        state["completed_max_epoch"] = target_max_epoch
                    _atomic_json(state_path, state)
                    if return_code != 0:
                        raise RuntimeError(
                            f"Stage {args.stage} train.py exited with code "
                            f"{return_code}; inspect {args.log}"
                        )
                    return 0

                log_mtime = args.log.stat().st_mtime
                stale_seconds = max(0.0, time.time() - log_mtime)
                # Scene creation for 4096 envs emits nothing for minutes while the
                # GPU is already busy.  Judging staleness before the training loop
                # exists would kill every startup and never make progress, so the
                # stall clock only applies once rl_games has printed throughput.
                training_started = _log_has_training_progress(args.log)
                state.update(
                    {
                        "last_poll_unix": time.time(),
                        "log_mtime_unix": log_mtime,
                        "log_stale_seconds": round(stale_seconds, 1),
                        "training_started": training_started,
                    }
                )
                _atomic_json(state_path, state)
                if training_started and stale_seconds > args.stall_seconds:
                    _write_log_marker(
                        args.log,
                        f"hang detected: log unchanged for "
                        f"{stale_seconds:.1f}s; SIGKILL process group "
                        f"{process.pid}",
                    )
                    _hard_kill_process_group(process)
                    _wait_for_gpu_release(args.gpu_release_timeout_seconds)

                    if args.stage == 1:
                        checkpoint = _latest_stage1_checkpoint(args.output)
                        if args.target_global_steps is None:
                            global_steps = _last_logged_steps(args.log)
                        else:
                            global_steps = _checkpoint_global_steps(
                                checkpoint,
                                epoch_step_offset=stage1_epoch_step_offset,
                                steps_per_epoch=stage1_steps_per_epoch,
                            )
                            if global_steps >= args.target_global_steps:
                                state["hang_count"] += 1
                                state["resume_events"].append(
                                    {
                                        "detected_unix": time.time(),
                                        "checkpoint": str(checkpoint),
                                        "checkpoint_global_steps": global_steps,
                                        "stale_seconds": round(stale_seconds, 1),
                                        "target_reached": True,
                                    }
                                )
                                state.update(
                                    {
                                        "status": "completed",
                                        "completed_from_checkpoint": str(checkpoint),
                                        "completed_global_steps": global_steps,
                                        "finished_unix": time.time(),
                                    }
                                )
                                _atomic_json(state_path, state)
                                _write_log_marker(
                                    args.log,
                                    "target reached by recovery checkpoint "
                                    f"{checkpoint} at global step {global_steps:,}; "
                                    "not relaunching",
                                )
                                return 0
                        command = _stage1_resume_command(
                            command,
                            checkpoint,
                            global_steps,
                        )
                        stage1_launch_global_steps = global_steps
                        if args.target_global_steps is not None:
                            (
                                command,
                                target_max_epoch,
                                planned_final_global_steps,
                            ) = _bounded_stage1_command(
                                command,
                                target_global_steps=args.target_global_steps,
                                epoch_step_offset=stage1_epoch_step_offset,
                                steps_per_epoch=stage1_steps_per_epoch,
                            )
                        resume_event = {
                            "detected_unix": time.time(),
                            "checkpoint": str(checkpoint),
                            "init_global_steps": global_steps,
                            "stale_seconds": round(stale_seconds, 1),
                        }
                        if args.target_global_steps is not None:
                            resume_event.update(
                                {
                                    "target_max_epoch": target_max_epoch,
                                    "planned_final_global_steps": (
                                        planned_final_global_steps
                                    ),
                                }
                            )
                    else:
                        checkpoint = _stage2_resume_checkpoint(args.output)
                        command = _stage2_resume_command(command, checkpoint)
                        resume_event = {
                            "detected_unix": time.time(),
                            "checkpoint": str(checkpoint),
                            "stale_seconds": round(stale_seconds, 1),
                        }
                    state["hang_count"] += 1
                    state["resume_events"].append(resume_event)
                    state["status"] = "resuming"
                    _atomic_json(state_path, state)
                    _write_log_marker(
                        args.log,
                        "GPU released; resuming from "
                        f"{checkpoint}"
                        + (
                            f" at global step {global_steps:,}"
                            if args.stage == 1
                            else ""
                        ),
                    )
                    break

                time.sleep(args.poll_seconds)


def main() -> int:
    return _run(_validated_args(_parser().parse_args()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[hora-supervisor] ERROR: {error}", file=sys.stderr, flush=True)
        raise
