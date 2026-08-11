"""Pure-Python contracts for the HORA full-training hang supervisor."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = ROOT / "tools/supervise_hora_full_training.py"
SPEC = importlib.util.spec_from_file_location(
    "hora_full_training_supervisor",
    SUPERVISOR_PATH,
)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


def _args(tmp_path: Path, *, stage: int) -> argparse.Namespace:
    checkpoint = tmp_path / "stage1.pth"
    checkpoint.write_bytes(b"stage1")
    return argparse.Namespace(
        stage=stage,
        output=SUPERVISOR.RUN_ROOT / "full_test",
        python=Path("/usr/bin/python3"),
        log=tmp_path / f"stage{stage}.log",
        seed=42,
        num_envs=4096 if stage == 1 else 2048,
        poll_seconds=60.0,
        stall_seconds=120.0,
        gpu_release_timeout_seconds=180.0,
        checkpoint=None if stage == 1 else checkpoint,
        init_global_steps=None,
        target_global_steps=None,
        horizon_length=32,
        adapt_iters=150,
        adapt_save_interval=20,
        adapt_resume_checkpoint=None,
        dry_run=True,
    )


def test_stage1_command_matches_handoff_and_has_no_checkpoint(tmp_path):
    command = SUPERVISOR._build_command(_args(tmp_path, stage=1))

    assert command[:2] == ["/usr/bin/python3", str(ROOT / "train.py")]
    assert command[command.index("--task") + 1] == SUPERVISOR.TASK_ID
    assert command[command.index("--stage") + 1] == "1"
    assert command[command.index("--num_envs") + 1] == "4096"
    assert command[command.index("--seed") + 1] == "42"
    assert "--headless" in command
    assert "--checkpoint" not in command
    assert "--init_global_steps" not in command


# Two tests covering free-cube supervision used to live here: the supervisor was
# expected to accept "Isaac-LinkerL20-Inhand-Rotation[-Topdown]" via an args.task
# override and to default those tasks to horizon_length == 8. The revision of
# tools/supervise_hora_full_training.py that implemented this was destroyed in the
# 2026-08-09 truncation; the surviving version is pinned to the single
# Topdown-Hora task. The contract is recorded in docs/DATA_LOSS_20260809.md so it
# can be reinstated together with the tool.


def test_stage1_resume_replaces_checkpoint_and_global_steps(tmp_path):
    checkpoint_a = tmp_path / "a.pth"
    checkpoint_b = tmp_path / "b.pth"
    base = ["python", "train.py", "--checkpoint", str(checkpoint_a)]

    resumed = SUPERVISOR._stage1_resume_command(
        base,
        checkpoint_b,
        25_165_824,
    )

    assert resumed.count("--checkpoint") == 1
    assert resumed[resumed.index("--checkpoint") + 1] == str(checkpoint_b)
    assert resumed.count("--init_global_steps") == 1
    assert resumed[resumed.index("--init_global_steps") + 1] == "25165824"


def test_stage1_target_sets_absolute_epoch_budget():
    bounded, target_max_epoch, final_steps = SUPERVISOR._bounded_stage1_command(
        ["python", "train.py"],
        target_global_steps=300_810_240,
        epoch_step_offset=13_762_560,
        steps_per_epoch=131_072,
    )

    assert target_max_epoch == 2_190
    assert final_steps == 300_810_240
    assert bounded[bounded.index("--max_epochs") + 1] == "2190"


def test_checkpoint_epoch_offset_matches_ep330_handoff(tmp_path):
    checkpoint = tmp_path / "last_task_ep_330_rew_1.0.pth"

    assert SUPERVISOR._checkpoint_epoch_step_offset(
        checkpoint,
        checkpoint_global_steps=57_016_320,
        steps_per_epoch=131_072,
    ) == 13_762_560


def test_checkpoint_global_steps_uses_epoch_offset(tmp_path):
    checkpoint = tmp_path / "last_task_ep_345_rew_1.0.pth"

    assert SUPERVISOR._checkpoint_global_steps(
        checkpoint,
        epoch_step_offset=13_762_560,
        steps_per_epoch=131_072,
    ) == 58_982_400


def test_stage2_command_is_off_policy_and_resume_is_unique(tmp_path):
    command = SUPERVISOR._build_command(_args(tmp_path, stage=2))
    resume_a = tmp_path / "adapt_a.pth"
    resume_b = tmp_path / "adapt_b.pth"

    resumed = SUPERVISOR._stage2_resume_command(command, resume_a)
    resumed = SUPERVISOR._stage2_resume_command(resumed, resume_b)

    assert command[command.index("--stage") + 1] == "2"
    assert command[command.index("--num_envs") + 1] == "2048"
    assert command[command.index("--adapt_iters") + 1] == "150"
    assert command[command.index("--adapt_save_interval") + 1] == "20"
    assert "--adapt_onpolicy" not in command
    assert resumed.count("--adapt_resume_checkpoint") == 1
    assert resumed[resumed.index("--adapt_resume_checkpoint") + 1] == str(
        resume_b
    )


def test_last_logged_steps_accepts_commas_and_uses_latest(tmp_path):
    log = tmp_path / "stage1.log"
    log.write_text("Step 1,024\nnoise\nStep 25,165,824\n", encoding="utf-8")

    assert SUPERVISOR._last_logged_steps(log) == 25_165_824


def test_latest_stage1_checkpoint_uses_mtime(tmp_path):
    old = tmp_path / "run_a/nn/last_task_ep_15.pth"
    new = tmp_path / "run_b/nn/last_task_ep_30.pth"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    old.touch()
    new.touch()
    old_mtime = old.stat().st_mtime_ns
    new_mtime = max(new.stat().st_mtime_ns, old_mtime + 1_000_000)
    import os

    os.utime(new, ns=(new_mtime, new_mtime))

    assert SUPERVISOR._latest_stage1_checkpoint(tmp_path) == new.resolve()
