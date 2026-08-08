"""Pure-Python contracts for bounded top-down training promotion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools/run_linker_topdown_training.py"
SPEC = importlib.util.spec_from_file_location("topdown_training_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 1_000_001)
    return path


def _report(
    path: Path,
    *,
    bucket_success: float = 0.95,
    wrong_surface_force: float = 0.02,
) -> Path:
    rows = [
        {
            "diameter_mm": diameter,
            "count": 32,
            "net_turns_mean": 3.0,
            "fall_rate": 0.03,
            "success_rate": bucket_success if diameter == 64.0 else 0.95,
        }
        for diameter in (60.0, 64.0, 68.0)
    ]
    payload = {
        "step_metrics": {
            "eval_wrong_surface_force": {"mean": wrong_surface_force},
        },
        "wrong_surface_distribution": {
            "fraction_above_0_05_n": 0.01,
            "p99_n": 1.0,
            "max_n": 20.0,
        },
        "episodes": {
            "count": 96,
            "net_turns_mean": 3.0,
            "fall_rate": 0.03,
            "success_rate": 0.95,
            "geometry_buckets": rows,
        }
    }
    path.write_text(json.dumps(payload))
    return path


def test_select_stage1_checkpoint_uses_canonical_best_not_last_epoch(tmp_path):
    run = tmp_path / "run"
    best = _checkpoint(
        run / "linker_l20_screwdriver_rotation_00-00-00-00/nn"
        / RUNNER.BEST_CHECKPOINT_NAME
    )
    final = _checkpoint(
        run / "linker_l20_screwdriver_rotation_00-00-00-00/nn"
        / "linker_l20_screwdriver_rotation_phase3.pth"
    )

    selected, phase3 = RUNNER._select_stage1_checkpoint(run)

    assert selected == best
    assert phase3 == final


def test_select_stage1_checkpoint_requires_canonical_best(tmp_path):
    _checkpoint(tmp_path / "nn/linker_l20_screwdriver_rotation_phase3.pth")

    with pytest.raises(FileNotFoundError, match="canonical best"):
        RUNNER._select_stage1_checkpoint(tmp_path)


def test_quality_gate_accepts_aggregate_and_every_diameter_bucket(tmp_path):
    report = _report(tmp_path / "pass.json")

    parsed = RUNNER._assert_quality_gate(
        report,
        label="deploy_dr",
        min_net_turns=3.0,
        max_fall_rate=0.05,
        min_success_rate=0.90,
        max_wrong_surface_force=0.05,
    )

    assert parsed["episodes"]["count"] == 96


def test_quality_gate_rejects_a_single_weak_diameter_bucket(tmp_path):
    report = _report(tmp_path / "fail.json", bucket_success=0.80)

    with pytest.raises(RuntimeError, match=r"64 mm bucket success rate"):
        RUNNER._assert_quality_gate(
            report,
            label="deploy_dr",
            min_net_turns=2.0,
            max_fall_rate=0.05,
            min_success_rate=0.90,
            max_wrong_surface_force=0.05,
        )


def test_quality_gate_rejects_wrong_surface_force(tmp_path):
    report = _report(tmp_path / "wrong_surface.json", wrong_surface_force=0.06)

    with pytest.raises(RuntimeError, match=r"wrong-surface mean force"):
        RUNNER._assert_quality_gate(
            report,
            label="deploy_dr",
            min_net_turns=2.0,
            max_fall_rate=0.05,
            min_success_rate=0.90,
            max_wrong_surface_force=0.05,
        )



def test_quality_gate_rejects_wrong_surface_tail(tmp_path):
    report = _report(tmp_path / "wrong_surface_tail.json")
    payload = json.loads(report.read_text())
    payload["wrong_surface_distribution"]["p99_n"] = 2.0
    report.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=r"wrong-surface p99"):
        RUNNER._assert_quality_gate(
            report,
            label="deploy_dr",
            min_net_turns=2.0,
            max_fall_rate=0.05,
            min_success_rate=0.90,
            max_wrong_surface_force=0.10,
            max_wrong_surface_p99=1.5,
        )


def test_stage2_command_exports_continuous_rollout_mode(tmp_path):
    args = SimpleNamespace(
        python="python", stage2_num_envs=256, stage2_iters=500,
        stage2_rollout_steps=128, stage2_continuous_rollouts=True, seed=42,
    )
    command = RUNNER._stage2_command(args, tmp_path, "stage1.pth")

    assert command[command.index("--num_envs") + 1] == "256"
    assert command[command.index("--adapt_rollout_steps") + 1] == "128"
    assert "--adapt_continuous_rollouts" in command
    assert "--no-adapt_onpolicy" not in command

def test_stage1_command_exports_stable_contact_policy_initialization(tmp_path):
    args = SimpleNamespace(
        python="python",
        stage1_num_envs=4096,
        stage1_max_epochs=1526,
        stage1_init_global_steps=0,
        stage1_resume_checkpoint=None,
        stage1_ppo_log_std=-2.525728644,
        stage1_ppo_sigma_override=None,
        stage1_ppo_entropy_coef=0.002,
        stage1_score_to_win=1.0e12,
        stage1_ppo_learning_rate=1.0e-4,
        stage1_ppo_lr_schedule="identity",
        stage1_ppo_mini_epochs=4,
        stage1_zero_mu_init=True,
        seed=42,
    )

    command = RUNNER._stage1_command(args, tmp_path)

    assert command.count("--max_epochs") == 1
    assert command[command.index("--ppo_sigma_init") + 1] == "-2.525728644"
    assert command[command.index("--ppo_entropy_coef") + 1] == "0.002"
    assert command[command.index("--ppo_score_to_win") + 1] == "1000000000000.0"
    assert command[command.index("--ppo_learning_rate") + 1] == "0.0001"
    assert command[command.index("--ppo_lr_schedule") + 1] == "identity"
    assert command[command.index("--ppo_mini_epochs") + 1] == "4"
    assert "--ppo_zero_mu_init" in command


def test_stage1_resume_command_and_sample_contract(tmp_path):
    checkpoint = _checkpoint(tmp_path / "safe_epoch_160.pth")
    args = RUNNER._parser().parse_args(
        [
            "--stage1-resume-checkpoint",
            str(checkpoint),
            "--stage1-resume-epoch",
            "160",
            "--stage1-init-global-steps",
            "41943040",
            "--stage1-ppo-learning-rate",
            "1e-5",
            "--stage1-ppo-sigma-override",
            "-2.302585093",
        ]
    )

    RUNNER._validate_stage1_schedule(args)
    command = RUNNER._stage1_command(args, tmp_path / "out")

    assert command[command.index("--checkpoint") + 1] == str(
        checkpoint.resolve()
    )
    assert command[command.index("--ppo_learning_rate") + 1] == "1e-05"
    assert command[command.index("--ppo_sigma_override") + 1] == "-2.302585093"
    assert (
        args.stage1_init_global_steps
        + args.stage1_num_envs
        * RUNNER.STAGE1_HORIZON
        * (args.stage1_max_epochs - args.stage1_resume_epoch)
        == 300_154_880
    )


def test_stage1_sigma_override_requires_resume_checkpoint():
    args = RUNNER._parser().parse_args(
        ["--stage1-ppo-sigma-override", "-2.302585093"]
    )

    with pytest.raises(ValueError, match="requires --stage1-resume-checkpoint"):
        RUNNER._validate_stage1_schedule(args)


def test_stage1_resume_rejects_mismatched_curriculum_steps(tmp_path):
    checkpoint = _checkpoint(tmp_path / "safe_epoch_160.pth")
    args = RUNNER._parser().parse_args(
        [
            "--stage1-resume-checkpoint",
            str(checkpoint),
            "--stage1-resume-epoch",
            "160",
            "--stage1-init-global-steps",
            "40000000",
        ]
    )

    with pytest.raises(ValueError, match="must exactly equal"):
        RUNNER._validate_stage1_schedule(args)


def test_stage1_production_defaults_preserve_samples_at_validated_parallelism():
    args = RUNNER._parser().parse_args([])

    assert args.stage1_num_envs == RUNNER.STAGE1_DEFAULT_NUM_ENVS
    assert args.stage1_num_envs == RUNNER.STAGE1_VALIDATED_MAX_ENVS
    assert args.stage1_max_epochs == RUNNER.STAGE1_DEFAULT_MAX_EPOCHS
    assert (
        args.stage1_num_envs
        * RUNNER.STAGE1_HORIZON
        * args.stage1_max_epochs
        == 300_154_880
    )
    RUNNER._validate_stage1_schedule(args)


def test_production_quality_gate_defaults_require_three_turns():
    args = RUNNER._parser().parse_args([])

    assert args.gate_success_turns == 3.0
    assert args.gate_min_net_turns == 3.0


def test_stage1_rejects_unvalidated_parallelism_without_explicit_override():
    args = RUNNER._parser().parse_args(["--stage1-num-envs", "16384"])

    with pytest.raises(ValueError, match="measured multi-asset solver ceiling"):
        RUNNER._validate_stage1_schedule(args)

    args.allow_unvalidated_stage1_parallelism = True
    RUNNER._validate_stage1_schedule(args)


def test_deploy_dry_run_binds_l20_calibration_overlay(tmp_path):
    command = RUNNER._deploy_command(
        SimpleNamespace(python="python"),
        tmp_path,
        tmp_path / "deploy.pth",
    )

    assert command[command.index("--hand-joint") + 1] == "L20"
    calib = Path(command[command.index("--calib") + 1])
    assert calib == ROOT / "linker_calib_deploy.json"
