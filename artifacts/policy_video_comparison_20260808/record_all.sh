#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python}
REPO=$(git rev-parse --show-toplevel)
TASK=Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora
S1=$REPO/runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/linker_l20_screwdriver_rotation_07-23-29-15/nn/last_linker_l20_screwdriver_rotation_ep_760_rew_22314.447.pth
S2=$REPO/runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth
ROOT=$REPO/artifacts/policy_video_comparison_20260808

record() {
    local policy=$1 view=$2 eye_x=$3 eye_y=$4 eye_z=$5 look_x=$6 look_y=$7 look_z=$8
    local out="$ROOT/${view}_${policy}"
    mkdir -p "$out"
    echo "START policy=$policy view=$view $(date --iso-8601=seconds)"
    local adapter=()
    if [[ "$policy" == stage2_adapter ]]; then
        adapter=(--adapter_checkpoint "$S2")
    fi
    env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" -u "$REPO/play.py" \
        --task "$TASK" --checkpoint "$S1" "${adapter[@]}" \
        --num_envs 1 --num_episodes 1 --seed 0 --eval_phase final \
        --fixed_start --no_domain_rand --headless \
        --video --video_length 300 --camera_env_index 0 \
        --camera_eye "$eye_x" "$eye_y" "$eye_z" \
        --camera_lookat "$look_x" "$look_y" "$look_z" \
        --output "$out" > "$out/play.log" 2>&1
    echo "DONE policy=$policy view=$view $(date --iso-8601=seconds)"
}

record stage1_oracle oblique 0.55 0.55 1.62 0.02 0.00 1.24
record stage2_adapter oblique 0.55 0.55 1.62 0.02 0.00 1.24
record stage1_oracle side 0.66 0.00 1.36 0.02 0.00 1.24
record stage2_adapter side 0.66 0.00 1.36 0.02 0.00 1.24
record stage1_oracle top 0.00 0.03 1.83 0.02 0.00 1.22
record stage2_adapter top 0.00 0.03 1.83 0.02 0.00 1.22

echo "ALL_DONE $(date --iso-8601=seconds)"
