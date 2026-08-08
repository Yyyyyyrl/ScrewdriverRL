#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python}
ROOT=$(git rev-parse --show-toplevel)
OUT=$ROOT/artifacts/full_training_20260808/stage2_acceptance
TASK=Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora
S1=$ROOT/runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/linker_l20_screwdriver_rotation_07-23-29-15/nn/last_linker_l20_screwdriver_rotation_ep_760_rew_22314.447.pth
AD=$ROOT/runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth
mkdir -p "$OUT"

run_eval() {
    local name=$1
    shift
    echo "START $name $(date --iso-8601=seconds)"
    env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" -u "$ROOT/eval.py" --task "$TASK" --checkpoint "$S1" \
        --num_envs 256 --seed 0 --json_output "$OUT/$name.json" "$@" \
        > "$OUT/$name.log" 2>&1
    echo "DONE $name $(date --iso-8601=seconds)"
}

run_eval oracle_native
run_eval adapter_native --deploy_eval --adapter_checkpoint "$AD"
run_eval oracle_bench --bench_dr --fixed_geometry_diameter_mm 64
run_eval adapter_bench --deploy_eval --adapter_checkpoint "$AD" --bench_dr --fixed_geometry_diameter_mm 64
echo "ALL_DONE $(date --iso-8601=seconds)"
