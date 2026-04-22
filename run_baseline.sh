#!/usr/bin/env bash
# =============================================================================
# run_baseline.sh — Unconstrained baseline runner (temperature=0)
# Models  : same 4 models as run_all.sh
# Dataset : mbpp
# Mode    : unconstrained only
# Resume  : skips any model whose output directory already has all tasks done
# =============================================================================

set -euo pipefail

MODELS=(
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "microsoft/phi-4"
    "deepseek-ai/deepseek-coder-6.7b-instruct"
)

TEMPERATURE=0.0
DATASET_NAME="mbpp"
INPUT_FILE="datasets/js_prompts_mbpp.jsonl"
OUTPUT_BASE="results"

# Total number of tasks in the input file (used to detect a complete run)
EXPECTED_TASKS=$(wc -l < "$INPUT_FILE")

# ── Helper: return 0 (true) if this model's output directory is complete ──────
is_complete() {
    local model="$1"
    # Mirror code_generator.py: replace '/' and '-' with '_'
    local model_clean
    model_clean=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local run_dir="${OUTPUT_BASE}/${DATASET_NAME}-js-${model_clean}-${TEMPERATURE}-unconstrained"

    [[ -d "$run_dir" ]] || return 1

    local done_count
    done_count=$(find "$run_dir" -maxdepth 1 -name "*.json" | wc -l)
    [[ "$done_count" -ge "$EXPECTED_TASKS" ]]
}

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL=${#MODELS[@]}
IDX=0

for model in "${MODELS[@]}"; do
    IDX=$(( IDX + 1 ))

    if is_complete "$model"; then
        echo "[${IDX}/${TOTAL}] SKIP (already complete): $model"
        continue
    fi

    echo ""
    echo "========================================================"
    echo "  [${IDX}/${TOTAL}] Model : $model"
    echo "  Mode  : unconstrained | Temp : $TEMPERATURE"
    echo "========================================================"

    python code_generator.py evaluate \
        --model        "$model" \
        --input_file   "$INPUT_FILE" \
        --dataset_name "$DATASET_NAME" \
        --mode         unconstrained \
        --temperature  "$TEMPERATURE" \
        --output_base  "$OUTPUT_BASE"

    echo "  ✓ Done: $model"
done

echo ""
echo "=========================================================="
echo "  All $TOTAL baseline jobs complete! Results in: $OUTPUT_BASE/"
echo "=========================================================="
