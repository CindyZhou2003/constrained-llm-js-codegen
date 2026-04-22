#!/usr/bin/env bash
# =============================================================================
# run_baseline.sh — Baseline runner (temperature=0)
# Models  : same 4 models as run_all.sh
# Dataset : mbpp
# Modes   : unconstrained / syncode
# Resume  : skips any model+mode whose output directory already has all tasks
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
GRAMMAR_SYNCODE="generators/grammars/javascript.lark"

# Total number of tasks in the input file (used to detect a complete run)
EXPECTED_TASKS=$(wc -l < "$INPUT_FILE")

# ── Helper: return 0 (true) if this model+mode output directory is complete ───
is_complete() {
    local model="$1"
    local mode="$2"
    # Mirror code_generator.py: replace '/' and '-' with '_'
    local model_clean
    model_clean=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local run_dir="${OUTPUT_BASE}/${DATASET_NAME}-js-${model_clean}-${TEMPERATURE}-${mode}"

    [[ -d "$run_dir" ]] || return 1

    local done_count
    done_count=$(find "$run_dir" -maxdepth 1 -name "*.json" | wc -l)
    [[ "$done_count" -ge "$EXPECTED_TASKS" ]]
}

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL=$(( ${#MODELS[@]} * 2 ))   # unconstrained + syncode
IDX=0

for model in "${MODELS[@]}"; do

    # 1. Unconstrained
    IDX=$(( IDX + 1 ))
    if is_complete "$model" "unconstrained"; then
        echo "[${IDX}/${TOTAL}] SKIP (already complete): $model | unconstrained"
    else
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
        echo "  ✓ Done: $model | unconstrained"
    fi

    # 2. Syncode
    IDX=$(( IDX + 1 ))
    if is_complete "$model" "syncode"; then
        echo "[${IDX}/${TOTAL}] SKIP (already complete): $model | syncode"
    else
        echo ""
        echo "========================================================"
        echo "  [${IDX}/${TOTAL}] Model : $model"
        echo "  Mode  : syncode | Temp : $TEMPERATURE"
        echo "========================================================"
        python code_generator.py evaluate \
            --model        "$model" \
            --input_file   "$INPUT_FILE" \
            --dataset_name "$DATASET_NAME" \
            --mode         syncode \
            --temperature  "$TEMPERATURE" \
            --output_base  "$OUTPUT_BASE" \
            --grammar      "$GRAMMAR_SYNCODE"
        echo "  ✓ Done: $model | syncode"
    fi

done

echo ""
echo "=========================================================="
echo "  All $TOTAL baseline jobs complete! Results in: $OUTPUT_BASE/"
echo "=========================================================="
