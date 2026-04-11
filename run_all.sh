#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Batch runner for code_generator.py evaluate
# Models  : 3-4 models (<7B)
# Dataset : mbpp
# Modes   : unconstrained / syncode / itergen / chopchop
# Temps   : 0.01 / 0.2
# =============================================================================

set -euo pipefail

# ── Model list (<7B, add/remove as needed) ───────────────────────────────────
MODELS=(
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "microsoft/phi-4"
    "deepseek-ai/deepseek-coder-6.7b-instruct"
)

# ── Temperature list ─────────────────────────────────────────────────────────
TEMPERATURES=(0.01 0.2)

# ── Dataset configuration ────────────────────────────────────────────────────
DATASET_NAME="mbpp"
INPUT_FILE="datasets/js_prompts_mbpp.jsonl"

# ── Other parameters ─────────────────────────────────────────────────────────
NUM_COMPLETIONS=3
MAX_NEW_TOKENS=512
OUTPUT_BASE="results"

# ── Grammar file paths ──────────────────────────────────────────────────���────
GRAMMAR_SYNCODE_ITERGEN="generators/grammars/javascript.lark"
GRAMMAR_CHOPCHOP="generators/grammars/javascript_chopchop.lark"

# ────────────────────────────────────────────────────────────────────────────

run_evaluate() {
    local model="$1"
    local temp="$2"
    local mode="$3"
    local grammar_args="${4:-}"

    echo ""
    echo "========================================================"
    echo "  Model   : $model"
    echo "  Dataset : $DATASET_NAME"
    echo "  Mode    : $mode"
    echo "  Temp    : $temp"
    echo "========================================================"

    python code_generator.py evaluate \
        --model        "$model" \
        --input_file   "$INPUT_FILE" \
        --dataset_name "$DATASET_NAME" \
        --mode         "$mode" \
        --temperature  "$temp" \
        --output_base  "$OUTPUT_BASE" \
        --num_completions "$NUM_COMPLETIONS" \
        --max_new_tokens  "$MAX_NEW_TOKENS" \
        $grammar_args

    echo "  ✓ Done: $model | $mode | temp=$temp"
}

# ── Main loop ────────────────────────────────────────────────────────────────
TOTAL=0
DONE=0

# Pre-calculate total number of jobs for progress display
N_MODELS=${#MODELS[@]}
N_TEMPS=${#TEMPERATURES[@]}
# unconstrained(1) + syncode(1) + itergen(1) + chopchop(1) = 4 modes
TOTAL=$(( N_MODELS * N_TEMPS * 4 ))

for model in "${MODELS[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do

        # 1. Unconstrained (no grammar required)
        run_evaluate "$model" "$temp" "unconstrained"
        DONE=$(( DONE + 1 ))
        echo "  [Progress: $DONE / $TOTAL]"

        # 2. Syncode (uses javascript.lark)
        run_evaluate "$model" "$temp" "syncode" "--grammar $GRAMMAR_SYNCODE_ITERGEN"
        DONE=$(( DONE + 1 ))
        echo "  [Progress: $DONE / $TOTAL]"

        # 3. Itergen (uses javascript.lark, same grammar as syncode)
        run_evaluate "$model" "$temp" "itergen" "--grammar $GRAMMAR_SYNCODE_ITERGEN"
        DONE=$(( DONE + 1 ))
        echo "  [Progress: $DONE / $TOTAL]"

        # 4. Chopchop (uses javascript_chopchop.lark)
        run_evaluate "$model" "$temp" "chopchop" "--grammar $GRAMMAR_CHOPCHOP --pruner basic"
        DONE=$(( DONE + 1 ))
        echo "  [Progress: $DONE / $TOTAL]"

    done
done

echo ""
echo "=========================================================="
echo "  All $TOTAL jobs complete! Results saved to: $OUTPUT_BASE/"
echo "=========================================================="
