#!/usr/bin/env bash
# Tier 1: complete the chopchop size-scaling ladder on mbpp / T0 / n=1.
#
# itergen ladder needs NO generation -- mbpp/T0 itergen already exists for
# all three sizes (3B in results.csv, 1.5B + 0.5B already generated). Just
# evaluate those dirs (see the echo hints at the end / Tier 0 eval).
#
# chopchop on mbpp/T0/n=1: 3B already exists (results.csv). Only 0.5B and
# 1.5B are missing. humaneval chopchop is intentionally dropped from the
# paper, so no humaneval runs here.
#
# Ordered cheapest-first: 0.5B (~2-2.5h) before 1.5B (~4-4.5h).

set -euo pipefail

M05="Qwen/Qwen2.5-Coder-0.5B"
M15="Qwen/Qwen2.5-Coder-1.5B-Instruct"
GRAMMAR_CHOP="generators/grammars/javascript_chopchop.lark"
OUTPUT_BASE="results"
MBPP="datasets/js_prompts_mbpp.jsonl"

run() {
  echo ">>> $*"
  python code_generator.py evaluate "$@"
}

# ---------- 0.5B chopchop (cheapest, run first) ----------

# mbpp, temp 0.0, n=1
run --model "$M05" --input_file "$MBPP" --dataset_name mbpp \
    --mode chopchop --grammar "$GRAMMAR_CHOP" --pruner basic \
    --output_base "$OUTPUT_BASE" \
    --temperature 0.0 --max_new_tokens 512 --num_completions 1

# ---------- 1.5B chopchop ----------

# mbpp, temp 0.0, n=1
run --model "$M15" --input_file "$MBPP" --dataset_name mbpp \
    --mode chopchop --grammar "$GRAMMAR_CHOP" --pruner basic \
    --output_base "$OUTPUT_BASE" \
    --temperature 0.0 --max_new_tokens 512 --num_completions 1

echo "Tier 1 chopchop generation complete."
echo ""
echo "Evaluate the new chopchop dirs (pass@1, n=1):"
echo "  python code_evaluation.py --input_dir results/Qwen_Qwen2.5_Coder_0.5B/mbpp-js/T0_chopchop --benchmark multipl-e"
echo "  python code_evaluation.py --input_dir results/Qwen_Qwen2.5_Coder_1.5B_Instruct/mbpp-js/T0_chopchop --benchmark multipl-e"
echo "  # 3B/mbpp/T0 chopchop is already in results.csv (row 57)"
