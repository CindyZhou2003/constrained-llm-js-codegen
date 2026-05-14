#!/usr/bin/env bash
# Itergen + chopchop runs for Qwen2.5-Coder-1.5B-Instruct.
# Matches the itergen/chopchop coverage already collected on Coder-3B,
# so you get a clean 3B-vs-1.5B comparison for these methods.
#
# Run this BEFORE run_missing_coder_1_5b.sh — chopchop is the slowest method,
# so kick it off first and let it cook.

set -euo pipefail

MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
GRAMMAR_STD="generators/grammars/javascript.lark"
GRAMMAR_CHOP="generators/grammars/javascript_chopchop.lark"
OUTPUT_BASE="results"

HUMANEVAL="datasets/js_prompts_humaneval.jsonl"
MBPP="datasets/js_prompts_mbpp.jsonl"

run() {
  echo ">>> $*"
  python code_generator.py evaluate "$@"
}

# ---------------- itergen ----------------

# humaneval, temp 0.0, n=1
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode itergen --grammar "$GRAMMAR_STD" --output_base "$OUTPUT_BASE" \
    --temperature 0.0 --max_new_tokens 512 --num_completions 1

# humaneval, temp 0.2, n=3 (pass@1 and pass@3)
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode itergen --grammar "$GRAMMAR_STD" --output_base "$OUTPUT_BASE" \
    --temperature 0.2 --max_new_tokens 512 --num_completions 3

# mbpp, temp 0.0, n=1
run --model "$MODEL" --input_file "$MBPP" --dataset_name mbpp \
    --mode itergen --grammar "$GRAMMAR_STD" --output_base "$OUTPUT_BASE" \
    --temperature 0.0 --max_new_tokens 512 --num_completions 1

# ---------------- chopchop ----------------

# humaneval, temp 0.2, n=3 — yields pass@1 and pass@3.
# Chose humaneval (161 problems) over mbpp (397) to keep wall-clock tractable;
# expect ~4-6 hours on 1.5B.
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode chopchop --grammar "$GRAMMAR_CHOP" --pruner basic \
    --output_base "$OUTPUT_BASE" \
    --temperature 0.2 --max_new_tokens 512 --num_completions 3

echo "Itergen + chopchop runs complete."
echo "Next: compress + evaluate each output dir, e.g.:"
echo "  python code_evaluation.py --input_dir raw_results/<run_name> --benchmark multipl-e"
echo "  python code_evaluation.py --input_dir raw_results/humaneval-js-Qwen_Qwen2.5_Coder_1.5B_Instruct-0.2-itergen --benchmark multipl-e --pass_k 3"
