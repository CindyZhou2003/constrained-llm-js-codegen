#!/usr/bin/env bash
# Fill in Qwen2.5-Coder-1.5B-Instruct runs to match the coverage that was
# previously (mistakenly) collected on the non-Coder Qwen2.5-1.5B-Instruct.
#
# After this finishes, the Coder-1.5B model will have parallel coverage with
# the Coder-3B baseline for {unconstrained, syncode} × {temp 0, 0.2, 0.5}
# on humaneval-js, plus the one missing mbpp-js temp=0 unconstrained run.

set -euo pipefail

MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
GRAMMAR="generators/grammars/javascript.lark"
OUTPUT_BASE="results"

HUMANEVAL="datasets/js_prompts_humaneval.jsonl"
MBPP="datasets/js_prompts_mbpp.jsonl"

run() {
  echo ">>> $*"
  python code_generator.py evaluate "$@"
}

# ---------------- humaneval-js ----------------

# temp 0.0, unconstrained, n=1
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode unconstrained --output_base "$OUTPUT_BASE" \
    --temperature 0.0 --max_new_tokens 512 --num_completions 1

# temp 0.0, syncode, n=1
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode syncode --grammar "$GRAMMAR" --output_base "$OUTPUT_BASE" \
    --temperature 0.0 --max_new_tokens 512 --num_completions 1

# temp 0.2, unconstrained, n=3 (pass@1 and pass@3)
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode unconstrained --output_base "$OUTPUT_BASE" \
    --temperature 0.2 --max_new_tokens 512 --num_completions 3

# temp 0.2, syncode, n=3
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode syncode --grammar "$GRAMMAR" --output_base "$OUTPUT_BASE" \
    --temperature 0.2 --max_new_tokens 512 --num_completions 3

# temp 0.5, unconstrained, n=5 (pass@1 and pass@5)
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode unconstrained --output_base "$OUTPUT_BASE" \
    --temperature 0.5 --max_new_tokens 512 --num_completions 5

# temp 0.5, syncode, n=5
run --model "$MODEL" --input_file "$HUMANEVAL" --dataset_name humaneval \
    --mode syncode --grammar "$GRAMMAR" --output_base "$OUTPUT_BASE" \
    --temperature 0.5 --max_new_tokens 512 --num_completions 5

# ---------------- mbpp-js ----------------

# temp 0.0, unconstrained, n=1
run --model "$MODEL" --input_file "$MBPP" --dataset_name mbpp \
    --mode unconstrained --output_base "$OUTPUT_BASE" \
    --temperature 0.0 --max_new_tokens 512 --num_completions 1

echo "All missing Coder-1.5B runs complete."
echo "Next step: compress + evaluate each output dir, e.g.:"
echo "  python code_evaluation.py --input_dir raw_results/<run_name> --benchmark multipl-e --pass_k 3"
