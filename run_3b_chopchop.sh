#!/usr/bin/env bash
# Standalone: 3B chopchop on mbpp / T0.2 / n=3 (pass@1 + pass@3).
#
# Split out from run_tier1_itergen_chopchop.sh because this single run is
# the most expensive cell in the whole study -- mbpp chopchop on 3B was
# ~8h at T0/n=1, so T0.2/n=3 is roughly a full day. Keeping it separate
# means the cheap 0.5B/1.5B Tier 1 runs are not blocked behind it.
#
# Run this on its own (e.g. overnight / over a weekend slot).

set -euo pipefail

M3="Qwen/Qwen2.5-Coder-3B-Instruct"
GRAMMAR_CHOP="generators/grammars/javascript_chopchop.lark"
OUTPUT_BASE="results"
MBPP="datasets/js_prompts_mbpp.jsonl"

echo ">>> 3B chopchop mbpp T0.2 n=3 (expect ~1 day)"
python code_generator.py evaluate \
    --model "$M3" --input_file "$MBPP" --dataset_name mbpp \
    --mode chopchop --grammar "$GRAMMAR_CHOP" --pruner basic \
    --output_base "$OUTPUT_BASE" \
    --temperature 0.2 --max_new_tokens 512 --num_completions 3

echo "Done. Evaluate with pass@3:"
echo "  python code_evaluation.py --input_dir results/Qwen_Qwen2.5_Coder_3B_Instruct/mbpp-js/T0.2_chopchop --benchmark multipl-e --pass_k 3"
