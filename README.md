# Constrained-llm-js-codegen

A framework for systematically evaluating LLM code generation under **unconstrained** and **grammar-constrained** decoding across multiple programming languages, using [MultiPL-E](https://github.com/nuprl/MultiPL-E) and [HumanEval-X](https://github.com/THUDM/CodeGeeX) as benchmarks.

**JavaScript is used as the primary example** throughout this README. The same workflow applies to any language supported by MultiPL-E — replace the language-specific files (translation script, grammar, prompt dataset) at the relevant steps. See [Adding a New Language](#adding-a-new-language) for a checklist.

## Project Structure

- `code_generator.py`: Generation-only CLI with two sub-commands:
  - `generate`: Quick single-pass generation — outputs plain source files (`.js`, etc.).
  - `evaluate`: Batch generation — outputs MultiPL-E-compatible `.json` files in `raw_results/`.
- `code_evaluation.py`: Post-generation evaluation CLI — compresses `.json` outputs to `.json.gz` and optionally runs the MultiPL-E Docker benchmark.
- `compress_results.py`: Post-processing utility to convert `.json` to `.json.gz` for MultiPL-E.
- `generators/`: Backend implementations and constrained decoding frameworks (`syncode`, `itergen`, `chopchop`, etc.).
  - `grammars/`: Language grammar files (`.lark`) for constrained modes. Add a grammar here for each new language.
- `datasets/`: Translated prompt datasets in `.jsonl` format, one file per language/benchmark combination.
- `raw_results/`: Uncompressed generation outputs (`.json`).
- `results_eval/`: Compressed benchmark inputs and evaluation outputs (`.json.gz`).
- `tools/`: Analysis scripts for generated and evaluated results.

### Utilities in `tools/`

- `count.py`: Summarize evaluation cases.

```bash
# Auto-generate output file name
python tools/count.py raw_results/mbpp-js-microsoft_phi_2-0.2

# Specify output file name (optional)
python tools/count.py raw_results/mbpp-js-microsoft_phi_2-0.2 custom_name.txt
```

- `diff.py`: Compare two summary files.

```bash
python tools/diff.py summary/mbpp-js-microsoft_phi_2-0.0.txt summary/mbpp-js-microsoft_phi_2-0.0-syncode.txt

# Specify output file name (optional)
python tools/diff.py summary/mbpp-js-microsoft_phi_2-0.0.txt summary/mbpp-js-microsoft_phi_2-0.0-syncode.txt custom_diff.txt
```

- `extract_prompts.py`: Extract cases that regress from one run to another.

```bash
# Use default prompts file and output path
python tools/extract_prompts.py mbpp-js-microsoft_phi_2-0.0-unconstrained_vs_mbpp-js-microsoft_phi_2-0.0-syncode_diff.txt

# Specify custom prompts file and output path
python tools/extract_prompts.py diff.txt datasets/js_prompts_mbpp.jsonl datasets/tem_custom.jsonl
```

## Setup

Requires Python 3.10 to 3.12.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## MultiPL-E Benchmark

This project uses [MultiPL-E](https://github.com/nuprl/MultiPL-E) at `benchmark/MultiPL-E`.

### 1) Generate Prompt Datasets

MultiPL-E translates benchmark problems from Python into many target languages. Run these commands inside `benchmark/MultiPL-E/dataset_builder`.

Replace `--lang` with the translator for your target language (e.g., `humaneval_to_py.py`, `humaneval_to_cpp.py`). All available translators are in `benchmark/MultiPL-E/dataset_builder/`.

**HumanEval** (164 cases) — JavaScript example:

```bash
python prepare_prompts_for_hfhub.py \
  --lang humaneval_to_js.py \
  --doctests transform \
  --prompt-terminology reworded \
  --output jsonl:../datasets/js_prompts_humaneval.jsonl \
  --original-dataset humaneval \
  --originals ../datasets/originals-with-cleaned-doctests
```

**MBPP** (397 cases) — JavaScript example:

```bash
python prepare_prompts_for_hfhub.py \
  --lang humaneval_to_js.py \
  --doctests transform \
  --prompt-terminology reworded \
  --output jsonl:../datasets/js_prompts_mbpp.jsonl \
  --originals ../datasets/mbpp-typed \
  --original-dataset mbpp
```

The translated JavaScript prompts are already included under `datasets/`. For other languages, run the corresponding translator and save to `datasets/<lang>_prompts_<benchmark>.jsonl`.

### 2) Basic Generation

Use `code_generator.py generate` for quick iteration. It reads `.jsonl` prompts and outputs plain source code files.

Key parameters to adapt for your language:
- `--input_file`: your language's prompt dataset (e.g., `datasets/py_prompts_mbpp.jsonl`)
- `--mode`: `unconstrained` | `syncode` | `itergen` | `chopchop`
- `--grammar`: path to the `.lark` grammar file; required for all constrained modes

```bash
python code_generator.py generate \
  --model microsoft/phi-2 \
  --input_file datasets/js_prompts_mbpp.jsonl \
  --mode syncode \
  --grammar generators/grammars/javascript.lark \
  --output_dir ./raw_outputs/ \
  --temperature 0.2 \
  --max_new_tokens 512
```

**Input**: `.jsonl` prompt file

**Output**: individual source code files

### 3) Batch Generation for Evaluation

Use `code_generator.py evaluate` for task-wise generation with MultiPL-E-compatible JSON content.

The output directory is automatically named `<dataset_name>-<lang>-<model>-<temperature>-<mode>`. Adjust `--input_file` and `--dataset_name` to switch languages or benchmarks.

Key parameters to adapt for your language:
- `--input_file`: your language's prompt dataset
- `--mode`: `unconstrained` | `syncode` | `itergen` | `chopchop`
- `--grammar`: path to the `.lark` grammar file; required for all constrained modes
- `--pruner`: pruner mode for `chopchop` (`none` = grammar-only, `basic` = env-aware JS pruning)
- `--dataset_name`: used only as a label in the output directory name (`humaneval` | `mbpp` | any string)

```bash
python code_generator.py evaluate \
  --model microsoft/phi-2 \
  --input_file datasets/js_prompts_mbpp.jsonl \
  --mode chopchop \
  --grammar generators/grammars/javascript_chopchop.lark \
  --pruner basic \
  --dataset_name mbpp \
  --output_base raw_results \
  --temperature 0.2 \
  --max_new_tokens 512
```

**Input**: `.jsonl` prompt file (and a grammar file for constrained modes)

**Output**: uncompressed `.json` files in `raw_results/<run_name>/`

To compress and evaluate, pass the output directory to `code_evaluation.py` (see steps 4–5).

### 4) Compress JSON Outputs

Convert the `.json` files produced by `code_generator.py evaluate` to `.json.gz` for MultiPL-E.

```bash
# Output defaults to results_eval/<run_name>/
python code_evaluation.py --input_dir raw_results/mbpp-js-microsoft_phi_2-0.2-chopchop

# Specify a custom output directory
python code_evaluation.py \
  --input_dir raw_results/mbpp-js-microsoft_phi_2-0.2-chopchop \
  --gz_output_dir results_eval/mbpp-js-microsoft_phi_2-0.2-chopchop
```

### 5) Evaluate with MultiPL-E

#### Option A — via code_evaluation.py (recommended)

Compress and run the full MultiPL-E Docker pipeline in one step:

```bash
python code_evaluation.py \
  --input_dir raw_results/mbpp-js-microsoft_phi_2-0.2-chopchop \
  --benchmark multipl-e
```

#### Option B — Manual

1. Pull the evaluation image:

```bash
docker pull ghcr.io/nuprl/multipl-e-evaluation
```

2. Tag the image:

```bash
docker tag ghcr.io/nuprl/multipl-e-evaluation multipl-e-eval
```

3. Run evaluation on a directory containing `*.json.gz` completions:

```bash
docker run --rm --network none \
  -v "/absolute/path/to/results_eval:/tutorial:rw" \
  multipl-e-eval --dir /tutorial --output-dir /tutorial --recursive
```

4. Compute pass@k:

```bash
python benchmark/MultiPL-E/pass_k.py /absolute/path/to/results_eval
```

After `pass_k.py`, related `.results.json.gz` files are written to that same `results_eval` directory.

## Adding a New Language

Follow these steps to extend the evaluation to another language:

1. **Generate prompts** — inside `benchmark/MultiPL-E/dataset_builder`, run `prepare_prompts_for_hfhub.py` with `--lang humaneval_to_<lang>.py` and save the output to `datasets/<lang>_prompts_<benchmark>.jsonl`.

2. **Add a grammar** — write or obtain a `.lark` grammar for the target language and place it in `generators/grammars/<lang>.lark`. This is required when using constrained modes (`syncode`, `itergen`, `chopchop`).

3. **Run generation** — pass the new prompt file and grammar to `code_generator.py evaluate`:
   ```bash
   python code_generator.py evaluate \
     --model <your_model> \
     --input_file datasets/<lang>_prompts_<benchmark>.jsonl \
     --mode <mode> \
     --grammar generators/grammars/<lang>.lark \
     --dataset_name <benchmark>
   ```

4. **Compress and evaluate** — pass the output directory to `code_evaluation.py`:
   ```bash
   python code_evaluation.py \
     --input_dir raw_results/<run_name> \
     --benchmark multipl-e
   ```

## HumanEval-X Benchmark Usage

### Code Generation

1. Use dataset:

`benchmark/CodeGeeX/codegeex/benchmark/humaneval-x/js/data/humaneval_js.jsonl`

2. Generate code with `benchmark/CodeGeeX/test_generate.py` and write to `benchmark/CodeGeeX/input_data`.

Each line must be JSON with `task_id` and `generation` fields, for example:

```json
{"task_id": "JavaScript/0", "generation": "..."}
```

### Evaluation

1. Change directory:

```bash
cd benchmark/CodeGeeX
```

2. Build image:

```bash
docker build -t humanevalx .
```

3. Start container and mount input directory:

```bash
docker run -it --mount type=bind,source=./input_data,target=/workspace/CodeGeeX/input_data --name jseval humanevalx
```

4. Run evaluation inside container:

```bash
./scripts/evaluate_humaneval_x.sh input_data/generations.jsonl js
```

5. Results are appended to each line of the input file.

6. Exit container:

```bash
exit
```
