# MultiPL-E Dataset Builder

## Introduction

This directory contains the code to build the MultiPL-E dataset from source
Python programs. You only need to work with this code if you're trying to
add a new dataset or programming language to MultiPL-E.

If your goal is to evaluate a model with MultiPL-E, you can ignore this
code and use the pre-built dataset on the HuggingFace hub:

https://huggingface.co/datasets/nuprl/MultiPL-E

## Requirements

Python 3.10+

## Usage

### Build all prompts

Generates JSONL files in `../prompts`:

```bash
python3 all_prepare_prompts.py
```

### Build prompts for a single language

**Local JSONL file:**

```bash
python3 prepare_prompts_for_hfhub.py \
  --lang humaneval_to_py.py \
  --original-dataset humaneval \
  --originals ../datasets/originals-with-cleaned-doctests \
  --output jsonl:../prompts/humaneval-py-reworded.jsonl
```

**Hugging Face Hub:**

```bash
python3 prepare_prompts_for_hfhub.py \
  --lang humaneval_to_py.py \
  --original-dataset humaneval \
  --originals ../datasets/originals-with-cleaned-doctests \
  --output hub:nuprl/MultiPL-E:humaneval-py:test
```

### Using with chat_completions.py

**From local JSONL:**

```bash
python3 chat_completions.py bench \
  --name "openai/gpt-4.1-nano" \
  --lang py \
  --max-completions 20 \
  --root-dataset humaneval \
  --temperature 0.2 \
  --dataset jsonl:../prompts/humaneval-py-reworded.jsonl
```

**From custom Hub dataset:**

```bash
python3 chat_completions.py bench \
  --name "openai/gpt-4.1-nano" \
  --lang py \
  --max-completions 20 \
  --root-dataset humaneval \
  --temperature 0.2 \
  --dataset hub:my-org/my-dataset:humaneval-py:test
```

Default `--dataset` is `hub:nuprl/MultiPL-E:{root_dataset}-{lang}:test`.

## Dataset Specification Format

Dataset specs use the format `type:parameters`:

- **JSONL**: `jsonl:path/to/file.jsonl`
- **Hub**: `hub:dataset_name:config:split` (all parts required)

Examples:
- `jsonl:../prompts/humaneval-py.jsonl`
- `hub:nuprl/MultiPL-E:humaneval-py:test`
