"""
Shared library for loading and saving MultiPL-E datasets.

Supports both:
- Hugging Face Hub (datasets.Dataset)
- Local JSONL files

Dataset Specification Format:
- Hub: hub:dataset_name:config:split (all parts required)
  - Example: hub:nuprl/MultiPL-E:humaneval-py:test
- JSONL: jsonl:path/to/file.jsonl
  - Examples: jsonl:../prompts/humaneval-py.jsonl
              jsonl:/absolute/path/to/data.jsonl

This eliminates code duplication between dataset preparation and evaluation scripts.
"""

import datasets
from pathlib import Path
import json
from typing import List, Dict, Any, Union, Tuple, Optional
import re


class DatasetSpec:
    """Parsed dataset specification"""
    def __init__(self, spec_type: str, **kwargs):
        self.type = spec_type
        self.params = kwargs

    def __repr__(self):
        return f"DatasetSpec(type={self.type}, params={self.params})"


def parse_dataset_spec(spec: str) -> DatasetSpec:
    """
    Parse a dataset specification string.

    Format:
    - hub:dataset_name:config:split
    - jsonl:path/to/file.jsonl

    Examples:
    - hub:nuprl/MultiPL-E:humaneval-py:test
    - jsonl:../prompts/humaneval-py.jsonl

    Returns:
        DatasetSpec with type and params
    """
    if not spec:
        raise ValueError("Dataset specification cannot be empty")

    parts = spec.split(":", 1)
    if len(parts) < 2:
        raise ValueError(
            f"Invalid dataset spec '{spec}'. Must start with 'hub:' or 'jsonl:'"
        )

    spec_type = parts[0].lower()

    if spec_type == "jsonl":
        # Format: jsonl:path/to/file.jsonl
        path = parts[1]
        return DatasetSpec("jsonl", path=path)

    elif spec_type == "hub":
        # Format: hub:dataset_name:config:split (all required)
        remaining = parts[1]
        hub_parts = remaining.split(":")

        if len(hub_parts) != 3:
            raise ValueError(
                f"Invalid hub spec '{spec}'. Expected format: hub:dataset_name:config:split"
            )

        dataset_name = hub_parts[0]
        config = hub_parts[1]
        split = hub_parts[2]

        return DatasetSpec(
            "hub",
            dataset_name=dataset_name,
            config=config,
            split=split
        )

    else:
        raise ValueError(
            f"Unknown dataset type '{spec_type}'. Must be 'hub' or 'jsonl'"
        )


def save_dataset(results: List[Dict[str, Any]], spec: str) -> None:
    """
    Save a dataset to either Hugging Face Hub or a local JSONL file.

    Args:
        results: List of problem dictionaries to save
        spec: Dataset specification string (all parts required)
              - For JSONL: jsonl:../prompts/humaneval-py.jsonl
              - For Hub: hub:dataset_name:config:split (e.g., hub:nuprl/MultiPL-E:humaneval-py:test)
    """
    parsed = parse_dataset_spec(spec)

    if parsed.type == "jsonl":
        # Save to local JSONL file
        output_path = Path(parsed.params["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

        print(f"Saved {len(results)} problems to {output_path}")

    elif parsed.type == "hub":
        # Save to Hugging Face Hub
        dataset_name = parsed.params["dataset_name"]
        config_name = parsed.params["config"]
        split = parsed.params["split"]

        dataset = datasets.Dataset.from_list(results)
        dataset.push_to_hub(dataset_name, split=split, config_name=config_name)
        print(f"Pushed {len(results)} problems to {dataset_name} ({config_name})")


def load_dataset(spec: str) -> Union[datasets.Dataset, List[Dict[str, Any]]]:
    """
    Load a dataset from either Hugging Face Hub or a local JSONL file.

    Args:
        spec: Dataset specification string (all parts required)
              - For JSONL: jsonl:../prompts/humaneval-py.jsonl
              - For Hub: hub:nuprl/MultiPL-E:humaneval-py:test

    Returns:
        Either a datasets.Dataset (from Hub) or a list of dictionaries (from JSONL)
    """
    parsed = parse_dataset_spec(spec)

    if parsed.type == "jsonl":
        # Load from local JSONL file
        source_path = Path(parsed.params["path"])
        if not source_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {source_path}")

        results = []
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        print(f"Loaded {len(results)} problems from {source_path}")
        return results

    elif parsed.type == "hub":
        # Load from Hugging Face Hub
        dataset_name = parsed.params["dataset_name"]
        config = parsed.params["config"]
        split = parsed.params["split"]

        dataset = datasets.load_dataset(dataset_name, config, split=split)
        print(f"Loaded {len(dataset)} problems from {dataset_name} ({config})")
        return dataset
