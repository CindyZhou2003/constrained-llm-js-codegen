from code_generator import UnifiedCodeGenerator
import json
from typing import List, Dict
import json
import os
import gzip
import argparse
from pathlib import Path
from tqdm import tqdm
import subprocess
from code_generator import UnifiedCodeGenerator

def run_evaluation_pipeline(args):
    # Initialize generator API
    generator = UnifiedCodeGenerator(
        model_name=args.model, 
        **vars(args)
    )

    # ouput dir naming: dataset-js-model-temp-mode
    model_name_clean = args.model.replace("/", "_").replace("-", "_")
    output_dir_name = f"{args.dataset_name}-js-{model_name_clean}-{args.temperature}-{args.mode}"
    output_path = Path(args.output_base) / output_dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Also save compressed .json.gz for MultiPL-E if requested
    gz_output_path = None
    if args.save_gz:
        gz_output_path = Path(args.gz_output_base) / output_dir_name
        gz_output_path.mkdir(parents=True, exist_ok=True)
    
    if args.mode in {"syncode", "itergen", "chopchop"} and not args.grammar:
        parser.error("--grammar is required for constrained modes: syncode, itergen, chopchop")
    
    print(f"\n>>> Step 1: Loading Dataset from {args.input_file}")
    tasks = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    print(f">>> Step 2: Generating Code (Mode: {args.mode})...")
    
    # generate code for each task
    for task in tqdm(tasks):
        
        task_name = task.get('name', task.get('task_id', task.get('id')))
        prompt = task['prompt']
        stop_tokens = task.get('stop_tokens', ["\nfunction", "\n//", "\n/*"]) # 默认 JS 停止符
        
        # unified generation interface
        code = generator.generate(
            prompt=prompt,
            mode=args.mode,
            grammar=args.grammar, # use if mode=syncode
            stop_tokens=stop_tokens,
            temperature=args.temperature
        )
        # ----------------
        
        # Build MultiPL-E result format
        result_item = task.copy()
        result_item["completions"] = [code]
        
        # save as uncompressed .json (easy to inspect)
        safe_name = str(task_name).replace("/", "_")
        json_file = output_path / (safe_name + ".json")
        with open(json_file, "w", encoding="utf-8") as f_json:
            json.dump(result_item, f_json, indent=2)
        
        # optionally also save compressed .json.gz
        if gz_output_path is not None:
            gz_file = gz_output_path / (safe_name + ".json.gz")
            with gzip.open(gz_file, "wt", encoding="utf-8") as f_gz:
                json.dump(result_item, f_gz)

    print(f"\n>>> Generation Finished! Uncompressed JSON saved to: {output_path}")
    if gz_output_path:
        print(f"    Compressed .json.gz also saved to: {gz_output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # base configuration
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="HuggingFace model ID")
    parser.add_argument("--input_file", type=str, required=True, help="Path to jsonl prompts (e.g., js_prompts_mbpp.jsonl)")
    parser.add_argument("--dataset_name", type=str, default="mbpp", help="Name for folder generation")
    parser.add_argument("--output_base", type=str, default="raw_results", help="Base output directory for uncompressed .json")
    parser.add_argument("--save_gz", action="store_true", help="Also save compressed .json.gz for MultiPL-E")
    parser.add_argument("--gz_output_base", type=str, default="results", help="Base output directory for compressed .json.gz")
    
    # generation configuration
    parser.add_argument("--mode", type=str, default="unconstrained", 
                        choices=["unconstrained", "syncode", "itergen", "chopchop"], help="Generation mode")
    parser.add_argument("--grammar", type=str, default=None, help="Path to grammar file (.lark) for constrained modes")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")

    args = parser.parse_args()
    run_evaluation_pipeline(args)
