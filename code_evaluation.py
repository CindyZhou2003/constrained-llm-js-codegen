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
        mode=args.mode, 
        model_name=args.model, 
        grammar=args.grammar
    )

    # ouput dir naming: dataset-js-model-temp-mode
    model_name_clean = args.model.replace("/", "_").replace("-", "_")
    output_dir_name = f"{args.dataset_name}-js-{model_name_clean}-{args.temperature}-{args.mode}"
    output_path = Path(args.output_base) / output_dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    
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
        
        # save as .json.gz (one file per task)
        safe_filename = str(task_name).replace("/", "_") + ".json.gz"
        save_file = output_path / safe_filename
        
        with gzip.open(save_file, "wt", encoding="utf-8") as f_gz:
            json.dump(result_item, f_gz)

    print(f"\n>>> Generation Finished! Files saved to: {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # base configuration
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="HuggingFace model ID")
    parser.add_argument("--input_file", type=str, required=True, help="Path to jsonl prompts (e.g., js_prompts_mbpp.jsonl)")
    parser.add_argument("--dataset_name", type=str, default="mbpp", help="Name for folder generation")
    parser.add_argument("--output_base", type=str, default="results", help="Base output directory")
    
    # generation configuration
    parser.add_argument("--mode", type=str, default="unconstrained", 
                        choices=["unconstrained", "syncode", "itergen"], help="Generation mode")
    parser.add_argument("--grammar", type=str, default="syncode/javascript.lark", help="Path to grammar file (for syncode)")
    parser.add_argument("--temperature", type=float, default=0.2)

    args = parser.parse_args()
    run_evaluation_pipeline(args)
