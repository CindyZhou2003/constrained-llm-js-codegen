from typing import Optional, Dict, Any, List
from generators.hf_generator import HFGenerator
from generators.syncode_generator import SyncodeGenerator
from generators.itergen_generator import ItergenGenerator
import argparse
import json
from tqdm import tqdm
from pathlib import Path

class UnifiedCodeGenerator:
    def __init__(self, mode: str, model_name: str, grammar: Optional[str] = None):
        self.mode = mode
        self.model_name = model_name
        self.generator = self._build_generator(grammar)

    def _build_generator(self, grammar):
        if self.mode == "syncode":
            return SyncodeGenerator(self.model_name, grammar)
        elif self.mode == "itergen":
            return ItergenGenerator(self.model_name, grammar)
        elif self.mode == "unconstrained":
            return HFGenerator(self.model_name)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def generate(self, prompt: str, stop_tokens: Optional[List[str]] = None, **kwargs) -> str:
        """Returns ONLY the raw generated string."""
        return self.generator.generate(prompt, stop_tokens=stop_tokens, **kwargs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw Code Generator for .jsonl datasets")
    # model and generation configuration
    parser.add_argument("--model", type=str, default="microsoft/phi-2")
    parser.add_argument("--mode", type=str, default="unconstrained", 
                        choices=["unconstrained", "syncode", "itergen"], help="Generation mode")
    parser.add_argument("--grammar", type=str, default="javascript")
    
    # input and output configuration
    parser.add_argument("--input_file", type=str, required=True, help="Path to a .jsonl prompts file")
    parser.add_argument("--output_dir", type=str, default="raw_outputs", help="Base directory for results")
    parser.add_argument("--dataset_name", type=str, default="mbpp", help="Dataset name for folder naming")
    
    # generation parameters
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)
    
    args = parser.parse_args()
    
    model_name_clean = args.model.replace("/", "_").replace("-", "_")
    output_dir_name = f"{args.dataset_name}-js-{model_name_clean}-{args.temperature}-{args.mode}"
    final_output_path = Path(args.output_dir) / output_dir_name
    final_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating directory: {final_output_path.absolute()}")
    
    gen = UnifiedCodeGenerator(args.mode, args.model, args.grammar)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"--- Generating raw code for {len(tasks)} tasks ---")
    print(f"--- Results will be saved to: {final_output_path} ---")
    
    for task in tqdm(tasks):
        task_id = str(task.get('name', task.get('task_id', 'output'))).replace("/", "_")
        prompt_text = task['prompt']
        
        result = gen.generate(
            prompt=prompt_text,
            stop_tokens=task.get('stop_tokens', ["\nfunction", "\n//", "\n/*"]),
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
        
        combined_output = f"{prompt_text}\n\n{result}"
        
        file_path = final_output_path / f"{task_id}.js"
        file_path.write_text(combined_output, encoding='utf-8')

    print(f"\nDone! All files saved in {final_output_path}")