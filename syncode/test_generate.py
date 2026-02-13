from syncode import Syncode
import json
import gzip
import os
from pathlib import Path
from tqdm import tqdm

MODEL_NAME = "microsoft/phi-2"
GRAMMAR_FILE = "javascript.lark"

def run_syncode_benchmark(dataset_path, output_base="syncode_results"):
    """
    dataset_path: MultiPL-E prompts, eg. humaneval-js.jsonl
    lang_code: abbrevation like js, py, rs, c, java, go
    output_base: base directory to save the results
    """
    if not os.path.exists(GRAMMAR_FILE):
        print(f"Error: Grammar file not found at '{GRAMMAR_FILE}'")
        return

    print(f"Initializing SynCode with model: '{MODEL_NAME}' and grammar: '{GRAMMAR_FILE}'")

    # llm = Syncode(model = MODEL_NAME, mode='original', max_new_tokens=250)
    syn_llm = Syncode(
        model=MODEL_NAME, 
        mode='grammar_mask', 
        grammar=GRAMMAR_FILE, 
        parse_output_only=True
    )

    # MultiPL-E output directory structure: {dataset_name}-{model_name}-syncode/{task_id}.json.gz
    dataset_name = Path(dataset_path).stem
    TEMPERATURE = 0.2 # Not sure
    sub_dir = f"{dataset_name}-{MODEL_NAME.replace('/', '_')}-{TEMPERATURE}-syncode"
    save_path = Path(output_base) / sub_dir
    save_path.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    print(f"Start processing dataset: {dataset_name}, total {len(tasks)} tasks...")

    for task in tqdm(tasks):
        task_id = task['name']
        prompt_text = task['prompt']
        stop_tokens = task.get('stop_tokens', [])
        completion = syn_llm.infer(prompt_text, stop_words=stop_tokens)[0]
        
        test_code = task.get('tests', "")
        
        # if not test_code.startswith("}"):
        #     test_code = "}\n" + test_code
            
        output_data = {
            "name": task_id,
            "language": lang_code,
            "prompt": prompt_text,
            "completions": [completion+"\n\n"],
            "tests": test_code,
            "stop_tokens": stop_tokens
        }
        
        # Saved to .json.gz
        file_name = f"{task_id.replace('/', '_')}.json.gz"
        with gzip.open(save_path / file_name, 'wt', encoding='utf-8') as zf:
            json.dump(output_data, zf)

    print(f"\nFinished! Results saved to: {save_path}")

if __name__ == "__main__":
    # change the path to your local MultiPL-E dataset
    run_syncode_benchmark("D:\\code\\constrained-llm-js-codegen\\benchmark\\MultiPL-E\\datasets\\js_prompts.jsonl", "js")