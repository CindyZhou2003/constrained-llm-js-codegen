import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_code(
    input_path="benchmark/CodeGeeX/codegeex/benchmark/humaneval-x/js/data/humaneval_js.jsonl",
    output_dir="benchmark/CodeGeeX/input_data", # output directory for generated results
    model_id="microsoft/phi-2", 
    temperature=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu"
):

    model_name_clean = model_id.replace("/", "_").replace("-", "_")
    output_filename = f"humaneval-x-js-{model_name_clean}-{temperature}.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Output will be saved to: {output_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).to(device)

    tasks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    print(f"Starting generation for {len(tasks)} tasks...")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for task in tasks:
            task_id = task["task_id"]
            prompt = task["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][input_length:]
            generation_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            result = {
                "task_id": task_id, 
                "generation": generation_text
            }
            f_out.write(json.dumps(result) + "\n")
            
            print(f"Completed: {task_id}")

    print(f"Finished! Results saved to {output_path}")

if __name__ == "__main__":
    # change model_id and temperature as needed
    generate_code(model_id="microsoft/phi-2", temperature=0.2)