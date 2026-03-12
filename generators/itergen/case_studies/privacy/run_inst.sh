#!/bin/bash

# List of Hugging Face model IDs with assumed usernames
models=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

# Loop through each model and run the commands
for model in "${models[@]}"; do
    echo "Running privacy evaluation for model: $model"
    
    # Run the original mode
    python privacy_evaluation.py --model_id="$model" --mode='original'
    
    # Run the itergen mode
    python privacy_evaluation.py --model_id="$model" --mode='itergen'
    
    echo "Completed evaluation for model: $model"
done
