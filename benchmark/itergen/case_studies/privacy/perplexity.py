import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

import os
from datasets import load_dataset
from tqdm import tqdm
import torch
torch.manual_seed(11)
os.environ['HF_CACHE'] = '/data/share/models/hugging_face'
os.environ['HF_ACCESS_TOKEN'] = "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"
from detoxify import Detoxify
import sys, os
sys.path.append(os.getcwd() + '/../../')
from itergen import common
import json
import argparse
import time

device='cuda:1'

import torch
import numpy as np

def calculate_perplexity(prompt, response, model, tokenizer):
    # Combine prompt and response
    input_text = prompt + response
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Calculate the loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    # Calculate perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

def calculate_average_perplexity_from_jsonl(jsonl_file, model, tokenizer):
    total_perplexity = 0.0
    num_responses = 0
    
    # First, we need to count the number of lines in the file for tqdm
    with open(jsonl_file, 'r') as file:
        total_lines = sum(1 for _ in file)
    
    with open(jsonl_file, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Processing lines"):
            data = json.loads(line)
            if 'prompt' in data and 'response' in data:
                perplexity = calculate_perplexity(data['prompt'], data['response'], model, tokenizer)
                total_perplexity += perplexity
                num_responses += 1
    
    average_perplexity = total_perplexity / num_responses if num_responses > 0 else 0
    return average_perplexity


model_name = 'Qwen/Qwen2.5-0.5B'  # Specify your model name
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='/data/share/models/hugging_face/')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/data/share/models/hugging_face/')

prompt = "Once upon a time"
response = " there was a brave knight."
perplexity_score = calculate_perplexity(prompt, response, model, tokenizer)
print(f'Perplexity Score: {perplexity_score}')
fname='/home/rohan/itergen/case_studies/privacy/results/privacy/generations/enron/email_extraction_known_fiveshot_a_s100_meta-llama_Llama-3.2-3B_1_itergen_False_completions.jsonl'
print(calculate_average_perplexity_from_jsonl(fname, model, tokenizer))