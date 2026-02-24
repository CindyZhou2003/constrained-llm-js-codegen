## IterGen: Iterative structured LLM Generation with Backtracking

<p align="center">
  <img width="400" alt="syncode" src="https://github.com/user-attachments/assets/0845333b-8c54-4887-abd0-b4a7550fb8a6" />
</p>

IterGen enables precise control of text generation by integrating grammar rules and offering three core functions: `forward`, `backward`, and `view`. These functions allow programs to navigate the generation process with fine-grained control while decoding with Large Language Models (LLMs).

## Key Features

1. **Grammar-Aware Decoding**  
   Define custom grammars to structure generation using terminal and non-terminal symbols. IterGen is built on top of [SynCode](https://github.com/uiuc-focal-lab/syncode), a grammar-constrained generation framework.

2. **Forward Generation**  
   The `forward` function generates content up to a specified count of grammar symbols (e.g., sentences, words) or until meeting stopping conditions such as EOS tokens or a maximum token limit. Each call supports customized decoding parameters like temperature and sampling methods.

3. **Backtracking Support**  
   The `backward` function removes generated content by a specified count of grammar symbols, allowing corrections and refinements in real time.

4. **Output Inspection**  
   The `view` function retrieves parts of the generated text corresponding to specific grammar symbols, enabling validation or further analysis.

---
### Python Installation and Usage Instructions
Simply install SynCode via PyPi using the following command:
``` bash
pip install git+https://github.com/uiuc-arc/itergen.git
```


## 📜 Citation
<p>
    <a href="https://arxiv.org/abs/2410.07295"><img src="https://img.shields.io/badge/Paper-arXiv-blue"></a>
</p>

```
@misc{ugare2024itergeniterativestructuredllm,
      title={IterGen: Iterative Structured LLM Generation}, 
      author={Shubham Ugare and Rohan Gumaste and Tarun Suresh and Gagandeep Singh and Sasa Misailovic},
      year={2024},
      eprint={2410.07295},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2410.07295}, 
}
```

## Example: Using IterGen 

```python
from itergen.main import IterGen

# Define the grammar
grammar = """
paragraph: sentence+
sentence: word+ sentence_end
word: /[a-zA-Z0-9]+/ | other_punctuations
sentence_end: "." | "!" | "?"
other_punctuations: "," | ";" | ":" | "'" | "\""
%ignore " "
"""

# Initialize IterGen with the grammar and a model with Hugging Face model ID
iter_gen = IterGen(grammar=grammar, model_id="microsoft/Phi-3-mini-128k-instruct")
prompt = "Once upon a time"

# Start generation
iter_gen.start(prompt)

# Generate one sentence
generated_sentence = iter_gen.forward(stop_symbol="sentence", num=1)
print("Generated Sentence:", generated_sentence)

# Backtrack by 2 words
iter_gen.backward("word", num=2)

# Inspect all words in the current generation
current_words = iter_gen.view("word")
print("Current Words:", current_words)
```


## Case Studies
The following case studies demonstrate the utility of IterGen in various applications as described in the [paper](https://arxiv.org/abs/2410.07295).

<details>
  <summary> SQL Case Study</summary>

IterGen can be used to improve SQL generation by backtracking and correcting errors when the table names and column names are not present in the database schema. The SQL case study demonstrates how IterGen improves the accuracy of SQL generation by correcting errors in the generated SQL code.

```python

To run all SQL experiments, use the following command:
```bash
python3 case_studies/sql/eval_sql.py
```

### Results
The results of the experiments will be stored in the `results/sql_results/results.jsonl` file. Each line in this file contains detailed information about the performance of a model on different SQL tasks.

Field Explanation Table:

| Field | Description |
| --- | --- |
| `model_id` | Model id used for the experiment |
| `evaluation_mode` | Evaluation mode used for the experiment |
| `temperature` | Temperature used for the experiment |
| `recurrence_penalty` | Recurrence penalty used for the experiment |
| `seed` | Seed used for the experiment |
| `execution_accuracy` | Accuracy of the model on different difficulty levels (easy, medium, hard, extra, all) |
| `Counts` | Number of problems in each difficulty level |
| `error_types` | Error types of the model |
| `results_jsonl_file` | Path to the result jsonl file |
| `avg_num_tokens` | Average number of tokens in the generated code |
| `num_of_problems` | Number of problems in the experiment |
| `Average time` | Average time taken to generate the code |

An example of a result entry looks like this:
```javascript
{
    'model_id': 'model_id', 
    'evaluation_mode': 'itergen', 
    'temperature': None, 
    'recurrence_penalty': 0.3, 
    'seed': 0, 
    'execution_accuracy': [
        ('easy', 0.5), 
        ('medium', 0.667), 
        ('hard', 0), 
        ('extra', 0), 
        ('all', 0.6)
    ], 
    'Counts': [
        ('easy', 4), 
        ('medium', 6), 
        ('hard', 0), 
        ('extra', 0), 
        ('all', 10)
    ], 
    'error_types': {
        'Valid': 10
    }, 
    'results_jsonl_file': 'results/sql_results/model_id/itergen_tempt:None_seed:0_rp:0.3_maxiter_20_num:10.jsonl', 
    'avg_num_tokens': 26.6, 
    'num_of_problems': 10, 
    'Average time': 1.313
}
```

#### Comparison script

To run the script to compare two result jsonl file the command line, use the following format:

```bash
python case_studies/sql/compare.py --file1 <path_to_file1> --file2 <path_to_file2>
```
</details>

<details>
  <summary> Privacy Leakage Case Study </summary>
   
To run a single privacy case study, use the following command:
```bash
python3 case_studies/privacy/privacy_evaluation.py
```

To run all the experiments from section 4.2, run: 
To run all a privacy case study, use the following command:
```bash
chmod +x run_inst.sh
./run_inst.sh
```

### Results
The results of the experiments will be stored in the `results/privacy/` folder, as a `...completions.json` file containing generation data, along with `...results.jsonl` file, containing evaluation metrics. 

Field Explanation Table:

| Field | Description |
| --- | --- |
| `dataset` | Dataset used for the experiment|
| `model_id` | Model id used for the experiment |
| `evaluation_mode` | Evaluation mode used for the experiment |
| `recurrence_penalty` | Recurrence penalty used for the experiment |
| `max_tokens` | Maximum number of tokens allowed |
| `leak_rate_email` | Percentage of emails generated that were leaks | 
| `avg_tokens` | Average number of tokens in the generated code |
| `avg_time` | Average amount of time taken per generation |
| `num_prompts` | Number of prompts used for evaluation |
| `perplexity` | Perplexity score of the generated response|

An example of a result entry looks like this:
```javascript
{"dataset": "enron", "model": "Qwen/Qwen2.5-0.5B", "recurrence_penalty": 0.3, "max_tokens": 19, "leak_rate_email": 0.0, "avg_tokens": 24.14, "avg_time": 0.46, "num_prompts": 100, "mode": "itergen", "perplexity": 6.314}
```
