## Privacy Leakage Case Study

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

