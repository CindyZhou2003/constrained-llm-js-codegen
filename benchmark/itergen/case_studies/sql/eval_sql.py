import sys, os
import tempfile
import time
from typing import Union

import torch
sys.path.append(os.getcwd())
from itergen.syncode.syncode.infer import Syncode
from itergen.main import IterGen
from itergen import Dataset
from itergen.syncode.syncode.evaluation.sql_eval import SQLEval
from tqdm import tqdm
from mxeval.data import write_jsonl

sql_dataset = Dataset('spider', language='sql')

def format_prompt(prompt: str, model: str) -> Union[str, list]:
    """
    If the model is instruct-tuned model then we format the prompt as follows:
    [{role: user, content: prompt}]

    else:
    prompt = prompt
    """
    prompt = prompt[:-5] + " Only output the SQL quey. \nSQL:"
    if 'instruct' in model or 'Instruct' in model or 'it' in model or 'chat' in model:
        return [{"role": "user", "content": prompt}]
    return prompt


def evaluate_sql_with_syncode(samples, f, problems, pbar, mode='original', temperature=0.2, model_id="meta-llama/Llama-2-7b-chat-hf", device='cuda:0'):
    do_sample = True if temperature is not None else False
    syncode = Syncode(mode=mode, model=model_id, parse_output_only=True, do_sample=do_sample, temperature=temperature, max_new_tokens=200, grammar='sql', device=device)
    total_tokens = 0

    for task_id, problem in enumerate(problems):
        prompt = format_prompt(problem['prompt'], model_id)
        out = syncode.infer(prompt, stop_words=['\n\n'])[0]
        completion = sql_dataset.post_process_answer(out)
        total_tokens += syncode.model.total_tokens
        res = dict(
            task_id=task_id,
            completion=completion,
        )
        samples += [res]
        f.write(completion + '\n')
        pbar.update(1)
        print([out])

    average_tokens = total_tokens / len(problems)
    pbar.close()
    f.close()
    return average_tokens

def evaluate_sql_with_itergen(samples, f, problems, pbar, temperature=0.2, model_id="meta-llama/Llama-2-7b-chat-hf", recurrence_penalty=0.1, max_iter=25, device='cuda:0'):
    do_sample = True if temperature is not None else False

    iter_gen = IterGen(grammar='sql', model_id=model_id, parse_output_only=True, do_sample=do_sample, temperature=temperature, stop_strings=['\n\n'], max_new_tokens=200, recurrence_penalty=recurrence_penalty, device=device)

    sum_tokens = 0

    for task_id, problem in enumerate(problems):
        # iter_gen.start(problem['prompt'])        
        # out = iter_gen.forward()

        out, metadata = generate_sql_with_itergen(iter_gen, problem, max_iter=max_iter)
        sum_tokens += metadata['total_tokens']

        raw_completion = out[0]
        completion = raw_completion

        completion = sql_dataset.post_process_answer(raw_completion)
        res = dict(
            task_id=task_id,
            completion=completion,
        )
        samples += [res]
        f.write(completion + '\n')
        pbar.update(1)

    print('Average number of tokens:', sum_tokens/len(problems))
    pbar.close()
    f.close()
    return samples, sum_tokens/len(problems)

def parse_sql_schema(problem):
    """
    problem['db_info'] is a string with the schema information as follows:
    # stadium ( stadium_id , location , name , capacity , highest , lowest , average )\n# singer ( singer_id , name , country , song_name , song_release_year , age , is_male )\n# concert ( concert_id , concert_name , theme , stadium_id , year )\n# singer_in_concert ( concert_id , singer_id )\n# concert.stadium_id = stadium.stadium_id\n# singer_in_concert.singer_id = singer.singer_id\n# singer_in_concert.concert_id = concert.concert_id\n

    Returns:
    schema: dict with tables and columns
    """
    schema = {}
    db_info = problem['db_info']
    tables = db_info.split('#')[1:]
    for table in tables:
        try:
            table_name, columns = table.split('(')
            columns = columns.split(')')[0].split(',')
        except:
            continue
        # Since SQL is case-insensitive, we convert everything to lowercase
        table_name = table_name.strip().lower()
        schema[table_name] = [col.strip().lower() for col in columns]
    return schema

## Helper functions to check if a column or table exists in the schema
def exists_column(schema: dict, column_name: str) -> bool:
    column_name = column_name.strip().lower()
    if column_name == '*': return True

    if '.' in column_name:
        try:
            table_name, column_name = column_name.split('.')
            if table_name in schema:
                return column_name in schema[table_name]
        except:
            pass
        return False
    else:
        # We don't know exactly which table the column belongs to so here we check all tables
        for _, columns in schema.items():
            if column_name in columns:
                return True
    return False
    
def exists_table(schema: dict, table_name: str) -> bool:
    table_name = table_name.strip().lower()
    return table_name in schema.keys()


def generate_sql_with_itergen(iter_gen: IterGen, problem: dict, max_iter: int):
    message = format_prompt(problem['prompt'], iter_gen.model_id)
    iter_gen.start(message)

    schema = parse_sql_schema(problem)
    num_backwards = 0
    backwards_limit = 10
    iter = 0

    while not iter_gen.finished() and iter < max_iter:
        iter += 1
        out = iter_gen.forward(units=['column_name', 'table_name'], num=1)

        column_names = iter_gen.view('column_name')[0]
        last_column = column_names[-1] if column_names else None        
        if last_column!=None and not exists_column(schema, last_column) and num_backwards < backwards_limit:
            iter_gen.backward('column_name')
            num_backwards += 1
            continue
        
        table_names = iter_gen.view('table_name')[0]
        last_table = table_names[-1] if table_names else None
        if last_table != None and not exists_table(schema, last_table) and num_backwards < backwards_limit:
            iter_gen.backward('table_name')
            num_backwards += 1
            continue     
        print(out)
    print(iter_gen._metadata)
    return out, iter_gen._metadata


def eval_sql():
    # Configuration
    models = ["Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-7B", "meta-llama/Llama-2-7b-chat-hf", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Meta-Llama-3-8B"]
    evaluation_modes = ['original', 'syncode', 'itergen']
    temperature = None
    
    models = ["Qwen/Qwen2.5-0.5B"]
    # try grammar_mask for both
    evaluation_modes = ['itergen']

    # 1034 problems in the dataset
    num_of_problems = 1034
    device = 'cuda:0'
    
    # Set to None to evaluate all problems or temporarily set to a specific problem to debug
    debug_problem, long_experiment = None, False

    # Fix seed for reproducibility
    seed = 0
    torch.manual_seed(seed)

    for model_id in models:
        for evaluation_mode in evaluation_modes:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    recurrence_penalty = 0.3 if evaluation_mode == 'itergen' else None
                    max_iter = 20 if evaluation_mode == 'itergen' else None

                    time_start = time.time()
                    samples, avg_num_tokens = [], None

                    results_jsonl_file = f'results/sql_results/{model_id}/{evaluation_mode}_tempt:{temperature}_seed:{seed}_rp:{recurrence_penalty}_maxiter_{max_iter}_num:{num_of_problems}.jsonl'
                    
                    # Open a temporary file to store the results
                    tmp_res_file = str(tmp_dir) + '/sql_res_file.txt'  
                    tmpf = open(tmp_res_file, 'w')

                    problems = sql_dataset.problems[:num_of_problems]

                    if debug_problem is not None: problems = [problems[debug_problem]]
                    pbar = tqdm(total=len(problems))

                    if evaluation_mode == 'syncode':
                        avg_num_tokens = evaluate_sql_with_syncode(samples, tmpf, problems, pbar, mode='grammar_strict', temperature=temperature, model_id=model_id, device=device)
                    elif evaluation_mode == 'original':
                        avg_num_tokens = evaluate_sql_with_syncode(samples, tmpf, problems, pbar, mode='original', temperature=temperature, model_id=model_id, device=device)
                    elif evaluation_mode == 'itergen':
                        _, avg_num_tokens = evaluate_sql_with_itergen(samples, tmpf, problems, pbar, temperature=temperature, model_id=model_id, recurrence_penalty=recurrence_penalty, max_iter=max_iter, device=device)

                    scores, error_types, results_jsonl = SQLEval.compute_accuracy(samples, tmp_res_file)

                    print("Execution accuracy:", scores['all']['exec']) 
                    print(f"Compilation error types: {error_types}")

                    if debug_problem is None:
                        os.makedirs(os.path.dirname(results_jsonl_file), exist_ok=True)
                        write_jsonl(results_jsonl_file, results_jsonl)

                        # Store the result summary in a file
                        result_summary = {
                            "model_id": model_id,
                            "evaluation_mode": evaluation_mode,
                            "temperature": temperature,
                            "recurrence_penalty": recurrence_penalty,
                            "seed": seed,
                            "execution_accuracy": [(lvl, round(scores[lvl]['exec'], 3)) for lvl in scores.keys()],
                            "Counts": [(lvl, round(scores[lvl]['count'], 3)) for lvl in scores.keys()],
                            "error_types": error_types,
                            "results_jsonl_file": results_jsonl_file,
                            "avg_num_tokens": avg_num_tokens,
                            "num_of_problems": num_of_problems,
                            "Average time": round((time.time() - time_start) / num_of_problems, 3)
                        }

                        # Add a line to the results file
                        with open('results/sql_results/results.jsonl', 'a') as f:
                            f.write(str(result_summary) + '\n')
            except Exception as e:
                # Store the error in the results file and continue 
                with open('results/sql_results/results.jsonl', 'a') as f:
                    result_summary = {
                        "model_id": model_id,
                        "evaluation_mode": evaluation_mode,
                        "temperature": temperature,
                        "recurrence_penalty": recurrence_penalty,
                        "seed": seed,
                        "error": str(e)
                    }
                    f.write(str(result_summary) + '\n')
                if not long_experiment: raise e

if __name__ == '__main__':
    eval_sql()
