import pandas as pd
import os, sys
sys.path.append(os.getcwd())
from itergen.main import IterGen
from itergen.syncode.syncode.infer import Syncode
import json
from vega_lite_grammar import vega_lite_grammar
from vega_compile import run_vega_lite, check_json_equivalence
from tqdm import tqdm


def create_prompt(utterence, dataset):
    sys_prompt = """You are an expert AI model in data visualization, skilled at converting natural language descriptions into Vega-Lite JSON specifications. Vega-Lite is a high-level JSON-based visualization grammar for creating interactive and multi-view visualizations. Its specifications describe a single view or complex composed views, using properties such as mark (visual type) and encoding (mapping data fields to visual properties). Each JSON specification should begin with the following structure. 
{
    "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
    "data": {
        "url": "datasets/{dataset}.csv"
    }
}
    
Given a natural language request, output a Vega-Lite JSON object that meets the requirements of the request. Only include "$schema", "data", "mark", and "encoding" keys in the JSON object.

For example:
Request: "Show a bar chart of the number of houses in each city."
Dataset: houses
Data fields: "City", "Price", "Size"
Vega-Lite JSON Specification:
{
    "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
    "data": {
        "url": "datasets/houses.csv"
    },
    "mark": {"type": "bar"},
    "encoding": {
        "x": {"field": "City", "type": "nominal"},
        "y": {"aggregate": "count", "type": "quantitative", "axis": {"title": "COUNT"}}
	}
}


Each JSON object should accurately reflect the intent of the query, using appropriate Vega-Lite encoding, marks, and transformations. Use "datasets/{dataset}.csv" as the data source.
"""
    # Get the data fields from the CSV file
    file_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(file_dir, "nlvcorpus/datasets", f"{dataset}.csv"), 'r') as f:
        data_fields = f.readline().strip().split(',')
    data_fields = ", ".join(data_fields)

    task_prompt =f"Can you convert the given utterence into a VEGA-Lite specification?\nUtterence: {utterence}\nDataset:{dataset}\nData fields: {data_fields}\nVega-Lite JSON Specification:\n"

    prompt = sys_prompt + task_prompt
    print(f"\n{task_prompt}")
    return prompt


def eval_vgl_with_syncode(df, vl_specs, mode='original', temperature=None, model_id="meta-llama/Llama-3.2-3B", device='cuda:1'):
    # Initialize Syncode
    do_sample = True if temperature is not None else False
    syncode = Syncode(mode=mode, model=model_id, parse_output_only=True, do_sample=do_sample, temperature=temperature, max_new_tokens=400, grammar=vega_lite_grammar, device=device)

    # Result summary
    results = []
    pbar = tqdm(total=len(df))

    for i, row in df.iterrows():
        print(f"Problem {i}:")
        ut = row['Utterance Set']
        dataset = row['dataset'].lower()

        prompt = create_prompt(ut, dataset)

        # Get the ground truth Vega-Lite specification
        gt_name = dataset + "-" + str(row['visId'])
        gt_json = vl_specs[gt_name]

        out = syncode.infer(prompt, stop_words=['\n\n'])[0]
        # print(f"Utterence: {ut}")
        print(f"Vega-Lite: {out}")

        acc = check_json_equivalence(out, gt_json, dataset)
        warnings, is_warn = run_vega_lite(out, f"out/output_{i}.pdf")

        print(f"Errors: {warnings}")
        print(f"Has errors: {is_warn}")
        print(f"Exact match: {acc}")

        results.append({
            'Utterance': ut,
            'Dataset': dataset,
            'Ground Truth': gt_json,
            'Generated': out,
            'Match': acc,
            'Warnings': warnings,
            'Has Warnings': is_warn
        })
        pbar.update(1)
        print("-"*50)
    return results

def generate_vgl_with_itergen(iter_gen: IterGen, prompt, dataset):
    # Compute the temporal fields in the dataset
    all_fields, temporal_fields = extract_fields(dataset)

    iter_gen.start(prompt)
    iter = 0

    while not iter_gen.finished() and iter < 200:
        iter += 1
        out = iter_gen.forward(units=['other_property'])

        key = iter_gen.view("key")[0][-1]
        value = iter_gen.view("value")[0][-1]

        if key == '"aggregate"' and value not in ['"count"', '"mean"', '"average"', '"sum"']:
            iter_gen.backward('value')
        
        if key == '"field"':
            if value[1:-1] not in all_fields:
                iter_gen.backward('value')

        if key == '"type"':
            n = len(iter_gen.view("key")[0])
            for i in range(n-1, -1, -1):
                if iter_gen.view("key")[0][i] == '"field"':
                    field = iter_gen.view("value")[0][i]

            # If the field is a temporal field, only allow temporal types
            if field[1:-1] in temporal_fields and value not in ['"temporal"']:
                iter_gen.backward('value')
                # Need this for faster convergence
                # iter_gen.append('"temporal"')
    
    return out

def extract_fields(dataset):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nlvcorpus/datasets", f"{dataset}.csv"), 'r') as f:
        all_fields = f.readline().strip().split(',')

        temporal_fields = []
        for i, field in enumerate(all_fields):
            if any(s in field.lower() for s in ["date", "year", "month", "day"]):
                temporal_fields.append(field)
    return all_fields, temporal_fields

def eval_vgl_with_itergen(df, vl_specs, model_id="meta-llama/Llama-3.2-3B", device='cuda:1', temperature=None, recurrence_penalty=0.9):
    do_sample = True if temperature is not None else False
    
    iter_gen = IterGen(grammar=vega_lite_grammar, model_id=model_id, parse_output_only=True, do_sample=do_sample, temperature=temperature, stop_strings=['\n\n'], max_new_tokens=200, recurrence_penalty=recurrence_penalty, device=device)
    pbar = tqdm(total=len(df))

    results = []

    for i, row in df.iterrows():
        print(f"Problem {i}:")
        dataset = row['dataset'].lower()
        prompt = create_prompt(row['Utterance Set'], dataset)
        out = generate_vgl_with_itergen(iter_gen, prompt, dataset)[0]

        gt_name = dataset + "-" + str(row['visId'])
        gt_json = vl_specs[gt_name]

        acc = check_json_equivalence(out, gt_json, dataset)
        warnings, is_warn = run_vega_lite(out, f"out/output_{i}.pdf")
        print(f"Vega-Lite: {out}")
        print(f"Errors: {warnings}")
        print(f"Has errors: {is_warn}")
        print(f"Exact match: {acc}")

        results.append({
            'Utterance': row['Utterance Set'],
            'Dataset': dataset,
            'Ground Truth': gt_json,
            'Generated': out,
            'Match': acc,
            'Warnings': warnings,
            'Has Warnings': is_warn
        })
        pbar.update(1)
        print("-"*50)
    
    return results

def main():
    # Get the current file directory
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(file_dir, "nlv_data.csv"))

    # Extract ground truth jsons from nlvcorpus/vlSpecs.json
    with open(os.path.join(file_dir, "nlvcorpus/vlSpecs.json"), 'r') as f:
        vl_specs = json.load(f)

    # Models 
    models = ["meta-llama/Llama-3.2-3B"]
    evaluation_modes = ['itergen']

    # 830 problems in the dataset
    num_of_problems = 100
    device = 'cuda:1'
    df = df.head(num_of_problems)

    debug_problem = None
    
    for model_id in models:
        for evaluation_mode in evaluation_modes:
            if debug_problem is not None:
                df = df.iloc[[debug_problem]]
                
            if evaluation_mode == 'syncode':
                # Evaluate the Vega-Lite generation with Syncode
                results = eval_vgl_with_syncode(df, vl_specs, mode='grammar_strict', temperature=None, model_id=model_id, device=device)
            
            elif evaluation_mode == 'original':
                results = eval_vgl_with_syncode(df, vl_specs, mode='original', temperature=None, model_id=model_id, device=device)
            
            elif evaluation_mode == 'itergen':
                results = eval_vgl_with_itergen(df, vl_specs, model_id=model_id, device=device, temperature=None, recurrence_penalty=0.9)

            if debug_problem is None:
                sum_acc = sum([r['Match'] for r in results])    
                print(f"Average Accuracy: {sum_acc/len(df)}")

                avg_valid = (num_of_problems - sum([r['Has Warnings'] for r in results]))/num_of_problems
                print(f"Average Validity: {avg_valid}")

                # Store the result summary in a file
                result_summary = {
                    "model_id": model_id,
                    "evaluation_mode": evaluation_mode,
                    "num_of_problems": num_of_problems,
                    "Exact Accuracy": sum_acc/len(df),
                    "Average Validity": avg_valid,
                }

                # Make sure the results directory exists
                os.makedirs('results/nlv_results', exist_ok=True)

                # Add a line to the results file
                with open('results/nlv_results/results.jsonl', 'a') as f:
                    f.write(str(result_summary) + '\n')

if __name__ == "__main__":
    main()
