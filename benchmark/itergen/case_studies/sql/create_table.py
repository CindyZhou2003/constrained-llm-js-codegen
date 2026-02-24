import json
from collections import defaultdict

# Sample input string where each line is a JSON object
json_string = """
{'model_id': 'Qwen/Qwen2.5-0.5B', 'evaluation_mode': 'original', 'temperature': None, 'recurrence_penalty': None, 'seed': 0, 'execution_accuracy': [('easy', 0.416), ('medium', 0.264), ('hard', 0.259), ('extra', 0.1), ('all', 0.273)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 472, 'Other Error': 494, 'Syntax Error': 68}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-0.5B/original_tempt:None_seed:0_rp:None_maxiter_None_num:1034.jsonl', 'avg_num_tokens': 56.823984526112184, 'num_of_problems': 1034, 'Average time': 0.881}
{'model_id': 'Qwen/Qwen2.5-0.5B', 'evaluation_mode': 'syncode', 'temperature': None, 'recurrence_penalty': None, 'seed': 0, 'execution_accuracy': [('easy', 0.424), ('medium', 0.275), ('hard', 0.264), ('extra', 0.094), ('all', 0.279)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 487, 'Other Error': 483, 'Syntax Error': 64}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-0.5B/syncode_tempt:None_seed:0_rp:None_maxiter_None_num:1034.jsonl', 'avg_num_tokens': 55.05899419729207, 'num_of_problems': 1034, 'Average time': 1.299}
{'model_id': 'Qwen/Qwen2.5-0.5B', 'evaluation_mode': 'itergen', 'temperature': None, 'recurrence_penalty': 0.3, 'seed': 0, 'execution_accuracy': [('easy', 0.548), ('medium', 0.314), ('hard', 0.339), ('extra', 0.124), ('all', 0.343)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 627, 'Syntax Error': 133, 'Other Error': 274}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-0.5B/itergen_tempt:None_seed:0_rp:0.3_maxiter_20_num:1034.jsonl', 'avg_num_tokens': 49.188588007736946, 'num_of_problems': 1034, 'Average time': 1.231}
{'model_id': 'Qwen/Qwen2.5-1.5B', 'evaluation_mode': 'original', 'temperature': None, 'recurrence_penalty': None, 'seed': 0, 'execution_accuracy': [('easy', 0.712), ('medium', 0.473), ('hard', 0.385), ('extra', 0.276), ('all', 0.484)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 813, 'Other Error': 190, 'Syntax Error': 31}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-1.5B/original_tempt:None_seed:0_rp:None_maxiter_None_num:1034.jsonl', 'avg_num_tokens': 42.36750483558994, 'num_of_problems': 1034, 'Average time': 0.747}
{'model_id': 'Qwen/Qwen2.5-1.5B', 'evaluation_mode': 'syncode', 'temperature': None, 'recurrence_penalty': None, 'seed': 0, 'execution_accuracy': [('easy', 0.724), ('medium', 0.48), ('hard', 0.391), ('extra', 0.282), ('all', 0.491)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 822, 'Other Error': 181, 'Syntax Error': 31}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-1.5B/syncode_tempt:None_seed:0_rp:None_maxiter_None_num:1034.jsonl', 'avg_num_tokens': 42.25918762088975, 'num_of_problems': 1034, 'Average time': 1.025}
{'model_id': 'Qwen/Qwen2.5-1.5B', 'evaluation_mode': 'itergen', 'temperature': None, 'recurrence_penalty': 0.3, 'seed': 0, 'execution_accuracy': [('easy', 0.736), ('medium', 0.486), ('hard', 0.397), ('extra', 0.282), ('all', 0.498)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 843, 'Syntax Error': 66, 'Other Error': 125}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-1.5B/itergen_tempt:None_seed:0_rp:0.3_maxiter_20_num:1034.jsonl', 'avg_num_tokens': 44.54061895551257, 'num_of_problems': 1034, 'Average time': 1.19}
{'model_id': 'Qwen/Qwen2.5-0.5B-Instruct', 'evaluation_mode': 'original', 'temperature': None, 'recurrence_penalty': None, 'seed': 0, 'execution_accuracy': [('easy', 0.028), ('medium', 0.002), ('hard', 0.006), ('extra', 0.006), ('all', 0.01)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Syntax Error': 947, 'Other Error': 63, 'Valid': 24}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-0.5B-Instruct/original_tempt:None_seed:0_rp:None_maxiter_None_num:1034.jsonl', 'avg_num_tokens': 58.648936170212764, 'num_of_problems': 1034, 'Average time': 0.883}
    {'model_id': 'Qwen/Qwen2.5-0.5B-Instruct', 'evaluation_mode': 'syncode', 'temperature': None, 'recurrence_penalty': None, 'seed': 0, 'execution_accuracy': [('easy', 0.18), ('medium', 0.059), ('hard', 0.103), ('extra', 0.053), ('all', 0.095)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Other Error': 552, 'Valid': 298, 'Syntax Error': 184}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-0.5B-Instruct/syncode_tempt:None_seed:0_rp:None_maxiter_None_num:1034.jsonl', 'avg_num_tokens': 103.04642166344294, 'num_of_problems': 1034, 'Average time': 2.801}
    {'model_id': 'Qwen/Qwen2.5-0.5B-Instruct', 'evaluation_mode': 'itergen', 'temperature': None, 'recurrence_penalty': 0.3, 'seed': 0, 'execution_accuracy': [('easy', 0.368), ('medium', 0.234), ('hard', 0.31), ('extra', 0.124), ('all', 0.261)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 670, 'Other Error': 306, 'Syntax Error': 58}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-0.5B-Instruct/itergen_tempt:None_seed:0_rp:0.3_maxiter_20_num:1034.jsonl', 'avg_num_tokens': 39.688588007736946, 'num_of_problems': 1034, 'Average time': 1.046}
{'model_id': 'Qwen/Qwen2.5-1.5B-Instruct', 'evaluation_mode': 'original', 'temperature': None, 'recurrence_penalty': None, 'seed': 0, 'execution_accuracy': [('easy', 0.0), ('medium', 0.0), ('hard', 0.0), ('extra', 0.0), ('all', 0.0)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Syntax Error': 1031, 'Other Error': 3}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-1.5B-Instruct/original_tempt:None_seed:0_rp:None_maxiter_None_num:1034.jsonl', 'avg_num_tokens': 45.36943907156673, 'num_of_problems': 1034, 'Average time': 0.819}
{'model_id': 'Qwen/Qwen2.5-1.5B-Instruct', 'evaluation_mode': 'syncode', 'temperature': None, 'recurrence_penalty': None, 'seed': 0, 'execution_accuracy': [('easy', 0.436), ('medium', 0.298), ('hard', 0.339), ('extra', 0.259), ('all', 0.332)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 639, 'Other Error': 261, 'Syntax Error': 134}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-1.5B-Instruct/syncode_tempt:None_seed:0_rp:None_maxiter_None_num:1034.jsonl', 'avg_num_tokens': 74.31721470019342, 'num_of_problems': 1034, 'Average time': 2.023}
{'model_id': 'Qwen/Qwen2.5-1.5B-Instruct', 'evaluation_mode': 'itergen', 'temperature': None, 'recurrence_penalty': 0.3, 'seed': 0, 'execution_accuracy': [('easy', 0.616), ('medium', 0.477), ('hard', 0.506), ('extra', 0.429), ('all', 0.508)], 'Counts': [('easy', 250), ('medium', 440), ('hard', 174), ('extra', 170), ('all', 1034)], 'error_types': defaultdict(<class 'int'>, {'Valid': 829, 'Other Error': 158, 'Syntax Error': 47}), 'results_jsonl_file': 'results/sql_results/Qwen/Qwen2.5-1.5B-Instruct/itergen_tempt:None_seed:0_rp:0.3_maxiter_20_num:1034.jsonl', 'avg_num_tokens': 40.24564796905222, 'num_of_problems': 1034, 'Average time': 1.143}
"""

# Function to parse JSON data from the string (each line is a JSON object)
def parse_json_string(json_string):
    json_lines = json_string.strip().split('\n')
    json_data = []
    for line in json_lines:
        # Replace defaultdict(<class 'int'>, {'Valid': 813, 'Other Error': 190, 'Syntax Error': 31}) with a valid JSON object
        import regex
        line = regex.sub(r'defaultdict\(<class \'int\'>, ({.*})\)', r'\1', line)

        line = line.replace("(", '[')
        line = line.replace(")", ']')

        line = line.replace("'", '"')
        line = line.replace("None", '0')

        line = line.replace("syncode", 'SynCode')
        line = line.replace("original", 'Original')
# 
        print(line[79:])
        json_data.append(json.loads(line))
    return json_data

# Function to create a LaTeX table from the JSON data
def json_to_latex(json_data):
    table = r"""
\begin{table}[htbp]
    \scriptsize
    \centering
    \begin{tabular}{llcccccccc}
        \toprule
        \multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{5}{c}{\textbf{Accuracy (\%)}} & \multirow{2}{*}{\textbf{Compile (\%)}} & \multirow{2}{*}{\textbf{Tokens}} & \multirow{2}{*}{\textbf{Time (s)}} \\
        \cmidrule(lr){3-7}
        & & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} & \textbf{Extra} & \textbf{Overall} & & & \\
        \midrule
    """
    
    # Sort the JSON data by model_id
    # Give the order as "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B", "microsoft/Phi-3-mini-128k-instruct", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/CodeLlama-7b-hf"

    order = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B", "microsoft/Phi-3-mini-128k-instruct", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/CodeLlama-7b-hf"]
    json_data = sorted(json_data, key=lambda x: order.index(x['model_id']))

    for id, entry in enumerate(json_data):
        if id % 3 == 0 and id != 0:
            table += r"""
            \midrule
            """
        model_id = entry['model_id'].split('/')[-1] if entry['evaluation_mode'] == 'SynCode' else ""
        method = entry['evaluation_mode'] if entry['evaluation_mode'] != "itergen" else "\\Tool{}"
        accuracy = {level: acc * 100 for level, acc in entry['execution_accuracy']}
        avg_tokens = entry.get('avg_num_tokens', '-')
        avg_time = entry.get('Average time', '-')
        
        # Calculate Compile (%) as the percentage of valid runs
        num_problems = entry.get('num_of_problems', 1)
        valid_count = entry['error_types'].get('Valid', 0)
        compile_percentage = (valid_count / num_problems) * 100 if num_problems > 0 else 0

        # Add a row to the LaTeX table
        if entry['evaluation_mode'] == 'itergen':
            # Make the IterGen row bold
            table += f"\\textbf{{{model_id}}} & {method} & \\textbf{{{accuracy['easy']:.1f}}} & \\textbf{{{accuracy['medium']:.1f}}} & \\textbf{{{accuracy['hard']:.1f}}} & \\textbf{{{accuracy['extra']:.1f}}} & \\textbf{{{accuracy['all']:.1f}}} & \\textbf{{{compile_percentage:.1f}}} & {avg_tokens:.2f} & {avg_time:.3f} \\\\\n"
        else:
            table += f"{model_id} & {method} & {accuracy['easy']:.1f} & {accuracy['medium']:.1f} & {accuracy['hard']:.1f} & {accuracy['extra']:.1f} & {accuracy['all']:.1f} & {compile_percentage:.1f} & {avg_tokens:.2f} & {avg_time:.3f} \\\\\n"

    table += r"""
        \bottomrule
    \end{tabular}
    \caption{Comparison of models on SQL based on accuracy, compile percentage, number of tokens, and average time.}
    \label{tab:model_comparison}
\end{table}
    """
    return table

# Main function to process the input and generate the LaTeX table
def main():
    # Parse JSON string into a list of dictionaries
    json_data = parse_json_string(json_string)
    
    # Convert the JSON data to LaTeX table
    latex_table = json_to_latex(json_data)
    
    # Print or save the LaTeX table
    print(latex_table)

# Run the main function
main()
