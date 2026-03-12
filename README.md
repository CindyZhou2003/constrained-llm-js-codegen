# Constrained-llm-js-codegen

## Project Structure

`code_generator.py`: The core library and router. It handles model initialization and provides a CLI for raw code generation.

`code_evaluation.py`: The evaluation wrapper. It processes .jsonl datasets (like MBPP or HumanEval) and outputs files in the compressed .json.gz format required for the MultiPL-E eval pipeline.

`generators/`: A subdirectory containing the specific implementation strategies (e.g., `hf_generator.py` and `syncode_generator.py`).

`datasets/`: The directory that contains datasets of prompts we use to generate code.

`results/`: The directory that stores the results of code generation and evaluation for different models.


`tools/`: Files that help analyze code genration results of (un)constrained models.

- `unzip.py`: Unzip the results.json.gz files to the raw_results directory.
```bash
python tools/unzip.py ./results/mbpp-js-microsoft_phi_2-0.2`
```
- `count.py`: Summarize the cases results.
 ```bash
 # auto name the output file
python tools/count.py raw_results/mbpp-js-microsoft_phi_2-0.2
# specify the output name(optional)
python tools/count.py raw_results/mbpp-js-microsoft_phi_2-0.2 custom_name.txt 
```
- `diff.py`: Show the differences between different models 
```bash
python tools/diff.py summary/mbpp-js-microsoft_phi_2-0.0.txt summary/mbpp-js-microsoft_phi_2-0.0-syncode.txt
# specify the output name(optional)
python tools/diff.py summary/mbpp-js-microsoft_phi_2-0.0.txt summary/mbpp-js-microsoft_phi_2-0.0-syncode.txt custom_diff.txt 
```
- `extract_prompts.py`: extract OK->error cases from unconstrained models to constrained ones(one model to another)
```bash
# use default prompts file and output path
python tools/extract_prompts.py mbpp-js-microsoft_phi_2-0.0-unconstrained_vs_mbpp-js-microsoft_phi_2-0.0-syncode_diff.txt
# specify other prompts or output path
python tools/extract_prompts.py diff.txt datasets/js_prompts_mbpp.jsonl datasets/tem_custom.jsonl
```

## Setup

Create a python virtual environment and install all the packages in `requirements.txt`.
Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## MultiPL-E benchmark

We use MultiPL-E framework(`./benchmark/MultiPL-E`) to evaluate the code accuracy.

### Code generation

#### Generate target prompts

Go to the `./benchmark/MultiPL-E/dataset_builder` directory and run the following code, we can generate the target language prompts(change the lang and output directory if needed):

For humaneval dataset, we got a total 164 test cases:

``` bash
   python prepare_prompts_for_hfhub.py
      --lang humaneval_to_js.py # language translation file in MultiPL-E\dataset_builder
      --doctests transform
      --prompt-terminology reworded
      --output jsonl:../datasets/js_prompts_humaneval.jsonl # translated dataset file
      --original-dataset humaneval
      --originals ../datasets/originals-with-cleaned-doctests # original data dir
   ```

   For MBPP dataset, we got a total 397 test cases:

   ```bash
   python prepare_prompts_for_hfhub.py --lang humaneval_to_js.py --doctests transform --prompt-terminology reworded --output jsonl:../datasets/js_prompts_mbpp.jsonl --originals ../datasets/mbpp-typed --original-dataset mbpp
   ```

   The translated javascript prompts are already in `./datasets`.

#### Basic Generation
   
Use the `code_generator.py` CLI to quickly test how a model handles prompts. It reads a `.jsonl` file but outputs plain text files (or prints to the terminal).


```bash
python code_generator.py
  --model microsoft/phi-2 # model name
  --input_file datasets/js_prompts_mbpp.jsonl
  --mode syncode  # "unconstrained" or "syncode" or "itergen"
  --grammar javascript 
  --output_dir ./raw_outputs/ # output dir
  --temperature 0.2 # optional, default=0.0
  --max_new_tokens 512 # optional, default=512
```

**Input**: `.jsonl` file.
**Output**: Individual `.js` files containing only the code.

#### Batch Evaluation

Use the the `code_evaluation.py`.It processes the same `.jsonl` file but follows the strict MultiPL-E formatting and compression rules required for scoring.

```bash
   python code_evaluation.py 
      --model microsoft/phi-2  # model name
      --input_file datasets/js_prompts_mbpp.jsonl # input file
      --mode syncode # "unconstrained" or "syncode" or "itergen"
      --grammar syncode/javascript.lark # extra grammar file for constrained models, unnecessary for unconstrained ones
      --dataset_name mbpp # "humaneval" or "mbpp"
      --output_base results # ouput dir
      --temperature 0.2 # optional, default=0.0
      --max_new_tokens 512 # optional, default=512
   ```
**Input**: `.jsonl` file and possible grammar file

**Output**: Compressed `.json.gz` files containing code, task IDs, and prompt metadata.

### Evaluation

1. Go to the MultiPL-E directory:
```bash
   cd benchmark/MultiPL-E
   ```
2. Pull the evaluation image:
```bash
   docker pull ghcr.io/nuprl/multipl-e-evaluation
   ```
3. Tag the image as `multipl-e-eval`:
```bash
   docker tag ghcr.io/nuprl/multipl-e-evaluation multipl-e-eval
   ```
4. Run the evaluation container for your results directory. Replace `/absolute/path/to/results` with the absolute path to the directory containing your generated completions (the directory that has the `*.jsonl.gz` files):
```bash
   docker run --rm --network none \
     -v "/absolute/path/to/results:/tutorial:rw" \
     multipl-e-eval --dir /tutorial --output-dir /tutorial --recursive
   ```
   This maps your local results directory to `/tutorial` inside the Docker container.

5. Compute pass@k metrics on the evaluated results:
```bash
   python pass_k.py /absolute/path/to/results
   ```
   After running `pass_k.py`, it will output related `.results.json.gz` files in your results directory.

## HumanEval-X Benchmark Usage
### Code generation
1. Use the dataset at `benchmark/CodeGeeX/codegeex/benchmark/humaneval-x/js/data/humaneval_js.jsonl` for code generation. An example data entry is as follows:
   ```json
   {
      "task_id": "JavaScript/0", 
      "prompt": "/* Check if in given list of numbers, are any two numbers closer to each other than\n  given threshold.\n  >>> hasCloseElements([1.0, 2.0, 3.0], 0.5)\n  false\n  >>> hasCloseElements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n  true\n  */\nconst hasCloseElements = (numbers, threshold) => {\n", 
      "canonical_solution": "  for (let i = 0; i < numbers.length; i++) {\n    for (let j = 0; j < numbers.length; j++) {\n      if (i != j) {\n        let distance = Math.abs(numbers[i] - numbers[j]);\n        if (distance < threshold) {\n          return true;\n        }\n      }\n    }\n  }\n  return false;\n}\n\n", 
      "test": "const testHasCloseElements = () => {\n  console.assert(hasCloseElements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) === true)\n  console.assert(\n    hasCloseElements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) === false\n  )\n  console.assert(hasCloseElements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) === true)\n  console.assert(hasCloseElements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) === false)\n  console.assert(hasCloseElements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) === true)\n  console.assert(hasCloseElements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) === true)\n  console.assert(hasCloseElements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) === false)\n}\n\ntestHasCloseElements()\n", 
      "declaration": "\nconst hasCloseElements = (numbers, threshold) => {\n", 
      "example_test": "const testHasCloseElements = () => {\n  console.assert(hasCloseElements([1.0, 2.0, 3.0], 0.5) === false)\n  console.assert(\n    hasCloseElements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) === true\n  )\n}\ntestHasCloseElements()\n"
   }
   ```
2. Use the `benchmark/CodeGeeX/test_generate.py` to generate code and store the code under `benchmark/CodeGeeX/input_data`. Each **line** of the file should be a JSON object with `task_id` and `generation` fields. An example line in the file is as follows:
   ```json
   {"task_id": "JavaScript/0", "generation": "  for (let i = 0; i < numbers.length; i++) {\n    for (let j = 0; j < numbers.length; j++) {\n      if (i != j) {\n        let distance = Math.abs(numbers[i] - numbers[j]);\n        if (distance < threshold) {\n          return true;\n        }\n      }\n    }\n  }\n  return false;\n}\n\n"}
   ```
### Evaluation
1. Change directory to `benchmark/CodeGeeX`
   ```bash
   cd benchmark/CodeGeeX
   ```
2. Build the Docker image `humanevalx`
   ```bash
   docker build -t humanevalx .
   ``` 
3. Run the Docker container `jseval` and mount the `benchmark/CodeGeeX/input_data` directory to `/workspace/CodeGeeX/input_data` in the container
   ```bash
   docker run -it --mount type=bind,source=./input_data,target=/workspace/CodeGeeX/input_data --name jseval humanevalx
   ```
4. Inside the container, run the evaluation script to evaluate the generated code. 
    ```bash
    ./scripts/evaluate_humaneval_x.sh input_data/generations.jsonl js
    ```
5. The evaluation results will be appended to each line of the input file. An example output line is as follows:
   ```json
   {"task_id": "JavaScript/0", "completion_id": 0, "test_code": "...", "prompt": "...", "generation": "...", "result": "failed: ...", "passed": false, "finish": -1, "file": "", "output": []}
   ```
6. Exit the Docker container and the data is available in the `benchmark/CodeGeeX/input_data` directory.
   ```bash
   exit
   ```
