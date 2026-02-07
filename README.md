# constrained-llm-js-codegen

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
2. Store the generated code under `benchmark/CodeGeeX/input_data`. Each **line** of the file should be a JSON object with `task_id` and `generation` fields. An example line in the file is as follows:
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
    ./scripts/evaluate_humaneval_x.sh input_data/<filename> js
    ```
5. The evaluation results will be appended to each line of the input file. An example output line is as follows:
```json
{"task_id": "JavaScript/0", "completion_id": 0, "test_code": "...", "prompt": "...", "generation": "...", "result": "failed: ...", "passed": false, "finish": -1, "file": "", "output": []}
```
6. Exit the Docker container and the data is available in the `benchmark/CodeGeeX/input_data` directory.
   ```bash
   exit
   ```


## MultiPL-E
### Code generation
1. After running the following code, we can generate the target languange prompts:
``` bash
python prepare_prompts_for_hfhub.py
 --lang humaneval_to_js.py # replace
 --doctests transform
 --prompt-terminology reworded
 --output jsonl:../datasets/js_prompts.jsonl # replace
 --original-dataset humaneval
 --originals ../datasets/originals-with-cleaned-doctests
```
(baseline can just skip the step)

2. Use the dataset in `benchmark\MultiPL-E\datasets\js_prompts.jsonl` for code generation.
   
   ``` python
   python automodel.py
   --name bigcode/starcoder2-3b
   --root-dataset humaneval
   --lang js
   --temperature 0.2
   --batch-size 8
   --completion-limit 20
   --output-dir-prefix experiment
   ```
3. To use Syncode to generate results, see the syncode repo.
   