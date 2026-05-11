from typing import Optional, Dict, Any, List
from generators.hf_generator import HFGenerator
from generators.syncode_generator import SyncodeGenerator
from generators.itergen_generator import ItergenGenerator
from generators.chopchop_generator import ChopchopGenerator
import argparse
import csv
import json
import time
from tqdm import tqdm
from pathlib import Path


class UnifiedCodeGenerator:
    def __init__(self, mode: str, model_name: str, grammar: Optional[str] = None, **kwargs):
        self.mode = mode
        self.model_name = model_name
        self.kwargs = kwargs
        self.generator = self._build_generator(grammar)

    def _build_generator(self, grammar):
        if self.mode == "syncode":
            return SyncodeGenerator(self.model_name, grammar, **self.kwargs)
        elif self.mode == "itergen":
            return ItergenGenerator(self.model_name, grammar, **self.kwargs)
        elif self.mode == "chopchop":
            from generators.javascript_chopchop import (
                CONSTRUCTORS, JS_START_RULE, JS_CONTEXT,
                make_js_pruner, extract_js_prefix, build_js_prompt,
            )
            pruner_mode = self.kwargs.get("pruner", "none")
            return ChopchopGenerator(
                self.model_name, grammar,
                constructors=CONSTRUCTORS,
                start_rule=JS_START_RULE,
                pruner_fn=make_js_pruner(pruner_mode),
                extract_prefix_fn=extract_js_prefix,
                build_prompt_fn=build_js_prompt,
                context=JS_CONTEXT,
                **self.kwargs,
            )
        elif self.mode == "unconstrained":
            return HFGenerator(self.model_name, **self.kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def generate(self, prompt: str, stop_tokens: Optional[List[str]], **kwargs) -> str:
        """Returns ONLY the raw generated string."""
        return self.generator.generate(prompt, stop_tokens=stop_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Sub-command: generate
# ---------------------------------------------------------------------------

def cmd_generate(args, parser):
    if args.mode in {"syncode", "itergen", "chopchop"} and not args.grammar:
        parser.error("--grammar is required for constrained modes: syncode, itergen, chopchop")

    model_name_clean = args.model.replace("/", "_").replace("-", "_")
    output_dir_name = f"{args.dataset_name}-js-{model_name_clean}-{args.temperature}-{args.mode}"
    final_output_path = Path(args.output_dir) / output_dir_name
    final_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating directory: {final_output_path.absolute()}")

    gen = UnifiedCodeGenerator(
        args.mode, args.model, args.grammar,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"--- Generating raw code for {len(tasks)} tasks ---")
    print(f"--- Results will be saved to: {final_output_path} ---")

    log_dir = Path(args.output_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timing_path = log_dir / f"{output_dir_name}_timing.csv"
    total_time = 0.0
    with open(timing_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["task_id", "task_name", "temperature", "execution_time_s"])

        for task in tqdm(tasks):
            task_name = task.get("name", task.get("task_id"))
            task_id = str(task_name).replace("/", "_")
            prompt_text = task["prompt"]

            t0 = time.perf_counter()
            result = gen.generate(
                prompt=prompt_text,
                stop_tokens=task.get("stop_tokens", ["\nfunction ", "\n/*", "\n//", "\nconsole.log"]),
                **vars(args),
            )
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            combined_output = f"{prompt_text.rstrip()}\n\n{result.rstrip()}"
            file_path = final_output_path / f"{task_id}.js"
            file_path.write_text(combined_output, encoding="utf-8")

            writer.writerow([task_id, task_name, args.temperature, f"{elapsed:.4f}"])

        writer.writerow(["TOTAL", "", "", f"{total_time:.4f}"])

    print(f"\nDone! All files saved in {final_output_path}")
    print(f"Timing saved to: {timing_path}")
    print(f"Total generation time: {total_time:.2f}s ({total_time/60:.1f}min)")


# ---------------------------------------------------------------------------
# Sub-command: evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args, parser):
    if args.mode in {"syncode", "itergen", "chopchop"} and not args.grammar:
        parser.error("--grammar is required for constrained modes: syncode, itergen, chopchop")

    generator = UnifiedCodeGenerator(model_name=args.model, **vars(args))

    model_name_clean = args.model.replace("/", "_").replace("-", "_")
    run_name = f"{args.dataset_name}-js-{model_name_clean}-{args.temperature}-{args.mode}"
    output_path = Path(args.output_base) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Step 1: Loading Dataset from {args.input_file}")
    tasks = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    print(f">>> Step 2: Generating Code (Mode: {args.mode})...")
    log_dir = Path(args.output_base) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timing_path = log_dir / f"{run_name}_timing.csv"
    total_time = 0.0
    with open(timing_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["task_name", "temperature", "num_completions", "execution_time_s", "time_per_completion_s"])

        for task in tqdm(tasks):
            task_name = task.get("name", task.get("task_id", task.get("id")))
            prompt = task["prompt"]
            stop_tokens = task.get("stop_tokens", ["\nfunction", "\n//", "\n/*"])
            safe_name = str(task_name).replace("/", "_")

            completions = []
            task_time = 0.0
            for _ in range(args.num_completions):
                t0 = time.perf_counter()
                code = generator.generate(
                    prompt=prompt,
                    mode=args.mode,
                    grammar=args.grammar,
                    stop_tokens=stop_tokens,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
                elapsed = time.perf_counter() - t0
                task_time += elapsed
                total_time += elapsed
                completions.append(code)

            result_item = task.copy()
            result_item["completions"] = completions

            json_file = output_path / (safe_name + ".json")
            with open(json_file, "w", encoding="utf-8") as f_json:
                json.dump(result_item, f_json, indent=2)

            writer.writerow([task_name, args.temperature, args.num_completions,
                             f"{task_time:.4f}", f"{task_time / args.num_completions:.4f}"])

        writer.writerow(["TOTAL", "", "", f"{total_time:.4f}", ""])

    print(f"\n>>> Generation Finished! JSON saved to: {output_path}")
    print(f"    Timing saved to: {timing_path}")
    print(f"    Total generation time: {total_time:.2f}s ({total_time/60:.1f}min)")
    print(f"    To compress and evaluate, run:")
    print(f"    python code_evaluation.py --input_dir {output_path} --benchmark multipl-e --pass_k {args.num_completions}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code generator and evaluator for constrained LLM benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- generate sub-command ------------------------------------------------
    gen_p = subparsers.add_parser("generate", help="Generate raw source-code files (.js / etc.)")
    gen_p.add_argument("--model", type=str, default="microsoft/phi-2")
    gen_p.add_argument("--mode", type=str, default="unconstrained",
                       choices=["unconstrained", "syncode", "itergen", "chopchop"])
    gen_p.add_argument("--grammar", type=str, default=None)
    gen_p.add_argument("--input_file", type=str, required=True)
    gen_p.add_argument("--output_dir", type=str, default="raw_outputs")
    gen_p.add_argument("--dataset_name", type=str, default="mbpp")
    gen_p.add_argument("--temperature", type=float, default=0.0)
    gen_p.add_argument("--max_new_tokens", type=int, default=512)
    gen_p.add_argument("--pruner", type=str, default="none",
                       choices=["none", "basic"],
                       help="Pruner mode for chopchop (none=identity, basic=env-aware JS pruning)")

    # -- evaluate sub-command ------------------------------------------------
    eval_p = subparsers.add_parser("evaluate", help="Batch generation — outputs MultiPL-E-compatible .json files")
    eval_p.add_argument("--model", type=str, default="microsoft/phi-2")
    eval_p.add_argument("--input_file", type=str, required=True)
    eval_p.add_argument("--dataset_name", type=str, default="mbpp")
    eval_p.add_argument("--output_base", type=str, default="results",
                        help="Base output directory for .json outputs")
    eval_p.add_argument("--mode", type=str, default="unconstrained",
                        choices=["unconstrained", "syncode", "itergen", "chopchop"])
    eval_p.add_argument("--grammar", type=str, default=None)
    eval_p.add_argument("--max_new_tokens", type=int, default=512)
    eval_p.add_argument("--temperature", type=float, default=0.0)
    eval_p.add_argument("--pruner", type=str, default="none",
                        choices=["none", "basic"],
                        help="Pruner mode for chopchop (none=identity, basic=env-aware JS pruning)")
    eval_p.add_argument("--num_completions", "-n", type=int, default=1,
                        help="Number of completions to generate per task (for pass@k with k>1)")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args, gen_p)
    elif args.command == "evaluate":
        cmd_evaluate(args, eval_p)
