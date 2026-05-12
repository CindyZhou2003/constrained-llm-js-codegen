"""
Post-generation evaluation utilities: result compression and MultiPL-E benchmark execution.

Usage:
    python code_evaluation.py --input_dir raw_results/<run_name> [--gz_output_dir results_eval/<run_name>] [--benchmark multipl-e]
"""

import argparse
import csv
import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def compress_folder(src_dir: Path, dst_dir: Path) -> int:
    """Compress all .json files in src_dir to .json.gz in dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        print(f"  No .json files found in {src_dir}")
        return 0
    count = 0
    for json_file in json_files:
        gz_file = dst_dir / (json_file.stem + ".json.gz")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        with gzip.open(gz_file, "wt", encoding="utf-8") as f_gz:
            json.dump(data, f_gz)
        count += 1
    print(f"  Compressed {count} files: {src_dir} -> {dst_dir}")
    return count


def _parse_run_path(gz_dir: Path):
    """Extract (dataset, model, temperature, mode) from a path like .../<model>/<dataset>-js/T<temp>_<mode>."""
    run_dir = gz_dir.name
    dataset = gz_dir.parent.name
    model = gz_dir.parent.parent.name
    m = re.match(r'^T(\d+(?:\.\d+)?)_(.+)$', run_dir)
    if m:
        return dataset, model, m.group(1), m.group(2)
    return dataset, model, '', ''


def _lookup_total_time(gz_dir: Path) -> str:
    """Look up total generation time from the timing CSV for a given gz directory."""
    run_dir = gz_dir.name
    dataset = gz_dir.parent.name
    model = gz_dir.parent.parent.name
    timing_path = (
        Path(__file__).parent / "results" / model / dataset / "log" / f"{run_dir}_timing.csv"
    )
    if not timing_path.exists():
        return ""
    total = 0.0
    with open(timing_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("task_name", row.get("task_id", "")) == "TOTAL":
                return row["execution_time_s"]
            total += float(row["execution_time_s"])
    return f"{total:.4f}"


def _append_to_results_csv(pass_k_output: str, gz_dir: Path) -> None:
    """Append pass@k result rows to results.csv, creating the file with a header if needed."""
    csv_path = Path(__file__).parent / "results.csv"
    lines = [line for line in pass_k_output.splitlines() if line.strip()]
    # First line is the header emitted by pass_k.py; the rest are data rows.
    data_lines = lines[1:] if len(lines) > 1 else []
    if not data_lines:
        return
    total_time = _lookup_total_time(gz_dir)
    dataset, model, temperature, mode = _parse_run_path(gz_dir)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        if write_header:
            f.write("Dataset,Model,Temperature,Mode,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions,TotalTime_s\n")
        for line in data_lines:
            # Replace the full absolute path (first CSV field) with the parsed columns.
            _, _, rest = line.partition(",")
            f.write(f"{dataset},{model},{temperature},{mode},{rest},{total_time}\n")
    print(f"  Results appended to: {csv_path}")


def run_multipl_e_benchmark(gz_dir: Path, pass_k: int = 1):
    """Run MultiPL-E evaluation via Docker, then compute pass@k."""
    abs_gz_dir = gz_dir.resolve()

    print("\n>>> Running MultiPL-E Benchmark...")

    if shutil.which("docker") is None:
        print(
            "Error: 'docker' executable not found in PATH.\n"
            "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/ "
            "and ensure it is running before re-running this command.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("  Pulling Docker image...")
    subprocess.run(
        ["docker", "pull", "ghcr.io/nuprl/multipl-e-evaluation"],
        check=True,
    )

    print("  Tagging image...")
    subprocess.run(
        ["docker", "tag", "ghcr.io/nuprl/multipl-e-evaluation", "multipl-e-eval"],
        check=True,
    )

    print(f"  Running evaluation on {abs_gz_dir}...")
    subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{abs_gz_dir}:/tutorial:rw",
            "multipl-e-eval",
            "--dir", "/tutorial",
            "--output-dir", "/tutorial",
            "--recursive",
        ],
        check=True,
    )

    pass_k_script = Path(__file__).parent / "benchmark" / "MultiPL-E" / "pass_k.py"
    if pass_k_script.exists():
        k_label = f"pass@1" if pass_k <= 1 else f"pass@1 and pass@{pass_k}"
        print(f"  Computing {k_label}...")
        cmd = [sys.executable, str(pass_k_script), str(abs_gz_dir)]
        if pass_k > 1:
            cmd += ["-k", str(pass_k)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout, end="")
        _append_to_results_csv(result.stdout, abs_gz_dir)
    else:
        print(f"  Warning: pass_k.py not found at {pass_k_script}. Run manually:")
        print(f"    python benchmark/MultiPL-E/pass_k.py {abs_gz_dir}")

    print(f"\n  Benchmark complete! Results in: {abs_gz_dir}")


BENCHMARK_RUNNERS = {
    "multipl-e": run_multipl_e_benchmark,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress .json generation outputs and optionally run benchmark evaluation."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing uncompressed .json generation outputs.",
    )
    parser.add_argument(
        "--gz_output_dir", type=str, default=None,
        help="Output directory for .json.gz files. "
             "Defaults to results_eval/<run_name> where <run_name> is the last component of --input_dir.",
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        choices=list(BENCHMARK_RUNNERS),
        help="Benchmark to run after compression (e.g. multipl-e).",
    )
    parser.add_argument(
        "--pass_k", type=int, default=1,
        help="Compute pass@1 and pass@k (e.g. --pass_k 5). Requires at least k completions per problem.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.gz_output_dir:
        gz_output_dir = Path(args.gz_output_dir)
    else:
        # Mirror the trailing <model>/<dataset>-js/<run_dir> structure under results_eval/
        parts = input_dir.parts[-3:]
        gz_output_dir = Path("results_eval", *parts)

    print(f">>> Compressing results: {input_dir} -> {gz_output_dir}")
    compress_folder(input_dir, gz_output_dir)

    if args.benchmark:
        runner = BENCHMARK_RUNNERS[args.benchmark]
        runner(gz_output_dir, pass_k=args.pass_k)
