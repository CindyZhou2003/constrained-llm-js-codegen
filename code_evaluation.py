"""
Post-generation evaluation utilities: result compression and MultiPL-E benchmark execution.

Usage:
    python code_evaluation.py --input_dir raw_results/<run_name> [--gz_output_dir results/<run_name>] [--benchmark multipl-e]
"""

import argparse
import gzip
import json
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


def run_multipl_e_benchmark(gz_dir: Path):
    """Run MultiPL-E evaluation via Docker, then compute pass@k."""
    abs_gz_dir = gz_dir.resolve()

    print("\n>>> Running MultiPL-E Benchmark...")

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
        print("  Computing pass@k...")
        subprocess.run([sys.executable, str(pass_k_script), str(abs_gz_dir)], check=True)
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
             "Defaults to results/<run_name> where <run_name> is the last component of --input_dir.",
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        choices=list(BENCHMARK_RUNNERS),
        help="Benchmark to run after compression (e.g. multipl-e).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    gz_output_dir = (
        Path(args.gz_output_dir) if args.gz_output_dir
        else Path("results") / input_dir.name
    )

    print(f">>> Compressing results: {input_dir} -> {gz_output_dir}")
    compress_folder(input_dir, gz_output_dir)

    if args.benchmark:
        runner = BENCHMARK_RUNNERS[args.benchmark]
        runner(gz_output_dir)
