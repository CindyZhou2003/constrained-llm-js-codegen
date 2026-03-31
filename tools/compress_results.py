"""
Compress .json files to .json.gz for MultiPL-E evaluation.

Usage:
    # Compress a specific raw_outputs folder into results/
    python compress_results.py raw_outputs/mbpp-js-microsoft_phi_2-0.0-itergen

    # Specify custom output base directory
    python compress_results.py raw_outputs/mbpp-js-microsoft_phi_2-0.0-itergen --output_base results

    # Compress all subdirectories under raw_outputs/
    python compress_results.py raw_outputs --all
"""

import json
import gzip
import argparse
from pathlib import Path


def compress_folder(src_dir: Path, dst_dir: Path):
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


def main():
    parser = argparse.ArgumentParser(description="Compress .json to .json.gz for MultiPL-E")
    parser.add_argument("input_dir", type=str, help="Directory containing .json files (or parent dir with --all)")
    parser.add_argument("--output_base", type=str, default="results", help="Base output directory")
    parser.add_argument("--all", action="store_true", help="Compress all subdirectories under input_dir")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_base)

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return

    if args.all:
        # Compress each subdirectory
        subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"No subdirectories found in {input_dir}")
            return
        total = 0
        for subdir in subdirs:
            dst = output_base / subdir.name
            total += compress_folder(subdir, dst)
        print(f"\nDone! Total: {total} files compressed.")
    else:
        # Compress single directory
        dst = output_base / input_dir.name
        count = compress_folder(input_dir, dst)
        print(f"\nDone! {count} files compressed.")


if __name__ == "__main__":
    main()
