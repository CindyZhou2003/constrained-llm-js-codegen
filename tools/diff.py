from collections import defaultdict
import re
import argparse
from pathlib import Path

def parse_results(path):
    """
    Parse a results summary file into:
    { test_name : status }
    """
    results = {}
    current_status = None

    status_pattern = re.compile(r"^\[(.+?)\]")
    file_pattern = re.compile(r"\s+(mbpp_\d+_.+?)\.results\.json")

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()

            status_match = status_pattern.match(line)
            if status_match:
                current_status = status_match.group(1)
                continue

            file_match = file_pattern.match(line)
            if file_match and current_status:
                test_name = file_match.group(1)
                results[test_name] = current_status

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two results summary files and output changes.")
    parser.add_argument("file1", help="First results summary file")
    parser.add_argument("file2", help="Second results summary file")
    parser.add_argument("output", nargs="?", help="Optional output file name. If omitted, name is auto-generated.")
    args = parser.parse_args()

    file1_path = args.file1
    file2_path = args.file2

    # Parse both files
    results1 = parse_results(file1_path)
    results2 = parse_results(file2_path)

    # Find changes
    changes = {}
    for test in results1:
        if test in results2 and results1[test] != results2[test]:
            changes[test] = (results1[test], results2[test])

    # Generate output file name if not provided
    if args.output:
        output_file = args.output
    else:
        name1 = Path(file1_path).stem
        name2 = Path(file2_path).stem
        output_file = f"{name1}_vs_{name2}_diff.txt"

    # Write results to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Comparison: {file1_path} vs {file2_path}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total changed test cases: {len(changes)}\n\n")

        for test, (old, new) in sorted(changes.items()):
            f.write(f"{test}: {old} -> {new}\n")

    print(f"Total changed test cases: {len(changes)}")
    print(f"Results saved to: {output_file}")