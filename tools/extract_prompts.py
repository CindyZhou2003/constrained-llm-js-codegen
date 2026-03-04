import argparse
import json
import re
from pathlib import Path


def parse_diff_file(diff_path):
    """Return set of test names where status changed from OK to non-OK."""
    changes = set()
    pattern = re.compile(r"^(mbpp_\d+_.+?):\s+OK\s*->\s*(.+)")
    with open(diff_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if m:
                name = m.group(1)
                new_stat = m.group(2)
                if new_stat and new_stat != 'OK':
                    changes.add(name)
    return changes


def extract_prompts(prompts_path, names, output_path):
    """Copy lines from prompts file whose 'name' field is in names set."""
    with open(prompts_path, 'r', encoding='utf-8') as inp, \
         open(output_path, 'w', encoding='utf-8') as outp:
        for line in inp:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get('name') in names:
                outp.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prompts for tests that went from OK to error.")
    parser.add_argument("diff_file", help="Path to the diff summary file")
    parser.add_argument("prompts_file", nargs="?", default="datasets/js_prompts_mbpp.jsonl",
                        help="Path to js prompts jsonl (default datasets/js_prompts_mbpp.jsonl)")
    parser.add_argument("output_file", nargs="?", default="datasets/tem.jsonl",
                        help="Output file to write selected prompts")
    args = parser.parse_args()

    diff_path = args.diff_file
    prompts_path = args.prompts_file
    output_path = args.output_file

    names = parse_diff_file(diff_path)
    if not names:
        print("No OK->error changes found in diff file.")
    else:
        print(f"Found {len(names)} tests with OK->error status.")
        extract_prompts(prompts_path, names, output_path)
        print(f"Prompts written to {output_path}")
