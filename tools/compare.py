"""
compare.py — Count, diff, and optionally extract regressions for two result directories.

Usage:
    python tools/compare.py results_eval/A results_eval/B [--prompts datasets/js_prompts_mbpp.jsonl] [--out-dir out]

This replaces running count.py + diff.py + extract_prompts.py separately.
Warns when .results.json.gz evaluation data is stale relative to the .json.gz completions.
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def check_staleness(eval_folder):
    """Return (stale_count, total) by comparing .json.gz completions vs .results.json.gz programs."""
    stale = 0
    total = 0
    for fname in os.listdir(eval_folder):
        if not fname.endswith(".results.json.gz"):
            continue
        name = fname.replace(".results.json.gz", "")
        gz_path = os.path.join(eval_folder, f"{name}.json.gz")
        res_path = os.path.join(eval_folder, fname)
        if not os.path.exists(gz_path):
            continue
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                d = json.load(f)
            with gzip.open(res_path, "rt", encoding="utf-8") as f:
                r = json.load(f)
            current = d["completions"][0]
            prompt = r["prompt"]
            tests = r["tests"]
            program = r["results"][0]["program"]
            evaluated = program[len(prompt): len(program) - len(tests)]
            total += 1
            if current.rstrip() != evaluated.rstrip():
                stale += 1
        except Exception:
            pass
    return stale, total


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def count_results(folder):
    """Return {status: [test_name, ...]} from all .results.json.gz files."""
    status_files = defaultdict(list)
    for root, _, files in os.walk(folder):
        for fname in files:
            if not (fname.endswith(".results.json") or fname.endswith(".results.json.gz")):
                continue
            fpath = os.path.join(root, fname)
            try:
                if fname.endswith(".gz"):
                    with gzip.open(fpath, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                if "results" in data and isinstance(data["results"], list):
                    for result in data["results"]:
                        if "status" in result:
                            rel = os.path.relpath(fpath, folder)
                            status_files[result["status"]].append(rel)
            except Exception as e:
                print(f"  Warning: could not read {fpath}: {e}", file=sys.stderr)
    return status_files


def format_count(status_files):
    lines = []
    lines.append("=" * 80)
    lines.append("Status summary")
    lines.append("=" * 80)
    total = sum(len(v) for v in status_files.values())
    lines.append(f"Total: {total}")
    for s in sorted(status_files.keys()):
        pct = 100 * len(status_files[s]) / total if total else 0
        lines.append(f"  {s}: {len(status_files[s])} ({pct:.1f}%)")
    return "\n".join(lines)


def save_count(status_files, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Results organized by status\n")
        f.write("=" * 80 + "\n\n")
        for status in sorted(status_files.keys()):
            files = status_files[status]
            f.write(f"\n[{status}] - {len(files)} file(s)\n")
            f.write("-" * 80 + "\n")
            for fn in sorted(files):
                f.write(f"  {fn}\n")
            f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Summary\n")
        f.write("=" * 80 + "\n")
        total = sum(len(v) for v in status_files.values())
        f.write(f"Total files: {total}\n")
        for s in sorted(status_files.keys()):
            f.write(f"{s}: {len(status_files[s])}\n")


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def parse_count_file(path):
    """Parse a saved count file into {test_name: status}."""
    results = {}
    current_status = None
    status_pattern = re.compile(r"^\[(.+?)\]")
    file_pattern = re.compile(r"\s+(mbpp_\d+_.+?)\.results\.json")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            m = status_pattern.match(line)
            if m:
                current_status = m.group(1)
                continue
            m = file_pattern.match(line)
            if m and current_status:
                results[m.group(1)] = current_status
    return results


def compute_diff(results1, results2):
    """Return {test_name: (status_in_1, status_in_2)} for changed tests."""
    changes = {}
    for test in results1:
        if test in results2 and results1[test] != results2[test]:
            changes[test] = (results1[test], results2[test])
    return changes


def format_diff(changes, label1, label2):
    lines = []
    lines.append("=" * 80)
    lines.append(f"Diff: {label1}  vs  {label2}")
    lines.append("=" * 80)
    lines.append(f"Total changed: {len(changes)}")
    lines.append("")

    improved = {t: v for t, v in changes.items() if v[1] == "OK"}
    regressed = {t: v for t, v in changes.items() if v[0] == "OK" and v[1] != "OK"}
    other = {t: v for t, v in changes.items() if t not in improved and t not in regressed}

    if improved:
        lines.append(f"[{label2} gains OK — {len(improved)} cases]")
        for t, (a, b) in sorted(improved.items()):
            lines.append(f"  {t}: {a} -> {b}")
        lines.append("")
    if regressed:
        lines.append(f"[{label2} loses OK — {len(regressed)} cases]")
        for t, (a, b) in sorted(regressed.items()):
            lines.append(f"  {t}: {a} -> {b}")
        lines.append("")
    if other:
        lines.append(f"[Other changes — {len(other)} cases]")
        for t, (a, b) in sorted(other.items()):
            lines.append(f"  {t}: {a} -> {b}")
        lines.append("")
    return "\n".join(lines)


def save_diff(changes, label1, label2, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Comparison: {label1} vs {label2}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total changed test cases: {len(changes)}\n\n")
        for test, (old, new) in sorted(changes.items()):
            f.write(f"{test}: {old} -> {new}\n")


# ---------------------------------------------------------------------------
# Extract regressions (OK in A, non-OK in B)
# ---------------------------------------------------------------------------

def extract_regressions(changes, prompts_path, output_path):
    names = {t for t, (a, b) in changes.items() if a == "OK" and b != "OK"}
    if not names:
        print("No OK->error regressions found.")
        return 0
    written = 0
    with open(prompts_path, "r", encoding="utf-8") as inp, \
         open(output_path, "w", encoding="utf-8") as outp:
        for line in inp:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("name") in names:
                outp.write(line)
                written += 1
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare two results_eval directories end-to-end.")
    parser.add_argument("dir1", help="First results_eval directory (baseline)")
    parser.add_argument("dir2", help="Second results_eval directory (comparison)")
    parser.add_argument("--prompts", default="datasets/js_prompts_mbpp.jsonl",
                        help="JSONL prompts file for regression extraction (default: datasets/js_prompts_mbpp.jsonl)")
    parser.add_argument("--out-dir", default="summary",
                        help="Output directory for generated files (default: summary)")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip regression prompt extraction")
    args = parser.parse_args()

    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    label1 = dir1.name
    label2 = dir2.name

    # --- Staleness check ---
    print(f"Checking staleness...")
    for label, d in [(label1, dir1), (label2, dir2)]:
        stale, total = check_staleness(d)
        if stale > 0:
            print(f"  WARNING: {label}: {stale}/{total} eval files are STALE "
                  f"(evaluated from older completions). Re-run evaluation before comparing.")
        else:
            print(f"  OK: {label}: all {total} eval files are current.")

    # --- Count ---
    print(f"\nCounting {label1}...")
    counts1 = count_results(dir1)
    count_file1 = out_dir / f"{label1}.txt"
    save_count(counts1, count_file1)
    print(format_count(counts1))

    print(f"\nCounting {label2}...")
    counts2 = count_results(dir2)
    count_file2 = out_dir / f"{label2}.txt"
    save_count(counts2, count_file2)
    print(format_count(counts2))

    # --- Diff ---
    print(f"\nDiff: {label1} vs {label2}")
    results1 = parse_count_file(count_file1)
    results2 = parse_count_file(count_file2)
    changes = compute_diff(results1, results2)
    diff_file = out_dir / f"{label1}_vs_{label2}_diff.txt"
    save_diff(changes, label1, label2, diff_file)
    print(format_diff(changes, label1, label2))

    # --- Regression extraction ---
    if not args.no_extract and Path(args.prompts).exists():
        regressions_file = out_dir / f"{label1}_vs_{label2}_regressions.jsonl"
        n = extract_regressions(changes, args.prompts, regressions_file)
        if n:
            print(f"Regression prompts ({n}) written to: {regressions_file}")
    elif not args.no_extract:
        print(f"(Skipping regression extraction — {args.prompts} not found)")

    print(f"\nOutput files written to: {out_dir}/")
    print(f"  {count_file1.name}")
    print(f"  {count_file2.name}")
    print(f"  {diff_file.name}")


if __name__ == "__main__":
    main()