import json
import os
from pathlib import Path
from collections import defaultdict

def organize_results(folder_path):
    """
    Scan all results.json files in the folder and organize by status.
    """
    status_files = defaultdict(list)
    
    # Find all results.json files recursively
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.results.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Extract status from results
                    if 'results' in data and isinstance(data['results'], list):
                        for result in data['results']:
                            if 'status' in result:
                                status = result['status']
                                # Store filename relative to the folder
                                rel_path = os.path.relpath(file_path, folder_path)
                                status_files[status].append(rel_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return status_files

def save_summary(status_files, output_file):
    """
    Save the organized results to a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Results.json Files Organized by Status\n")
        f.write("=" * 80 + "\n\n")
        
        for status in sorted(status_files.keys()):
            files = status_files[status]
            f.write(f"\n[{status}] - {len(files)} file(s)\n")
            f.write("-" * 80 + "\n")
            for file in sorted(files):
                f.write(f"  {file}\n")
            f.write("\n")
        
        # Statistics
        f.write("=" * 80 + "\n")
        f.write("Summary\n")
        f.write("=" * 80 + "\n")
        total_files = sum(len(files) for files in status_files.values())
        f.write(f"Total files: {total_files}\n")
        for status in sorted(status_files.keys()):
            f.write(f"{status}: {len(status_files[status])}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan and organize results.json files by status.")
    parser.add_argument("folder", help="Folder path containing results.json files (relative or absolute)")
    parser.add_argument("output_file", nargs="?", help="Optional output text file for summary. If omitted a name is generated in 'summary' folder.")
    args = parser.parse_args()

    folder = args.folder
    if args.output_file:
        output_file = args.output_file
    else:
        # derive name from folder
        base = Path(folder).name
        summary_dir = Path("summary")
        summary_dir.mkdir(exist_ok=True)
        output_file = summary_dir / f"{base}.txt"

    print(f"Scanning folder: {folder}")
    status_files = organize_results(folder)

    print(f"Found {len(status_files)} different statuses")
    save_summary(status_files, output_file)

    print(f"Summary saved to: {output_file}")
