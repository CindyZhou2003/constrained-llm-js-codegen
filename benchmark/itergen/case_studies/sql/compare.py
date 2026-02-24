import argparse
import json

def compare_jsonl_files(file1, file2):
    with open(file1, 'r') as f:
        lines1 = f.readlines()
    with open(file2, 'r') as f:
        lines2 = f.readlines()
    
    if len(lines1) != len(lines2):
        raise ValueError(f"Files have different number of lines: {len(lines1)} vs {len(lines2)}")

    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        # Each line is a JSON object
        res_json1 = json.loads(line1)
        res_json2 = json.loads(line2)

        if res_json1["exec"] == True and res_json2["exec"] == False:
            # Add color to the output
            
            print(f"Line {i} worked in file1 but not in file2")
            print(f"File1: {res_json1}")
            print(f"File2: {res_json2}")
    
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        # Each line is a JSON object
        res_json1 = json.loads(line1)
        res_json2 = json.loads(line2)

        if res_json1["exec"] == False and res_json2["exec"] == True:
            print(f"Line {i} worked in file2 but not in file1")
            print(f"File1: {res_json1}")
            print(f"File2: {res_json2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two JSONL files')
    parser.add_argument('--file1', type=str, help='Path to first JSONL file')
    parser.add_argument('--file2', type=str, help='Path to second JSONL file')
    args = parser.parse_args()
    compare_jsonl_files(args.file1, args.file2)
