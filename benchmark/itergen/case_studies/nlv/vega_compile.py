import json
import subprocess
import re
import os

def check_json_equivalence(generated_json, gt_json, dataset) -> bool:
    # Check if the two Vega-Lite JSON specifications are equivalent
    try:
        generated_json = json.loads(generated_json)
    except json.JSONDecodeError:
        print("Fail: Generated JSON is not valid")
        return False

    # Get the data fields from the CSV file
    file_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(file_dir, "nlvcorpus/datasets", f"{dataset}.csv"), 'r') as f:
        data_fields = f.readline().strip().split(',')
    data_fields = ", ".join(data_fields)

     # Count number of unique entries for each field
    data_fields_to_entry = {}

    with open(os.path.join(file_dir, "nlvcorpus/datasets", f"{dataset}.csv"), 'r', errors='replace') as f:
        for line in f:
            data = line.strip().split(',')
            for i, field in enumerate(data_fields.split(', ')):
                if field not in data_fields_to_entry:
                    data_fields_to_entry[field] = set()
                data_fields_to_entry[field].add(data[i])
    
    data_fields_count = {}
    for field in data_fields_to_entry:
        data_fields_count[field] = len(data_fields_to_entry[field])

    # For mark - check if the type is the same
    if 'mark' not in generated_json:
        print("Fail: 'mark' key not found in generated JSON")
        return False
    
    if type(generated_json['mark']) != dict:
        if not( type(generated_json['mark']) == str and generated_json['mark'] == gt_json['mark']['type']):
            print("Fail: 'mark' key is not a dictionary")
            return False

    if generated_json['mark']['type'] != gt_json['mark']['type']:
        # Assume 'circle' and 'point' are equivalent
        if generated_json['mark']['type'] == 'circle' and gt_json['mark']['type'] == 'point':
            pass

        print(f"Fail: Mark type is different. Expected: {gt_json['mark']['type']}, Got: {generated_json['mark']['type']}")
        return False
    
    # For encoding - check if all fields are the same
    if 'encoding' not in generated_json:
        print("Fail: 'encoding' key not found in generated JSON")
        return False
    
    for key in generated_json['encoding']:
        if key not in gt_json['encoding']:
            print(f"Fail: Key {key} not found in ground truth JSON")
            return False
        
        # For each key check if field, type, and aggregate are the same
        if 'field' in gt_json['encoding'][key]:
            if 'field' not in generated_json['encoding'][key]:
                print(f"Fail: 'field' key not found in generated JSON")
                return False

            if generated_json['encoding'][key]['field'] != gt_json['encoding'][key]['field']:
                print(f"Fail: Field for key {key} is different. Expected: {gt_json['encoding'][key]['field']}, Got: {generated_json['encoding'][key]['field']}")
                return False
        
        if generated_json['encoding'][key]['type'] != gt_json['encoding'][key]['type']:
            print(f"Fail: Type for key {key} is different. Expected: {gt_json['encoding'][key]['type']}, Got: {generated_json['encoding'][key]['type']}")
            return False
        
        if 'aggregate' in generated_json['encoding'][key] and 'aggregate' in gt_json['encoding'][key]:
            if generated_json['encoding'][key]['aggregate'] != gt_json['encoding'][key]['aggregate']:
                print(f"Fail: Aggregate for key {key} is different. Expected: {gt_json['encoding'][key]['aggregate']}, Got: {generated_json['encoding'][key]['aggregate']}")
                return False
        else:
            # If one has aggregate and the other doesn't
            if 'aggregate' in generated_json['encoding'][key] or 'aggregate' in gt_json['encoding'][key]:
                print(f"Fail: Aggregate for key {key} is not found in both JSONs")
                return False
    
    return True

def run_vega_lite(json_string, output_pdf_path):
    # Get the directory of the current Python script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_pdf_path = os.path.join(current_directory, output_pdf_path)
    
    # Create a temporary file in the same directory as the Python script
    temp_json_file_path = os.path.join(current_directory, 'temp_data.json')
    
    # Write JSON data to the temporary file
    with open(temp_json_file_path, 'w') as temp_json_file:
        temp_json_file.write(json_string)

    # Define the command to execute
    command = [
        'npx', '-p', 'vega-lite', '-p', 'vega-cli', 'vl2pdf', temp_json_file_path, output_pdf_path, '--renderer', 'svg'
    ]
    
    try:
        # Run the command and capture output (both stdout and stderr)
        result = subprocess.run(command, capture_output=True, text=True)

        # Combine both stdout and stderr outputs into one string
        output = result.stdout + "\n" + result.stderr

        # Initialize flag for warnings/errors detection
        found_warning_or_error = False
        warnings_and_errors = []

        # Define regex patterns for warnings and errors
        warning_pattern = r'WARN.*'
        error_pattern = r'Error.*'

        # Search for warnings and errors using regex
        warnings = re.findall(warning_pattern, output)
        errors = re.findall(error_pattern, output)

        # Combine warnings and errors into one list
        warnings_and_errors = warnings + errors

        # If there are any warnings or errors, set the flag to True
        if warnings_and_errors:
            found_warning_or_error = True

        # Return results: list of warnings/errors and the boolean flag
        return warnings_and_errors, found_warning_or_error
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_json_file_path):
            os.remove(temp_json_file_path)

# Example usage
json_string = '''{
    "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
    "data": {
        "url": "/datasets/cars.csv"
    },
    "mark": {"type": "bar"},
    "encoding": {
        "x": {"field": "Origin", "type": "ordinal"},
        "y": {"aggregate": "count", "type": "quantitative", "axis": {"title": "COUNT"}},
        "color": {"field": "Origin", "type": "nominal"}
	}
}'''

pdf_file = "example_output.pdf"

# Run the function
warnings, has_warnings_or_errors = run_vega_lite(json_string, pdf_file)

if has_warnings_or_errors:
    print("Warnings/Errors found:")
    for warning_or_error in warnings:
        print(warning_or_error)
else:
    print("No warnings or errors found.")
