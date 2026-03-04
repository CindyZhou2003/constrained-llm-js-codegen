import gzip
import os
import sys
import argparse
from pathlib import Path

def unzip_json_gz_files(source_dir, output_dir):
    """
    Decompress all .json.gz files from source_dir to output_dir
    
    Args:
        source_dir: Source directory containing .json.gz files
        output_dir: Output directory for decompressed .json files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all json.gz files in source directory
    source_path = Path(source_dir)
    gz_files = list(source_path.glob('*.json.gz'))
    
    if not gz_files:
        print(f"No .json.gz files found in {source_dir}")
        return
    
    # Process each .json.gz file
    for gz_file in gz_files:
        output_file = Path(output_dir) / gz_file.stem  # Remove .gz extension
        
        try:
            # Read compressed file and write decompressed content
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            # print(f"✓ Decompressed: {gz_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"✗ Error decompressing {gz_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress .json.gz files from source directory to raw_results directory")
    parser.add_argument("source_dir", help="Source directory path (relative or absolute), e.g., ./results/mbpp-js-microsoft_phi_2-0.2")
    
    args = parser.parse_args()
    
    # Get the source directory and extract the folder name
    source_directory = args.source_dir
    folder_name = Path(source_directory).name
    
    # Construct output directory
    output_directory = Path("./raw_results") / folder_name
    
    print(f"Source: {source_directory}")
    print(f"Output: {output_directory}")
    
    unzip_json_gz_files(source_directory, str(output_directory))
