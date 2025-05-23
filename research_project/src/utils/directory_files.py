# This file contains utility functions for extracting the names of all files in a directory and export it into a json list

import json
import os
from typing import List


def get_all_files_in_directory(directory: str) -> List[str]:
    """
    Get all files in a directory and its subdirectories.

    Args:
        directory (str): The path to the directory.

    Returns:
        List[str]: A list of file paths.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def export_file_paths_to_json(file_paths: List[str], output_file: str) -> None:
    """
    Export a list of file paths to a JSON file.
    Args:
        file_paths (List[dict { "filename": str, "path": str }]): The list of file paths.
        output_file (str): The path to the output JSON file.
    """
    data = [{"filename": os.path.basename(path), "path": path} for path in file_paths]
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def main():
    directory = "/Users/home/Desktop/University/MOD12/git/research-project/research_project/demos/.dust2_unlisted_demos"  # Replace with your directory path
    output_file = "file_paths.json"  # Replace with your desired output file name

    file_paths = get_all_files_in_directory(directory)
    export_file_paths_to_json(file_paths, output_file)
    print(f"Exported {len(file_paths)} file paths to {output_file}")


if __name__ == "__main__":
    main()
