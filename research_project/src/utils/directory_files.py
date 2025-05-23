# This file contains utility functions for extracting the names of all files in a directory and export it into a json list

import json
import os
from typing import List


def get_all_files_in_directory(directory: str) -> List[str]:
    """
    Get all files in a directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        List[str]: A list of filenames.
    """
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    return file_names


def export_file_names_to_json(file_names: List[str], output_file: str) -> None:
    """
    Export a list of filenames to a JSON file.
    Args:
        file_names (List[str]): The list of file names.
        output_file (str): The path to the output JSON file.
    """
    data = [{"filename": name} for name in file_names]
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def main():
    directory = "/Users/home/Desktop/University/MOD12/git/research-project/research_project/demos/.dust2_unlisted_demos"  # Replace with your directory path
    output_file = "file_paths.json"  # Replace with your desired output file name

    file_names = get_all_files_in_directory(directory)
    export_file_names_to_json(file_names, output_file)
    print(f"Exported {len(file_names)} file names to {output_file}")


if __name__ == "__main__":
    main()
