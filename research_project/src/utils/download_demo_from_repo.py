# This file contains utility functions for downloading demo files from a repository and extracting them to a specified directory.

import json
import os
from typing import List

import requests
from tqdm import tqdm
from utils.extract_demos import extract_single_xz_json_file


def download_file(url: str, output_path: str) -> None:
    """
    Download a file from a URL and save it to a specified path using GitHub raw API.

    Args:
        url (str): The URL of the file to download.
        output_path (str): The path to save the downloaded file.
    """
    headers = {"Accept": "application/vnd.github.v3.raw"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def get_demo_files_from_list(demo_files_list_path: str, compressed: bool) -> List[str]:
    """
    Get a list of demo files from a repository.
    Args:
        demo_files_list_path (str): Path to the JSON file containing demo file information.
        compressed (bool): Whether to return compressed file names.
    Returns:
        List[str]: A list of demo file names.
    """
    demo_files_list = []
    with open(demo_files_list_path, "r") as f:
        demo_files_list = json.load(f)

    if not demo_files_list:
        print("No demo files found in the repository.")
        return []

    names = []
    for demo_file in demo_files_list:
        if compressed:
            names.append(demo_file["filename"] + ".xz")
        else:
            names.append(demo_file["filename"])

    return names


def main():
    repo_url = "https://github.com/pnxenopoulos/esta/raw/refs/heads/main/data/"
    folders = ["lan/", "online/"]

    demo_files_list = "file_paths.json"
    output_directory = "research_project/demos/dust2"
    os.makedirs(output_directory, exist_ok=True)

    demo_files = get_demo_files_from_list(demo_files_list, compressed=True)

    print(f"Found {len(demo_files)} demo files in the repository.")
    print(f"Downloading demo files to {output_directory}")

    for demo_file in tqdm(demo_files, desc="Processing demos"):
        file_path = os.path.join(output_directory, os.path.basename(demo_file))
        if not os.path.exists(file_path[:-3]):
            try:
                # lan directory
                full_url = os.path.join(repo_url, folders[0], demo_file)
                download_file(full_url, file_path)
            except requests.exceptions.HTTPError:
                # online directory
                full_url = os.path.join(repo_url, folders[1], demo_file)
                try:
                    download_file(full_url, file_path)
                except Exception as e:
                    print(f"Error downloading {demo_file}: {e}")
                    return
            # Extract the downloaded file
            extract_single_xz_json_file(
                file_path, file_path[:-3]
            )  # Remove ".xz" extension for output file

            # Remove the downloaded .xz file
            os.remove(file_path)

    print(
        f"✅ Downloaded and extracted {len(demo_files)} demo files from the repository."
    )


if __name__ == "__main__":
    main()
