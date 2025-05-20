import json
import lzma
import os


def extract_xz_json_files(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".xz"):
                input_path = os.path.join(root, file)
                output_filename = file[:-3]  # remove ".xz"
                output_path = os.path.join(output_dir, output_filename)
                extract_single_xz_json_file(input_path, output_path)


def extract_single_xz_json_file(input_path, output_path):
    try:
        with lzma.open(input_path, "rt", encoding="utf-8") as xz_file:
            data = json.load(xz_file)

        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2)

        print(f"Extracted: {input_path} → {output_path}")
    except Exception as e:
        print(f"Failed to extract {input_path}: {e}")


if __name__ == "__main__":
    extract_xz_json_files(
        "/Users/home/Downloads/dust2_xz", "./research_project/demos/dust2"
    )
