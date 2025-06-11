import json
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_average_frames_per_round():
    """
    Calculate the average number of frames per round from the dataset.
    """
    total_frames = 0
    total_rounds = 0

    dataset_dir = os.getenv(
        "CREATE_GRAPHS_DEMO_DIR",
        "/Users/home/Desktop/University/MOD12/git/research-project/research_project/demos/dust2",
    )
    list_of_files = os.listdir(dataset_dir)

    print(f"Processing {len(list_of_files)} files in directory: {dataset_dir}")
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith(".json"):
            try:
                file_path = os.path.join(dataset_dir, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    total_rounds += len(data.get("gameRounds", []))
                    print(
                        f"Total rounds in file: {len(data.get('gameRounds', []))}, total frames in game: {sum(len(round.get('frames', [])) for round in data.get('gameRounds', []))}"
                    )
                    for round in data.get("gameRounds", []):
                        frames_in_round = len(round.get("frames", []))
                        total_frames += frames_in_round

            except json.decoder.JSONDecodeError:
                print(f"Skipping file {file_path} due to JSON decode error.")
                continue

    if total_rounds == 0:
        return 0

    return total_frames / total_rounds


def get_total_rounds():
    """
    Calculate the total number of rounds from the dataset.
    """
    total_rounds = 0
    dataset_dir = os.getenv(
        "CREATE_GRAPHS_DEMO_DIR",
        "/Users/home/Desktop/University/MOD12/git/research-project/research_project/demos/dust2",
    )

    for file_name in os.listdir(dataset_dir):
        if file_name.endswith(".json"):
            try:
                file_path = os.path.join(dataset_dir, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    total_rounds += len(data.get("gameRounds", []))
            except json.decoder.JSONDecodeError:
                print(f"Skipping file {file_path} due to JSON decode error.")
                continue

    return total_rounds


def get_total_frames_labeled():
    """
    Calculate the total number of labeled frames from the dataset.
    """
    total_frames = 0
    dataset_dir = os.getenv(
        "LABELS_OUTPUT_DIR",
        "/Users/home/Desktop/University/MOD12/git/research-project/research_project/tactic_labels",
    )

    de_dust2_dir = os.path.join(dataset_dir, "de_dust2")
    subfolders = [f.path for f in os.scandir(de_dust2_dir) if f.is_dir()]

    for subfolder in subfolders:
        for file_name in os.listdir(subfolder):
            if file_name.endswith(".json"):
                path = os.path.join(subfolder, file_name)
                total_frames += get_labeled_frames_per_round(path)

    return total_frames


def get_labeled_frames_per_round(path):
    """
    Calculate the number of labeled frames per round.
    """
    total_frames = 0
    try:
        with open(path, "r") as f:
            data = json.load(f)
            no_frames_labeled = len(list(data.keys()))
            total_frames += no_frames_labeled

    except json.decoder.JSONDecodeError:
        print(f"Skipping file {path} due to JSON decode error.")
        return 0

    return total_frames


def get_total_frames_per_game(path):
    """
    Calculate the total number of frames per game from the dataset.
    This function reads a JSON file and counts the total number of frames across all rounds.
    Args:
        path (str): The path to the JSON file containing game data. (demo not tactic labels)
    Returns:
        int: The total number of frames across all rounds in the game.
    Raises:
        json.decoder.JSONDecodeError: If the JSON file cannot be decoded.
    Returns:
        int: The total number of frames in the game, or 0 if the file cannot be read or decoded.
    """
    total_frames = 0
    try:
        with open(path, "r") as f:
            data = json.load(f)
            for round in data.get("gameRounds", []):
                frames_in_round = len(round.get("frames", []))
                total_frames += frames_in_round

    except json.decoder.JSONDecodeError:
        print(f"Skipping file {path} due to JSON decode error.")
        return 0

    return total_frames


def get_unlabeled_frames_per_game(path):
    """
    Calculate the number of unlabelled frames per game.
    """
    total_unlabeled_frames = 0
    labeled_frames = 0
    filename = os.path.basename(path)

    try:
        with open(path, "r") as f:
            data = json.load(f)
            for round in data.get("gameRounds", []):
                frames_in_round = len(round.get("frames", []))
                total_unlabeled_frames += frames_in_round

        print("found frames in game:", total_unlabeled_frames)

        labels_dir = os.path.join(
            os.getenv(
                "LABELS_OUTPUT_DIR",
                "/Users/home/Desktop/University/MOD12/git/research-project/research_project/tactic_labels",
            ),
            "de_dust2",
            filename.replace(".json", ""),
        )
        if not os.path.exists(labels_dir):
            print(f"No labels found for game: {filename}")
            return total_unlabeled_frames

        for file_name in os.listdir(labels_dir):
            if file_name.endswith(".json"):
                path_to_labels = os.path.join(labels_dir, file_name)
                labeled_frames += get_labeled_frames_per_round(path_to_labels)

        print("found labeled frames in game:", labeled_frames)

        total_unlabeled_frames -= labeled_frames

    except json.decoder.JSONDecodeError:
        print(f"Skipping file {path} due to JSON decode error.")
        return 0

    return total_unlabeled_frames


def get_all_unlabeled_frames():
    """
    Calculate the total number of unlabeled frames across all games in the dataset.
    """
    total_unlabeled_frames = 0

    # get all files in labels output directory. they are subdirectories with the game name
    # and each subdirectory contains json files with the labels for each round
    # get all files in labels output directory
    labels_output_dir = os.getenv(
        "LABELS_OUTPUT_DIR",
        "/Users/home/Desktop/University/MOD12/git/research-project/research_project/tactic_labels",
    )
    de_dust2_dir = os.path.join(labels_output_dir, "de_dust2")
    labeled_files = [f.name for f in os.scandir(de_dust2_dir) if f.is_dir()]
    print(f"Found {len(labeled_files)} labeled files in {de_dust2_dir}")

    # get all files in dataset directory
    dataset_dir = os.getenv(
        "CREATE_GRAPHS_DEMO_DIR",
        "/Users/home/Desktop/University/MOD12/git/research-project/research_project/demos/dust2",
    )
    demo_files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    print(f"Found {len(demo_files)} demo files in {dataset_dir}")

    # filter out files that are not in the labels output directory
    demo_files = [f for f in demo_files if f.replace(".json", "") in labeled_files]
    print(f"Filtered demo files to {len(demo_files)} that have labels")

    # calculate total unlabeled frames
    for file_name in demo_files:
        path = os.path.join(dataset_dir, file_name)
        total_unlabeled_frames += get_unlabeled_frames_per_game(path)

    return total_unlabeled_frames


def get_all_unique_tactic_labels():
    """
    Get all unique tactic labels from the dataset.
    """
    tactic_labels = set()
    dataset_dir = os.getenv(
        "LABELS_OUTPUT_DIR",
        "/Users/home/Desktop/University/MOD12/git/research-project/research_project/tactic_labels",
    )

    de_dust2_dir = os.path.join(dataset_dir, "de_dust2")
    subfolders = [f.path for f in os.scandir(de_dust2_dir) if f.is_dir()]

    for subfolder in subfolders:
        for file_name in os.listdir(subfolder):
            if file_name.endswith(".json"):
                path = os.path.join(subfolder, file_name)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        tactic_labels.update(data.values())
                except json.decoder.JSONDecodeError:
                    print(f"Skipping file {path} due to JSON decode error.")
                    continue

    return list(tactic_labels)


def get_most_common_tactic_labels():
    """
    Get the most common tactic labels from the dataset.
    """
    tactic_labels = get_all_unique_tactic_labels()
    label_counts = {label: 0 for label in tactic_labels}

    dataset_dir = os.getenv(
        "LABELS_OUTPUT_DIR",
        "/Users/home/Desktop/University/MOD12/git/research-project/research_project/tactic_labels",
    )

    de_dust2_dir = os.path.join(dataset_dir, "de_dust2")
    subfolders = [f.path for f in os.scandir(de_dust2_dir) if f.is_dir()]

    for subfolder in subfolders:
        for file_name in os.listdir(subfolder):
            if file_name.endswith(".json"):
                path = os.path.join(subfolder, file_name)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        for label in data.values():
                            if label in label_counts:
                                label_counts[label] += 1
                except json.decoder.JSONDecodeError:
                    print(f"Skipping file {path} due to JSON decode error.")
                    continue

    return sorted(label_counts.items(), key=lambda x: x[1], reverse=True)


def create_bar_chart_labels_frequency():
    """
    Create a bar chart of the frequency of tactic labels.
    """

    label_counts = get_most_common_tactic_labels()
    labels, counts = zip(*label_counts)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel("Tactic Labels")
    plt.ylabel("Frequency")
    plt.title("Frequency of Tactic Labels")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    create_bar_chart_labels_frequency()
