import argparse
import pickle


def unpack_pkl(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)

    txt_path = pkl_file_path.replace(".pkl", ".txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(str(data))

    print(f"Saved unpacked data to {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unpack a pickle file and print its contents."
    )
    parser.add_argument(
        "pkl_file_path", type=str, help="Path to the .pkl file to unpack"
    )
    args = parser.parse_args()

    unpack_pkl(args.pkl_file_path)
