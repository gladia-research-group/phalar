import random
import shutil
from pathlib import Path

def split_and_move_moisesdb(
    dataset_path,
    output_path,
    train_ratio=0.8,
    valid_ratio=0.1,
    seed=42
):
    """
    Splits MoisesDB dataset into train/valid/test folders and moves the data.

    Args:
        dataset_path (str): Path to the original MoisesDB dataset.
        output_path (str): Path where train/valid/test folders will be created.
        train_ratio (float): Fraction of data for training.
        valid_ratio (float): Fraction of data for validation.
        seed (int): Random seed for reproducibility.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # Create output folders
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Get all tracks (assuming each track is a folder)
    all_tracks = [p for p in dataset_path.iterdir() if p.is_dir()]
    # Filter train/valid/test from input folder in case it is also output folder
    all_tracks = [p for p in all_tracks if p.name not in {"train", "valid", "test"}]
    all_tracks.sort()

    random.seed(seed)
    random.shuffle(all_tracks)

    n_total = len(all_tracks)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = n_total - n_train - n_valid

    splits = {
        "train": all_tracks[:n_train],
        "valid": all_tracks[n_train:n_train+n_valid],
        "test": all_tracks[n_train+n_valid:]
    }

    # Move folders
    for split_name, folders in splits.items():
        for folder in folders:
            dest = output_path / split_name / folder.name
            shutil.move(str(folder), str(dest))

    print(f"Data split complete!")
    print(f"Train: {n_train}, Valid: {n_valid}, Test: {n_test}")
    print(f"Output folders created at: {output_path}")

if __name__ == "__main__":
    split_and_move_moisesdb("./moisesdb_contrastive/moisesdb_v0.1", "./moisesdb_contrastive/moisesdb_v0.1")