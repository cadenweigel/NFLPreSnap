import os
import time
import argparse
import numpy as np
from preprocess import load_data, extract_all_frames

# Path to the full dataset output
SAVE_PATH = "data/full_play_data.npy"

def compute_stats(play_data):
    """
    Print descriptive statistics about the number of frames per play.

    Args:
        play_data (dict): Dictionary of (gameId, playId) -> play data
    """
    frame_lengths = [v['trajectory'].shape[0] for v in play_data.values()]
    print(f"📊 Frame count stats:")
    print(f"   Min:    {min(frame_lengths)}")
    print(f"   Max:    {max(frame_lengths)}")
    print(f"   Avg:    {sum(frame_lengths) / len(frame_lengths):.2f}")
    print(f"   Median: {np.median(frame_lengths)}")
    print(f"   Std Dev:{np.std(frame_lengths):.2f}")

def split_dict(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split a dataset dictionary into train, validation, and test sets.

    Args:
        data (dict): Full play dictionary
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        seed (int): Random seed for reproducibility

    Returns:
        (train, val, test): three dictionaries
    """
    keys = list(data.keys())
    np.random.seed(seed)
    np.random.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Slice keys based on proportions
    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]

    # Reconstruct subsets as dicts
    train = {k: data[k] for k in train_keys}
    val = {k: data[k] for k in val_keys}
    test = {k: data[k] for k in test_keys}

    return train, val, test

def save_splits(base_path, train, val, test):
    """
    Save the train, validation, and test splits as separate .npy files.

    Args:
        base_path (str): Base name (no extension) for saving files
        train, val, test (dict): The data splits
    """
    np.save(f"{base_path}_train.npy", train)
    np.save(f"{base_path}_val.npy", val)
    np.save(f"{base_path}_test.npy", test)
    print(f"💾 Saved splits:")
    print(f"    {base_path}_train.npy")
    print(f"    {base_path}_val.npy")
    print(f"    {base_path}_test.npy")

def main(overwrite=False, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Main function to run the full preprocessing + split pipeline.

    Args:
        overwrite (bool): Whether to regenerate data if file already exists
        train_ratio (float): Train set proportion
        val_ratio (float): Validation set proportion
        seed (int): Seed for random shuffling
    """
    # If data already exists and not forced to overwrite, skip
    if os.path.exists(SAVE_PATH) and not overwrite:
        print(f"✅ {SAVE_PATH} already exists. Use --overwrite to regenerate.")
        return

    # Load raw CSVs
    print("📦 Loading raw CSVs...")
    tracking, plays, players, games = load_data()

    # Extract all plays and normalize them
    print("⚙️  Extracting all frames (pre + post snap)...")
    start_time = time.time()
    play_data = extract_all_frames(tracking, plays)
    duration = time.time() - start_time
    print(f"✅ Extracted {len(play_data)} plays in {duration:.2f} seconds.")

    # Print stats about play lengths
    compute_stats(play_data)

    # Save full processed dataset
    np.save(SAVE_PATH, play_data)
    print(f"💾 Saved full play data to: {SAVE_PATH}")

    # Split into train/val/test
    train, val, test = split_dict(play_data, train_ratio, val_ratio, seed)
    base = os.path.splitext(SAVE_PATH)[0]
    save_splits(base, train, val, test)

if __name__ == "__main__":
    # Argument parser for CLI usage
    parser = argparse.ArgumentParser(description="Preprocess and split NFL tracking data.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate full dataset even if file exists")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proportion of plays for training set")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Proportion of plays for validation set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    main(overwrite=args.overwrite,
         train_ratio=args.train_ratio,
         val_ratio=args.val_ratio,
         seed=args.seed)
