import os
import numpy as np

# Set directory where the .npy batches are stored
BATCH_DIR = "data/batches/"
OUT_BASE = "data/full_play_data"

def load_all_batches(batch_dir):
    all_data = {}
    for filename in sorted(os.listdir(batch_dir)):
        if filename.endswith(".npy"):
            batch_path = os.path.join(batch_dir, filename)
            print(f"Loading {batch_path}")
            batch_data = np.load(batch_path, allow_pickle=True).item()
            all_data.update(batch_data)
    print(f"Loaded {len(all_data)} total plays.")
    return all_data

def split_dict(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    keys = list(data.keys())
    np.random.seed(seed)
    np.random.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]

    return (
        {k: data[k] for k in train_keys},
        {k: data[k] for k in val_keys},
        {k: data[k] for k in test_keys}
    )

def save_splits(base_path, train, val, test):
    np.save(f"{base_path}_train.npy", train)
    np.save(f"{base_path}_val.npy", val)
    np.save(f"{base_path}_test.npy", test)
    print(f"Saved splits:")
    print(f"    {base_path}_train.npy")
    print(f"    {base_path}_val.npy")
    print(f"    {base_path}_test.npy")

if __name__ == "__main__":
    full_data = load_all_batches(BATCH_DIR)
    train, val, test = split_dict(full_data)
    save_splits(OUT_BASE, train, val, test)
