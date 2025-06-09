import numpy as np
import torch
from torch.utils.data import Dataset

class FullPlayDataset(Dataset):
    """
    Dataset that loads full-length, variable-frame NFL plays from a .npy file.
    Each item includes a play's full trajectory and outcome.
    """
    def __init__(self, data_path):
        """
        Args:
            data_path (str): path to .npy file containing the play dictionary
        """
        self.play_dict = np.load(data_path, allow_pickle=True).item()
        self.keys = list(self.play_dict.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        play = self.play_dict[key]

        return {
            'trajectory': play['trajectory'],  # shape: [frames, 22, 6]
            'outcome': play['outcome']         # scalar
        }
