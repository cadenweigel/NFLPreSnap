import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to handle variable-length play sequences.
    Pads each play in the batch to match the longest play (in frames).
    
    Args:
        batch: list of items, each from FullPlayDataset:
            {
                'trajectory': Tensor [frames, 22, 6],
                'outcome': scalar
            }

    Returns:
        padded_trajectories: Tensor [B, T_max, 22, 6]
            All plays padded to max frame length in batch
        lengths: Tensor [B]
            Original length (frames) of each play before padding
        outcomes: Tensor [B]
            Yards gained or other label
    """
    # Extract tensors and outcomes
    trajectories = [torch.tensor(item['trajectory'], dtype=torch.float32) for item in batch]
    outcomes = torch.tensor([item['outcome'] for item in batch], dtype=torch.float32)

    # Track original lengths (frame counts) for each play
    lengths = torch.tensor([t.shape[0] for t in trajectories], dtype=torch.long)

    # Pad all plays along the time (frame) dimension to max in batch
    # Result: shape [B, T_max, 22, 6]
    padded_trajectories = pad_sequence(trajectories, batch_first=True)

    return padded_trajectories, lengths, outcomes
