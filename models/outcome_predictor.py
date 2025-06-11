# models/outcome_predictor.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

class OutcomePredictor(nn.Module):
    """
    LSTM-based outcome predictor for NFL plays.
    Accepts packed padded sequences of player trajectories.
    Input: PackedSequence of shape [batch, T, 22*6]
    Output: Scalar prediction per play (yards gained)
    """
    def __init__(self, player_count=22, feature_dim=6, lstm_hidden=64, dense_hidden=128):
        super(OutcomePredictor, self).__init__()

        self.input_dim = player_count * feature_dim  # Each timestep has 22 players x 6 features

        # Bidirectional LSTM to model temporal dynamics
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers to predict scalar outcome
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden, dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 1)
        )

    def forward(self, packed_x):
        """
        Args:
            packed_x (PackedSequence): Packed padded trajectory input [B, T, 132]
        Returns:
            Tensor [B]: predicted yards gained for each play
        """
        # Pass through LSTM
        packed_output, _ = self.lstm(packed_x)

        # Unpack padded sequence for final timestep extraction
        unpacked, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # Collect final timestep output for each sequence (accounting for padding)
        batch_size = unpacked.size(0)
        final_timestep_indices = lengths - 1  # shape [B]
        final_outputs = unpacked[torch.arange(batch_size), final_timestep_indices]

        # Pass through dense layers to get predictions
        return self.fc(final_outputs).squeeze()
