import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TrajectoryVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder LSTM for encoding trajectory sequences
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Fully connected layers to derive mean and log-variance for latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Optional conditioning on play outcome (e.g., yards gained)
        self.condition_to_latent = nn.Linear(latent_dim + 1, latent_dim)

        # Map latent vector to decoder's hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # Decoder LSTM for reconstructing the trajectory sequence
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Final projection to original input dimensions
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, lengths):
        """
        Encode padded input sequence using LSTM. Uses pack_padded_sequence
        so the encoder ignores padding during computation.
        Args:
            x: Padded input tensor (batch, seq_len, input_dim)
            lengths: Actual lengths of each sequence in the batch (batch,)
        Returns:
            mu, logvar: Mean and log-variance vectors for the latent distribution
        """
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed)  # Use final hidden state
        h_n = h_n.squeeze(0)  # Remove LSTM layer dimension
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, sigma^2)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len, outcome=None):
        """
        Decode latent vector into trajectory sequence.
        Args:
            z: Latent vector (batch, latent_dim)
            seq_len: Number of time steps to decode
            outcome: Optional outcome for conditioning (batch,)
        Returns:
            Reconstructed sequence (batch, seq_len, input_dim)
        """
        if outcome is not None:
            outcome = outcome.unsqueeze(1)
            z = torch.cat([z, outcome], dim=1)
            z = self.condition_to_latent(z)

        h_dec = self.latent_to_hidden(z).unsqueeze(0)  # Initial hidden state
        c_dec = torch.zeros_like(h_dec)  # Initial cell state

        # Zero-input decoder, LSTM generates from hidden state only
        dec_input = torch.zeros((z.size(0), seq_len, self.hidden_dim), device=z.device)
        output, _ = self.decoder(dec_input, (h_dec, c_dec))

        # Project to original input space
        output = self.output_projection(output)
        return output

    def forward(self, x, outcome=None, lengths=None):
        """
        Full VAE forward pass.
        Args:
            x: Input sequence (batch, seq_len, input_dim)
            outcome: Optional scalar outcome (batch,)
            lengths: Sequence lengths before padding (batch,)
        Returns:
            recon_x: Reconstructed sequence
            mu, logvar: Latent distribution parameters
        """
        mu, logvar = self.encode(x, lengths)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, x.size(1), outcome=outcome)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE loss combines MSE reconstruction loss and KL divergence.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div, recon_loss, kl_div
