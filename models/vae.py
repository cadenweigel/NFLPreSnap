# models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        """
        Args:
            input_dim (int): Number of input features per time step (flattened player features).
            hidden_dim (int): Hidden dimension size for LSTMs.
            latent_dim (int): Size of the latent vector z.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder LSTM: processes input sequence into a final hidden state
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Linear layers to produce mean and log-variance of the latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: starts from latent vector, expands into hidden state, then LSTM decodes sequence
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def encode(self, x):
        """
        Encodes the input sequence into latent distribution parameters.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            mu (Tensor): Mean vector of latent distribution (batch, latent_dim)
            logvar (Tensor): Log variance vector (batch, latent_dim)
        """
        _, (h_n, _) = self.encoder(x)  # h_n: (1, batch, hidden_dim)
        h_n = h_n.squeeze(0)           # -> (batch, hidden_dim)
        mu = self.fc_mu(h_n)           # -> (batch, latent_dim)
        logvar = self.fc_logvar(h_n)   # -> (batch, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from N(mu, sigma^2)

        Args:
            mu (Tensor): Mean of latent distribution
            logvar (Tensor): Log variance of latent distribution

        Returns:
            z (Tensor): Latent sample (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """
        Decodes a latent vector into a reconstructed sequence.

        Args:
            z (Tensor): Latent vector (batch, latent_dim)
            seq_len (int): Desired sequence length

        Returns:
            output (Tensor): Reconstructed sequence (batch, seq_len, input_dim)
        """
        h_dec = self.latent_to_hidden(z).unsqueeze(0)  # (1, batch, hidden_dim)
        c_dec = torch.zeros_like(h_dec)                # Initial cell state
        # Initial input to decoder is zeros
        dec_input = torch.zeros((z.size(0), seq_len, self.input_dim), device=z.device)
        output, _ = self.decoder(dec_input, (h_dec, c_dec))
        return output

    def forward(self, x):
        """
        Full forward pass through the VAE.

        Args:
            x (Tensor): Input sequence (batch, seq_len, input_dim)

        Returns:
            recon_x (Tensor): Reconstructed sequence
            mu (Tensor): Mean of latent distribution
            logvar (Tensor): Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, x.size(1))
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Computes total VAE loss = reconstruction loss + KL divergence.

        Args:
            recon_x (Tensor): Reconstructed sequence
            x (Tensor): Original input sequence
            mu (Tensor): Mean of latent distribution
            logvar (Tensor): Log variance of latent distribution

        Returns:
            total_loss (Tensor): Combined loss
            recon_loss (Tensor): MSE reconstruction loss
            kl_div (Tensor): KL divergence loss
        """
        # Mean squared error between reconstructed and original input
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        # KL divergence between latent distribution and standard normal
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div, recon_loss, kl_div
