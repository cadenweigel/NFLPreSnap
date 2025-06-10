import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        """
        Conditional Variational Autoencoder for trajectory sequences.

        Args:
            input_dim (int): Flattened input feature size per time step (e.g., 132 for 22 players × 6 features).
            hidden_dim (int): Hidden size of encoder/decoder LSTMs.
            latent_dim (int): Dimension of latent representation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder LSTM: processes input sequence into hidden state
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Latent space transformations
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # If using conditioning (e.g., yards gained), apply before latent-to-hidden
        self.condition_to_latent = nn.Linear(latent_dim + 1, latent_dim)

        # Map latent vector to decoder's initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # Decoder LSTM now outputs hidden_dim, not input_dim
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Final projection to reconstruct original input dimension
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Encodes the input into latent mean and log-variance."""
        _, (h_n, _) = self.encoder(x)  # h_n shape: (1, batch, hidden_dim)
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Samples from N(mu, sigma^2) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len, outcome=None):
        """
        Decodes the latent code into a trajectory sequence, optionally conditioned on outcome.

        Args:
            z: latent vector (batch, latent_dim)
            seq_len: number of time steps to decode
            outcome: optional scalar conditioning input (batch,)
        """
        if outcome is not None:
            outcome = outcome.unsqueeze(1)  # shape: (batch, 1)
            z = torch.cat([z, outcome], dim=1)  # shape: (batch, latent + 1)
            z = self.condition_to_latent(z)  # back to latent_dim

        # Initialize decoder hidden and cell states
        h_dec = self.latent_to_hidden(z).unsqueeze(0)  # shape: (1, batch, hidden_dim)
        c_dec = torch.zeros_like(h_dec)

        # Start decoding with zero input
        dec_input = torch.zeros((z.size(0), seq_len, self.hidden_dim), device=z.device)
        output, _ = self.decoder(dec_input, (h_dec, c_dec))  # (batch, seq_len, hidden_dim)

        # Project output back to original input dimension
        output = self.output_projection(output)  # (batch, seq_len, input_dim)
        return output

    def forward(self, x, outcome=None):
        """Full VAE forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, x.size(1), outcome=outcome)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss: MSE reconstruction loss + KL divergence."""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div, recon_loss, kl_div
