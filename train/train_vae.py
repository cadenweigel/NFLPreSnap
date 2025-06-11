# --- train/train_vae.py ---

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models.vae import TrajectoryVAE

# --- Configuration Constants ---

PLAYER_FEATURES = 6
ROLE_FILTERS = ["offense", "defense"]
PLAYER_COUNT = 22
INPUT_DIM = PLAYER_COUNT * PLAYER_FEATURES
HIDDEN_DIM = 128
LATENT_DIM = 64
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 10
MODEL_PATH = "models/trajectory_vae.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading Function ---

def load_variable_length_plays(path, role_filters=ROLE_FILTERS):
    """
    Loads a dataset of variable-length trajectory sequences with player features.
    Filters out plays that don't have exactly 22 players with valid roles.
    Returns a list of (sequence_tensor, outcome) tuples.
    """
    data_dict = np.load(path, allow_pickle=True).item()
    sequences, outcomes = [], []

    for item in data_dict.values():
        traj = item["trajectory"]
        roles = item["roles"]
        outcome = item["outcome"]

        mask = [i for i, r in enumerate(roles) if r in role_filters]
        if len(mask) != PLAYER_COUNT:
            continue

        traj = traj[:, mask, :]  # (T, 22, 6)
        flattened = traj.reshape(traj.shape[0], -1)  # (T, 132)

        sequences.append(torch.tensor(flattened, dtype=torch.float32))
        outcomes.append(torch.tensor(outcome, dtype=torch.float32))

    return list(zip(sequences, outcomes))

def collate_fn(batch):
    """
    Pads a batch of variable-length sequences for batching.
    Returns padded sequences, outcome labels, and original lengths.
    """
    x_list, y_list = zip(*batch)
    x_padded = pad_sequence(x_list, batch_first=True)  # (B, T_max, 132)
    y_tensor = torch.stack(y_list)  # (B,)
    lengths = torch.tensor([seq.size(0) for seq in x_list])  # (B,)
    return x_padded, y_tensor, lengths

# --- Training and Evaluation ---

def train(model, dataloader, optimizer, epoch):
    model.train()
    total_loss, recon_total, kl_total = 0, 0, 0

    for x, y, lengths in dataloader:
        x, y, lengths = x.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
        optimizer.zero_grad()

        # Forward pass with outcome conditioning and sequence lengths
        recon_x, mu, logvar = model(x, outcome=y, lengths=lengths)

        # Compute VAE loss
        loss, recon_loss, kl_div = model.loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_total += recon_loss.item()
        kl_total += kl_div.item()

    print(f"[Epoch {epoch}][Train] Total: {total_loss:.4f} | Recon: {recon_total:.4f} | KL: {kl_total:.4f}")

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss, recon_total, kl_total = 0, 0, 0

    for x, y, lengths in dataloader:
        x, y, lengths = x.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)

        # Forward pass with outcome conditioning and sequence lengths
        recon_x, mu, logvar = model(x, outcome=y, lengths=lengths)

        # Compute VAE loss
        loss, recon_loss, kl_div = model.loss_function(recon_x, x, mu, logvar)
        total_loss += loss.item()
        recon_total += recon_loss.item()
        kl_total += kl_div.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Validation] Total: {total_loss:.4f} | Recon: {recon_total:.4f} | KL: {kl_total:.4f}")
    return avg_loss

# --- Main Entrypoint ---

def main():
    # Load training and validation datasets
    train_data = load_variable_length_plays("data/npy/full_play_data_train.npy")
    val_data = load_variable_length_plays("data/npy/full_play_data_val.npy")

    # Use custom collate_fn to pad sequences
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Initialize VAE model and optimizer
    model = TrajectoryVAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Training loop with early stopping
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)
        val_loss = evaluate(model, val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model saved to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                break

    print("Training complete.")

if __name__ == "__main__":
    main()
