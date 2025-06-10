# train/train_vae.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.vae import TrajectoryVAE

# --- Configuration Constants ---

SEQ_LEN = 50                        # Number of frames per play
PLAYER_FEATURES = 6                # ['x', 'y', 's', 'a', 'dir', 'o']
ROLE_FILTERS = ["offense", "defense"]  # Include both offense and defense players
PLAYER_COUNT = 22                  # 11 offense + 11 defense
INPUT_DIM = PLAYER_COUNT * PLAYER_FEATURES  # 132
HIDDEN_DIM = 128
LATENT_DIM = 64
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 10                       # Early stopping patience
MODEL_PATH = "models/trajectory_vae.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Data Loading Function ---

def load_npy_dict_conditioned(path, seq_len=SEQ_LEN, role_filters=ROLE_FILTERS):
    """
    Load .npy dictionary of trajectory data and outcomes.
    Filters for plays with exactly 22 known-role players (offense + defense),
    and returns fixed-length, flattened trajectory tensors + outcome labels.
    """
    data_dict = np.load(path, allow_pickle=True).item()
    sequences, outcomes = [], []

    for item in data_dict.values():
        traj = item["trajectory"]  # (T, 22, 6)
        roles = item["roles"]      # list of 22 role strings
        outcome = item["outcome"]  # yards gained (float)

        # Filter players by role
        mask = [i for i, r in enumerate(roles) if r in role_filters]
        if len(mask) != PLAYER_COUNT:
            continue  # skip incomplete plays

        traj = traj[:, mask, :]  # (T, 22, 6)

        # Truncate or zero-pad to SEQ_LEN
        if traj.shape[0] >= seq_len:
            clipped = traj[:seq_len]
        else:
            pad = np.zeros((seq_len - traj.shape[0], PLAYER_COUNT, PLAYER_FEATURES))
            clipped = np.concatenate([traj, pad], axis=0)

        # Flatten player data for each frame
        sequences.append(clipped.reshape(seq_len, -1))  # (SEQ_LEN, 132)
        outcomes.append(outcome)

    # Convert to PyTorch tensors
    x = torch.tensor(np.stack(sequences), dtype=torch.float32)
    y = torch.tensor(outcomes, dtype=torch.float32)
    return TensorDataset(x, y)


# --- Training and Evaluation ---

def train(model, dataloader, optimizer, epoch):
    model.train()
    total_loss, recon_total, kl_total = 0, 0, 0

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        # Forward pass with outcome conditioning
        recon_x, mu, logvar = model(x, outcome=y)
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

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        recon_x, mu, logvar = model(x, outcome=y)
        loss, recon_loss, kl_div = model.loss_function(recon_x, x, mu, logvar)

        total_loss += loss.item()
        recon_total += recon_loss.item()
        kl_total += kl_div.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Validation] Total: {total_loss:.4f} | Recon: {recon_total:.4f} | KL: {kl_total:.4f}")
    return avg_loss


# --- Main Entrypoint ---

def main():
    # Load datasets
    train_data = load_npy_dict_conditioned("data/npy/full_play_data_train.npy")
    val_data = load_npy_dict_conditioned("data/npy/full_play_data_val.npy")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Initialize model and optimizer
    model = TrajectoryVAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)

    # Early stopping and tracking best model
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)
        val_loss = evaluate(model, val_loader)
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Best model saved to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"⏹️ Early stopping triggered after {PATIENCE} epochs without improvement.")
                break

    print("🏁 Training complete.")

if __name__ == "__main__":
    main()
