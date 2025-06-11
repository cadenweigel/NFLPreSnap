# train/train_predictor.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from models.outcome_predictor import OutcomePredictor
from data.collate import collate_fn

# --- Configuration ---

PLAYER_FEATURES = 6
PLAYER_COUNT = 22
INPUT_DIM = PLAYER_COUNT * PLAYER_FEATURES
LSTM_HIDDEN = 64
DENSE_HIDDEN = 128
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 10
MODEL_PATH = "models/outcome_predictor.pt"

TRAIN_PATH = "data/npy/full_play_data_train.npy"
VAL_PATH = "data/npy/full_play_data_val.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Dataset Class (compatible with collate_fn) ---

class PlayDataset(Dataset):
    def __init__(self, data_dict):
        self.plays = list(data_dict.values())

    def __len__(self):
        return len(self.plays)

    def __getitem__(self, idx):
        play = self.plays[idx]
        return {
            'trajectory': play['trajectory'].astype(np.float32),  # [T, 22, 6]
            'outcome': play['outcome']                            # scalar
        }


def load_dataset(path):
    data = np.load(path, allow_pickle=True).item()
    return PlayDataset(data)


# --- Training / Evaluation ---

def train(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    i = 0

    for x, lengths, y in dataloader:
        #print(f"⏱️ Batch {i}: x={x.shape}, lengths={lengths}, y={y.shape}")
        x, y, lengths = x.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
        x = x.view(x.shape[0], x.shape[1], -1)  # [B, T, 132]
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        preds = model(packed)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        i+=1

    print(f"[Epoch {epoch}][Train] MSE Loss: {total_loss / len(dataloader):.4f}")


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    for x, lengths, y in dataloader:
        x, y, lengths = x.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
        x = x.view(x.shape[0], x.shape[1], -1)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        preds = model(packed)
        loss = criterion(preds, y)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Validation] MSE Loss: {avg_loss:.4f}")
    return avg_loss


# --- Entrypoint ---

def main():
    print("📦 Loading data...")
    train_ds = load_dataset(TRAIN_PATH)
    val_ds = load_dataset(VAL_PATH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = OutcomePredictor(
        player_count=PLAYER_COUNT,
        feature_dim=PLAYER_FEATURES,
        lstm_hidden=LSTM_HIDDEN,
        dense_hidden=DENSE_HIDDEN
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)

    best_val_loss = float("inf")
    no_improvement = 0

    print("🚀 Beginning training...")
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)
        val_loss = evaluate(model, val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Best model saved to {MODEL_PATH}")
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                print(f"⏹️ Early stopping after {PATIENCE} epochs without improvement.")
                break

    print("🏁 Training complete.")

if __name__ == "__main__":
    main()
