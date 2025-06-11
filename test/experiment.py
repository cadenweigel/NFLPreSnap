import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from models.vae import TrajectoryVAE
from models.outcome_predictor import OutcomePredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Constants ---
INPUT_DIM = 132
LATENT_DIM = 64
HIDDEN_DIM = 128
MODEL_PATH_VAE = "models/trajectory_vae.pt"
MODEL_PATH_PREDICTOR = "models/outcome_predictor.pt"
TEST_PATH = "data/npy/full_play_data_test.npy"
CSV_PATH = "test/predictions.csv"
VIS_DIR = "test/visualizations"
NUM_VISUALS = 5

os.makedirs(VIS_DIR, exist_ok=True)

# --- Load Models ---
def load_models():
    vae = TrajectoryVAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(MODEL_PATH_VAE, map_location=DEVICE))
    vae.eval()

    predictor = OutcomePredictor().to(DEVICE)
    predictor.load_state_dict(torch.load(MODEL_PATH_PREDICTOR, map_location=DEVICE))
    predictor.eval()

    return vae, predictor

# --- Load and preprocess test data ---
def load_test_data():
    data = np.load(TEST_PATH, allow_pickle=True).item()
    real_plays, real_outcomes, lengths = [], [], []

    for item in data.values():
        traj = item["trajectory"].astype(np.float32)  # [T, 22, 6]
        outcome = item["outcome"]
        T = traj.shape[0]
        lengths.append(T)
        traj_flat = traj.reshape(T, -1)  # [T, 132]
        real_plays.append(torch.tensor(traj_flat))
        real_outcomes.append(torch.tensor(outcome))

    return real_plays, real_outcomes, lengths

# --- Generate synthetic plays using VAE ---
def generate_plays(vae, outcomes, lengths):
    z = torch.randn(len(outcomes), LATENT_DIM).to(DEVICE)
    y = torch.tensor(outcomes, dtype=torch.float32).to(DEVICE)
    max_len = max(lengths)
    synthetic = vae.decode(z, max_len, outcome=y).detach().cpu()
    return synthetic  # [B, T, 132]

# --- Evaluate predicted outcomes and return results ---
def evaluate_predictions(predictor, trajectories, lengths, true_outcomes, title=""):
    packed = pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)
    preds = predictor(packed).detach().cpu()
    true = torch.tensor(true_outcomes)
    mse = torch.mean((preds - true) ** 2).item()
    print(f"{title} MSE: {mse:.4f}")
    return preds.numpy(), true.numpy(), mse

# --- Visualize a trajectory ---
def visualize_trajectory(traj_tensor, name):
    """
    Visualizes a play trajectory for 22 players over time.
    Offense = blue, Defense = red.
    Start point marked with 'X'.
    Arrows show direction from first to last position.
    If coordinates are small (e.g., synthetic), rescale to match typical field dimensions.
    """
    import matplotlib.pyplot as plt

    traj = traj_tensor.view(traj_tensor.size(0), 22, 6)  # [T, 22, 6]

    # --- Optional rescaling for synthetic output ---
    if traj[:, :, 0:2].abs().max() < 5:  # Likely a synthetic, tiny-scale trajectory
        traj[:, :, 0:2] *= 50  # Heuristic scale-up (you can tune this)

    plt.figure(figsize=(10, 6))
    for pid in range(22):
        x = traj[:, pid, 0].numpy()
        y = traj[:, pid, 1].numpy()

        # Offense: first 11 players, Defense: next 11
        if pid < 11:
            color = "blue"
            label = "Offense" if pid == 0 else None
        else:
            color = "red"
            label = "Defense" if pid == 11 else None

        # Draw trajectory
        plt.plot(x, y, color=color, alpha=0.6, label=label)

        # Starting point
        plt.scatter(x[0], y[0], color=color, edgecolors='k', marker="X", s=60)

        # Direction arrow (start to end)
        dx, dy = x[-1] - x[0], y[-1] - y[0]
        plt.quiver(
            x[0], y[0], dx, dy,
            angles='xy', scale_units='xy', scale=1,
            color=color, alpha=0.6, width=0.003
        )

    plt.title(f"Trajectory: {name}")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/{name}.png")
    plt.close()



def compute_ade_fde(real_seqs, gen_seqs, lengths):
    """
    real_seqs, gen_seqs: Tensors [B, T, 132]
    lengths: Tensor [B], true sequence lengths
    Returns: ADE, FDE
    """
    ade_list = []
    fde_list = []

    for i in range(len(real_seqs)):
        T = lengths[i]
        real = real_seqs[i, :T].view(T, 22, 6)[:, :, :2]  # [T, 22, 2]
        gen = gen_seqs[i, :T].view(T, 22, 6)[:, :, :2]    # [T, 22, 2]

        dists = torch.norm(real - gen, dim=2)  # [T, 22]
        ade = dists.mean().item()
        fde = dists[-1].mean().item()

        ade_list.append(ade)
        fde_list.append(fde)

    return np.mean(ade_list), np.mean(fde_list)


# --- Save results to CSV ---
def save_to_csv(real_preds, real_true, syn_preds, syn_true):
    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Type", "TrueOutcome", "PredictedOutcome"])
        for y, y_hat in zip(real_true, real_preds):
            writer.writerow(["Real", y, y_hat])
        for y, y_hat in zip(syn_true, syn_preds):
            writer.writerow(["Synthetic", y, y_hat])
    print(f"Saved predictions to {CSV_PATH}")

# --- Run full experiment ---
def run_experiment():
    vae, predictor = load_models()
    real_plays, real_outcomes, lengths = load_test_data()

    # Pad and evaluate real
    padded_real = pad_sequence(real_plays, batch_first=True).to(DEVICE)
    real_lengths = torch.tensor(lengths)
    print("Evaluating on real plays...")
    real_preds, real_true, _ = evaluate_predictions(predictor, padded_real, real_lengths, real_outcomes, title="Real")

    # Generate and evaluate synthetic
    print("Generating and evaluating synthetic plays...")
    synthetic = generate_plays(vae, real_outcomes, lengths)
    syn_preds, syn_true, _ = evaluate_predictions(predictor, synthetic, real_lengths, real_outcomes, title="Synthetic")

     # Compute ADE/FDE
    print("Computing ADE/FDE for synthetic plays...")
    ade, fde = compute_ade_fde(padded_real.cpu(), synthetic, real_lengths)
    print(f"ADE: {ade:.3f} | FDE: {fde:.3f}")

    # Save predictions
    save_to_csv(real_preds, real_true, syn_preds, syn_true)

    # Visualize plays
    print(f"Saving {NUM_VISUALS} sample play visualizations...")
    for i in range(NUM_VISUALS):
        visualize_trajectory(real_plays[i], f"real_{i}")
        visualize_trajectory(synthetic[i], f"synthetic_{i}")

if __name__ == "__main__":
    run_experiment()
