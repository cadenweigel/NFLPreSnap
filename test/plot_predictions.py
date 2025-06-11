import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import os

CSV_PATH = "test/predictions.csv"
OUT_DIR = "test/prediction_plots"
os.makedirs(OUT_DIR, exist_ok=True)

def load_predictions(path):
    return pd.read_csv(path)

def plot_histograms(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="TrueOutcome", hue="Type", kde=True, bins=20, stat="density", alpha=0.6)
    sns.histplot(data=df, x="PredictedOutcome", hue="Type", kde=True, bins=20, stat="density", alpha=0.3, linestyle="--", element="step")
    plt.title("Distribution of True vs Predicted Outcomes")
    plt.xlabel("Yards Gained")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/histogram_true_vs_predicted.png")
    plt.close()

def plot_scatter(df):
    plt.figure(figsize=(6, 6))

    # Plot scatter by type
    sns.scatterplot(data=df, x="TrueOutcome", y="PredictedOutcome", hue="Type", alpha=0.6)

    # Plot regression lines and compute R² separately
    r2_texts = []
    for type_name, color in zip(["Real", "Synthetic"], ["blue", "red"]):
        subset = df[df["Type"] == type_name]
        sns.regplot(
            data=subset,
            x="TrueOutcome", y="PredictedOutcome",
            scatter=False,
            color=color,
            line_kws={"label": f"{type_name} Fit"},
        )
        r2 = r2_score(subset["TrueOutcome"], subset["PredictedOutcome"])
        r2_texts.append(f"{type_name} $R^2$ = {r2:.3f}")

    # Perfect prediction reference line
    plt.plot(
        [df["TrueOutcome"].min(), df["TrueOutcome"].max()],
        [df["TrueOutcome"].min(), df["TrueOutcome"].max()],
        'k--', label="Perfect Prediction"
    )

    # Display R² scores in upper left
    full_r2 = r2_score(df["TrueOutcome"], df["PredictedOutcome"])
    r2_texts.append(f"Combined $R^2$ = {full_r2:.3f}")
    plt.text(0.05, 0.95, "\n".join(r2_texts),
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top')

    plt.xlabel("True Outcome (Yards)")
    plt.ylabel("Predicted Outcome (Yards)")
    plt.title("True vs Predicted Outcomes by Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/scatter_true_vs_predicted.png")
    plt.close()

def plot_error_distribution(df):
    df["AbsError"] = (df["TrueOutcome"] - df["PredictedOutcome"]).abs()
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="AbsError", hue="Type", kde=True, bins=20)
    plt.title("Absolute Prediction Error Distribution")
    plt.xlabel("Absolute Error (Yards)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/histogram_absolute_error.png")
    plt.close()

def main():
    df = load_predictions(CSV_PATH)
    print(f"✅ Loaded {len(df)} predictions")

    plot_histograms(df)
    plot_scatter(df)
    plot_error_distribution(df)

    print(f"📊 Plots saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
