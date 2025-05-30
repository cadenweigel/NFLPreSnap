# basic_analysis.py

import pandas as pd
from pathlib import Path

# Paths
data_dir = Path("data")
report_file = Path("reports/basic_analysis.txt")
report_file.parent.mkdir(parents=True, exist_ok=True)

# Load data
games = pd.read_csv(data_dir / "games.csv")
plays = pd.read_csv(data_dir / "plays.csv")
players = pd.read_csv(data_dir / "players.csv")
tracking = pd.read_csv(data_dir / "test/test_tracking_week_1.csv")
player_play = pd.read_csv(data_dir / "test/test_player_play.csv")

# Open the report file for writing
with open(report_file, "w") as report:

    def write_and_print(text):
        print(text)
        report.write(text + "\n")

    def summarize_dataframes():
        write_and_print("\n--- Dataset Overview ---\n")
        for name, df in {
            "Games": games,
            "Plays": plays,
            "Players": players,
            "Tracking": tracking,
            "Player Play": player_play
        }.items():
            write_and_print(f"{name}:\nShape: {df.shape}")
            write_and_print(df.describe(include='all').T.to_string())
            write_and_print(df.isna().sum().to_string() + "\n")

    def top_offensive_contributors():
        write_and_print("\n--- Top Offensive Players ---\n")
        summary = player_play.groupby("nflId")[[
            "passingYards", "rushingYards", "receivingYards"
        ]].sum()
        summary = summary.merge(players[["nflId", "displayName", "position"]], on="nflId", how="left")
        write_and_print("Top Passers:\n" + summary.sort_values(by="passingYards", ascending=False).head(10).to_string())
        write_and_print("Top Rushers:\n" + summary.sort_values(by="rushingYards", ascending=False).head(10).to_string())
        write_and_print("Top Receivers:\n" + summary.sort_values(by="receivingYards", ascending=False).head(10).to_string())

    def play_type_distribution():
        write_and_print("\n--- Play Type Distribution ---\n")
        plays["isPass"] = plays["passResult"].notna()
        plays["isRush"] = plays["rushLocationType"].notna()
        write_and_print(f"Pass Plays: {plays['isPass'].sum()}")
        write_and_print(f"Rush Plays: {plays['isRush'].sum()}")
        write_and_print(f"Other/Unknown: {len(plays) - (plays['isPass'].sum() + plays['isRush'].sum())}")

    def merge_tracking_and_players():
        write_and_print("\n--- Sample Merged Tracking & Player Info ---\n")
        merged = tracking.merge(players[["nflId", "displayName", "position"]], on="nflId", how="left")
        write_and_print(merged[["gameId", "playId", "displayName", "position", "x", "y", "s", "a"]].head().to_string())

    # Run all analysis functions
    summarize_dataframes()
    top_offensive_contributors()
    play_type_distribution()
    merge_tracking_and_players()
