import pandas as pd
import numpy as np
import os

# File paths for raw data
TRACKING_CSV = "data/tracking_all_weeks.csv"
PLAYS_CSV = "data/plays.csv"
PLAYERS_CSV = "data/players.csv"
GAMES_CSV = "data/games.csv"

# Configuration for trajectory structure
PLAYER_COUNT = 22
PLAYER_FEATURES = ['x', 'y', 's', 'a', 'dir', 'o']
FEATURE_DIM = len(PLAYER_FEATURES)

def load_data():
    """
    Load all required CSV files into pandas DataFrames.
    """
    tracking = pd.read_csv(TRACKING_CSV)
    plays = pd.read_csv(PLAYS_CSV)
    players = pd.read_csv(PLAYERS_CSV)
    games = pd.read_csv(GAMES_CSV)
    return tracking, plays, players, games

def normalize_coordinates(play_df, snap_frame):
    """
    Normalize coordinates so that the football at the snap is at (0, 0).
    """
    snap_event = play_df[(play_df['event'] == 'ball_snap') & (play_df['frameId'] == snap_frame)]
    ball = snap_event[snap_event['displayName'] == 'football']
    if ball.empty:
        return play_df
    x0, y0 = ball.iloc[0][['x', 'y']]
    play_df['x'] -= x0
    play_df['y'] -= y0
    return play_df

def extract_all_frames(tracking, plays):
    """
    Extract and process play trajectories in batches, each with up to 500 plays.
    Saves batches to 'data/batches/batch_###.npy'.

    Each play is represented as:
        - trajectory: np.array of shape [num_frames, 22, 6]
        - outcome: yards gained
        - roles: list of 22 labels ('offense', 'defense', 'unknown')
    """
    # Batch configuration
    batch_size = 500
    batch_counter = 0
    batch_index = 0
    os.makedirs("data/batches", exist_ok=True)

    # Group tracking data by play
    play_group = tracking.groupby(['gameId', 'playId'])
    play_data = {}

    for (gameId, playId), df in play_group:
        df = df.sort_values('frameId')

        # Find the snap frame
        snap_events = df[df['event'] == 'ball_snap']
        if snap_events.empty:
            continue
        snap_frame = snap_events['frameId'].min()

        # Normalize player coordinates relative to snap
        df = normalize_coordinates(df, snap_frame)

        # Get yards gained from plays metadata
        play_info = plays[(plays['gameId'] == gameId) & (plays['playId'] == playId)]
        if play_info.empty or pd.isna(play_info.iloc[0]['yardsGained']):
            continue
        yards = play_info.iloc[0]['yardsGained']

        # Prepare frame-level data
        frame_ids = sorted(df['frameId'].unique())
        frame_count = len(frame_ids)
        play_tensor = np.zeros((frame_count, PLAYER_COUNT, FEATURE_DIM))

        valid = True
        for f_idx, frame in enumerate(frame_ids):
            frame_df = df[df['frameId'] == frame]

            # Drop the football row
            players_frame = frame_df[~frame_df['displayName'].isin(['football'])]

            # Sort players by team (club) and jersey number
            if 'club' in players_frame.columns and 'jerseyNumber' in players_frame.columns:
                players_frame = players_frame.sort_values(['club', 'jerseyNumber'], na_position='last')

                # Assign roles using possession team
                possession_team = play_info.iloc[0]['possessionTeam']
                def get_role(club):
                    if club == possession_team:
                        return 'offense'
                    elif club in [play_info.iloc[0]['defensiveTeam']] if 'defensiveTeam' in play_info.columns else []:
                        return 'defense'
                    else:
                        return 'unknown'
                players_frame['role'] = players_frame['club'].apply(get_role)
            else:
                print(f"Missing 'club' or 'jerseyNumber' in frame (gameId={gameId}, playId={playId}), skipping.")
                valid = False
                break

            # Require 22 players (exclude incomplete frames)
            if players_frame.shape[0] < PLAYER_COUNT:
                valid = False
                break

            # Fill tensor for this frame
            for p_idx, (_, p) in enumerate(players_frame.iloc[:PLAYER_COUNT].iterrows()):
                play_tensor[f_idx, p_idx, :] = [p[feat] for feat in PLAYER_FEATURES]

        if valid:
            # Capture player roles once per play
            roles = players_frame.iloc[:PLAYER_COUNT]['role'].values.tolist()

            # Save play to current batch
            play_data[(gameId, playId)] = {
                'trajectory': play_tensor,
                'outcome': yards,
                'roles': roles
            }

            batch_counter += 1

            # If batch is full, save to disk
            if batch_counter >= batch_size:
                batch_file = f"data/batches/batch_{batch_index:03}.npy"
                np.save(batch_file, play_data)
                print(f"💾 Saved batch: {batch_file} ({len(play_data)} plays)")
                batch_index += 1
                batch_counter = 0
                play_data = {}

    # Save any remaining plays in the final batch
    if play_data:
        batch_file = f"data/batches/batch_{batch_index:03}.npy"
        np.save(batch_file, play_data)
        print(f"Saved final batch: {batch_file} ({len(play_data)} plays)")

    return None

if __name__ == "__main__":
    print("Loading raw CSV data...")
    tracking, plays, players, games = load_data()

    print("Extracting full-length trajectories (batched)...")
    extract_all_frames(tracking, plays)

    print("Done.")
