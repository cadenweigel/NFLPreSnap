import sys
import pandas as pd
import os

def create_sample_file(input_filename, percent):
    # Validate percent
    if not (0 < percent <= 1):
        raise ValueError("Percent must be a float between 0 and 1 (exclusive of 0).")

    # Read the CSV
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{input_filename}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

    # Sample the DataFrame
    sample_df = df.sample(frac=percent, random_state=42)

    # Construct output filename
    base_name = os.path.basename(input_filename)
    output_filename = f"test_{base_name}"

    # Save to new file
    sample_df.to_csv(output_filename, index=False)
    print(f"Sampled file saved as '{output_filename}' with {len(sample_df)} rows.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_sample_data.py filename.csv percent")
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        fraction = float(sys.argv[2])
    except ValueError:
        print("Percent must be a number between 0 and 1.")
        sys.exit(1)

    try:
        create_sample_file(input_file, fraction)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
