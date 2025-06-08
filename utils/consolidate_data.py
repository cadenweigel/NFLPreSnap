import csv
import os
import pandas as pd

def consolidate_data():
    weeks = range(1, 10)
    file_template = "data/tracking_week_{}.csv"
    output_path = "data/tracking_all_weeks.csv"
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header_written = False

    with open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = None

        for week in weeks:
            file_path = file_template.format(week)
            with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                header = next(reader)

                if not header_written:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    header_written = True

                for row in reader:
                    writer.writerow(row)

if __name__ == "__main__":
    consolidate_data()
