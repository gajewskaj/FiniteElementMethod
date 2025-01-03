import argparse
import csv
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.helpers.config import output_path, scripts_path

times = []
min_temp = []
max_temp = []

def parse_arguments() -> str:
    parser = argparse.ArgumentParser(description="Temperature results to CSV")
    parser.add_argument("--log-dir", type=str, default="1000x1000_quad",
                        help="Input log directory")
    parser.add_argument("--file", type=str, default="temperatures.csv",
                        help="Output filename")
    args = parser.parse_args()
    return args.log_dir, args.file

def parse_log_file(filename: str):
    pattern = re.compile(r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            match_pattern = pattern.search(line)
            if match_pattern:
                times.append(float(match_pattern.group(1)))
                min_temp.append(float(match_pattern.group(2)))
                max_temp.append(float(match_pattern.group(3)))

def interpret_data():
    column_names = ["Time", "Min temperature", "Max temperature"]
    rows = []
    for i in range(len(times)):
        rows.append([times[i], min_temp[i], max_temp[i]])
    return column_names, rows

def save_to_csv(filename: str, column_names: list[str], rows: list[list[str]]) -> None:
    with open(os.path.join(scripts_path, filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(rows)

input_directory, output_filename = parse_arguments()
parse_log_file(os.path.join(output_path, input_directory, "log.log"))
column_names, rows = interpret_data()
save_to_csv(output_filename, column_names, rows)