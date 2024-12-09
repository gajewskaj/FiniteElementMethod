import argparse
import csv
import os
import re
import sys
from glob import glob

import prettytable

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.helpers.config import output_path, scripts_path

def parse_arguments() -> str:
    parser = argparse.ArgumentParser(description="Log parser")
    parser.add_argument("--file", type=str, default="time_stats.csv",
                        help="Output filename")
    parser.add_argument("--log-dir-template", type=str, default="*", choices=["*", "*_cpu*", "*_gpu*"],
                        help="Output directory name template")
    args = parser.parse_args()
    return args.file, args.log_dir_template

class CaseData:
    def __init__(self, num_elements: int, num_nodes: int):
        self.num_elements = num_elements
        self.num_nodes = num_nodes
        self.functions = {}

    def add_time(self, funcion_name: str, time: float):
        if funcion_name in self.functions:
            self.functions[funcion_name].append(time)
        else:
            self.functions[funcion_name] = [time]

case_data: list[CaseData] = []

def parse_log_file(filename: str):
    pattern_num_elements = re.compile(r"Number of elements: (\d+)")
    pattern_num_nodes = re.compile(r"Number of nodes: (\d+)")
    pattern_func_time = re.compile(r"Function '(\w+)' executed in (.+) seconds.")
    with open(filename, "r") as f:
        lines = f.readlines()
        num_elements = None
        num_nodes = None
        for line in lines:
            match_num_elements = pattern_num_elements.search(line)
            match_num_nodes = pattern_num_nodes.search(line)
            match_func_time = pattern_func_time.search(line)
            if match_num_elements:
                num_elements = int(match_num_elements.group(1))
            if match_num_nodes:
                num_nodes = int(match_num_nodes.group(1))
            if match_func_time:
                function_name = match_func_time.group(1)
                exec_time = float(match_func_time.group(2))
                try:
                    i = [case.num_elements for case in case_data].index(num_elements)
                except ValueError:
                    i = -1
                if i == -1:
                    case_data.append(CaseData(num_elements, num_nodes))
                case_data[i].add_time(function_name, exec_time)

def interpret_data():
    column_names = ["Elements", "Nodes"]
    column_names.extend([function_name for function_name in case_data[0].functions.keys()])
    rows = []
    for case in case_data:
        row = [case.num_elements, case.num_nodes]
        for times in case.functions.values():
            avg_time = sum(times)/len(times)
            row.append(avg_time)
        rows.append(row)
    return column_names, rows

def print_table(column_names: list[str], rows: list[list[str]]) -> None:
    print("Execution time on average [s]:")
    table = prettytable.PrettyTable()
    table.field_names = column_names
    table.add_rows(rows)
    print(table)

def save_to_csv(file_name: str, column_names: list[str], rows: list[list[str]]) -> None:
    with open(os.path.join(scripts_path, file_name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(rows)

file_name, log_dir_template = parse_arguments()
for directory in glob(output_path + f"/{log_dir_template}"):
    for filename in glob(os.path.join(output_path, directory, "*.log")):
        parse_log_file(filename)
column_names, rows = interpret_data()
print_table(column_names, rows)
save_to_csv(file_name, column_names, rows)