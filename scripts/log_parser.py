import sys
import os
import re
from glob import glob

import prettytable

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.helpers.config import output_path

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
    pattern_func_time = re.compile(r"Function '(\w+)' executed in (\d+(\.\d+)?) seconds.")
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
                exec_time = match_func_time.group(2)
                try:
                    i = [case.num_elements for case in case_data].index(num_elements)
                except ValueError:
                    i = -1
                if i == -1:
                    case_data.append(CaseData(num_elements, num_nodes))
                case_data[i].add_time(function_name, float(exec_time))

def interpret_data():
    print("Execution time on average:")
    table = prettytable.PrettyTable()
    table.header = True
    names = ["Elements", "Nodes"]
    names.extend([f"{function_name} [s]" for function_name in case_data[0].functions.keys()])
    names.append("Total time [s]")
    table.field_names = names
    for case in case_data:
        row = [case.num_elements, case.num_nodes]
        total_time: float = 0
        for times in case.functions.values():
            time = sum(times)
            total_time += time
            row.append(time/len(times))
        row.append(total_time)
        table.add_row(row, divider=True)
    print(table)

for directory in glob(output_path + "/*"):
    for filename in glob(os.path.join(output_path, directory, "*.log")):
        parse_log_file(filename)
interpret_data()