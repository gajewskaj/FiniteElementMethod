import os
import sys
import csv
import prettytable
from glob import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.helpers.config import Settings

os.chdir(Settings.project_path)
results = {}
for solver in ["cudss_v1", "cudss_v2", "cudss_v3", "cudss_v4"]:
    results[solver] = {}
    for mesh_file in glob(os.path.join(Settings.input_path, "*.msh")):
        os.system("ncu --csv --profile-from-start off --metrics " \
        "sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum " \
        f"python main.py --solver {solver} --mesh {mesh_file} > temp.csv")

        add = 0
        mul = 0
        fma = 0

        with open("temp.csv", "r") as f:
            reader = csv.reader(f)
            data = list(reader)
            nodes = data[3][0].split()[2]
            results[solver][nodes] = {}
            for row in data[8:]:
                match row[12]:
                    case "sm__sass_thread_inst_executed_op_dadd_pred_on.sum":
                        add += int(row[14])
                    case "sm__sass_thread_inst_executed_op_dmul_pred_on.sum":
                        mul += int(row[14])
                    case "sm__sass_thread_inst_executed_op_dfma_pred_on.sum":
                        fma += int(row[14])
                    case _:
                        raise ValueError(f"Unexpected metric: {row[12]}")
        results[solver][nodes] = (add, mul, fma, add + mul + 2 * fma)
        add = 0
        mul = 0
        fma = 0
        
        table = prettytable.PrettyTable()
        table.field_names = ["Nodes", "ADD", "MUL", "FMA", "flop"]
        table.add_rows([[nodes, *results[solver][nodes]] for nodes in results[solver]])

        with open(f"{solver}_factorization_ncu_profile.txt", "w") as f:
            print(table, file=f)
        os.system("rm temp.csv")

for solver in results:
    print(f"Solver: {solver}")
    table = prettytable.PrettyTable()
    table.field_names = ["Nodes", "ADD", "MUL", "FMA", "flop"]
    table.add_rows([[nodes, *results[solver][nodes]] for nodes in results[solver]])
    print(table)
    
