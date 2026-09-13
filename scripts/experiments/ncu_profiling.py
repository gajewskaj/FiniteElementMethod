import os
import sys
import csv
import prettytable
from glob import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.helpers.config import Settings

os.chdir(Settings.project_path)
results = {}
for solver in ["cudss_v4"]:
    results[solver] = {}
    for mesh_file in glob(os.path.join(Settings.input_path, "1000x1000_quad.msh")):
        os.system("ncu --csv --profile-from-start 0 --metrics " \
        "sm__sass_thread_inst_executed_op_dadd_pred_on.sum," \
        "sm__sass_thread_inst_executed_op_dmul_pred_on.sum," \
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum," \
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum," \
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum," \
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum " \
        f"python main.py --solver {solver} --mesh {mesh_file} > temp.csv")

        dadd = 0
        dmul = 0
        dfma = 0
        fadd = 0
        fmul = 0
        ffma = 0

        with open("temp.csv", "r") as f:
            reader = csv.reader(f)
            data = list(reader)
            nodes = data[3][0].split()[2]
            results[solver][nodes] = {}
            for row in data[8:]:
                match row[12]:
                    case "sm__sass_thread_inst_executed_op_dadd_pred_on.sum":
                        dadd += int(row[14])
                    case "sm__sass_thread_inst_executed_op_dmul_pred_on.sum":
                        dmul += int(row[14])
                    case "sm__sass_thread_inst_executed_op_dfma_pred_on.sum":
                        dfma += int(row[14])
                    case "sm__sass_thread_inst_executed_op_fadd_pred_on.sum":
                        fadd += int(row[14])
                    case "sm__sass_thread_inst_executed_op_fmul_pred_on.sum":
                        fmul += int(row[14])
                    case "sm__sass_thread_inst_executed_op_ffma_pred_on.sum":
                        ffma += int(row[14])
                    case _:
                        raise ValueError(f"Unexpected metric: {row[12]}")
        results[solver][nodes] = (dadd, dmul, dfma, dadd + dmul + 2 * dfma, fadd, fmul, ffma, fadd + fmul + 2 * ffma)
        
        table = prettytable.PrettyTable()
        table.field_names = ["Nodes", "DADD", "DMUL", "DFMA", "FP64 FLOP", "FADD", "FMUL", "FFMA", "FP32 FLOP"]
        table.add_rows([[nodes, *results[solver][nodes]] for nodes in results[solver]])
        print(f"Solver: {solver}")
        print(table)
        os.system("rm ncu_profile.csv")