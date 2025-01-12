import os
import sys
from glob import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.helpers.config import input_path, output_path, project_path

os.chdir(project_path)

for i in range(10):
    for mesh_file in glob(input_path + "/*.msh"):
        os.system(f"py main.py --mesh {mesh_file} --data {input_path}/global_data.json")
        log_dir_name: str = os.path.join(output_path, os.path.basename(mesh_file).strip(".msh"))
        os.rename(log_dir_name, f"{log_dir_name}_gpu{i+1}")
os.system(f"py scripts/create_time_stats.py --file gpu_stats.csv --log-dir-template *_gpu*")

for i in range(10):
    for mesh_file in glob(input_path + "/*.msh"):
        os.system(f"py main.py --mesh {mesh_file} --data {input_path}/global_data.json --force-cpu")
        log_dir_name: str = os.path.join(output_path, os.path.basename(mesh_file).strip(".msh"))
        os.rename(log_dir_name, f"{log_dir_name}_cpu{i+1}")
os.system(f"py scripts/create_time_stats.py --file cpu_stats.csv --log-dir-template *_cpu*")