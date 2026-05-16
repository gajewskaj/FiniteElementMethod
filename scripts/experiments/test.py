import os
import sys
from glob import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.helpers.config import Settings

os.chdir(Settings.project_path)

for i in range(10):
    for mesh_file in glob(Settings.input_path + "/*.msh"):
        os.system(f"python main.py --mesh {mesh_file} --data {os.path.join(Settings.input_path, 'global_data.json')} --mc cpu --solver scipy")
        log_dir_name: str = os.path.join(Settings.output_path, os.path.basename(mesh_file).strip(".msh"))
        os.rename(log_dir_name, f"{log_dir_name}_scipy{i+1}")
os.system(f"python scripts/experiments/create_time_stats.py --file scipy.csv --log-dir-template *_scipy*")

# for i in range(10):
#     for mesh_file in glob(Settings.input_path + "/*.msh"):
#         os.system(f"python main.py --mesh {mesh_file} --data {Settings.input_path}/global_data.json --force-cpu")
#         log_dir_name: str = os.path.join(Settings.output_path, os.path.basename(mesh_file).strip(".msh"))
#         os.rename(log_dir_name, f"{log_dir_name}_cpu{i+1}")
# os.system(f"python scripts/experiments/create_time_stats.py --file cpu_stats.csv --log-dir-template *_cpu*")