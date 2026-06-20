import os
import sys
from glob import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.helpers.config import Settings

os.chdir(Settings.project_path)

for solver in ("cudss_v1", "cudss_v2", "cudss_v3", "cudss_v4"):
    for i in range(10):
        for mesh_file in glob(os.path.join(Settings.input_path, "*.msh")):
            os.system(f"python main.py --mesh {os.path.join(Settings.input_path, mesh_file)} --solver {solver}")
            log_dir_name: str = os.path.join(Settings.output_path, os.path.basename(mesh_file).strip(".msh"))
            os.rename(log_dir_name, f"{log_dir_name}_{solver}_{i+1}")
    os.system(f"python scripts/experiments/create_time_stats.py --file {solver}.csv --log-dir-template *_{solver}_{i+1}*")