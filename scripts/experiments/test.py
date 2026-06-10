import os
import sys
from glob import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.helpers.config import Settings

os.chdir(Settings.project_path)

for solver in ("scipy_v1", "scipy_v2", "scipy_v3", "cupy_v1", "cupy_v2", "cupy_v3"):
    for i in range(10):
        mesh_file = os.path.join(Settings.input_path, "1000x1000_quad.msh")
        os.system(f"python main.py --mesh {mesh_file} --solver {solver}")
        log_dir_name: str = os.path.join(Settings.output_path, os.path.basename(mesh_file).strip(".msh"))
        os.rename(log_dir_name, f"{log_dir_name}_{solver}_{i+1}")
    os.system(f"python scripts/experiments/create_time_stats.py --file {solver}.csv --log-dir-template *_{solver}_{i+1}*")