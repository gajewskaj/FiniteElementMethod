import argparse
import os

import numpy as np

from src.helpers.helpers import set_use_gpu, create_or_clear_directory, init_logging
from src.helpers import config

def parse_arguments() -> tuple[str, str, bool]:
    parser = argparse.ArgumentParser(description="Finite Element Method Simulation")
    parser.add_argument("--mesh", type=str, default=os.path.join(config.input_path, "50x50_quad.msh"),
                        help="Path to the input grid file")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the input data file")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force the simulation to run on CPU")
    args = parser.parse_args()
    return args.mesh, args.data, args.force_cpu

def run() -> None:
    try:
        mesh_filepath, data_filepath, force_cpu = parse_arguments()
        # In output directory, create a subdirectory with the name of the input mesh file
        output_dir_path = create_or_clear_directory(os.path.join(config.output_path,
                                                                 os.path.basename(mesh_filepath).split(".")[0]))
        config.logger = init_logging(logger_name=config.MAIN_LOGGER_NAME, log_dirpath=output_dir_path)
        set_use_gpu(force_cpu)

        from src.grid.grid import Grid
        grid = Grid(mesh_filepath, data_filepath)

        from src.mc.matrices_calculation import calculate_and_assemble_matrices
        from src.soe.temperature_simulation import simulate

        calculate_and_assemble_matrices(grid)
        times, temperatures = simulate(grid)

        config.logger.info(f"Time        Min temp    Max temp")
        for i in range(len(temperatures)):
            config.logger.info(f"{(times[i]):<12}{round(np.min(temperatures[i]), 3):<12}{round(np.max(temperatures[i]), 3):<12}")

        # Generate .vtk files for ParaView
        from src.helpers.vtk_generator import generate_vtk_files
        generate_vtk_files(output_dir_path, grid, temperatures)
    except Exception:
        config.logger.error(f"Script execution failed due to an exception. Check log file for details.", exc_info=True)

if __name__ == "__main__":
    run()