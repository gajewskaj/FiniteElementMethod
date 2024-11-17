import argparse
import os
import pathlib
import time

import numpy as np

from src.helpers.helpers import set_use_gpu, create_or_clear_directory, init_logging
from src.helpers import config
from src.helpers.vtk_generator import generate_vtk_files

def parse_arguments() -> tuple[str, str, bool]:
    """
    Parses command line arguments.

    Returns:
        Settings: Parsed arguments as a Settings object.
    """

    parser = argparse.ArgumentParser(description="Finite Element Method Simulation")
    parser.add_argument('--mesh', type=str, default=os.path.join(config.test_path, "test4_grid.txt"),
                        help='Path to the input grid file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to the input data file')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force the simulation to run on CPU')
    args = parser.parse_args()
    return args.mesh, args.data, args.force_cpu



def run() -> None:
    """
    Runs all the necessary functions to calculate max and min temperature of the element in time.
    """
    try:
        mesh_filepath, data_filepath, force_cpu = parse_arguments()
        # In output directory, create a subdirectory with the name of the input mesh file
        output_dir_path = create_or_clear_directory(os.path.join(config.output_path,
                                                                 os.path.basename(mesh_filepath).split(".")[0]))
        config.logger = init_logging(logger_name=config.MAIN_LOGGER_NAME, log_dirpath=output_dir_path)
        set_use_gpu(force_cpu)

        from src.grid.grid import Grid

        def _create_grid(mesh_filepath: str, data_filepath: str) -> Grid:
            mesh_filepath_ext: str = pathlib.Path(mesh_filepath).suffix
            data_filepath_ext: str = pathlib.Path(data_filepath).suffix if data_filepath is not None else None
            if mesh_filepath_ext == ".msh":
                if data_filepath_ext != ".json":
                    config.logger.error("Data file path in .json format is required for .msh files.")
                    raise Exception
                return Grid.create_from_msh_and_json(mesh_filepath, data_filepath)
            elif mesh_filepath_ext == ".txt":
                if data_filepath_ext is not None:
                    config.logger.warning(f"Data file path is not required for .txt files. Data from {data_filepath} will be ignored.")
                return Grid.create_from_txt(mesh_filepath)

        grid: Grid = _create_grid(mesh_filepath, data_filepath)

        from src.lmc.lmc import calculate_local_matrices
        from src.soe.soe import simulate

        calculate_local_matrices(5, grid)
        times, temperatures = simulate(grid)

        config.logger.debug(f"Time        Min temp    Max temp")
        for i in range(len(temperatures)):
            config.logger.debug(f"{(times[i]):<12}{round(np.min(temperatures[i]), 3):<12}{round(np.max(temperatures[i]), 3):<12}")

        # Generate .vtk files for ParaView
        generate_vtk_files(output_dir_path, grid, temperatures)
    except Exception:
        config.logger.error(f"Script execution failed due to an exception. Check log file for details.", exc_info=True)

if __name__ == "__main__":
    run()