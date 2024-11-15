import argparse
import os
import pathlib
import time

import numpy as np

from src.helpers import create_or_clear_directory, init_logging
from src import config
from src.grid import Grid
from src.local_matrices_calculation import LocalMatricesCalculation
from src.local_matrices_calculation_gpu import calculate_local_matrices
from src.vtk_generator import generate_vtk_files

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
        mesh_filepath, data_filepath, config.force_cpu = parse_arguments()
        # In output directory, create a subdirectory with the name of the input mesh file
        output_dir_path = create_or_clear_directory(os.path.join(config.output_path,
                                                                 os.path.basename(mesh_filepath).split(".")[0]))
        config.logger = init_logging(logger_name=config.MAIN_LOGGER_NAME, log_dirpath=output_dir_path)
        # Check the types of given input files and creates a grid object
        mesh_filepath_ext: str = pathlib.Path(mesh_filepath).suffix
        data_filepath_ext: str = pathlib.Path(data_filepath).suffix if data_filepath is not None else None
        if mesh_filepath_ext == ".msh":
            if data_filepath_ext != ".json":
                config.logger.error("Data file path in .json format is required for .msh files.")
                raise Exception
            grid = Grid.create_from_msh_and_json(mesh_filepath, data_filepath)
        elif mesh_filepath_ext == ".txt":
            if data_filepath_ext is not None:
                config.logger.warning(f"Data file path is not required for .txt files. Data from {data_filepath} will be ignored.")
            grid = Grid.create_from_txt(mesh_filepath)
        # Calculate matrices stored in elements
        start: float = time.time() # Start measuring time
        # LocalMatricesCalculation.calculate(2, grid)
        calculate_local_matrices(2, grid)
        end: float = time.time() # Stop measuring time
        config.logger.info(f"Local matrices calculated in {end-start} seconds.")
        # Simulate temperatures in the grid for given timeframes
        # from src.system_of_equations import simulate
        # times, temperatures = simulate(grid)
        # config.logger.debug(f"Time        Min temp    Max temp")
        # for i in range(len(temperatures)):
        #     config.logger.debug(f"{(times[i]):<12}{round(np.min(temperatures[i]), 3):<12}{round(np.max(temperatures[i]), 3):<12}")
        # # Generate .vtk files for ParaView
        # generate_vtk_files(output_dir_path, grid, temperatures)
    except Exception:
        config.logger.error(f"Script execution failed due to an exception. Check log file for details.", exc_info=True)

if __name__ == "__main__":
    run()