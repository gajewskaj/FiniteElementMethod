import os
import pathlib

import numpy as np
import argparse
from jinja2 import Template

from src import common
from src.common import *
from src.grid import Grid
from src.local_matrices_calculation import LocalMatricesCalculation

def parse_arguments() -> Settings:
    """
    Parses command line arguments.

    Returns:
        Settings: Parsed arguments as a Settings object.
    """
    parser = argparse.ArgumentParser(description="Finite Element Method Simulation")
    parser.add_argument('--mesh', type=str, default=os.path.join(test_path, "test1_grid.txt"),
                        help='Path to the input grid file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to the input data file')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force the simulation to run on CPU')
    args = parser.parse_args()
    return Settings(args.mesh, args.data, args.force_cpu)

def generate_vtk_files(output_dir_path: str, grid: Grid, temperatures: list[np.ndarray]) -> None:
    """
    Creates files for simulation in ParaView environment.

    Args:
        - output_dir_path (str): Path to the output directory.
        - grid (Grid): Grid object containing simulation data.
        - temperatures (list[np.ndarray]): List of temperature arrays for each time step.
    """
    num_files: int = len(temperatures)
    element_nodes_number: int = []
    for element in grid.elements:
        element_nodes_number.append(len(element.node_ids))

    data: dict = {}
    data["nodes_number"] = grid.global_data.nodes_number
    data["nodes"] = grid.nodes
    data["elements_number"] = grid.global_data.elements_number
    data["elements"] = grid.elements
    data["element_nodes_number"] = element_nodes_number
    data["sum_elements_data"] = grid.global_data.elements_number + sum(element_nodes_number)

    template: Template = initialize_jinja_environment("temperatures.vtk.jinja")
    for i in range(0, num_files):
        data["temperatures"] = temperatures[i]
        filename: str = f"frame{i+1}.vtk"
        generate_file(data, template, output_dir_path, filename)
    common.logger.info(f"Output files generated in '{output_dir_path}'.")

def run() -> None:
    """
    Runs all the necessary functions to calculate max and min temperature of the element in time.
    """
    global settings
    try:
        settings = parse_arguments()
        common.settings = settings
        create_or_clear_directory(output_path)
        common.logger = init_logging()
        output_dir_path = create_or_clear_directory(os.path.join(output_path,
                                                                 os.path.basename(settings.mesh_filepath).split(".")[0]))
        mesh_filepath_ext: str = pathlib.Path(settings.mesh_filepath).suffix
        data_filepath_ext: str = pathlib.Path(settings.data_filepath).suffix if settings.data_filepath is not None else None
        if mesh_filepath_ext == ".msh":
            if data_filepath_ext != ".json":
                common.logger.error("Data file path in .json format is required for .msh files.")
                raise Exception
            grid = Grid.create_from_msh_and_json(settings.mesh_filepath, settings.data_filepath)
        elif mesh_filepath_ext == ".txt":
            if data_filepath_ext is not None:
                common.logger.warning(f"Data file path is not required for .txt files. Data from {settings.data_filepath} will be ignored.")
            grid = Grid.create_from_txt(settings.mesh_filepath)
        LocalMatricesCalculation.calculate(5, grid)
        from src.system_of_equations import simulate
        temperatures: list[float] = simulate(grid)
        common.logger.debug(temperatures)
        generate_vtk_files(output_dir_path, grid, temperatures)
    except Exception:
        common.logger.error(f"Script execution failed due to an exception. Check log file for details.", exc_info=True)

if __name__ == "__main__":
    run()