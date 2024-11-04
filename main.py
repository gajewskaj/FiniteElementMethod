import os
import numpy as np
import argparse
from jinja2 import Template
import src.common as common
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
    parser.add_argument('--input', type=str, default=os.path.join(input_path, "example_grid.txt"),
                        help='Path to the input grid file')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force the simulation to run on CPU')
    args = parser.parse_args()
    return Settings(args.input, args.force_cpu)

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
    data["nodesNumber"] = grid.global_data.nodes_number
    data["nodes"] = grid.nodes
    data["elementsNumber"] = grid.global_data.elements_number
    data["elements"] = grid.elements
    data["elementNodesNumber"] = element_nodes_number
    data["sumOfElementsData"] = grid.global_data.elements_number + sum(element_nodes_number)

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
                                                                 os.path.basename(settings.input_filepath).split(".")[0]))
        grid = Grid.create_from_file(settings.input_filepath)
        LocalMatricesCalculation.calculate(5, grid)
        from src.system_of_equations import simulate
        temperatures: list[float] = simulate(grid)
        common.logger.debug(temperatures)
        generate_vtk_files(output_dir_path, grid, temperatures)
    except HandledException:
        common.logger.info("Script execution failed due to an exception.")
    except Exception as e:
        common.logger.error(f"Script execution failed due to an unexpected exception. Check log file for details.", exc_info=True)

if __name__ == "__main__":
    run()