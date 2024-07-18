import time
import common
from assertion_data import *
from common import *
from grid import Grid
from jinja2 import Template
from local_matrices_calculation import LocalMatricesCalculation
from system_of_equations import simulate
import numpy as np
from test import *

def get_input_filepath() -> tuple[str]:
    """
    Gets path to grid file from user.
    """
    input_filepath: str = os.path.join(input_path, "example_grid.txt")
    output_dir_path: str = create_or_clear_directory(input_filepath)
    return input_filepath, output_dir_path

def generate_vtk_files(output_dir_path: str, grid: Grid, temperatures: list[np.ndarray]) -> None:
    """
    Creates files for simulation in ParaView environment.
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
    try:
        common.logger = init_logging()
        # sanity_check()
        input_filepath, output_dir_path = get_input_filepath()
        grid = Grid.create_from_file(input_filepath)

        start: float = time.time() # Start measuring time
        LocalMatricesCalculation.calculate(5, grid)
        temperatures: list[float] = simulate(grid)
        common.logger.debug(temperatures)
        end: float = time.time() # Stop measuring time

        common.logger.info(f"Calculated in {end-start} seconds.")
        generate_vtk_files(output_dir_path, grid, temperatures)
    except HandledException:
        common.logger.info("Script execution failed due to an exception.")
    except Exception as e:
        common.logger.error(f"Script execution failed due to an unexpected exception.", exc_info=True)

if __name__ == "__main__":
    run()