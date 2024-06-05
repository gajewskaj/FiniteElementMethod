import time
import common
from common import *
from grid import Grid, Element, Node
from jinja2 import Template
from local_matrices_calculation import LocalMatricesCalculation
from system_of_equations import SystemOfEquations
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

def get_input_fileath() -> str:
    """
    Gets path to grid file from user.
    """
    # Tk().withdraw()
    # inputFilePath: str = askopenfilename()
    input_fileath: str = os.path.join(script_path, 'Data', 'example_grid.txt')
    return input_fileath

def simulate(grid: Grid) -> list[np.ndarray]:
    """
    Returns temeratures in element nodes for all time steps.
    """
    temperatures: list[np.ndarray] = []
    soe = SystemOfEquations(grid)
    tau0: int = 0
    tauk: float = grid.global_data.simulation_time
    step: float = grid.global_data.simulation_step_time
    common.main_logger.info(f'Time        Min temp    Max temp')
    while tau0 < tauk:
        result: np.ndarray = soe.solve()
        temperatures.append(result)
        common.main_logger.info(f'{(soe.dtau):<12}{round(min(result)[0], 3):<12}{round(max(result)[0], 3):<12}')
        tau0+=step
    return temperatures

def generate_vtk_files(input_filename: str, grid: Grid, temperatures: list[np.ndarray]) -> None:
    """
    Creates files for simulation in ParaView environment.
    """
    num_files: int = len(temperatures)
    element_nodes_number: int = []
    for element in grid.elements:
        element_nodes_number.append(len(element.node_ids))

    data: dict = {}
    data['nodesNumber'] = grid.global_data.nodes_number
    data['nodes'] = grid.nodes
    data['elementsNumber'] = grid.global_data.elements_number
    data['elements'] = grid.elements
    data['elementNodesNumber'] = element_nodes_number
    data['sumOfElementsData'] = grid.global_data.elements_number + sum(element_nodes_number)

    destination_dir: str = create_or_clear_directory(input_filename)
    template: Template = initialize_jinja_environment('temperatures.vtk.jinja')
    for i in range(0, num_files):
        data['temperatures'] = temperatures[i]
        filename: str = f'frame{i+1}.vtk'
        generate_file(data, template, destination_dir, filename)
    common.main_logger.info(f"Output files generated in '{destination_dir}'.")

def run() -> None:
    """
    Runs all the necessary functions to calculate max and min temperature of the element in time.
    """
    try:
        common.main_logger = init_logging(os.path.join(output_path, 'log.log'))
        inputFilePath: str = get_input_fileath()
        start: float = time.time() # Start measuring time after user input
        grid = Grid.create_from_file(inputFilePath)
        LocalMatricesCalculation.calculate(5, grid)
        temperatures: list[float] = simulate(grid)
        end: float = time.time() # Stop measuring time after finishing the calculations
        common.main_logger.info(f'Calculated in {end-start} seconds.')
        generate_vtk_files(inputFilePath, grid, temperatures)
    except FiniteElementMethodException as e:
        common.main_logger.error(e)
    except Exception as e:
        common.main_logger.error(f'Catched unexpected exception:\n{e}')

if __name__ == '__main__':
    run()