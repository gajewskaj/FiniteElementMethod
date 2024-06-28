import time
import common
from assertion_data import *
from common import *
from grid import Grid
from jinja2 import Template
from local_matrices_calculation import LocalMatricesCalculation
from system_of_equations import simulate
import numpy as np

def sanity_check() -> None:
    """
    Performs basing unit testing.
    """
    try:
        common.main_logger.info("Performing a sanity check.")
        temperatures_calculated_correctly("test1_grid.txt", 1, test1_temperatures_n_1)
        temperatures_calculated_correctly("test1_grid.txt", 2, test1_temperatures_n_2)
        temperatures_calculated_correctly("test1_grid.txt", 3, test1_temperatures_n_3)
        temperatures_calculated_correctly("test1_grid.txt", 4, test1_temperatures_n_4)
        temperatures_calculated_correctly("test1_grid.txt", 5, test1_temperatures_n_5)
        temperatures_calculated_correctly("test2_grid.txt", 1, test2_temperatures_n_1)
        temperatures_calculated_correctly("test2_grid.txt", 2, test2_temperatures_n_2)
        temperatures_calculated_correctly("test2_grid.txt", 3, test2_temperatures_n_3)
        temperatures_calculated_correctly("test2_grid.txt", 4, test2_temperatures_n_4)
        temperatures_calculated_correctly("test2_grid.txt", 5, test2_temperatures_n_5)
        temperatures_calculated_correctly("test3_grid.txt", 5, test3_temperatures_n_5)
    except AssertionError as e:
        common.main_logger.error(f"Sanity check failed. {e}")
        raise HandledException
    else:
        common.main_logger.info("Sanity check performed successfully!")

def temperatures_calculated_correctly(input_filename: str, n: int, correct_temperatures: list[np.ndarray[float]]) -> None:
    input_filename: str = os.path.join(test_path, input_filename)
    grid = Grid.create_from_file(input_filename)
    LocalMatricesCalculation.calculate(n, grid)
    temperatures: list[float] = simulate(grid)
    assert np.allclose(temperatures, correct_temperatures), f"Temperatures calculated for '{input_filename}' are different than expected."
