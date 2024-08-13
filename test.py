import unittest
import common
from assertion_data import *
from common import *
from grid import Grid
from local_matrices_calculation import LocalMatricesCalculation
from system_of_equations import simulate
import numpy as np

def assert_output_temperatures(input_filename: str, n: int, correct_temperatures: list[np.ndarray[float]]):
    input_filename: str = os.path.join(test_path, input_filename)
    grid = Grid.create_from_file(input_filename)
    LocalMatricesCalculation.calculate(n, grid)
    temperatures: list[float] = simulate(grid)
    assert np.allclose(temperatures, correct_temperatures, rtol=1e-08), f"Temperatures calculated for '{input_filename}' are different than expected."

class TestOutputTemperatures(unittest.TestCase):
    def test_init(self):
        common.logger = init_logging(TEST_LOGGER_NAME)

    def test_output_temperatures_grid1_n1(self):
        assert_output_temperatures("test1_grid.txt", 1, test1_temperatures_n_1)

    def test_output_temperatures_grid1_n2(self):
        assert_output_temperatures("test1_grid.txt", 1, test1_temperatures_n_1)

    def test_output_temperatures_grid1_n3(self):
        assert_output_temperatures("test1_grid.txt", 3, test1_temperatures_n_3)

    def test_output_temperatures_grid1_n4(self):
        assert_output_temperatures("test1_grid.txt", 4, test1_temperatures_n_4)

    def test_output_temperatures_grid1_n5(self):
        assert_output_temperatures("test1_grid.txt", 5, test1_temperatures_n_5)

    def test_output_temperatures_grid2_n1(self):
        assert_output_temperatures("test2_grid.txt", 1, test2_temperatures_n_1)

    def test_output_temperatures_grid2_n2(self):
        assert_output_temperatures("test2_grid.txt", 2, test2_temperatures_n_2)

    def test_output_temperatures_grid2_n3(self):
        assert_output_temperatures("test2_grid.txt", 3, test2_temperatures_n_3)

    def test_output_temperatures_grid2_n4(self):
        assert_output_temperatures("test2_grid.txt", 4, test2_temperatures_n_4)

    def test_output_temperatures_grid2_n5(self):
        assert_output_temperatures("test2_grid.txt", 5, test2_temperatures_n_5)

    def test_output_temperatures_grid3_n5(self):
        assert_output_temperatures("test3_grid.txt", 5, test3_temperatures_n_5)

    # def test_negative_output_temperatures_grid1_n4(self):
    #     with self.assertRaises(AssertionError):
    #         assert_output_temperatures("test1_grid.txt", 4, test1_temperatures_n_1)

if __name__ == '__main__':
    unittest.main()