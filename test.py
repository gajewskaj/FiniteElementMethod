import unittest
import common
from assertion_data import *
from common import *
from grid import GlobalData, Node, Element, Grid
from local_matrices_calculation import LocalMatricesCalculation
from system_of_equations import simulate
import numpy as np

def get_global_data(input_filename: str) -> GlobalData:
    input_filename: str = os.path.join(test_path, input_filename)
    grid = Grid.create_from_file(input_filename)
    global_data: GlobalData = grid.global_data
    return global_data

def get_nodes(input_filename: str) -> np.ndarray[Node]:
    input_filename: str = os.path.join(test_path, input_filename)
    grid = Grid.create_from_file(input_filename)
    nodes: np.ndarray[Node] = grid.nodes
    return nodes

def get_elements(input_filename: str) -> np.ndarray[Element]:
    input_filename: str = os.path.join(test_path, input_filename)
    grid = Grid.create_from_file(input_filename)
    elements: np.ndarray[Element] = grid.elements
    return elements

class TestInputData(unittest.TestCase):
    def test1_global_data(self):
        global_data_dict: dict[str, int | float] = {
            "SimulationTime": 500,
            "SimulationStepTime": 50,
            "Conductivity": 25,
            "Alfa": 300,
            "Tot": 1200,
            "InitialTemp": 100,
            "Density": 7800,
            "SpecificHeat": 700,
            "Nodesnumber": 16,
            "Elementsnumber": 9
        }
        correct_global_data: GlobalData = GlobalData(global_data_dict)
        read_global_data: GlobalData = get_global_data("test1_grid.txt")
        self.assertEqual(correct_global_data, read_global_data, "Global data was read incorrectly from the input file.")

    def test2_global_data(self):
        global_data_dict: dict[str, int | float] = {
            "SimulationTime": 20,
            "SimulationStepTime": 1,
            "Conductivity": 25,
            "Alfa": 300,
            "Tot": 1200,
            "InitialTemp": 100,
            "Density": 7800,
            "SpecificHeat": 700,
            "Nodesnumber": 961,
            "Elementsnumber": 900
        }
        correct_global_data: GlobalData = GlobalData(global_data_dict)
        read_global_data: GlobalData = get_global_data("test3_grid.txt")
        self.assertEqual(correct_global_data, read_global_data, \
                         "Global data was read incorrectly from the input file.")

    def test3_nodes(self):
        correct_nodes: list[Node] = [
            Node(1, 0.100000001, 0.00499999989, 1),
            Node(2, 0.0666666701, 0.00499999989, 1),
            Node(3, 0.0333333351, 0.00499999989, 1),
            Node(11, 0.0333333351, -0.0616666675),
            Node(12, 0, -0.0616666675, 1),
            Node(13, 0.100000001, -0.0949999988, 1)
        ]
        read_nodes: list[Node] = get_nodes("test1_grid.txt")
        read_nodes = [
            read_nodes[0],
            read_nodes[1],
            read_nodes[2],
            read_nodes[10],
            read_nodes[11],
            read_nodes[12]
        ]
        self.assertListEqual(correct_nodes, read_nodes, \
                             "Nodes were read incorrectly from the input file.")

    def test4_nodes(self):
        correct_nodes: list[Node] = [
            Node(1, 0.100000001, 0.00499999989, 1),
            Node(2, 0.0546918176, 0.00499999989, 1),
            Node(3, 0.0226540919, 0.00499999989, 1),
            Node(11, 0.0376100652, -0.0573899336),
            Node(12, 0, -0.0496918149, 1),
            Node(13, 0.100000001, -0.0949999988, 1)
        ]
        read_nodes: list[Node] = get_nodes("test2_grid.txt")
        read_nodes = [
            read_nodes[0],
            read_nodes[1],
            read_nodes[2],
            read_nodes[10],
            read_nodes[11],
            read_nodes[12]
        ]
        self.assertListEqual(correct_nodes, read_nodes, \
                             "Nodes were read incorrectly from the input file.")

    def test5_nodes(self):
        correct_nodes: list[Node] = [
            Node(1, 0.100000001, 0.00499999989, 1),
            Node(2, 0.0966666639, 0.00499999989, 1),
            Node(3, 0.0933333337, 0.00499999989, 1),
            Node(92, 0.00333333341, -0.00166666671),
            Node(93, 0, -0.00166666671, 1),
            Node(94, 0.100000001, -0.00499999989, 1)
        ]
        read_nodes: list[Node] = get_nodes("test3_grid.txt")
        read_nodes = [
            read_nodes[0],
            read_nodes[1],
            read_nodes[2],
            read_nodes[91],
            read_nodes[92],
            read_nodes[93]
        ]
        self.assertListEqual(correct_nodes, read_nodes, \
                             "Nodes were read incorrectly from the input file.")

    def test6_elements(self):
        correct_elements: list[Element] = [
            Element(1, [1, 2, 6, 5]),
            Element(2, [2, 3, 7, 6]),
            Element(3, [3, 4, 8, 7]),
            Element(6, [7, 8, 12, 11]),
            Element(7, [9, 10, 14, 13]),
            Element(8, [10, 11, 15, 14])
        ]
        read_elements: list[Element] = get_elements("test1_grid.txt")
        read_elements = [
            read_elements[0],
            read_elements[1],
            read_elements[2],
            read_elements[5],
            read_elements[6],
            read_elements[7]
        ]
        self.assertListEqual(correct_elements, read_elements, \
                             "Elements were read incorrectly from the input file.")

    def test7_elements(self):
        correct_elements: list[Element] = [
            Element(1, [1, 2, 33, 32]),
            Element(2, [2, 3, 34, 33]),
            Element(3, [3, 4, 35, 34]),
            Element(126, [130, 131, 162, 161]),
            Element(127, [131, 132, 163, 162]),
            Element(128, [132, 133, 164, 163])
        ]
        read_elements: list[Element] = get_elements("test3_grid.txt")
        read_elements = [
            read_elements[0],
            read_elements[1],
            read_elements[2],
            read_elements[125],
            read_elements[126],
            read_elements[127]
        ]
        self.assertListEqual(correct_elements, read_elements, \
                             "Elements were read incorrectly from the input file.")

def assert_output_temperatures(input_filename: str, n: int, correct_temperatures: list[np.ndarray[float]]) -> None:
    input_filename: str = os.path.join(test_path, input_filename)
    grid = Grid.create_from_file(input_filename)
    LocalMatricesCalculation.calculate(n, grid)
    temperatures: list[float] = simulate(grid)
    assert np.allclose(temperatures, correct_temperatures, rtol=1e-08), f"Temperatures calculated for '{input_filename}' are different than expected."

class TestOutputTemperatures(unittest.TestCase):
    def test8_output_temperatures_grid1_n1(self):
        assert_output_temperatures("test1_grid.txt", 1, test1_temperatures_n_1)

    def test9_output_temperatures_grid1_n2(self):
        assert_output_temperatures("test1_grid.txt", 2, test1_temperatures_n_2)

    def test10_output_temperatures_grid1_n3(self):
        assert_output_temperatures("test1_grid.txt", 3, test1_temperatures_n_3)

    def test11_output_temperatures_grid1_n4(self):
        assert_output_temperatures("test1_grid.txt", 4, test1_temperatures_n_4)

    def test12_output_temperatures_grid1_n5(self):
        assert_output_temperatures("test1_grid.txt", 5, test1_temperatures_n_5)

    def test13_output_temperatures_grid2_n1(self):
        assert_output_temperatures("test2_grid.txt", 1, test2_temperatures_n_1)

    def test14_output_temperatures_grid2_n2(self):
        assert_output_temperatures("test2_grid.txt", 2, test2_temperatures_n_2)

    def test15_output_temperatures_grid2_n3(self):
        assert_output_temperatures("test2_grid.txt", 3, test2_temperatures_n_3)

    def test16_output_temperatures_grid2_n4(self):
        assert_output_temperatures("test2_grid.txt", 4, test2_temperatures_n_4)

    def test17_output_temperatures_grid2_n5(self):
        assert_output_temperatures("test2_grid.txt", 5, test2_temperatures_n_5)

    def test18_output_temperatures_grid3_n5(self):
        assert_output_temperatures("test3_grid.txt", 5, test3_temperatures_n_5)

if __name__ == '__main__':
    common.logger = init_logging(TEST_LOGGER_NAME)
    unittest.main()