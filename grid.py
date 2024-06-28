import common
from common import *
from universal_element import *
import numpy as np

class GlobalData:
    """
    Stores general information like simulation time, conductivity, initial temperature, density etc.
    """
    def __init__(self, global_data_dict: dict):
        self.simulation_time: float = global_data_dict["SimulationTime"]
        self.simulation_step_time: float = global_data_dict["SimulationStepTime"]
        self.conductivity: float = global_data_dict["Conductivity"]
        self.alfa: float = global_data_dict["Alfa"]
        self.tot: float = global_data_dict["Tot"]
        self.initial_temp: float = global_data_dict["InitialTemp"]
        self.density: float = global_data_dict["Density"]
        self.specific_heat: float = global_data_dict["SpecificHeat"]
        self.nodes_number: int = global_data_dict["Nodesnumber"]
        self.elements_number: int = global_data_dict["Elementsnumber"]

    def print(self) -> None:
        common.main_logger.debug(f"Simulation time: \t{self.simulation_time}")
        common.main_logger.debug(f"Simulation step time: \t{self.simulation_step_time}")
        common.main_logger.debug(f"Conductivity: \t\t{self.conductivity}")
        common.main_logger.debug(f"Alfa: \t\t\t{self.alfa}")
        common.main_logger.debug(f"Tot: \t\t\t{self.tot}")
        common.main_logger.debug(f"Initial temp: \t\t{self.initial_temp}")
        common.main_logger.debug(f"Density: \t\t{self.density}")
        common.main_logger.debug(f"Specific heat: \t\t{self.specific_heat}")
        common.main_logger.debug(f"Nodes number: \t\t{self.nodes_number}")
        common.main_logger.debug(f"Elements number: \t{self.elements_number}")

class Node:
    """
    Stores information about a single node of the grid.

    id:      Node"s ID
    x:       x coord
    y:       y coord
    BC:      border condition (0 or 1)
    """
    def __init__(self, id: int, x: float, y: float):
        self.id: int = id
        self.x: float = x
        self.y: float = y
        self.BC: float = 0

    def print(self) -> None:
        common.main_logger.debug(f"Node {self.id}: \t({self.x}, {self.y})")

class Element:
    """
    Stores information about a single 4-node element of the grid.

    id:         Element"s ID
    IDs:        IDs od nodes belonging to the element
    H:          H matrix for the element (4x4)
    Hbc:        Hbc matrix for the element (4x4)
    P:          P vector for the element (4X1)
    C:          C matrix for the element (4x4)
    """
    def __init__(self, id: int, node_ids: np.ndarray[int]):
        self.id: int = id
        self.node_ids: np.ndarray[int] = node_ids
        self.H: np.ndarray = np.empty((4, 4), dtype=float)
        self.Hbc: np.ndarray = np.empty((4, 4), dtype=float)
        self.P: np.ndarray = np.empty((4, 1), dtype=float)
        self.C: np.ndarray = np.empty((4, 4), dtype=float)

    def print(self) -> None:
        common.main_logger.debug(f"Element {self.id}: \t{self.node_ids}")

class Grid:
    """
    Stores information allowing to recreate the grid.

    globalData:     i.e. simulation time, conductivity, initial temperature, density etc.
    nodes:          list of nodes in the grid
    elements:       list of elements in the grid
    BC:             list of nodes with border condition
    """
    def __init__(self, global_data: GlobalData = None, elements: np.ndarray[Element] = None, nodes: np.ndarray[Node] = None):
        self.global_data: GlobalData = global_data
        self.nodes: np.ndarray[Node] = nodes
        self.elements: np.ndarray[Element] = elements

    @classmethod
    def create_from_file(cls, input_filepath: str):
        def _read_global_data(input: str) -> GlobalData:
            """
            Reads global data from input file.
            """
            global_data_dict = {}
            for i in range (0, 8):
                line = input[i]
                line = line.split(" ")
                for i in range (0, 2):
                    line[i] = line[i].strip()
                global_data_dict[line[0]] = int(line[1])
            for i in range (8, 10):
                line = input[i]
                line = line.split(" ")
                for i in range (0, 3):
                    line[i] = line[i].strip()
                line[0] = line[0] + line[1]
                global_data_dict[line[0]] = int(line[2])
            return GlobalData(global_data_dict)

        def _read_nodes(input: str, nodes_start_line: int, nodes_number: int) -> np.ndarray[Node]:
            """
            Reads nodes from input file.
            """
            node_list = np.empty(nodes_number, dtype=Node)
            for i in range (nodes_start_line, nodes_start_line + nodes_number):
                line = input[i].split(", ")
                node_list[i-nodes_start_line] = Node(int(line[0]), float(line[1]), float(line[2]))
            return node_list

        def _read_elements(input: str, elements_start_line: int, elements_number: int) -> np.ndarray[Element]:
            """
            Reads elements data from input file.
            """
            elements = np.empty(elements_number, dtype=Element)
            node_ids = []
            for i in range (elements_start_line, elements_start_line + elements_number):
                line = input[i].split(", ")
                for j in range (1, len(line)):
                    node_ids.append(int(line[j]))
                elements[i-elements_start_line] = Element(int(line[0]), node_ids)
                node_ids = []
            return elements

        def _read_bc(input: str, bc_start_line: int) -> list[int]:
            """
            Reads nodes with border condition from input file.
            """
            node_ids = []
            line = input[bc_start_line]
            line = line.split(", ")
            for i in range(0, len(line)):
                node_ids.append(int(line[i]))
            return node_ids

        def _add_bc_to_node(nodes: np.ndarray[Node], BC: list[int]) -> None:
            """
            Adds border condition to node.
            """
            for node_id in BC:
                nodes[node_id - 1].BC = 1

        try:
            common.main_logger.info(f"Creating a grid from input file: '{os.path.basename(input_filepath)}'.")
            f = open(input_filepath, "r")
            file_content: str = f.readlines()
            global_data: GlobalData = _read_global_data(file_content)
            nodes_start_line: int = 11
            elements_start_line: int = nodes_start_line + global_data.nodes_number + 1
            bc_start_line: int = elements_start_line + global_data.elements_number + 1

            nodes: np.ndarray[Node] = _read_nodes(file_content, nodes_start_line, global_data.nodes_number)
            elements: np.ndarray[Element] = _read_elements(file_content, elements_start_line, global_data.elements_number)
            BC: list[Node] = _read_bc(file_content, bc_start_line)
            _add_bc_to_node(nodes, BC)
            f.close()
            return cls(global_data, elements, nodes)
        except FileNotFoundError:
            common.main_logger.error(f"Cannnot find an input file: '{input_filepath}'", exc_info=True)
            raise HandledException
        except Exception:
            common.main_logger.error(f"Unknown exception while creating a grid from input file.", exc_info=True)
            raise HandledException

    def print(self) -> None:
        self.global_data.print()
        common.main_logger.debug("\nNodes:")
        for node in self.nodes:
            node.print()
        common.main_logger.debug("\nElements:")
        for element in self.elements:
            element.print()
        common.main_logger.debug(f"\nBC:\n{self.BC}\n")