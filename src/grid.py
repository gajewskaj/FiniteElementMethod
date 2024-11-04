import numpy as np
from . import common
from .common import *
from .universal_element import *

class GlobalData:
    """
    Stores general information like simulation time, conductivity, initial temperature, density, etc.

    Attributes:
        simulation_time (float): Total simulation time.
        simulation_step_time (float): Time step for the simulation.
        conductivity (float): Thermal conductivity.
        alfa (float): Heat transfer coefficient.
        tot (float): Ambient temperature.
        initial_temp (float): Initial temperature.
        density (float): Material density.
        specific_heat (float): Specific heat capacity.
        nodes_number (int): Number of nodes in the grid.
        elements_number (int): Number of elements in the grid.
    """
    def __init__(self, global_data_dict: dict[str, int | float]):
        """
        Initializes GlobalData with values from a dictionary.

        Args:
            global_data_dict (dict): Dictionary containing global data.
        """
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

    def __eq__(self, other: 'GlobalData') -> bool:
        """
        Checks if two GlobalData instances are equal.

        Args:
            other (GlobalData): Another GlobalData instance.

        Returns:
            bool: True if equal, False otherwise.
        """
        result: bool = isclose(self.simulation_time, other.simulation_time) and \
                       isclose(self.simulation_step_time, other.simulation_step_time) and \
                       isclose(self.conductivity, other.conductivity) and \
                       isclose(self.alfa, other.alfa) and \
                       isclose(self.tot, other.tot) and \
                       isclose(self.initial_temp, other.initial_temp) and \
                       isclose(self.density, other.density) and \
                       isclose(self.specific_heat, other.specific_heat) and \
                       self.nodes_number == other.nodes_number and \
                       self.elements_number == other.elements_number
        return result

    def print(self) -> None:
        """
        Prints the global data.
        """
        common.logger.debug(f"Simulation time: \t{self.simulation_time}")
        common.logger.debug(f"Simulation step time: \t{self.simulation_step_time}")
        common.logger.debug(f"Conductivity: \t\t{self.conductivity}")
        common.logger.debug(f"Alfa: \t\t\t{self.alfa}")
        common.logger.debug(f"Tot: \t\t\t{self.tot}")
        common.logger.debug(f"Initial temp: \t\t{self.initial_temp}")
        common.logger.debug(f"Density: \t\t{self.density}")
        common.logger.debug(f"Specific heat: \t\t{self.specific_heat}")
        common.logger.debug(f"Nodes number: \t\t{self.nodes_number}")
        common.logger.debug(f"Elements number: \t{self.elements_number}")

class Node:
    """
    Stores information about a single node of the grid.

    Attributes:
        id (int): Node's ID.
        x (float): x coordinate.
        y (float): y coordinate.
        BC (float): Border condition (0 or 1).
    """
    def __init__(self, id: int, x: float, y: float, BC: float = 0):
        """
        Initializes a Node instance.

        Args:
            id (int): Node's ID.
            x (float): x coordinate.
            y (float): y coordinate.
            BC (float, optional): Border condition. Defaults to 0.
        """
        self.id: int = id
        self.x: float = x
        self.y: float = y
        self.BC: float = BC

    def __eq__(self, other: 'Node') -> bool:
        """
        Checks if two Node instances are equal.

        Args:
            other (Node): Another Node instance.

        Returns:
            bool: True if equal, False otherwise.
        """
        result: bool = self.id == other.id and \
                       isclose(self.x, other.x) and \
                       isclose(self.y, other.y) and \
                       isclose(self.BC, other.BC)
        return result

    def print(self) -> None:
        """
        Prints the node data.
        """
        common.logger.debug(f"Node {self.id}: \t({self.x}, {self.y})")

class Element:
    """
    Stores information about a single 4-node element of the grid.

    Attributes:
        id (int): Element's ID.
        node_ids (np.ndarray[int]): IDs of nodes belonging to the element.
        H (np.ndarray): H matrix for the element (4x4).
        Hbc (np.ndarray): Hbc matrix for the element (4x4).
        P (np.ndarray): P vector for the element (4x1).
        C (np.ndarray): C matrix for the element (4x4).
    """
    def __init__(self, id: int, node_ids: np.ndarray[int]):
        """
        Initializes an Element instance.

        Args:
            id (int): Element's ID.
            node_ids (np.ndarray[int]): IDs of nodes belonging to the element.
        """
        self.id: int = id
        self.node_ids: np.ndarray[int] = node_ids
        self.H: np.ndarray = np.zeros((4, 4), dtype=float)
        self.Hbc: np.ndarray = np.zeros((4, 4), dtype=float)
        self.P: np.ndarray = np.zeros((4, 1), dtype=float)
        self.C: np.ndarray = np.zeros((4, 4), dtype=float)

    def __eq__(self, other: 'Element') -> bool:
        """
        Checks if two Element instances are equal.

        Args:
            other (Element): Another Element instance.

        Returns:
            bool: True if equal, False otherwise.
        """
        result: bool = self.id == other.id and \
                       self.node_ids == other.node_ids and \
                       np.allclose(self.H, other.H) and \
                       np.allclose(self.Hbc, other.Hbc) and \
                       np.allclose(self.P, other.P) and \
                       np.allclose(self.C, other.C)
        return result

    def print(self) -> None:
        """
        Prints the element data.
        """
        common.logger.debug(f"Element {self.id}: \t{self.node_ids}")

class Grid:
    """
    Stores information allowing to recreate the grid.

    Attributes:
        global_data (GlobalData): General simulation data.
        nodes (np.ndarray[Node]): List of nodes in the grid.
        elements (np.ndarray[Element]): List of elements in the grid.
        BC (list[int]): List of nodes with border condition.
    """
    def __init__(self, global_data: GlobalData = None, elements: np.ndarray[Element] = None, nodes: np.ndarray[Node] = None):
        """
        Initializes a Grid instance.

        Args:
            global_data (GlobalData, optional): General simulation data. Defaults to None.
            elements (np.ndarray[Element], optional): List of elements in the grid. Defaults to None.
            nodes (np.ndarray[Node], optional): List of nodes in the grid. Defaults to None.
        """
        self.global_data: GlobalData = global_data
        self.nodes: np.ndarray[Node] = nodes
        self.elements: np.ndarray[Element] = elements

    @classmethod
    def create_from_file(cls, input_filepath: str):
        """
        Creates a Grid instance from an input file.

        Args:
            input_filepath (str): Path to the input file.

        Returns:
            Grid: A Grid instance.

        Raises:
            HandledException: If the input file is not found or an unknown error occurs.
        """
        def _read_global_data(input: str) -> GlobalData:
            """
            Reads global data from input file.

            Args:
                input (str): Content of the input file.

            Returns:
                GlobalData: An instance of GlobalData.
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

            Args:
                input (str): Content of the input file.
                nodes_start_line (int): Line number where nodes data starts.
                nodes_number (int): Number of nodes.

            Returns:
                np.ndarray[Node]: Array of Node instances.
            """
            node_list = np.empty(nodes_number, dtype=Node)
            for i in range (nodes_start_line, nodes_start_line + nodes_number):
                line = input[i].split(", ")
                node_list[i-nodes_start_line] = Node(int(line[0]), float(line[1]), float(line[2]))
            return node_list

        def _read_elements(input: str, elements_start_line: int, elements_number: int) -> np.ndarray[Element]:
            """
            Reads elements data from input file.

            Args:
                input (str): Content of the input file.
                elements_start_line (int): Line number where elements data starts.
                elements_number (int): Number of elements.

            Returns:
                np.ndarray[Element]: Array of Element instances.
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

            Args:
                input (str): Content of the input file.
                bc_start_line (int): Line number where border condition data starts.

            Returns:
                list[int]: List of node IDs with border condition.
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

            Args:
                nodes (np.ndarray[Node]): Array of Node instances.
                BC (list[int]): List of node IDs with border condition.
            """
            for node_id in BC:
                nodes[node_id - 1].BC = 1

        try:
            common.logger.info(f"Creating a grid from input file: '{os.path.basename(input_filepath)}'.")
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
            common.logger.error(f"Cannnot find an input file: '{input_filepath}'", exc_info=True)
            raise HandledException
        except Exception:
            common.logger.error(f"Unknown exception while creating a grid from input file.", exc_info=True)
            raise HandledException

    def print(self) -> None:
        """
        Prints the grid data.
        """
        self.global_data.print()
        common.logger.debug("\nNodes:")
        for node in self.nodes:
            node.print()
        common.logger.debug("\nElements:")
        for element in self.elements:
            element.print()
        common.logger.debug(f"\nBC:\n{self.BC}\n")