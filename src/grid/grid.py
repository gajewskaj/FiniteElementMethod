import json
from math import isclose
import os

from src.helpers import config
from src.universal_element.universal_element import NUM_OF_SHAPE_FUNCTIONS

import gmshparser
from gmshparser import Mesh
import numpy as np

class GlobalData:
    """
    Stores general information like simulation time, conductivity, initial temperature, density, etc.

    Attributes:
        simulation_time (float): Total simulation time.
        simulation_step_time (float): Time step for the simulation.
        conductivity (float): Thermal conductivity.
        alpha (float): Heat transfer coefficient.
        ambient_temp (float): Ambient temperature.
        initial_temp (float): Initial temperature.
        density (float): Material density.
        specific_heat (float): Specific heat capacity.
        nodes_number (int): Number of nodes in the grid.
        elements_number (int): Number of elements in the grid.
    """
    def __init__(self, simulation_time: float,
                 simulation_step_time: float,
                 conductivity: float,
                 alpha: float,
                 ambient_temp: float,
                 initial_temp: float,
                 density: float,
                 specific_heat: float,
                 nodes_number: int,
                 elements_number: int):
        """
        Initializes GlobalData with values from a dictionary.

        Args:
            global_data_dict (dict): dictionary containing global data.
        """
        self.simulation_time: float = simulation_time
        self.simulation_step_time: float = simulation_step_time
        self.conductivity: float = conductivity
        self.alpha: float = alpha
        self.ambient_temp: float = ambient_temp
        self.initial_temp: float = initial_temp
        self.density: float = density
        self.specific_heat: float = specific_heat
        self.nodes_number: int = nodes_number
        self.elements_number: int = elements_number

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
                       isclose(self.alpha, other.alpha) and \
                       isclose(self.ambient_temp, other.ambient_temp) and \
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
        config.logger.debug(f"Simulation time: \t{self.simulation_time}")
        config.logger.debug(f"Simulation step time: \t{self.simulation_step_time}")
        config.logger.debug(f"Conductivity: \t\t{self.conductivity}")
        config.logger.debug(f"alpha: \t\t\t{self.alpha}")
        config.logger.debug(f"Ambient temperature: \t\t\t{self.ambient_temp}")
        config.logger.debug(f"Initial temperature: \t\t{self.initial_temp}")
        config.logger.debug(f"Density: \t\t{self.density}")
        config.logger.debug(f"Specific heat: \t\t{self.specific_heat}")
        config.logger.debug(f"Nodes number: \t\t{self.nodes_number}")
        config.logger.debug(f"Elements number: \t{self.elements_number}")

class Node:
    """
    Stores information about a single node of the grid.

    Attributes:
        id (int): Node's ID.
        x (float): x coordinate.
        y (float): y coordinate.
        BC (float): Border condition (0 or 1).
    """
    def __init__(self, id: int, x: float, y: float, z: float = 0, BC: float = 0):
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
        self.z: float = z
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
        config.logger.debug(f"Node {self.id}: \t({self.x}, {self.y})")

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
        self.H: np.ndarray = np.zeros((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
        self.Hbc: np.ndarray = np.zeros((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
        self.P: np.ndarray = np.zeros((NUM_OF_SHAPE_FUNCTIONS, 1), dtype=np.float32)
        self.C: np.ndarray = np.zeros((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)

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
        config.logger.debug(f"Element {self.id}: \t{self.node_ids}")

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
    def create_from_txt(cls: 'Grid', mesh_path: str) -> 'Grid':
        """
        Creates a Grid instance from a text file.

        Args:
            mesh_path (str): Path to the mesh .txt file.

        Returns:
            Grid: A Grid instance.
        """
        def _read_global_data(input: str) -> GlobalData:
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
            simulation_time: float = global_data_dict["SimulationTime"]
            simulation_step_time: float = global_data_dict["SimulationStepTime"]
            conductivity: float = global_data_dict["Conductivity"]
            alpha: float = global_data_dict["Alfa"]
            ambient_temp: float = global_data_dict["Tot"]
            initial_temp: float = global_data_dict["InitialTemp"]
            density: float = global_data_dict["Density"]
            specific_heat: float = global_data_dict["SpecificHeat"]
            nodes_number: int = global_data_dict["Nodesnumber"]
            elements_number: int = global_data_dict["Elementsnumber"]
            return GlobalData(simulation_time, simulation_step_time, conductivity, alpha, ambient_temp,
                              initial_temp, density, specific_heat, nodes_number, elements_number)

        def _read_nodes(input: str, nodes_start_line: int, nodes_number: int) -> np.ndarray[Node]:
            node_list = np.empty(nodes_number, dtype=Node)
            for i in range (nodes_start_line, nodes_start_line + nodes_number):
                line = input[i].split(", ")
                node_list[i-nodes_start_line] = Node(int(line[0]), float(line[1]), float(line[2]))
            return node_list

        def _read_elements(input: str, elements_start_line: int, elements_number: int) -> np.ndarray[Element]:
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
            node_ids = []
            line = input[bc_start_line]
            line = line.split(", ")
            for i in range(0, len(line)):
                node_ids.append(int(line[i]))
            return node_ids

        def _add_bc_to_node(nodes: np.ndarray[Node], BC: list[int]) -> None:
            for node_id in BC:
                nodes[node_id - 1].BC = 1

        try:
            config.logger.info(f"Creating a grid from input file: '{os.path.basename(mesh_path)}'.")
            f = open(mesh_path, "r")
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
        except Exception:
            err_msg: str = f"Unknown exception while creating a Grid instance from input files: '{os.path.basename(mesh_path)}'."
            config.logger.error(err_msg, exc_info=True)
            raise RuntimeError(err_msg)

    @classmethod
    def create_from_msh_and_json(cls: 'Grid', mesh_path: str, data_path: str) -> 'Grid':
        """
        Creates a Grid instance from a .msh file and a .json file.

        Args:
            mesh_path (str): Path to the .msh file.
            data_path (str): Path to the .json file.

        Returns:
            Grid: A Grid instance.
        """
        def _read_global_data(input: dict, nodes_number: int, elements_number: int) -> GlobalData:
            global_data_dict: dict = input
            simulation_time: float = global_data_dict["simulation_time"]
            simulation_step_time: float = global_data_dict["simulation_step_time"]
            conductivity: float = global_data_dict["conductivity"]
            alpha: float = global_data_dict["alpha"]
            ambient_temp: float = global_data_dict["ambient_temp"]
            initial_temp: float = global_data_dict["initial_temp"]
            density: float = global_data_dict["density"]
            specific_heat: float = global_data_dict["specific_heat"]
            return GlobalData(simulation_time, simulation_step_time, conductivity, alpha, ambient_temp,
                              initial_temp, density, specific_heat, nodes_number, elements_number)

        def _read_nodes(mesh: Mesh) -> np.ndarray[Node]:
            config.logger.info("Reading nodes from mesh.")
            nodes_number: int = mesh.get_number_of_nodes()
            config.logger.info(f"Number of nodes: {nodes_number}.")
            nodes = np.empty(nodes_number, dtype=Node)
            for entity in mesh.get_node_entities():
                entity_dim = 1 if entity.get_dimension() in [0, 1] else 0
                for node in entity.get_nodes():
                    n_id = node.get_tag()
                    n_coords = node.get_coordinates()
                    nodes[n_id - 1] = Node(n_id, n_coords[0], n_coords[1], n_coords[2], entity_dim)
            return nodes

        def _read_elements(mesh: Mesh) -> np.ndarray[Element]:
            config.logger.info("Reading elements from mesh.")
            elements_list = []
            for entity in mesh.get_element_entities():
                if entity.get_element_type() in [2, 3]:
                    for element in entity.get_elements():
                        el_id = element.get_tag()
                        el_con = np.asarray(element.get_connectivity())
                        elements_list.append(Element(el_id, el_con))
            elements_number = len(elements_list)
            config.logger.info(f"Number of elements: {elements_number}.")
            return np.asarray(elements_list)

        try:
            config.logger.info(f"Creating a grid from input files: '{os.path.basename(mesh_path)}', '{os.path.basename(data_path)}'")
            mesh: Mesh = gmshparser.parse(mesh_path)
            nodes: np.ndarray[Node] = _read_nodes(mesh)
            elements: np.ndarray[Element] = _read_elements(mesh)
            with open(data_path, "r") as data_file:
                data = json.load(data_file)
            global_data: GlobalData = _read_global_data(data, len(nodes), len(elements))
            return cls(global_data, elements, nodes)
        except FileNotFoundError as e:
            err_msg: str = f"File not found while creating a Grid instance from input files: '{os.path.basename(mesh_path)}', '{os.path.basename(data_path)}'."
            config.logger.error(err_msg, exc_info=True)
            raise FileNotFoundError(err_msg)
        except Exception as e:
            err_msg: str = f"Unknown exception while creating a Grid instance from input files: '{os.path.basename(mesh_path)}', '{os.path.basename(data_path)}'."
            config.logger.error(err_msg, exc_info=True)
            raise RuntimeError(err_msg)

    def print(self) -> None:
        """
        Prints the grid data.
        """
        self.global_data.print()
        config.logger.debug("\nNodes:")
        for node in self.nodes:
            node.print()
        config.logger.debug("\nElements:")
        for element in self.elements:
            element.print()
        config.logger.debug(f"\nBC:\n{self.BC}\n")