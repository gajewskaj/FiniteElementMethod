import common
from common import *
from universal_element import *
import numpy as np

class GlobalData:
    """
    Stores general information like simulation time, conductivity, initial temperature, density etc.
    """
    def __init__(self, global_data_dict: dict):
        self.simulation_time: float = global_data_dict['SimulationTime']
        self.simulation_step_time: float = global_data_dict['SimulationStepTime']
        self.conductivity: float = global_data_dict['Conductivity']
        self.alfa: float = global_data_dict['Alfa']
        self.tot: float = global_data_dict['Tot']
        self.initial_temp: float = global_data_dict['InitialTemp']
        self.density: float = global_data_dict['Density']
        self.specific_heat: float = global_data_dict['SpecificHeat']
        self.nodes_number: int = global_data_dict['Nodesnumber']
        self.elements_number: int = global_data_dict['Elementsnumber']

    def print(self) -> None:
        print(f'Simulation time: \t{self.simulation_time}')
        print(f'Simulation step time: \t{self.simulation_step_time}')
        print(f'Conductivity: \t\t{self.conductivity}')
        print(f'Alfa: \t\t\t{self.alfa}')
        print(f'Tot: \t\t\t{self.tot}')
        print(f'Initial temp: \t\t{self.initial_temp}')
        print(f'Density: \t\t{self.density}')
        print(f'Specific heat: \t\t{self.specific_heat}')
        print(f'Nodes number: \t\t{self.nodes_number}')
        print(f'Elements number: \t{self.elements_number}')

class Node:
    """
    Stores information about a single node of the grid.

    id:      Node's ID
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
        print(f'Node {self.id}: \t({self.x}, {self.y})')

class Element:
    """
    Stores information about a single 4-node element of the grid.

    id:         Element's ID
    IDs:        IDs od nodes belonging to the element
    H:          H matrix for the element (4x4)
    Hbc:        Hbc matrix for the element (4x4)
    P:          P vector for the element (4X1)
    C:          C matrix for the element (4x4)
    """
    def __init__(self, id: int, node_ids: list[int]):
        self.id: int = id
        self.node_ids: list[int] = node_ids
        self.H: np.ndarray = np.zeros((4, 4))
        self.Hbc: np.ndarray = np.zeros((4, 4))
        self.P: np.ndarray = np.zeros((4, 1))
        self.C: np.ndarray = np.zeros((4, 4))

    def print(self) -> None:
        print(f'Element {self.id}: \t{self.node_ids}')

class Grid:
    """
    Stores information allowing to recreate the grid.

    globalData:     i.e. simulation time, conductivity, initial temperature, density etc.
    nodes:          list of nodes in the grid
    elements:       list of elements in the grid
    BC:             list of nodes with border condition
    """
    def __init__(self, global_data: GlobalData = None, elements: list[Element] = None, nodes: list[Node] = None):
        self.global_data: GlobalData = global_data
        self.nodes: list[Node] = nodes
        self.elements: list[Element] = elements

    @classmethod
    def create_from_file(cls, inputFilePath: str):
        try:
            common.main_logger.info(f"Creating a grid from input file: '{os.path.basename(inputFilePath)}'")
            f = open(inputFilePath, 'r')
            fileContent: str = f.readlines()
            globalData: GlobalData = cls._read_global_data(fileContent)
            nodes: list[Node] = cls._read_nodes(fileContent, globalData.nodes_number)
            elements: list[Element] = cls._read_elements(fileContent, globalData.nodes_number, globalData.elements_number)
            BC: list[Node] = cls._read_bc(fileContent, globalData.nodes_number, globalData.elements_number)
            cls._add_bc_to_node(nodes, BC)
            f.close()
            return cls(globalData, elements, nodes)
        except Exception as e:
            raise FiniteElementMethodException(f'Error while creating a grid. Check if your input file format is correct. Error:\n{e}')

    @staticmethod
    def _read_global_data(input: str) -> GlobalData:
        """
        Reads global data from input file.
        """
        global_data_dict = {}
        for i in range (0, 8):
            line = input[i]
            line = line.split(' ')
            for i in range (0, 2):
                line[i] = line[i].strip()
            global_data_dict[line[0]] = int(line[1])
        for i in range (8, 10):
            line = input[i]
            line = line.split(' ')
            for i in range (0, 3):
                line[i] = line[i].strip()
            line[0] = line[0] + line[1]
            global_data_dict[line[0]] = int(line[2])
        return GlobalData(global_data_dict)

    @staticmethod
    def _read_nodes(input: str, nodes_number: int) -> list[Node]:
        """
        Reads nodes from input file.
        """
        node_list = []
        for i in range (11, 11 + nodes_number):
            line = input[i].split(', ')
            node_list.append(Node(int(line[0]), float(line[1]), float(line[2])))
        return node_list

    @staticmethod
    def _read_elements(input: str, nodes_number: int, elements_number: int) -> list[Element]:
        """
        Reads elements data from input file.
        """
        elements = []
        node_ids = []
        for i in range (11 + nodes_number + 1, 11 + nodes_number + 1 + elements_number):
            line = input[i].split(', ')
            for i in range (1, len(line)):
                node_ids.append(int(line[i]))
            elements.append(Element(int(line[0]), node_ids))
            node_ids = []
        return elements

    @staticmethod
    def _read_bc(input: str, nodes_number: int, elements_number: int) -> list[int]:
        """
        Reads nodes with border condition from input file.
        """
        node_ids = []
        line = input[11 + nodes_number + 1 + elements_number + 1]
        line = line.split(', ')
        for i in range(0, len(line)):
            node_ids.append(int(line[i]))
        return node_ids

    @staticmethod
    def _add_bc_to_node(nodes: list[Node], BC: list[int]) -> None:
        """
        Adds border condition to node.
        """
        for node_id in BC:
            nodes[node_id - 1].BC = 1

    def print(self) -> None:
        self.global_data.print()
        print('\nNodes:')
        for node in self.nodes:
            node.print()
        print('\nElements:')
        for element in self.elements:
            element.print()
        print(f'\nBC:\n{self.BC}\n')