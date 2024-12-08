import json
import os

from src.helpers import config
from src.uel.universal_element import NUM_OF_SHAPE_FUNCTIONS

import gmshparser
from gmshparser import Mesh
import numpy as np

class GlobalData:
    def __init__(self, simulation_time: float,
                 simulation_step_time: float,
                 conductivity: float,
                 alpha: float,
                 ambient_temp: float,
                 initial_temp: float,
                 density: float,
                 specific_heat: float):
        self.simulation_time = float(simulation_time)
        self.simulation_step_time = float(simulation_step_time)
        self.conductivity = float(conductivity)
        self.alpha = float(alpha)
        self.ambient_temp = float(ambient_temp)
        self.initial_temp = float(initial_temp)
        self.density = float(density)
        self.specific_heat = float(specific_heat)

class Grid:
    def __init__(self, mesh_path: str, data_path: str):
        self.global_data: GlobalData
        # Nodes
        self.nodes_id: np.ndarray[int]
        self.nodes_x: np.ndarray[float]
        self.nodes_y: np.ndarray[float]
        self.nodes_bc: np.ndarray[int]
        # Elements
        self.elements_id: np.ndarray[int]
        self.elements_node_ids: np.ndarray[int]
        self.elements_H: np.ndarray[float]
        self.elements_C: np.ndarray[float]
        self.elements_Hbc: np.ndarray[float]
        self.elements_P: np.ndarray[float]
        try:
            config.logger.info(f"Creating a {self.__class__.__name__} from input files: '{os.path.basename(mesh_path)}', '{os.path.basename(data_path)}'")
            mesh: Mesh = gmshparser.parse(mesh_path)
            self._read_nodes(mesh)
            self._read_elements(mesh)
            with open(data_path, "r") as data_file:
                data = json.load(data_file)
            self._read_global_data(data)
            self.elements_H = np.zeros((len(self.elements_id), NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS))
            self.elements_C = np.zeros((len(self.elements_id), NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS))
            self.elements_Hbc = np.zeros((len(self.elements_id), NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS))
            self.elements_P = np.zeros((len(self.elements_id), NUM_OF_SHAPE_FUNCTIONS))
        except FileNotFoundError:
            err_msg: str = f"File not found while creating a {self.__class__.__name__} instance from input files: '{os.path.basename(mesh_path)}', '{os.path.basename(data_path)}'."
            config.logger.error(err_msg, exc_info=True)
            raise FileNotFoundError(err_msg)
        except Exception:
            err_msg: str = f"Unknown exception while creating a {self.__class__.__name__} instance from input files: '{os.path.basename(mesh_path)}', '{os.path.basename(data_path)}'."
            config.logger.error(err_msg, exc_info=True)
            raise RuntimeError(err_msg)

    def _read_global_data(self, input: dict):
        global_data_dict: dict = input
        simulation_time: float = global_data_dict["simulation_time"]
        simulation_step_time: float = global_data_dict["simulation_step_time"]
        conductivity: float = global_data_dict["conductivity"]
        alpha: float = global_data_dict["alpha"]
        ambient_temp: float = global_data_dict["ambient_temp"]
        initial_temp: float = global_data_dict["initial_temp"]
        density: float = global_data_dict["density"]
        specific_heat: float = global_data_dict["specific_heat"]
        self.global_data = GlobalData(simulation_time, simulation_step_time, conductivity, alpha, ambient_temp,
                                      initial_temp, density, specific_heat)

    def _read_nodes(self, mesh: Mesh):
        config.logger.info("Reading nodes from mesh.")
        nodes_number: int = mesh.get_number_of_nodes()
        self.nodes_id = np.empty(nodes_number, dtype=int)
        self.nodes_x = np.empty(nodes_number)
        self.nodes_y = np.empty(nodes_number)
        self.nodes_bc = np.empty(nodes_number, dtype=int)
        for entity in mesh.get_node_entities():
            bc = 1 if entity.get_dimension() in [0, 1] else 0
            for node in entity.get_nodes():
                n_id = node.get_tag()
                n_coords = node.get_coordinates()
                self.nodes_id[n_id - 1] = n_id
                self.nodes_x[n_id - 1] = n_coords[0]
                self.nodes_y[n_id - 1] = n_coords[1]
                self.nodes_bc[n_id - 1] = bc
        config.logger.info(f"Number of nodes: {nodes_number}.")

    def _read_elements(self, mesh: Mesh):
        config.logger.info("Reading elements from mesh.")
        element_ids_list = []
        element_node_ids_list = []
        for entity in mesh.get_element_entities():
            if entity.get_element_type() in [2, 3]:
                for element in entity.get_elements():
                    el_id = element.get_tag()
                    el_con = np.asarray(element.get_connectivity())
                    element_ids_list.append(el_id)
                    element_node_ids_list.append(el_con)
        elements_number = len(element_ids_list)
        self.elements_id = np.asarray(element_ids_list)
        self.elements_node_ids = np.asarray(element_node_ids_list)
        config.logger.info(f"Number of elements: {elements_number}.")