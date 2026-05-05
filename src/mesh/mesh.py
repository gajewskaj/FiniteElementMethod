import gmsh
import json
import os

from src.helpers.config import logger, Settings

import numpy as np
import cupy as cp

class GlobalData:
    def __init__(self, data_path: str):
        with open(data_path) as f:
            d = json.load(f)
        self.simulation_time = d["simulation_time"]
        self.simulation_step_time = d["simulation_step_time"]
        self.ambient_temp = d["ambient_temp"]
        self.initial_temp=d["initial_temp"]

        pg = gmsh.model.getPhysicalGroups()
        self.materials = np.empty((len(pg), 4), dtype=np.float64)
        for dim, phys_tag in gmsh.model.getPhysicalGroups(2):
            material_name = gmsh.model.getPhysicalName(dim, phys_tag)
            self.materials[phys_tag - 1] = np.array([d["materials"][material_name]["alpha"],
                                                     d["materials"][material_name]["conductivity"],
                                                     d["materials"][material_name]["density"],
                                                     d["materials"][material_name]["specific_heat"]], dtype=np.float64)

class Mesh:
    def __init__(self, mesh_path: str, data_path: str):
        logger.info(f"Creating a {self.__class__.__name__} from input files: '{os.path.basename(mesh_path)}', '{os.path.basename(data_path)}'")

        gmsh.initialize()
        gmsh.open(mesh_path)
        self.global_data: GlobalData = GlobalData(data_path)
        self.nodes_id, self.nodes_x, self.nodes_y, self.nodes_bc = self.read_nodes()
        self.elements_id, self.elements_node_ids, self.elements_material_ids = self.read_elements()
        gmsh.finalize()

        logger.info(f"Number of elements: {len(self.elements_id)}.")
        logger.info(f"Number of nodes: {len(self.nodes_id)}.")

    def read_nodes(self) -> tuple[np.ndarray[int], np.ndarray[float], np.ndarray[float], np.ndarray[int]]:
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes_number = len(node_tags)
        nodes_id = np.empty(nodes_number, dtype=np.int64)
        nodes_x = np.empty(nodes_number, dtype=np.float64)
        nodes_y = np.empty(nodes_number, dtype=np.float64)
        nodes_bc = np.empty(nodes_number, dtype=np.int64)
        for i, tag in enumerate(node_tags):
            nodes_id[tag - 1] = int(tag)
            nodes_x[tag - 1] = float(coords[3 * i])
            nodes_y[tag - 1] = float(coords[3 * i + 1])
            nodes_bc[tag - 1] = 0

        for dim, phys_tag in gmsh.model.getPhysicalGroups(1):
            gmsh.model.getPhysicalName(dim, phys_tag)
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
                bc_node_tags, _, _ = gmsh.model.mesh.getNodes(dim, entity, includeBoundary=True)
                for tag in bc_node_tags:
                    nodes_bc[tag - 1] = 1

        logger.info(f"Loaded {len(nodes_id)} unique nodes.")
        return nodes_id, nodes_x, nodes_y, nodes_bc

    def read_elements(self) -> tuple[np.ndarray[int], np.ndarray[np.ndarray[int]], np.ndarray[int]]:
        elements_id: list[int] = []
        elements_node_ids: list[list[int]] = []
        elements_material_ids: list[int] = []
        for dim, phys_tag in gmsh.model.getPhysicalGroups():
            gmsh.model.getPhysicalName(dim, phys_tag)
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim, entity)
                for elem_type, tags, node_tags in zip(elem_types, elem_tags, elem_node_tags):
                    if elem_type not in [2, 3]:
                        continue
                    for i, tag in enumerate(tags):
                        node_ids = [int(node_tags[Settings.MatricesCalculation.DOF * i + j]) for j in range(Settings.MatricesCalculation.DOF)]
                        elements_id.append(int(tag))
                        elements_node_ids.append(node_ids)
                        elements_material_ids.append(phys_tag)
        elements_id_array = np.asarray(elements_id, dtype=np.int64)
        elements_node_ids_array = np.asarray(elements_node_ids, dtype=np.int64)
        elements_material_ids_array = np.asarray(elements_material_ids, dtype=np.int64)

        logger.info(f"Loaded {len(elements_id_array)} elements.")
        return elements_id_array, elements_node_ids_array, elements_material_ids_array
