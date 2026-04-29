import os

from jinja2 import  Environment, FileSystemLoader, Template
import numpy as np

from src.helpers.config import logger, Settings
from src.mesh.mesh import Mesh

def initialize_jinja_environment(template_filepath: str) -> Template:
    environment = Environment(loader=FileSystemLoader(Settings.templates_path))
    template = environment.get_template(template_filepath)
    return template

def generate_file(data: dict, template: Template, dest_dir: str, output_fileame: str) -> None:
    output_filepath = os.path.join(dest_dir, output_fileame)
    content = template.render(data)
    with open(output_filepath, mode="w", encoding="utf-8") as file:
        file.write(content)

def generate_vtk_files(output_dir_path: str, mesh: Mesh, temperatures: list[np.ndarray]) -> None:
    num_files: int = len(temperatures)
    element_nodes_number: list[int] = [Settings.MatricesCalculation.DOF] * len(mesh.elements_id)

    data: dict = {}
    data["nodes_number"] = len(mesh.nodes_id)
    data["nodes_x"] = mesh.nodes_x
    data["nodes_y"] = mesh.nodes_y
    data["elements_number"] = len(mesh.elements_id)
    data["elements_node_ids"] = mesh.elements_node_ids
    data["element_nodes_number"] = element_nodes_number
    data["sum_elements_data"] = len(mesh.elements_id) + sum(element_nodes_number)

    template: Template = initialize_jinja_environment("temperatures.vtk.jinja")
    for i in range(0, num_files):
        data["temperatures"] = temperatures[i]
        filename: str = f"frame{i+1}.vtk"
        generate_file(data, template, output_dir_path, filename)
    logger.info(f"Output files generated in '{output_dir_path}'.")