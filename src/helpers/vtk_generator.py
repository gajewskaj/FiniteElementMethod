import os

from jinja2 import  Environment, FileSystemLoader, Template
import numpy as np

from src.helpers import config
from src.grid.grid import Grid

def initialize_jinja_environment(template_filepath: str) -> Template:
    environment = Environment(loader=FileSystemLoader(config.templates_path))
    template = environment.get_template(template_filepath)
    return template

def generate_file(data: dict, template: Template, dest_dir: str, output_fileame: str) -> None:
    output_filepath = os.path.join(dest_dir, output_fileame)
    content = template.render(data)
    with open(output_filepath, mode="w", encoding="utf-8") as file:
        file.write(content)

def generate_vtk_files(output_dir_path: str, grid: Grid, temperatures: list[np.ndarray]) -> None:
    num_files: int = len(temperatures)
    element_nodes_number: list[int] = [config.num_of_shape_functions] * len(grid.elements_id)

    data: dict = {}
    data["nodes_number"] = len(grid.nodes_id)
    data["nodes_x"] = grid.nodes_x
    data["nodes_y"] = grid.nodes_y
    data["elements_number"] = len(grid.elements_id)
    data["elements_node_ids"] = grid.elements_node_ids
    data["element_nodes_number"] = element_nodes_number
    data["sum_elements_data"] = len(grid.elements_id) + sum(element_nodes_number)

    template: Template = initialize_jinja_environment("temperatures.vtk.jinja")
    for i in range(0, num_files):
        data["temperatures"] = temperatures[i]
        filename: str = f"frame{i+1}.vtk"
        generate_file(data, template, output_dir_path, filename)
    config.logger.info(f"Output files generated in '{output_dir_path}'.")