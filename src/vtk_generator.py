import os

from jinja2 import  Environment, FileSystemLoader, Template
import numpy as np

from . import config
from .grid import Grid

def initialize_jinja_environment(template_filepath: str) -> Template:
    """
    Initialize the Jinja2 environment and load a template.

    Args:
        template_filepath (str): The path to the template file.

    Returns:
        Template: The loaded Jinja2 template.
    """
    environment = Environment(loader=FileSystemLoader(config.templates_path))
    template = environment.get_template(template_filepath)
    return template

def generate_file(data: dict, template: Template, dest_dir: str, output_fileame: str) -> None:
    """
    Generate a file from a template and data.

    Args:
        data (dict): The data to render the template with.
        template (Template): The Jinja2 template to use.
        dest_dir (str): The directory to save the generated file in.
        output_fileame (str): The name of the generated file.
    """
    output_filepath = os.path.join(dest_dir, output_fileame)
    content = template.render(data)
    with open(output_filepath, mode="w", encoding="utf-8") as file:
        file.write(content)

def generate_vtk_files(output_dir_path: str, grid: Grid, temperatures: list[np.ndarray]) -> None:
    """
    Creates files for simulation in ParaView environment.

    Args:
        output_dir_path (str): Path to the output directory.
        grid (Grid): Grid object containing simulation data.
        temperatures (list[np.ndarray]): List of temperature arrays for each time step.
    """
    num_files: int = len(temperatures)
    element_nodes_number: int = []
    for element in grid.elements:
        element_nodes_number.append(len(element.node_ids))

    data: dict = {}
    data["nodes_number"] = grid.global_data.nodes_number
    data["nodes"] = grid.nodes
    data["elements_number"] = grid.global_data.elements_number
    data["elements"] = grid.elements
    data["element_nodes_number"] = element_nodes_number
    data["sum_elements_data"] = grid.global_data.elements_number + sum(element_nodes_number)

    template: Template = initialize_jinja_environment("temperatures.vtk.jinja")
    for i in range(0, num_files):
        data["temperatures"] = temperatures[i]
        filename: str = f"frame{i+1}.vtk"
        generate_file(data, template, output_dir_path, filename)
    config.logger.info(f"Output files generated in '{output_dir_path}'.")