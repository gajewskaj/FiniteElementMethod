import os
import sys
import logging
from jinja2 import Environment, FileSystemLoader, Template

script_path: str = os.getcwd()
grids_path: str = os.path.join(script_path, 'Data', 'Grids')
output_path: str = os.path.join(script_path, 'Data', 'Output')
templates_path: str = os.path.join(script_path, 'Data', 'Templates')

main_logger: logging.Logger = None

class FiniteElementMethodException(Exception):
    pass

def init_logging(log_filepath: str) -> logging.Logger:
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(fmt='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter(fmt='[%(levelname)s] %(message)s')

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def create_or_clear_directory(input_filename: str) -> str:
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    dir_path = os.path.join(output_path, os.path.basename(input_filename).split('.')[0])

    try:
        os.mkdir(dir_path)
    except FileExistsError:
        for filename in os.listdir(dir_path):
            try:
                filepath = os.path.join(dir_path, filename)
                if os.path.isfile(filepath) or os.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    dir_path.rmtree(filepath)
            except Exception as e:
                raise FiniteElementMethodException(f'Error while claring output directory {dir_path}.\nFailed to delete {os.path.basename(filepath)}. Error:\n{e}')
    finally:
        return dir_path

def initialize_jinja_environment(template_filepath: str) -> Template:
    environment = Environment(loader=FileSystemLoader(templates_path))
    template = environment.get_template(template_filepath)
    return template

def generate_file(data: dict, template: Template, dest_dir: str, output_fileame: str) -> None:
    output_filepath = os.path.join(dest_dir, output_fileame)
    content = template.render(data)
    with open(output_filepath, mode='w', encoding='utf-8') as file:
        file.write(content)

def print2dTab(tab: list[list]) -> None:
    for inner_tab in tab:
        print(inner_tab)
    print('\n')