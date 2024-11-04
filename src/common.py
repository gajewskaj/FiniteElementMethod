import os
import logging
import shutil
from jinja2 import Environment, FileSystemLoader, Template

script_path: str = os.getcwd()
input_path: str = os.path.join(script_path, "Input")
output_path: str = os.path.join(script_path, "Output")
templates_path: str = os.path.join(script_path, "Templates")
test_path: str = os.path.join(script_path, "Test")

MAIN_LOGGER_NAME = "main_logger"
TEST_LOGGER_NAME = "test_logger"

logger: logging.Logger = None

class Settings:
    """
    Class to store settings provided as parameters.
    """
    def __init__(self, input_filepath: str, force_cpu: bool = False):
        self.input_filepath = input_filepath
        self.force_cpu = force_cpu

settings: Settings = None

class HandledException(Exception):
    """Custom exception class for handling specific errors."""
    pass

class NoTracebackFilter(logging.Filter):
    """Logging filter to remove traceback information from log records."""
    def filter(self, record):
        if record.exc_info:
            record.exc_info = None
            record.exc_text = None
        return True

def init_logging(logger_name: str = MAIN_LOGGER_NAME) -> logging.Logger:
    """
    Initialize and configure the logging system.

    Args:
        logger_name (str): The name of the logger to initialize.

    Returns:
        logging.Logger: Configured logger instance.
    """
    match(logger_name):
        case "main_logger":
            logger = logging.getLogger(logger_name)
            logger = logging.getLogger(MAIN_LOGGER_NAME)
            log_filepath = os.path.join(output_path, "log.log")
            logger.setLevel(logging.DEBUG)

            file_formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

            console_formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            console_handler.addFilter(NoTracebackFilter())
            logger.addHandler(console_handler)

        case "test_logger":
            logger = logging.getLogger(logger_name)
            logger = logging.getLogger(TEST_LOGGER_NAME)
            log_filepath = os.path.join(output_path, "test_log.log")
            logger.setLevel(logging.DEBUG)

            file_formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger

def create_or_clear_directory(dir_path: str) -> str:
    """
    Create a directory or clear its contents if it already exists.

    Args:
        dir_path (str): The path of the directory to create or clear.

    Returns:
        str: The path of the created or cleared directory.
    """
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    try:
        os.mkdir(dir_path)
    except FileExistsError:
        for filename in os.listdir(dir_path):
            try:
                filepath = os.path.join(dir_path, filename)
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception:
                logger.error(f"Error while clearing output directory: '{dir_path}'. Failed to delete '{os.path.basename(filepath)}'.", exc_info=True)
                raise HandledException
    finally:
        return dir_path

def initialize_jinja_environment(template_filepath: str) -> Template:
    """
    Initialize the Jinja2 environment and load a template.

    Args:
        template_filepath (str): The path to the template file.

    Returns:
        Template: The loaded Jinja2 template.
    """
    environment = Environment(loader=FileSystemLoader(templates_path))
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

def print2dTab(tab: list[list]) -> None:
    """
    Print a 2D list to the logger.

    Args:
        tab (list[list]): The 2D list to print.
    """
    for inner_tab in tab:
        logger.debug(inner_tab)