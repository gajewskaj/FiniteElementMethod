import logging
import os
import shutil
import time

from . import config

def measure_time(func):
    """
    Decorator for measuring the execution time of a function.

    Args:
        func (function): The function to measure the execution time of.

    Returns:
        function: The decorated function.
    """
    def wrapper(*args, **kwargs):
        start: float = time.time()
        result = func(*args, **kwargs)
        end: float = time.time()
        config.logger.info(f"'{func.__name__}' executed in {end  - start} seconds.")
        return result

    return wrapper

def set_use_gpu(force_cpu: bool = False) -> None:
    """
    Set the use_gpu flag based on the availability of CUDA and the force_cpu flag.

    Args:
        force_cpu (bool, optional): Defaults to False.
    """
    if force_cpu:
        config.use_gpu = False
        config.logger.warning("Forcing CPU usage instead of GPU.")
        return

    try:
        import cupy
        import numba
    except ImportError:
        config.logger.warning("CuPy or Numba is not installed. Using CPU instead.")
    else:
        if not cupy.cuda.is_available():
            config.use_gpu = False
            config.logger.warning("CUDA is not available. Using CPU instead.")
        else:
            config.use_gpu = True
            config.logger.info("CUDA is available. Using GPU.")

class NoTracebackFilter(logging.Filter):
    """Logging filter to remove traceback information from log records."""
    def filter(self, record):
        if record.exc_info:
            record.exc_info = None
            record.exc_text = None
        return True

def init_logging(logger_name: str = config.MAIN_LOGGER_NAME,
                 log_dirpath: str = config.output_path) -> logging.Logger:
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
            logger = logging.getLogger(config.MAIN_LOGGER_NAME)
            log_filepath = os.path.join(log_dirpath, "log.log")
            logger.setLevel(logging.DEBUG)

            file_formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

            console_formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            console_handler.addFilter(NoTracebackFilter())
            logger.addHandler(console_handler)

        case "test_logger":
            logger = logging.getLogger(logger_name)
            logger = logging.getLogger(config.TEST_LOGGER_NAME)
            log_filepath = os.path.join(log_dirpath, "test_log.log")
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
        os.mkdir(config.output_path)
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
                err_msg: str = f"Error while clearing output directory: '{dir_path}'. Failed to delete '{os.path.basename(filepath)}'."
                config.logger.error(err_msg, exc_info=True)
                raise RuntimeError(err_msg)
    finally:
        return dir_path

def print2dTab(tab: list[list]) -> None:
    """
    Print a 2D list to the logger.

    Args:
        tab (list[list]): The 2D list to print.
    """
    for inner_tab in tab:
        config.logger.debug(inner_tab)