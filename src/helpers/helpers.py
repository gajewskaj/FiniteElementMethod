import logging
import os
import shutil
import time

from src.helpers.config import MAIN_LOGGER_NAME, Settings
import src.helpers.config as config

def measure_time(func):
    def wrapper(*args, **kwargs):
        start: float = time.time()
        result = func(*args, **kwargs)
        end: float = time.time()
        config.logger.debug(f"Function '{func.__name__}' executed in {(end  - start)} seconds.")
        return result

    return wrapper

def set_algorithms(mc: str, solver: str) -> None:
    if mc == "gpu":
        config.logger.info("GPU selected for calculating matrices.")
        try: import numba
        except ImportError:
            Settings.MatricesCalculation.use_gpu = False
            config.logger.warning("Numba is not installed. CPU will be used for matrices calculation.")
        else:
            Settings.MatricesCalculation.use_gpu = True
    else:
        config.logger.info("CPU selected for calculating matrices.")

    Settings.Solver.solver = solver

    try:
        import cupy
    except ImportError:
        Settings.Solver.use_cupy = False
        Settings.Solver.use_cudss = False
        config.logger.warning("CuPy is not installed. CPU will be used for matrices calculation.")
    else:
        try:
            if not cupy.cuda.is_available():
                Settings.Solver.use_cupy = False
                Settings.Solver.use_cudss = False
                config.logger.warning("CUDA is not available, SciPy solver will be used.")
        except:
            config.logger.warning("CUDA is not available, SciPy solver will be used.")

class NoTracebackFilter(logging.Filter):
    def filter(self, record):
        if record.exc_info:
            record.exc_info = None
            record.exc_text = None
        return True

def init_logging(logger_name: str = MAIN_LOGGER_NAME,
                 log_dirpath: str = Settings.output_path) -> logging.Logger:
    config.logger = logging.getLogger(logger_name)
    config.logger = logging.getLogger(MAIN_LOGGER_NAME)
    log_filepath = os.path.join(log_dirpath, "log.log")
    config.logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    config.logger.addHandler(file_handler)

    console_formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    config.logger.addHandler(console_handler)

    return config.logger

def create_or_clear_directory(dir_path: str) -> str:
    try:
        os.mkdir(Settings.output_path)
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