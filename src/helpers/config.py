import logging
import os

MAIN_LOGGER_NAME = "main_logger"

logger: logging.Logger = logging.Logger(MAIN_LOGGER_NAME)

class Settings:
    project_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_path: str = os.path.join(project_path, "input")
    output_path: str = os.path.join(project_path, "output")
    templates_path: str = os.path.join(project_path, "templates")
    scripts_path: str = os.path.join(project_path, "scripts")
    class MatricesCalculation:
        use_gpu: bool = False
        DOF: int = 4
        threads_per_block: int = 512
    class Solver:
        use_cupy: bool = False
        use_cudss: bool = False