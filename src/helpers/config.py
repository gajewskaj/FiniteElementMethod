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
        DOF: int = None
        TPB: int = 512
        MAX_MATERIALS: int = 20
        MAX_IP: int = 25
    class Solver:
        solver = "cudss_v1"