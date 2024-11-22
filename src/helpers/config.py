import logging
import os
import sys

project_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
input_path: str = os.path.join(project_path, "input")
output_path: str = os.path.join(project_path, "output")
templates_path: str = os.path.join(project_path, "templates")
test_path: str = os.path.join(project_path, "test")
scripts_path: str = os.path.join(project_path, "scripts")

MAIN_LOGGER_NAME = "main_logger"
TEST_LOGGER_NAME = "test_logger"

logger: logging.Logger = logging.Logger(MAIN_LOGGER_NAME)

use_gpu: bool = False

