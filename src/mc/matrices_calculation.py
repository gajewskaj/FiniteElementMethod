from src.helpers import config
from src.grid.grid import Grid

def calculate_and_assemble_matrices(grid: Grid) -> None:
    if config.use_gpu:
        from src.mc.matrices_calculation_gpu import calculate_and_assemble_matrices
    else:
        from src.mc.matrices_calculation_cpu import calculate_and_assemble_matrices
    config.logger.info(f"Calculating and assembling matrices.")
    calculate_and_assemble_matrices(grid)