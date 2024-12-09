from src.helpers import config
from src.grid.grid import Grid

def calculate_and_assemble_matrices(grid: Grid) -> None:
    if config.use_gpu:
        from src.lmc.matrices_calculation_gpu import calculate_and_assemble_matrices
    else:
        from src.lmc.matrices_calculation_cpu import calculate_and_assemble_matrices
    config.logger.info(f"Calculating local matrices for every element of the grid.")
    calculate_and_assemble_matrices(grid)