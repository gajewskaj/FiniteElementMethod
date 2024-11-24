from src.helpers import config
from src.grid.grid import Grid

def calculate_local_matrices(grid: Grid) -> None:
    if config.use_gpu:
        from src.lmc.local_matrices_calculation_gpu import calculate_local_matrices
    else:
        from src.lmc.local_matrices_calculation_cpu import calculate_local_matrices
    config.logger.info(f"Calculating local matrices for every element of the grid.")
    calculate_local_matrices(grid)