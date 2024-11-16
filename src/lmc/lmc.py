from src.helpers import config
from src.grid.grid import Grid

def calculate_local_matrices(n: int, grid: Grid) -> None:
    if config.use_gpu:
        from src.lmc.lmc_gpu import calculate_local_matrices
    else:
        from src.lmc.lmc_cpu import calculate_local_matrices
    calculate_local_matrices(n, grid)