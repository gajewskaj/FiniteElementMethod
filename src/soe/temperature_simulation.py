import numpy as np

from src.helpers import config
from src.grid.grid import Grid

def simulate(grid: Grid) -> tuple[list[float], list[np.ndarray]]:
    if config.use_gpu:
        from src.soe.system_of_equations_gpu import simulate
    else:
        from src.soe.system_of_equations_cpu import simulate
    times, temperatures = simulate(grid)
    return times, temperatures