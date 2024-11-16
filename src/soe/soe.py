import numpy as np

from src.helpers import config
from src.grid.grid import Grid

def simulate(grid: Grid) -> tuple[list[float], list[np.ndarray]]:
    if config.use_gpu:
        from src.soe.soe_gpu import simulate
    else:
        from src.soe.soe_cpu import simulate
    times, temperatures = simulate(grid)
    return times, temperatures