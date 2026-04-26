import numpy as np

from src.helpers.config import logger, Settings
from src.grid.grid import Grid

def simulate(grid: Grid) -> tuple[list[float], list[np.ndarray[float]]]:
    if Settings.Solver.use_cudss:
        logger.info(f"Solving system of equations using cuDSS solver.")
        from src.soe.system_of_equations_cudss import SystemOfEquationsCuDSS as SystemOfEquations
    elif Settings.Solver.use_cupy:
        logger.info(f"Solving system of equations using CuPy solver.")
        from src.soe.system_of_equations_cupy import SystemOfEquationsCuPy as SystemOfEquations
    else:
        logger.info(f"Solving system of equations using SciPy solver.")
        from src.soe.system_of_equations_scipy import SystemOfEquationsSciPy as SystemOfEquations
    soe = SystemOfEquations(grid)
    times, temperatures = soe.simulate()
    return times, temperatures