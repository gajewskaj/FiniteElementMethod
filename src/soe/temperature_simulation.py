import numpy as np

from src.helpers.config import logger, Settings
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices

def simulate(mesh: Mesh, out_mat: OutMatrices) -> tuple[list[float], list[np.ndarray[float]]]:
    match Settings.Solver.solver:
        case "cudss_v1":
            logger.info(f"Solving system of equations using cuDSS V1 solver.")
            from src.soe.system_of_equations_cudss_v1 import SystemOfEquationsCuDSS as SystemOfEquations
        case "cudss_v2":
            logger.info(f"Solving system of equations using cuDSS V2 solver.")
            from src.soe.system_of_equations_cudss_v2 import SystemOfEquationsCuDSS as SystemOfEquations
        case "cudss_v3":
            logger.info(f"Solving system of equations using cuDSS V3 solver.")
            from src.soe.system_of_equations_cudss_v3 import SystemOfEquationsCuDSS as SystemOfEquations
        case "cudss_v4":
            logger.info(f"Solving system of equations using cuDSS V4 solver.")
            from src.soe.system_of_equations_cudss_v4 import SystemOfEquationsCuDSS as SystemOfEquations
        case "cupy":
            logger.info(f"Solving system of equations using CuPy solver.")
            from src.soe.system_of_equations_cupy import SystemOfEquationsCuPy as SystemOfEquations
        case _:
            logger.info(f"Solving system of equations using SciPy solver.")
            from src.soe.system_of_equations_scipy import SystemOfEquationsSciPy as SystemOfEquations
    soe = SystemOfEquations(mesh, out_mat)
    times, temperatures = soe.simulate()
    return times, temperatures