import numpy as np

from src.helpers.config import logger, Settings
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices

def simulate(mesh: Mesh, out_mat: OutMatrices) -> tuple[list[float], list[np.ndarray[float]]]:
    match Settings.Solver.solver:
        case "cudss_amd_reord_scipy":
            logger.info(f"Solving system of equations using SciPy with AMD cuDSS reordering")
            from src.soe.cudss_amd_reord_scipy import SystemOfEquationsCuDSSSciPy as SystemOfEquations
        case "cudss_amd_reord_cupy":
            logger.info(f"Solving system of equations using CuPy with AMD cuDSS reordering")
            from src.soe.cudss_amd_reord_cupy import SystemOfEquationsCuDSSCuPy as SystemOfEquations
        case "cudss_nd_reord_scipy":
            logger.info(f"Solving system of equations using SciPy with ND cuDSS reordering")
            from src.soe.cudss_nd_reord_scipy import SystemOfEquationsCuDSSSciPy as SystemOfEquations
        case "cudss_nd_reord_cupy":
            logger.info(f"Solving system of equations using CuPy with ND cuDSS reordering")
            from src.soe.cudss_nd_reord_cupy import SystemOfEquationsCuDSSCuPy as SystemOfEquations
        case "scipy_v1":
            logger.info(f"Solving system of equations using SciPy V1 solver.")
            from src.soe.system_of_equations_scipy_v1 import SystemOfEquationsSciPy as SystemOfEquations
        case "scipy_v2":
            logger.info(f"Solving system of equations using SciPy V2 solver.")
            from src.soe.system_of_equations_scipy_v2 import SystemOfEquationsSciPy as SystemOfEquations
        case "scipy_v3":
            logger.info(f"Solving system of equations using SciPy V3 solver.")
            from src.soe.system_of_equations_scipy_v3 import SystemOfEquationsSciPy as SystemOfEquations
        case "cupy_v1":
            logger.info(f"Solving system of equations using CuPy V1 solver.")
            from src.soe.system_of_equations_cupy_v1 import SystemOfEquationsCuPy as SystemOfEquations
        case "cupy_v2":
            logger.info(f"Solving system of equations using CuPy V2 solver.")
            from src.soe.system_of_equations_cupy_v2 import SystemOfEquationsCuPy as SystemOfEquations
        case "cupy_v3":
            logger.info(f"Solving system of equations using CuPy V3 solver.")
            from src.soe.system_of_equations_cupy_v3 import SystemOfEquationsCuPy as SystemOfEquations
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
        case _:
            raise ValueError(f"Unknown solver: {Settings.Solver.solver}.")
    soe = SystemOfEquations(mesh, out_mat)
    times, temperatures = soe.simulate()
    return times, temperatures