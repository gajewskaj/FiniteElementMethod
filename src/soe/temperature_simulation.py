import numpy as np

from src.helpers.config import logger, Settings
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices

from src.soe.scipy.system_of_equations_scipy import SystemOfEquationsSciPy
from src.soe.scipy.scipy_cudss_amd import SystemOfEquationsSciPyAMD
from src.soe.scipy.scipy_cudss_nd import SystemOfEquationsSciPyND
from src.soe.cupy.system_of_equations_cupy import SystemOfEquationsCuPy
from src.soe.cupy.cupy_cudss_amd import SystemOfEquationsCuPyAMD
from src.soe.cupy.cupy_cudss_nd import SystemOfEquationsCuPyND
from src.soe.cudss.cudss_v1 import SystemOfEquationsCuDSSV1
from src.soe.cudss.cudss_v2 import SystemOfEquationsCuDSSV2
from src.soe.cudss.cudss_v3 import SystemOfEquationsCuDSSV3
from src.soe.cudss.cudss_v4 import SystemOfEquationsCuDSSV4

d: dict = {
    "scipy_colamd": {"solver": SystemOfEquationsSciPy, "lib": "SciPy", "reordering_alg": "COLAMD"},
    "scipy_mmd_ata": {"solver": SystemOfEquationsSciPy, "lib": "SciPy", "reordering_alg": "MMD_ATA"},
    "scipy_mmd_at_plus_a": {"solver": SystemOfEquationsSciPy, "lib": "SciPy", "reordering_alg": "MMD_AT_PLUS_A"},
    "scipy_cudss_amd": {"solver": SystemOfEquationsSciPyAMD, "lib": "SciPy", "reordering_alg": "AMD (cuDSS)"},
    "scipy_cudss_nd": {"solver": SystemOfEquationsSciPyND, "lib": "SciPy", "reordering_alg": "ND (cuDSS)"},
    "cupy_colamd": {"solver": SystemOfEquationsCuPy, "lib": "CuPy", "reordering_alg": "COLAMD"},
    "cupy_mmd_ata": {"solver": SystemOfEquationsCuPy, "lib": "CuPy", "reordering_alg": "MMD_ATA"},
    "cupy_mmd_at_plus_a": {"solver": SystemOfEquationsCuPy, "lib": "CuPy", "reordering_alg": "MMD_AT_PLUS_A"},
    "cupy_cudss_amd": {"solver": SystemOfEquationsCuPyAMD, "lib": "CuPy", "reordering_alg": "AMD (cuDSS)"},
    "cupy_cudss_nd": {"solver": SystemOfEquationsCuPyND, "lib": "CuPy", "reordering_alg": "ND (cuDSS)"},
    "cudss_v1": {"solver": SystemOfEquationsCuDSSV1, "lib": "CuDSS (v1)", "reordering_alg": "ND"},
    "cudss_v2": {"solver": SystemOfEquationsCuDSSV2, "lib": "CuDSS (v2)", "reordering_alg": "AMD"},
    "cudss_v3": {"solver": SystemOfEquationsCuDSSV3, "lib": "CuDSS (v3)", "reordering_alg": "ND"},
    "cudss_v4": {"solver": SystemOfEquationsCuDSSV4, "lib": "CuDSS (v4)", "reordering_alg": "ND"}
}

def simulate(mesh: Mesh, out_mat: OutMatrices) -> tuple[list[float], list[np.ndarray[float]]]:
    lib: str = d[Settings.Solver.solver]["lib"]
    reordering_alg: str = d[Settings.Solver.solver]["reordering_alg"]
    solver = d[Settings.Solver.solver]["solver"]
    logger.info(f"Solving system of equations using {lib} with {reordering_alg} reordering.")
    soe = solver(mesh, out_mat, reordering_alg)
    times, temperatures = soe.simulate()
    return times, temperatures