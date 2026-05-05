import numpy as np
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_linalg

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices
from src.soe.system_of_equations import SystemOfEquations

class SystemOfEquationsSciPy(SystemOfEquations):
    def __init__(self, mesh: Mesh, out_mat: OutMatrices) -> None:
        super().__init__(mesh, out_mat)
        self.t0: np.ndarray = np.full((self.dim, 1), mesh.global_data.initial_temp)
        self.H, self.C, self.P = self._prepare_data()
        self.A = self.H + self.C/self.step
        self.solve_factorized = self._factorize()

    @measure_time
    def _prepare_data(self) -> tuple[scipy_sparse.csr_matrix, scipy_sparse.csr_matrix]:
        return super()._prepare_data()

    @measure_time
    def _factorize(self) -> callable:
        return scipy_linalg.factorized(self.A)

    @measure_time
    def solve(self) -> np.ndarray:
        B = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = self.solve_factorized(B)
        self.t0 = result
        return result

    def simulate(self) -> tuple[list[float], list[np.ndarray[float]]]:
        times: list[float] = []
        temperatures: list[np.ndarray] = []
        logger.info("Initializing system of equations.")
        tauk: float = self.mesh.global_data.simulation_time
        dtau: float = self.step
        logger.info("Calculating temperatures for every timestamp.")
        while dtau <= tauk:
            result: np.ndarray = self.solve()
            times.append(dtau)
            temperatures.append(result)
            dtau += self.step
        return times, temperatures