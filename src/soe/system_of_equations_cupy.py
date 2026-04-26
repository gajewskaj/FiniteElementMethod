import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import cupyx.scipy.sparse.linalg as cupy_linalg
import numpy as np

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.grid.grid import Grid
from src.soe.system_of_equations import SystemOfEquations

class SystemOfEquationsCuPy(SystemOfEquations):
    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp, dtype=cp.float64)
        self.H, self.C, self.P = self._prepare_data()
        self.A = (self.H + self.C/self.step)
        self.solve_factorized = self._factorize()

    @measure_time
    def _prepare_data(self) -> tuple[cupy_sparse.csr_matrix, cupy_sparse.csr_matrix, cp.ndarray]:
        H_cpu, C_cpu, P_cpu = super()._prepare_data()
        P = cp.array(P_cpu, dtype=cp.float64)
        H = cupy_sparse.csr_matrix(
            (
                cp.asarray(H_cpu.data, dtype=cp.float64),
                cp.asarray(H_cpu.indices, dtype=cp.int32),
                cp.asarray(H_cpu.indptr, dtype=cp.int32),
            ),
            shape=H_cpu.shape,
        )
        C = cupy_sparse.csr_matrix(
            (
                cp.asarray(C_cpu.data, dtype=cp.float64),
                cp.asarray(C_cpu.indices, dtype=cp.int32),
                cp.asarray(C_cpu.indptr, dtype=cp.int32),
            ),
            shape=C_cpu.shape,
        )

        return H, C, P

    @measure_time
    def _factorize(self) -> callable:
        return cupy_linalg.factorized(self.A)

    @measure_time
    def solve(self) -> cp.ndarray:
        b = self.P + self.C.dot(self.t0)/self.step
        result: cp.ndarray = self.solve_factorized(b)
        self.t0 = result
        return result

    def simulate(self) -> tuple[list[float], list[np.ndarray[float]]]:
        times: list[float] = []
        temperatures: list[cp.ndarray] = []
        logger.info("Initializing system of equations.")
        tauk = self.grid.global_data.simulation_time
        dtau = self.step
        logger.info("Calculating temperatures for every timestamp.")
        while dtau <= tauk:
            result: cp.ndarray = self.solve()
            times.append(dtau)
            temperatures.append(result)
            dtau += self.step
        return times, [temp.get() for temp in temperatures]