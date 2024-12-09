import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import cupyx.scipy.sparse.linalg as gpu_linalg
import numpy as np

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid
from src.soe.system_of_equations import SystemOfEquations

class SystemOfEquationsGPU(SystemOfEquations):
    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp)
        self.H, self.C, self.P = self._prepare_data()
        self.A = (self.H + self.C/self.step)
        self.solve_factorized = self._factorize()

    @measure_time
    def _factorize(self) -> callable:
        return gpu_linalg.factorized(self.A)

    @measure_time
    def _prepare_data(self) -> tuple[gpu_sparse.csr_matrix, gpu_sparse.csr_matrix]:
        H_cpu, C_cpu, P_cpu = super()._prepare_data()
        P = cp.array(P_cpu, dtype=cp.float64)
        H = gpu_sparse.csr_matrix((cp.array(H_cpu.data), cp.array(H_cpu.indices), cp.array(H_cpu.indptr)), shape=H_cpu.shape)
        C = gpu_sparse.csr_matrix((cp.array(C_cpu.data), cp.array(C_cpu.indices), cp.array(C_cpu.indptr)), shape=C_cpu.shape)

        return H, C, P

    @measure_time
    def solve(self) -> cp.ndarray:
        B = self.P + self.C.dot(self.t0)/self.step
        result: cp.ndarray = self.solve_factorized(B)
        self.t0 = result
        return result

def simulate(grid: Grid) -> tuple[list[float], list[np.ndarray[float]]]:
    times: list[float] = []
    temperatures: list[cp.ndarray] = []
    config.logger.info("Initializing system of equations.")
    soe = SystemOfEquationsGPU(grid)
    tauk = grid.global_data.simulation_time
    dtau = soe.step
    config.logger.info("Calculating temperatures for every timestamp on GPU.")
    while dtau <= tauk:
        result: np.ndarray = soe.solve()
        times.append(dtau)
        temperatures.append(result)
        dtau += soe.step
    if config.use_gpu:
        return cp.asnumpy(times), [temp.get() for temp in temperatures]
    return times, temperatures