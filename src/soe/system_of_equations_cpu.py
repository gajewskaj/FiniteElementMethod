import numpy as np
import scipy.sparse as cpu_sparse
import scipy.sparse.linalg as cpu_linalg

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid
from src.soe.system_of_equations import SystemOfEquations

class SystemOfEquationsCPU(SystemOfEquations):
    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.t0: np.ndarray = np.full((self.dim, 1), grid.global_data.initial_temp)
        self.H, self.C, self.P = self._prepare_data()
        self.A = self.H + self.C/self.step
        self.solve_factorized = self._factorize()

    @measure_time
    def _factorize(self) -> callable:
        return cpu_linalg.factorized(self.A)

    @measure_time
    def _prepare_data(self) -> tuple[cpu_sparse.csr_matrix, cpu_sparse.csr_matrix]:
        return super()._prepare_data()

    @measure_time
    def solve(self) -> np.ndarray:
        B = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = self.solve_factorized(B)
        self.t0 = result
        return result

def simulate(grid: Grid) -> tuple[list[float], list[np.ndarray[float]]]:
    times: list[float] = []
    temperatures: list[np.ndarray] = []
    config.logger.info("Initializing system of equations.")
    soe = SystemOfEquationsCPU(grid)
    tauk: float = grid.global_data.simulation_time
    dtau: float = soe.step
    config.logger.info("Calculating temperatures for every timestamp on CPU.")
    while dtau <= tauk:
        result: np.ndarray = soe.solve()
        times.append(dtau)
        temperatures.append(result)
        dtau += soe.step
    return times, temperatures