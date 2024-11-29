import numpy as np
import scipy.sparse as cpu_sparse
import scipy.sparse.linalg as cpu_linalg

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid, Element
from src.soe.system_of_equations import SystemOfEquations
from src.uel.universal_element import NUM_OF_SHAPE_FUNCTIONS

class SystemOfEquationsCPU(SystemOfEquations):
    def __init__(self, grid: Grid):
        self.dim: int = grid.global_data.nodes_number
        self.step: float = grid.global_data.simulation_step_time
        self.elements: list[Element] = grid.elements
        self.t0: np.ndarray = np.full((self.dim, 1), grid.global_data.initial_temp)
        self.P: np.ndarray = self._aggregate_P()
        self.H, self.C = self._aggregate_H_C()
        self.cpu_solve_factorized = self._factorize()

    @measure_time
    def _factorize(self) -> cpu_linalg.factorized:
        return cpu_linalg.factorized(self.H + self.C/self.step)

    @measure_time
    def _aggregate_H_C(self) -> tuple[cpu_sparse.csr_matrix, cpu_sparse.csr_matrix]:
        data_H, row_H, col_H = [], [], []
        data_C, row_C, col_C = [], [], []

        for element in self.elements:
            local_H = element.H + element.Hbc
            for i in range(NUM_OF_SHAPE_FUNCTIONS):
                for j in range(NUM_OF_SHAPE_FUNCTIONS):
                        data_H.append(local_H[i][j])
                        row_H.append(element.node_ids[i] - 1)
                        col_H.append(element.node_ids[j] - 1)
                        data_C.append(element.C[i][j])
                        row_C.append(element.node_ids[i] - 1)
                        col_C.append(element.node_ids[j] - 1)

        H: cpu_sparse.csr_matrix = cpu_sparse.coo_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim)).tocsr()
        C: cpu_sparse.csr_matrix = cpu_sparse.coo_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim)).tocsr()

        return H, C

    @measure_time
    def _aggregate_P(self) -> np.ndarray:
        P = np.zeros((self.dim, 1))
        for element in self.elements:
            for i in range(NUM_OF_SHAPE_FUNCTIONS):
                P[element.node_ids[i] - 1] += element.P[i]
        return P

    @measure_time
    def solve(self) -> np.ndarray:
        P = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = self.cpu_solve_factorized(P)
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