import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import cupyx.scipy.sparse.linalg as gpu_linalg
import numpy as np

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid
from src.soe.system_of_equations import SystemOfEquations
from src.uel.universal_element import NUM_OF_SHAPE_FUNCTIONS

class SystemOfEquationsGPU(SystemOfEquations):
    def __init__(self, grid: Grid):
        self.dim: int = len(grid.nodes_id)
        self.step: float = grid.global_data.simulation_step_time
        self.grid = grid
        self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp)
        self.H_g, self.C_g, self.P_g = self._assemble()
        self.A = self.H_g + self.C_g/self.step
        self.gpu_solve_factorized = self._factorize()

    @measure_time
    def _factorize(self) -> gpu_linalg.factorized:
        return gpu_linalg.factorized(self.A)

    @measure_time
    def _assemble(self) -> tuple[gpu_sparse.csr_matrix, gpu_sparse.csr_matrix]:
        data_H, row_H, col_H = [], [], []
        data_C, row_C, col_C = [], [], []
        P = np.zeros((self.dim, 1))

        for i in range(len(self.grid.elements_id)):
            local_H = self.grid.elements_H[i] + self.grid.elements_Hbc[i]
            for j in range(NUM_OF_SHAPE_FUNCTIONS):
                P[self.grid.elements_node_ids[i, j] - 1] += self.grid.elements_P[i, j]
                for k in range(NUM_OF_SHAPE_FUNCTIONS):
                        data_H.append(local_H[j, k])
                        row_H.append(self.grid.elements_node_ids[i, j] - 1)
                        col_H.append(self.grid.elements_node_ids[i, k] - 1)
                        data_C.append(self.grid.elements_C[i, j, k])
                        row_C.append(self.grid.elements_node_ids[i, j] - 1)
                        col_C.append(self.grid.elements_node_ids[i, k] - 1)
        data_H = cp.array(data_H)
        row_H = cp.array(row_H)
        col_H = cp.array(col_H)
        data_C = cp.array(data_C)
        row_C = cp.array(row_C)
        col_C = cp.array(col_C)

        H: gpu_sparse.csr_matrix = gpu_sparse.coo_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim)).tocsr()
        C: gpu_sparse.csr_matrix = gpu_sparse.coo_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim)).tocsr()
        return H, C, cp.array(P)

    @measure_time
    def solve(self) -> cp.ndarray:
        B = self.P_g + self.C_g.dot(self.t0)/self.step
        result: cp.ndarray = self.gpu_solve_factorized(B)
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