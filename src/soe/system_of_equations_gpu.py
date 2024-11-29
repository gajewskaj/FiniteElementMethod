import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import cupyx.scipy.sparse.linalg as gpu_linalg
import numpy as np

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid, Element
from src.soe.system_of_equations import SystemOfEquations
from src.uel.universal_element import NUM_OF_SHAPE_FUNCTIONS

import time
from cupyx.profiler import benchmark

class SystemOfEquationsGPU(SystemOfEquations):
    def __init__(self, grid: Grid):
        self.dim: int = grid.global_data.nodes_number
        self.step: float = grid.global_data.simulation_step_time
        self.elements: list[Element] = grid.elements
        self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp)
        self.P: cp.ndarray = self._aggregate_P()
        self.H, self.C = self._aggregate_H_C()
        self.gpu_solve_factorized = self._factorize()

    @measure_time
    def _factorize(self) -> gpu_linalg.factorized:
        return gpu_linalg.factorized(self.H + self.C/self.step)

    @measure_time
    def _aggregate_H_C(self) -> tuple[gpu_sparse.csr_matrix, gpu_sparse.csr_matrix]:
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
        data_H = cp.array(data_H)
        row_H = cp.array(row_H)
        col_H = cp.array(col_H)
        data_C = cp.array(data_C)
        row_C = cp.array(row_C)
        col_C = cp.array(col_C)

        H: gpu_sparse.csr_matrix = gpu_sparse.coo_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim)).tocsr()
        C: gpu_sparse.csr_matrix = gpu_sparse.coo_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim)).tocsr()
        return H, C

    @measure_time
    def _aggregate_P(self) -> cp.ndarray:
        P = np.zeros((self.dim, 1))
        for element in self.elements:
            for i in range(NUM_OF_SHAPE_FUNCTIONS):
                P[element.node_ids[i] - 1] += element.P[i]
        return cp.array(P)

    @measure_time
    def solve(self) -> cp.ndarray:
        P = self.P + self.C.dot(self.t0)/self.step
        result: cp.ndarray = self.gpu_solve_factorized(P)
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
        # b = benchmark(soe.solve, n_repeat=3)
        # print(b)
        times.append(dtau)
        temperatures.append(result)
        dtau += soe.step
    if config.use_gpu:
        return cp.asnumpy(times), [temp.get() for temp in temperatures]
    return times, temperatures