import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import cupyx.scipy.sparse.linalg as gpu_linalg
import numpy as np

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid, Element
from src.soe.system_of_equations import SystemOfEquations
from src.uel.universal_element import NUM_OF_SHAPE_FUNCTIONS

class SystemOfEquationsGPU(SystemOfEquations):
    """
    Class for calculating temperature in each node of the grid by creating and solving a system of equations using GPU and CuPy module.

    Attributes:
        H (cupyx.scipy.sparse.csr_matrix): Global matrix of H + Hbc of each element of the grid.
        P (cupy.ndarray): Global P vector.
        C (cupyx.scipy.sparse.csr_matrix): Global C matrix.
        t0 (cupy.ndarray): Vector filled with initial temperature values.
        step (cupy.float32): Simulation step time.
        dim (cupy.int32): Dimensions of H matrix and P vector.
        elements (list[Element]): List of elements in the grid.
    """
    def __init__(self, grid: Grid):
        self.dim: cp.int32 = cp.int32(grid.global_data.nodes_number)
        self.step: cp.float32 = cp.float32(grid.global_data.simulation_step_time)
        self.elements: list[Element] = grid.elements
        self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp)
        self.P: cp.ndarray = self._aggregate_P()
        self.H, self.C = self._aggregate_H_C()
        self.gpu_solve_factorized = gpu_linalg.factorized(self.H + self.C/self.step)

    @measure_time
    def _aggregate_H_C(self) -> tuple[gpu_sparse.csc_matrix, gpu_sparse.csc_matrix]:
        """
        Creates global H and C matrices using GPU.

        Returns:
            tuple[cupyx.scipy.sparse.csr_matrix.csc_matrix, cupyx.scipy.sparse.csr_matrix.csc_matrix]: A tuple containing the global H and C matrices.
        """

        data_H, row_H, col_H = [], [], []
        data_C, row_C, col_C = [], [], []

        for element in self.elements:
            local_H = element.H + element.Hbc
            for i in range(NUM_OF_SHAPE_FUNCTIONS):
                for j in range(NUM_OF_SHAPE_FUNCTIONS):
                    if local_H[i][j] != 0:
                        data_H.append(local_H[i][j])
                        row_H.append(element.node_ids[i] - 1)
                        col_H.append(element.node_ids[j] - 1)
                    if element.C[i][j] != 0:
                        data_C.append(element.C[i][j])
                        row_C.append(element.node_ids[i] - 1)
                        col_C.append(element.node_ids[j] - 1)

        data_H = cp.array(data_H, dtype=cp.float32)
        row_H = cp.array(row_H, dtype=cp.float32)
        col_H = cp.array(col_H, dtype=cp.float32)
        data_C = cp.array(data_C, dtype=cp.float32)
        row_C = cp.array(row_C, dtype=cp.float32)
        col_C = cp.array(col_C, dtype=cp.float32)

        H: gpu_sparse.csc_matrix = gpu_sparse.coo_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim)).tocsc()
        C: gpu_sparse.csc_matrix = gpu_sparse.coo_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim)).tocsc()
        return H, C

    @measure_time
    def _aggregate_P(self) -> cp.ndarray:
        """
        Creates global P vector using GPU.

        Returns:
            cp.ndarray: The global P vector.
        """
        P = np.zeros((self.dim, 1))

        for element in self.elements:
            for i in range(NUM_OF_SHAPE_FUNCTIONS):
                P[element.node_ids[i] - 1] += element.P[i]

        return cp.array(P, dtype=cp.float32)

    @measure_time
    def solve(self) -> cp.ndarray:
        """
        Solves the system of equations using GPU.

        Returns:
            cp.ndarray: The temperature values at each node.
        """
        P = self.P + self.C.dot(self.t0)/self.step
        result: cp.ndarray = self.gpu_solve_factorized(P)
        self.t0 = result
        return result

def simulate(grid: Grid) -> tuple[list[float], list[np.ndarray]]:
    """
    Returns temperatures in element nodes for all time steps.

    Args:
        grid (Grid): The grid containing elements.

    Returns:
        tuple[list[float], list[np.ndarray]]: A tuple containing a list of time steps and a list of temperature values at each node for all time steps.
    """
    times: list[cp.float32] = []
    temperatures: list[cp.ndarray] = []
    config.logger.info("Initializing system of equations.")
    soe = SystemOfEquationsGPU(grid)
    tauk = cp.float32(grid.global_data.simulation_time)
    dtau: cp.float32 = soe.step
    config.logger.info("Calculating temperatures for every timestamp on GPU.")
    while dtau <= tauk:
        result: np.ndarray = soe.solve()
        times.append(dtau)
        temperatures.append(result)
        dtau += soe.step
    if config.use_gpu:
        return cp.asnumpy(times), [temp.get() for temp in temperatures]
    return times, temperatures