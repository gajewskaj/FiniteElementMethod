import common
from common import *
from grid import Element, Grid
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
from abc import ABC, abstractmethod

try:
    import cupy as cp
    import cupyx.scipy.sparse as sparse
    import cupyx.scipy.sparse.linalg as sparse_linalg
    cp_available = cp.cuda.runtime.getDeviceCount() > 0
except ImportError:
    cp_available = False

class SystemOfEquations:
    """
    Class for calculating temperature in each node of the grid by creating and solving a system of equations.

    Attributes:
    H:      Global matrix of H + Hbc of each element of the grid.
    P:      Global P vector.
    C:      Global C matrix.
    t0:     Vector filled with initial temperature values.
    step:   Simulation step time.
    dTau:   Current time - start time.
    dim:    Dimensions of H matrix and P vector.
    """
    def __init__(self, grid: Grid):
        self.dim: int = grid.global_data.nodes_number
        self.step: float = grid.global_data.simulation_step_time
        self.dtau: float = 0.0
        self.elements: list[Element] = grid.elements

    @abstractmethod
    def _aggregate_H_C(self, grid: Grid) -> tuple[scipy.sparse.csr_array, scipy.sparse.csr_array] \
                                            | tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Creates global H and C matrices.
        """
        pass

    @abstractmethod
    def _aggregate_P(self, grid: Grid) -> np.ndarray | cp.ndarray:
        """
        Creates global P vector from local (per element) P vectors.
        """
        pass

    @abstractmethod
    def solve(self) -> np.ndarray:
        """
        Solves system of equations for calculating temperature in each node at any given time.

        H[0] + C[0]/dTau * t1[0] = C[0]/dTau * t0[0] + P[0]
        H[1] + C[1]/dTau * t1[1] = C[1]/dTau * t0[1] + P[1]
        ...
        H[n] + C[n]/dTau * t1[n] = C[n]/dTau * t0[n] + P[n]
        """
        pass

class SystemOfEquationsCPU(SystemOfEquations):
    def __init__(self, grid: Grid):
        super().__init__(self)
        self.t0: np.ndarray = np.full((self.dim, 1), grid.global_data.initial_temp)
        self.P = self._aggregate_P()
        self.H, self.C = self._aggregate_H_C()

    def _aggregate_H_C(self) -> tuple[scipy.sparse.csr_array, scipy.sparse.csr_array]:
        H = scipy.sparse.lil_matrix((self.dim, self.dim))
        C = scipy.sparse.lil_matrix((self.dim, self.dim))

        for element in self.elements:
            local_H: np.ndarray = element.H + element.Hbc
            for i in range (4):
                for j in range(4):
                    H[element.node_ids[i] - 1, element.node_ids[j] - 1] += local_H[i][j]
                    C[element.node_ids[i] - 1, element.node_ids[j] - 1] += element.C[i][j]

        H = H.tocsr()
        C = C.tocsr()

        return H, C
    
    def _aggregate_P(self) -> np.ndarray:
        P = np.zeros((self.dim, 1))
        
        for element in self.elements:
            for i in range(4):
                P[element.node_ids[i] - 1] += element.P[i]
        
        return P
    
    def solve(self) -> np.ndarray:
        self.dtau += self.step
        H = self.H + self.C/self.step
        P = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = scipy.sparse.linalg.spsolve(H, P)
        self.t0 = result.reshape(-1, 1)
        return result

class SystemOfEquationsGPU(SystemOfEquations):
    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp)
        self.P = self._aggregate_P()
        self.H, self.C = self._aggregate_H_C()

    def _aggregate_H_C(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        data_H, row_H, col_H = [], [], []
        data_C, row_C, col_C = [], [], []

        for element in self.elements:
            local_H = element.H + element.Hbc
            for i in range(4):
                for j in range(4):
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

        H = sparse.coo_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim)).tocsr()
        C = sparse.coo_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim)).tocsr()

        return H, C
    
    def _aggregate_P(self):
        P = cp.zeros((self.dim, 1))
        
        for element in self.elements:
            for i in range(0, 4):
                P[element.node_ids[i] - 1] += cp.asarray(element.P[i])
        
        return P
    
    def solve(self) -> np.ndarray:
        self.dtau += self.step
        H = self.H + self.C/self.step
        P = self.P + self.C.dot(self.t0)/self.step
        result: cp.ndarray = sparse_linalg.spsolve(H, P)
        self.t0 = result.reshape(-1, 1)
        return cp.asnumpy(result)
        
def simulate(grid: Grid) -> list[np.ndarray]:
    """
    Returns temperatures in element nodes for all time steps.
    """
    temperatures: list[np.ndarray] = []
    soe: SystemOfEquations
    if cp_available:
        soe = SystemOfEquationsGPU(grid)
    else:
        soe = SystemOfEquationsCPU(grid)
    tau0: int = 0
    tauk: float = grid.global_data.simulation_time
    step: float = grid.global_data.simulation_step_time
    common.logger.info(f"Time        Min temp    Max temp")
    start: float = time.time() # Start measuring time
    while tau0 < tauk:
        result: np.ndarray = soe.solve()
        temperatures.append(result)
        common.logger.info(f"{(soe.dtau):<12}{round(np.min(result), 3):<12}{round(np.max(result), 3):<12}")
        tau0 += step
    end: float = time.time() # Stop measuring time
    common.logger.info(f"Calculated in {end-start} seconds.")
    return temperatures