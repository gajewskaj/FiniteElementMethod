import common
from common import *
from grid import Grid
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

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
        if cp_available:
            self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp)
        else:
            self.t0: np.ndarray = np.full((self.dim, 1), grid.global_data.initial_temp)
        self.P = self._aggregate_p(grid)
        self.H, self.C = self._aggregate_h_c(grid)

    def _aggregate_h_c(self, grid: Grid):
        """
        Creates global H and C matrices.
        """
        if cp_available:
            data_H, row_H, col_H = [], [], []
            data_C, row_C, col_C = [], [], []
        else:
            H = scipy.sparse.lil_matrix((self.dim, self.dim))
            C = scipy.sparse.lil_matrix((self.dim, self.dim))
        
        for element in grid.elements:
            local_h = element.H + element.Hbc
            for j in range(0, 4):
                for i in range(0, 4):
                    if cp_available:
                        data_H.append(local_h[j][i])
                        row_H.append(element.node_ids[j] - 1)
                        col_H.append(element.node_ids[i] - 1)
                        data_C.append(element.C[j][i])
                        row_C.append(element.node_ids[j] - 1)
                        col_C.append(element.node_ids[i] - 1)
                    else:
                        H[element.node_ids[j] - 1, element.node_ids[i] - 1] += local_h[j][i]
                        C[element.node_ids[j] - 1, element.node_ids[i] - 1] += element.C[j][i]
        
        if cp_available:
            data_H = cp.array(data_H)
            row_H = cp.array(row_H)
            col_H = cp.array(col_H)
            data_C = cp.array(data_C)
            row_C = cp.array(row_C)
            col_C = cp.array(col_C)
            H = sparse.coo_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim)).tocsr()
            C = sparse.coo_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim)).tocsr()
        else:
            H = H.tocsr()
            C = C.tocsr()
        
        return H, C

    def _aggregate_p(self, grid: Grid):
        """
        Creates global P vector from local (per element) P vectors.
        """
        if cp_available:
            P = cp.zeros((self.dim, 1))
        else:
            P = np.zeros((self.dim, 1))
        
        for element in grid.elements:
            for i in range(0, 4):
                if cp_available:
                    P[element.node_ids[i] - 1] += cp.asarray(element.P[i])
                else:
                    P[element.node_ids[i] - 1] += element.P[i]
        
        return P

    def solve(self) -> np.ndarray:
        """
        Solves system of equations for calculating temperature in each node at any given time.

        H[0] + C[0]/dTau * t1[0] = C[0]/dTau * t0[0] + P[0]
        H[1] + C[1]/dTau * t1[1] = C[1]/dTau * t0[1] + P[1]
        ...
        H[n] + C[n]/dTau * t1[n] = C[n]/dTau * t0[n] + P[n]
        """
        self.dtau += self.step
        if cp_available:
            H = self.H + self.C/self.step
            P = self.P + self.C.dot(self.t0)/self.step
            result: cp.ndarray = sparse_linalg.spsolve(H, P)
            self.t0 = result
            self.t0 = result.reshape(-1, 1)
            return cp.asnumpy(result)
        else:
            H = self.H + self.C/self.step
            P = self.P + self.C.dot(self.t0)/self.step
            result: np.ndarray = scipy.sparse.linalg.spsolve(H, P)
            self.t0 = result
            self.t0 = result.reshape(-1, 1)
            return result

def simulate(grid: Grid) -> list[np.ndarray]:
    """
    Returns temperatures in element nodes for all time steps.
    """
    temperatures: list[np.ndarray] = []
    soe = SystemOfEquations(grid)
    tau0: int = 0
    tauk: float = grid.global_data.simulation_time
    step: float = grid.global_data.simulation_step_time
    common.logger.info(f"Time        Min temp    Max temp")
    while tau0 < tauk:
        result: np.ndarray = soe.solve()
        temperatures.append(result)
        common.logger.info(f"{(soe.dtau):<12}{round(np.min(result), 3):<12}{round(np.max(result), 3):<12}")
        tau0 += step
    return temperatures