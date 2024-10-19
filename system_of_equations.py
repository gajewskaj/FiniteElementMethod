import common
from common import *
from grid import Grid
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import cupy as cp

class SystemOfEquations:
    """
    Class for calculating temperature in each node of the grid by creating and solving a system of equations.

    H:      global matrix of H + Hbc od each element of the grid
    P:      global P vector
    C:      global C matrix
    t0:     vector filled with value of initail temperature
    step:   simulation step time
    dTau:   current time - start time
    dim:    dimensions of H matrix and P vector
    """
    def __init__(self, grid: Grid):
        self.dim: int = grid.global_data.nodes_number
        self.t0: np.ndarray = np.full((self.dim, 1), grid.global_data.initial_temp)
        self.step: float = grid.global_data.simulation_step_time
        self.dtau: float = 0.0
        self.P = self._aggregate_p(grid)
        self.H, self.C = self._aggregate_h_c(grid)

    def _aggregate_h_c(self, grid: Grid) -> tuple[scipy.sparse.csr_matrix]:
        """
        Creates global H and C matrices.
        """
        H = scipy.sparse.lil_matrix((self.dim, self.dim))
        C = scipy.sparse.lil_matrix((self.dim, self.dim))
        for element in grid.elements:
            local_h = element.H + element.Hbc
            for j in range(0, 4):
                for i in range(0, 4):
                    H[element.node_ids[j] - 1, element.node_ids[i] - 1] += local_h[j][i]
                    C[element.node_ids[j] - 1, element.node_ids[i] - 1] += element.C[j][i]
        H = H.tocsr()
        C = C.tocsr()
        return H, C

    def _aggregate_p(self, grid: Grid) -> np.ndarray:
        """
        Creates global P vector from local (per element) P vectors.
        """
        P: np.ndarray = np.zeros((self.dim, 1))
        for element in grid.elements:
            for i in range(0, 4):
                P[element.node_ids[i] - 1] += element.P[i]
        return P
        #common.main_logger.debug(f"Global P:\n{P}")

    def solve(self) -> np.ndarray:
        """
        Solves system of equations for calculating temperature in each node at any given time.

        H[0] + C[0]/dTau * t1[0] = C[0]/dTau * t0[0] + P[0]
        H[1] + C[1]/dTau * t1[1] = C[1]/dTau * t0[1] + P[1]
        ...
        H[n] + C[n]/dTau * t1[n] = C[n]/dTau * t0[n] + P[n]
        """
        self.dtau += self.step
        H = self.H + self.C/self.step
        # H = self.H + self.C/(self.step)
        P = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = scipy.sparse.linalg.spsolve(H, P)
        # result: np.ndarray = np.linalg.solve(H, P)
        self.t0 = result
        self.t0 = result.reshape(-1, 1)
        return result

def simulate(grid: Grid) -> list[np.ndarray]:
    """
    Returns temeratures in element nodes for all time steps.
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
        tau0+=step
    return temperatures