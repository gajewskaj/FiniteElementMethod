import common
from common import *
from grid import Grid
import numpy as np
from numpy import linalg

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
        self.H: np.ndarray = np.zeros((self.dim, self.dim))
        self.P: np.ndarray = np.zeros((self.dim, 1))
        self.C: np.ndarray = np.zeros((self.dim, self.dim))
        self._aggregate_h_c(grid)
        self._aggreagte_p(grid)

    def _aggregate_h_c(self, grid: Grid) -> None:
        """
        Creates global H and C matrices.
        """
        for element in grid.elements:
            local_h = element.H + element.Hbc
            for j in range(0, 4):
                for i in range(0, 4):
                    self.H[element.node_ids[j] - 1][element.node_ids[i] - 1] += local_h[j][i]
                    self.C[element.node_ids[j] - 1][element.node_ids[i] - 1] += element.C[j][i]
        #common.main_logger.debug(f"Global H:\n{self.H}\nGlobal C:{self.C}")

    def _aggreagte_p(self, grid: Grid) -> None:
        """
        Creates global P vector from local (per element) P vectors.
        """
        for element in grid.elements:
            for i in range(0, 4):
                self.P[element.node_ids[i] - 1] += element.P[i]
        #common.main_logger.debug(f"Global P:\n{self.P}")

    def solve(self) -> np.ndarray:
        """
        Solves system of equations for calculating temperature in each node at any given time.

        H[0] + C[0]/dTau * t1[0] = C[0]/dTau * t0[0] + P[0]
        H[1] + C[1]/dTau * t1[1] = C[1]/dTau * t0[1] + P[1]
        ...
        H[n] + C[n]/dTau * t1[n] = C[n]/dTau * t0[n] + P[n]
        """
        self.dtau += self.step
        H = self.H + self.C/(self.step)
        P = self.P + np.matmul(self.C/(self.step), self.t0)
        result: np.ndarray = linalg.solve(H, P)
        self.t0 = result
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
    common.main_logger.debug(f"Time        Min temp    Max temp")
    while tau0 < tauk:
        result: np.ndarray = soe.solve()
        temperatures.append(result)
        common.main_logger.debug(f"{(soe.dtau):<12}{round(min(result)[0], 3):<12}{round(max(result)[0], 3):<12}")
        tau0+=step
    return temperatures