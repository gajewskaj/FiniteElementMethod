import numpy as np
import scipy.sparse as cpu_sparse
import scipy.sparse.linalg as cpu_linalg

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid, Element
from src.soe.system_of_equations import SystemOfEquations
from src.uel.universal_element import NUM_OF_SHAPE_FUNCTIONS

class SystemOfEquationsCPU(SystemOfEquations):
    """
    Class for calculating temperature in each node of the grid by creating and solving a system of equations using CPU and NumPy, SciPy modules.

    Attributes:
        H (scipy.sparse.csr_matrix): Global matrix of H + Hbc of each element of the grid.
        P (numpy.ndarray): Global P vector.
        C (scipy.sparse.csr_matrix): Global C matrix.
        t0 (numpy.ndarray): Vector filled with initial temperature values.
        step (float): Simulation step time.
        dim (int): Dimensions of H matrix and P vector.
        elements (list[Element]): List of elements in the grid.
    """
    def __init__(self, grid: Grid):
        self.dim: int = grid.global_data.nodes_number
        self.step: float = grid.global_data.simulation_step_time
        self.elements: list[Element] = grid.elements
        self.t0: np.ndarray = np.full((self.dim, 1), grid.global_data.initial_temp)
        self.P: np.ndarray = self._aggregate_P()
        self.H, self.C = self._aggregate_H_C()
        self.cpu_solve_factorized = cpu_linalg.factorized(self.H + self.C/self.step)

    @measure_time
    def _aggregate_H_C(self) -> tuple[cpu_sparse.csc_matrix, cpu_sparse.csc_matrix]:
        """
        Creates global H and C matrices using CPU.

        Returns:
            tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix]: A tuple containing the global H and C matrices.
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

        H: cpu_sparse.csc_matrix = cpu_sparse.coo_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim)).tocsc()
        C: cpu_sparse.csc_matrix = cpu_sparse.coo_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim)).tocsc()

        return H, C

    @measure_time
    def _aggregate_P(self) -> np.ndarray:
        """
        Creates global P vector using CPU.

        Returns:
            np.ndarray: The global P vector.
        """
        P = np.zeros((self.dim, 1))

        for element in self.elements:
            for i in range(NUM_OF_SHAPE_FUNCTIONS):
                P[element.node_ids[i] - 1] += element.P[i]

        return P

    @measure_time
    def solve(self) -> np.ndarray:
        """
        Solves the system of equations using CPU.

        Returns:
            np.ndarray: The temperature values at each node.
        """
        P = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = self.cpu_solve_factorized(P)
        self.t0 = result
        return result

def simulate(grid: Grid) -> tuple[list[float], list[np.ndarray[float]]]:
    """
    Simulates the temperature distribution over time.

    Args:
        grid (Grid): The grid containing elements.

    Returns:
        tuple[list[float], list[np.ndarray]]: A tuple containing a list of time steps and a list of temperature values at each node for all time steps.
    """
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