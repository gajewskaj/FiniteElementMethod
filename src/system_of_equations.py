import time
from typing import Union
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as cpu_sparse
import scipy.sparse.linalg as cpu_linalg

cp_available = False
from . import common
if not common.settings.force_cpu:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as gpu_sparse
        import cupyx.scipy.sparse.linalg as gpu_linalg
        cp_available = cp.cuda.runtime.getDeviceCount() > 0
    except ImportError:
        common.logger.warning(f"Failed to import CuPy. GPU will NOT be used for the further calculations.", exc_info=True)
        # In future add prompt asking if the user wants to proceed in that case
        cp_available = False
    except Exception as e:
        common.logger.error(f"Exception while importing CuPy for GPU calculations. \
If you want to run the calculations on CPU instead, use: '--force-cpu' option.")
        raise RuntimeError from e

from .common import *
from .grid import Element, Grid

class SystemOfEquations(ABC):
    """
    Class for calculating temperature in each node of the grid by creating and solving a system of equations.

    Attributes:
        H (scipy.sparse.csr_matrix or cupyx.scipy.sparse.csr_matrix): Global matrix of H + Hbc of each element of the grid.
        P (np.ndarray or cp.ndarray): Global P vector.
        C (scipy.sparse.csr_matrix or cupyx.scipy.sparse.csr_matrix): Global C matrix.
        t0 (np.ndarray or cp.ndarray): Vector filled with initial temperature values.
        step (float): Simulation step time.
        dtau (float): Current time - start time.
        dim (int): Dimensions of H matrix and P vector.
        elements (list[Element]): List of elements in the grid.
    """
    def __init__(self, grid: Grid):
        self.dim: int = grid.global_data.nodes_number
        self.step: float = grid.global_data.simulation_step_time
        self.dtau: float = 0.0
        self.elements: list[Element] = grid.elements

    @abstractmethod
    def _aggregate_H_C(self) -> Union[tuple[cpu_sparse.csr_matrix, cpu_sparse.csr_matrix],
                                      'tuple[gpu_sparse.csr_matrix, gpu_sparse.csr_matrix]']:
        """
        Creates global H and C matrices.

        Returns:
            tuple: A tuple containing the global H and C matrices.
        """
        pass

    @abstractmethod
    def _aggregate_P(self) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Creates global P vector from local (per element) P vectors.

        Returns:
            np.ndarray or cp.ndarray: The global P vector.
        """
        pass

    @abstractmethod
    def solve(self) -> np.ndarray:
        """
        Solves system of equations for calculating temperature in each node at any given time.

        Returns:
            np.ndarray: The temperature values at each node.
        """
        pass

class SystemOfEquationsCPU(SystemOfEquations):
    """
    Class for calculating temperature in each node of the grid by creating and solving a system of equations using CPU and NumPy, SciPy modules.

    Attributes:
        H (scipy.sparse.csr_matrix): Global matrix of H + Hbc of each element of the grid.
        P (np.ndarray): Global P vector.
        C (scipy.sparse.csr_matrix): Global C matrix.
        t0 (np.ndarray): Vector filled with initial temperature values.
        step (float): Simulation step time.
        dtau (float): Current time - start time.
        dim (int): Dimensions of H matrix and P vector.
        elements (list[Element]): List of elements in the grid.
    """
    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.t0: np.ndarray = np.full((self.dim, 1), grid.global_data.initial_temp)
        self.P = self._aggregate_P()
        self.H, self.C = self._aggregate_H_C()

    def _aggregate_H_C(self) -> tuple[cpu_sparse.csr_matrix, cpu_sparse.csr_matrix]:
        """
        Creates global H and C matrices using CPU.

        Returns:
            tuple: A tuple containing the global H and C matrices.
        """
        H = cpu_sparse.lil_matrix((self.dim, self.dim))
        C = cpu_sparse.lil_matrix((self.dim, self.dim))

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
        """
        Creates global P vector using CPU.

        Returns:
            np.ndarray: The global P vector.
        """
        P = np.zeros((self.dim, 1))

        for element in self.elements:
            for i in range(4):
                P[element.node_ids[i] - 1] += element.P[i]

        return P

    def solve(self) -> np.ndarray:
        """
        Solves system of equations using CPU.

        Returns:
            np.ndarray: The temperature values at each node.
        """
        self.dtau += self.step
        H = self.H + self.C/self.step
        P = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = cpu_linalg.spsolve(H, P)
        self.t0 = result.reshape(-1, 1)
        return result

class SystemOfEquationsGPU(SystemOfEquations):
    """
    Class for calculating temperature in each node of the grid by creating and solving a system of equations using GPU and CuPy module.

    Attributes:
        H (cupyx.scipy.sparse.csr_matrix): Global matrix of H + Hbc of each element of the grid.
        P (cp.ndarray): Global P vector.
        C (cupyx.scipy.sparse.csr_matrix): Global C matrix.
        t0 (cp.ndarray): Vector filled with initial temperature values.
        step (float): Simulation step time.
        dtau (float): Current time - start time.
        dim (int): Dimensions of H matrix and P vector.
        elements (list[Element]): List of elements in the grid.
    """
    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp)
        self.P = self._aggregate_P()
        self.H, self.C = self._aggregate_H_C()

    def _aggregate_H_C(self) -> tuple['gpu_sparse.csr_matrix', 'gpu_sparse.csr_matrix']:
        """
        Creates global H and C matrices using GPU.

        Returns:
            tuple: A tuple containing the global H and C matrices.
        """
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

        H = gpu_sparse.coo_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim)).tocsr()
        C = gpu_sparse.coo_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim)).tocsr()

        return H, C

    def _aggregate_P(self) -> 'cp.ndarray':
        """
        Creates global P vector using GPU.

        Returns:
            cp.ndarray: The global P vector.
        """
        P = cp.zeros((self.dim, 1))

        for element in self.elements:
            for i in range(0, 4):
                P[element.node_ids[i] - 1] += cp.asarray(element.P[i])

        return P

    def solve(self) -> np.ndarray:
        """
        Solves system of equations using GPU.

        Returns:
            np.ndarray: The temperature values at each node.
        """
        self.dtau += self.step
        H = self.H + self.C/self.step
        P = self.P + self.C.dot(self.t0)/self.step
        result: cp.ndarray = gpu_linalg.spsolve(H, P)
        self.t0 = result.reshape(-1, 1)
        return cp.asnumpy(result)

def simulate(grid: Grid) -> list[np.ndarray]:
    """
    Returns temperatures in element nodes for all time steps.

    Args:
        grid (Grid): The grid containing elements.

    Returns:
        list[np.ndarray]: List of temperature values at each node for all time steps.
    """
    temperatures: list[np.ndarray] = []
    soe: SystemOfEquations
    if cp_available:
        common.logger.info("Starting calculations on GPU.")
        soe = SystemOfEquationsGPU(grid)
    else:
        common.logger.info("Starting calculations on CPU.")
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