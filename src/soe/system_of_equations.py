from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as cpu_sparse
from src.grid.grid import Grid

class SystemOfEquations(ABC):
    """
    Abstract class designed for calculating temperature in each node of the grid by creating and solving a system of equations.

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
    @abstractmethod
    def __init__(self, grid: Grid):
        pass

    @abstractmethod
    def _aggregate_H_C(self) -> tuple[cpu_sparse.csr_matrix, cpu_sparse.csr_matrix]:
        """
        Creates global H and C matrices.

        Returns:
            tuple: A tuple containing the global H and C matrices.
        """
        pass

    @abstractmethod
    def _aggregate_P(self) -> np.ndarray:
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