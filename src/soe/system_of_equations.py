from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as cpu_sparse
from src.grid.grid import Grid

class SystemOfEquations(ABC):
    @abstractmethod
    def __init__(self, grid: Grid):
        self.dim: int = len(grid.nodes_id)
        self.step: float = grid.global_data.simulation_step_time
        self.grid = grid

    @abstractmethod
    def _factorize(self) -> callable:
        pass

    @abstractmethod
    def _prepare_data(self) -> tuple:
        data_C, row_C, col_C = self.grid.C_val, self.grid.C_row, self.grid.C_col
        P = self.grid.P.reshape(-1, 1)
        data_H = np.concatenate((self.grid.H_val, self.grid.Hbc_val))
        row_H = np.concatenate((self.grid.H_row, self.grid.Hbc_row))
        col_H = np.concatenate((self.grid.H_col, self.grid.Hbc_col))
        H = cpu_sparse.csr_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim))
        C = cpu_sparse.csr_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim))
        return H, C, P

    @abstractmethod
    def solve(self) -> np.ndarray:
        pass