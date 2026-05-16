from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as scipy_sparse
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices

class SystemOfEquations(ABC):
    @abstractmethod
    def __init__(self, mesh: Mesh, out_mat: OutMatrices) -> None:
        self.dim: int = len(mesh.nodes_id)
        self.step: float = mesh.global_data.simulation_step_time
        self.mesh = mesh
        self.out_mat = out_mat

    @abstractmethod
    def prepare_data(self) -> tuple:
        self.out_mat.to_numpy()
        data_C, row_C, col_C = self.out_mat.C_val_out, self.out_mat.C_row_out, self.out_mat.C_col_out
        P = self.out_mat.P_out.reshape(-1, 1)
        data_H = np.concatenate((self.out_mat.H_val_out, self.out_mat.Hbc_val_out))
        row_H = np.concatenate((self.out_mat.H_row_out, self.out_mat.Hbc_row_out))
        col_H = np.concatenate((self.out_mat.H_col_out, self.out_mat.Hbc_col_out))
        H = scipy_sparse.csr_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim))
        C = scipy_sparse.csr_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim))
        return H, C, P

    @abstractmethod
    def factorize(self) -> callable:
        pass

    @abstractmethod
    def solve(self) -> np.ndarray:
        pass

    @abstractmethod
    def simulate(self) -> tuple[list[float], list[np.ndarray[float]]]:
        pass