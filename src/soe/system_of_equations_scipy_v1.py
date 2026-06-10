import numpy as np
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_linalg

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices
from src.soe.system_of_equations import SystemOfEquations

class SystemOfEquationsSciPy(SystemOfEquations):
    def __init__(self, mesh: Mesh, out_mat: OutMatrices) -> None:
        super().__init__(mesh, out_mat)
        self.t0: np.ndarray = np.full((self.dim, 1), mesh.global_data.initial_temp)
        self.prepare_data()
        self.solve_factorized = self.factorize()

    @measure_time
    def prepare_data(self) -> None:
        self.out_mat.to_numpy()
        data_C, row_C, col_C = self.out_mat.C_val_out, self.out_mat.C_row_out, self.out_mat.C_col_out
        self.P = self.out_mat.P_out.reshape(-1, 1)
        data_H = np.concatenate((self.out_mat.H_val_out, self.out_mat.Hbc_val_out))
        row_H = np.concatenate((self.out_mat.H_row_out, self.out_mat.Hbc_row_out))
        col_H = np.concatenate((self.out_mat.H_col_out, self.out_mat.Hbc_col_out))
        H = scipy_sparse.csc_matrix((data_H, (row_H, col_H)), shape=(self.dim, self.dim))
        self.C = scipy_sparse.csc_matrix((data_C, (row_C, col_C)), shape=(self.dim, self.dim))
        
        self.A = H + self.C/self.step
    
    @measure_time
    def factorize(self) -> callable:
        return scipy_linalg.splu(self.A, permc_spec='COL_AMD').solve

    @measure_time
    def solve(self) -> np.ndarray:
        B = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = self.solve_factorized(B)
        self.t0 = result
        return result

    def simulate(self) -> tuple[list[float], list[np.ndarray[float]]]:
        times: list[float] = []
        temperatures: list[np.ndarray] = []
        logger.info("Initializing system of equations.")
        tauk: float = self.mesh.global_data.simulation_time
        dtau: float = self.step
        logger.info("Calculating temperatures for every timestamp.")
        while dtau <= tauk:
            result: np.ndarray = self.solve()
            times.append(dtau)
            temperatures.append(result)
            dtau += self.step
        return times, temperatures