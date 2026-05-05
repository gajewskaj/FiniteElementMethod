import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import cupyx.scipy.sparse.linalg as cupy_linalg
import numpy as np

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices
from src.soe.system_of_equations import SystemOfEquations

class SystemOfEquationsCuPy(SystemOfEquations):
    def __init__(self, mesh: Mesh, out_mat: OutMatrices) -> None:
        super().__init__(mesh, out_mat)
        self.t0: cp.ndarray = cp.full((self.dim, 1), mesh.global_data.initial_temp, dtype=cp.float64)
        self.H, self.C, self.P = self._prepare_data()
        self.A = (self.H + self.C/self.step)
        self.solve_factorized = self._factorize()

    @measure_time
    def _prepare_data(self) -> tuple[cupy_sparse.csr_matrix, cupy_sparse.csr_matrix, cp.ndarray]:
        self.out_mat.to_cupy()
        P = self.out_mat.P_out.reshape(-1, 1)
        H_val = cp.concatenate((self.out_mat.H_val_out, self.out_mat.Hbc_val_out))
        H_row = cp.concatenate((self.out_mat.H_row_out, self.out_mat.Hbc_row_out))
        H_col = cp.concatenate((self.out_mat.H_col_out, self.out_mat.Hbc_col_out))
        H = cupy_sparse.csr_matrix((H_val,(H_row, H_col)), shape=(self.dim, self.dim))
        C = cupy_sparse.csr_matrix(
            (
                self.out_mat.C_val_out,
                (self.out_mat.C_row_out, self.out_mat.C_col_out)
            ),
            shape=(self.dim, self.dim)
        )

        return H, C, P

    @measure_time
    def _factorize(self) -> callable:
        return cupy_linalg.factorized(self.A)

    @measure_time
    def solve(self) -> cp.ndarray:
        b = self.P + self.C.dot(self.t0)/self.step
        result: cp.ndarray = self.solve_factorized(b)
        self.t0 = result
        return result

    def simulate(self) -> tuple[list[float], list[np.ndarray[float]]]:
        times: list[float] = []
        temperatures: list[cp.ndarray] = []
        logger.info("Initializing system of equations.")
        tauk = self.mesh.global_data.simulation_time
        dtau = self.step
        logger.info("Calculating temperatures for every timestamp.")
        while dtau <= tauk:
            result: cp.ndarray = self.solve()
            times.append(dtau)
            temperatures.append(result)
            dtau += self.step
        return times, [temp.get() for temp in temperatures]