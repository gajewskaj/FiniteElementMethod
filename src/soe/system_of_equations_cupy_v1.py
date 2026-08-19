import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import cupyx.scipy.sparse.linalg as cupy_linalg
import numpy as np
import nvtx
import ctypes

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices
from src.soe.system_of_equations import SystemOfEquations

class SystemOfEquationsCuPy(SystemOfEquations):
    def __init__(self, mesh: Mesh, out_mat: OutMatrices) -> None:
        super().__init__(mesh, out_mat)
        self.cudart = ctypes.CDLL('libcudart.so')
        self.t0: cp.ndarray = cp.full((self.dim, 1), mesh.global_data.initial_temp, dtype=cp.float64)
        self.prepare_data()
        self.solve_factorized = self.factorize()

    @measure_time
    def prepare_data(self) -> None:
        self.out_mat.to_cupy()
        self.P = self.out_mat.P_out.reshape(-1, 1)
        H_val = cp.concatenate((self.out_mat.H_val_out, self.out_mat.Hbc_val_out))
        H_row = cp.concatenate((self.out_mat.H_row_out, self.out_mat.Hbc_row_out))
        H_col = cp.concatenate((self.out_mat.H_col_out, self.out_mat.Hbc_col_out))
        H = cupy_sparse.csc_matrix((H_val,(H_row, H_col)), shape=(self.dim, self.dim))
        self.C = cupy_sparse.csc_matrix(
            (
                self.out_mat.C_val_out,
                (self.out_mat.C_row_out, self.out_mat.C_col_out)
            ),
            shape=(self.dim, self.dim)
        )
        self.A = (H + self.C/self.step)

    @measure_time
    def factorize(self) -> callable:
        with nvtx.annotate("cupy_factorization", color="red"):
            lu = cupy_linalg.splu(self.A, permc_spec='COLAMD')
            print(lu.L.nnz + lu.U.nnz)
            return lu.solve

    @measure_time
    def solve(self) -> cp.ndarray:
        b = self.P + self.C.dot(self.t0)/self.step

        with nvtx.annotate("solve", color="green"):
            # self.cudart.cudaProfilerStart()
            result: cp.ndarray = self.solve_factorized(b)
            cp.cuda.get_current_stream().synchronize() # Only for profiling
            # self.cudart.cudaProfilerStop()

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