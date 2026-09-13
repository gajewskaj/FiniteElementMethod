import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import nvmath
from nvmath.bindings import cudss as cudss
import numpy as np
import ctypes
import os
import torch

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices
from src.soe.system_of_equations import SystemOfEquations

class SystemOfEquationsCuDSS(SystemOfEquations):
    def __init__(self, mesh: Mesh, out_mat: OutMatrices, reordering_alg: str) -> None:
        # reordering_alg is redundant - added for integrity, to be modified in future
        super().__init__(mesh, out_mat)
        self.t0: cp.ndarray = cp.full((self.dim, 1), mesh.global_data.initial_temp, dtype=cp.float64)
        self.prepare_data()
        self.solve_factorized = self.factorize()

    @measure_time
    def factorize(self) -> callable:
        self._analyze()
        self._factorize()

        def solve_factorized(b: cp.ndarray) -> cp.ndarray:
            b = cp.asarray(b, dtype=cp.float64)
            x = cp.zeros_like(b)
            cudss.matrix_set_values(self.x_cudss, x.data.ptr)
            cudss.matrix_set_values(self.b_cudss, b.data.ptr)
            cudss.execute(self._handle, cudss.Phase.SOLVE, self._config, self._data, self.A_cudss, self.x_cudss, self.b_cudss)
            return x

        return solve_factorized

    def prepare_data(self) -> None:
        self.out_mat.to_cupy()
        self.P = self.out_mat.P_out.reshape(-1, 1)
        H_val = cp.concatenate((self.out_mat.H_val_out, self.out_mat.Hbc_val_out))
        H_row = cp.concatenate((self.out_mat.H_row_out, self.out_mat.Hbc_row_out))
        H_col = cp.concatenate((self.out_mat.H_col_out, self.out_mat.Hbc_col_out))
        H = cupy_sparse.csr_matrix((H_val,(H_row, H_col)), shape=(self.dim, self.dim))
        self.C = cupy_sparse.csr_matrix(
            (
                self.out_mat.C_val_out,
                (self.out_mat.C_row_out, self.out_mat.C_col_out)
            ),
            shape=(self.dim, self.dim)
        )
        self.A = (H + self.C/self.step)

        self._handle = cudss.create()
        
    @measure_time
    def _analyze(self) -> None:
        torch.cuda.nvtx.range_push("Analyze")
        cudss.execute(self._handle, cudss.Phase.ANALYSIS, self._config, self._data, self.A_cudss, self.x_cudss, self.b_cudss)
        cp.cuda.get_current_stream().synchronize() # Only for profiling
        torch.cuda.nvtx.range_pop()

    @measure_time
    def _factorize(self) -> None:
        torch.cuda.nvtx.range_push("Factorize")
        cudss.execute(self._handle, cudss.Phase.FACTORIZATION, self._config, self._data, self.A_cudss, self.x_cudss, self.b_cudss)
        cp.cuda.get_current_stream().synchronize() # Only for profiling
        torch.cuda.nvtx.range_pop()

    @measure_time
    def solve(self) -> cp.ndarray:
        b = self.P + self.C.dot(self.t0)/self.step

        torch.cuda.nvtx.range_push("Solve")
        torch.cuda.cudart().cudaProfilerStart()
        result: cp.ndarray = self.solve_factorized(b)
        cp.cuda.get_current_stream().synchronize() # Only for profiling
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()

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

    def __del__(self) -> None:
        for attr in ("b_cudss", "x_cudss", "A_cudss"):
            try: matrix = getattr(self, attr)
            except Exception as e: logger.warning(e)
            try: cudss.matrix_destroy(matrix)
            except Exception as e: logger.warning(e)

        try:
            if hasattr(self, "_handle") and hasattr(self, "_data"):
                cudss.data_destroy(self._handle, self._data)
        except Exception as e: logger.warning(e)

        try:
            if hasattr(self, "_config"):
                cudss.config_destroy(self._config)
        except Exception as e: logger.warning(e)

        try:
            if hasattr(self, "_handle"):
                cudss.destroy(self._handle)
        except Exception as e: logger.warning(e)