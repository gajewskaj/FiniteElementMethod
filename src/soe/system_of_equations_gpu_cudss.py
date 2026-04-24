import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import numpy as np
import nvmath
from nvmath.bindings import cudss as cudss

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid
from src.soe.system_of_equations import SystemOfEquations


class SystemOfEquationsGPU(SystemOfEquations):
    def __init__(self, grid: Grid):
        super().__init__(grid)

        self.t0: cp.ndarray = cp.full((self.dim, 1), grid.global_data.initial_temp, dtype=cp.float64)
        self.H, self.C, self.P = self._prepare_data()
        self.A = (self.H + self.C / self.step).tocsr()
        self.A.sum_duplicates()
        self.A.sort_indices()

        self.solve_factorized = self._factorize()

    def __del__(self) -> None:
        for attr in ("_b_dn", "_x_dn", "_A_cudss"):
            try:
                matrix = getattr(self, attr)
            except Exception:
                continue
            try:
                cudss.matrix_destroy(matrix)
            except Exception:
                pass

        try:
            if hasattr(self, "_handle") and hasattr(self, "_data"):
                cudss.data_destroy(self._handle, self._data)
        except Exception:
            pass

        try:
            if hasattr(self, "_config"):
                cudss.config_destroy(self._config)
        except Exception:
            pass

        try:
            if hasattr(self, "_handle"):
                cudss.destroy(self._handle)
        except Exception:
            pass

    @measure_time
    def _factorize(self) -> callable:
        try:
            self._handle = cudss.create()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize cuDSS (missing libcudss.so or misconfigured CUDA runtime). "
                "In Docker, install 'libcudss0-cuda-12' in the image."
            ) from exc

        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)

        A = self.A
        indptr = cp.asarray(A.indptr, dtype=cp.int32)
        indices = cp.asarray(A.indices, dtype=cp.int32)
        data = cp.asarray(A.data, dtype=cp.float64)
        A = gpu_sparse.csr_matrix((data, indices, indptr), shape=A.shape)
        A.sum_duplicates()
        A.sort_indices()
        self.A = A

        self._A_cudss = cudss.matrix_create_csr(
            nrows=A.shape[0],
            ncols=A.shape[1],
            nnz=A.nnz,
            row_start=A.indptr.data.ptr,
            row_end=0,
            col_indices=A.indices.data.ptr,
            values=A.data.data.ptr,
            index_type=nvmath.CudaDataType.CUDA_R_32I,
            value_type=nvmath.CudaDataType.CUDA_R_64F,
            mtype=cudss.MatrixType.GENERAL,
            mview=cudss.MatrixViewType.FULL,
            index_base=cudss.IndexBase.ZERO,
        )

        # Create dense (n x 1) wrappers; pointers will be updated per-solve.
        placeholder = cp.zeros((self.dim, 1), dtype=cp.float64)
        self._x_dn = cudss.matrix_create_dn(
            nrows=self.dim,
            ncols=1,
            ld=self.dim,
            values=placeholder.data.ptr,
            value_type=nvmath.CudaDataType.CUDA_R_64F,
            layout=cudss.Layout.COL_MAJOR,
        )
        self._b_dn = cudss.matrix_create_dn(
            nrows=self.dim,
            ncols=1,
            ld=self.dim,
            values=placeholder.data.ptr,
            value_type=nvmath.CudaDataType.CUDA_R_64F,
            layout=cudss.Layout.COL_MAJOR,
        )

        cudss.execute(self._handle, cudss.Phase.ANALYSIS, self._config, self._data, self._A_cudss, self._x_dn, self._b_dn)
        cudss.execute(self._handle, cudss.Phase.FACTORIZATION, self._config, self._data, self._A_cudss, self._x_dn, self._b_dn)

        def solve_factorized(rhs: cp.ndarray) -> cp.ndarray:
            rhs = cp.asarray(rhs, dtype=cp.float64)
            x = cp.zeros_like(rhs)
            cudss.matrix_set_values(self._x_dn, x.data.ptr)
            cudss.matrix_set_values(self._b_dn, rhs.data.ptr)
            cudss.execute(self._handle, cudss.Phase.SOLVE, self._config, self._data, self._A_cudss, self._x_dn, self._b_dn)
            return x

        return solve_factorized

    @measure_time
    def _prepare_data(self) -> tuple[gpu_sparse.csr_matrix, gpu_sparse.csr_matrix, cp.ndarray]:
        H_cpu, C_cpu, P_cpu = super()._prepare_data()

        P = cp.array(P_cpu, dtype=cp.float64)

        H = gpu_sparse.csr_matrix(
            (
                cp.asarray(H_cpu.data, dtype=cp.float64),
                cp.asarray(H_cpu.indices, dtype=cp.int32),
                cp.asarray(H_cpu.indptr, dtype=cp.int32),
            ),
            shape=H_cpu.shape,
        )
        C = gpu_sparse.csr_matrix(
            (
                cp.asarray(C_cpu.data, dtype=cp.float64),
                cp.asarray(C_cpu.indices, dtype=cp.int32),
                cp.asarray(C_cpu.indptr, dtype=cp.int32),
            ),
            shape=C_cpu.shape,
        )

        return H, C, P

    @measure_time
    def solve(self) -> cp.ndarray:
        B = self.P + self.C.dot(self.t0) / self.step

        result: cp.ndarray = self.solve_factorized(B)
        self.t0 = result
        return result


def simulate(grid: Grid) -> tuple[list[float], list[np.ndarray[float]]]:
    times: list[float] = []
    temperatures: list[cp.ndarray] = []
    config.logger.info("Initializing system of equations.")
    soe = SystemOfEquationsGPU(grid)
    tauk = grid.global_data.simulation_time
    dtau = soe.step
    config.logger.info("Calculating temperatures for every timestamp on GPU.")
    while dtau <= tauk:
        result: cp.ndarray = soe.solve()
        times.append(dtau)
        temperatures.append(result)
        dtau += soe.step
    return cp.asnumpy(times), [temp.get() for temp in temperatures]