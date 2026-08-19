import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_linalg
import nvmath
from nvmath.bindings import cudss as cudss
import nvtx
import ctypes
import numpy as np
import matplotlib.pyplot as plt

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.soe.system_of_equations_cupy_v1 import SystemOfEquationsCuPy

class SystemOfEquationsCuDSSSciPy(SystemOfEquationsCuPy):
    @measure_time
    def factorize(self) -> callable:
        n = self.A.shape[0]

        self.perm = cp.empty(n, dtype=cp.int32)
        size_written = np.zeros(1, dtype=np.int64)
        size_written_ptr = size_written.ctypes.data

        self._analyze()
        cudss.data_get(
            self._handle,
            self._data,
            cudss.DataParam.PERM_REORDER_ROW,
            self.perm.data.ptr,
            self.perm.nbytes,
            size_written_ptr
        )
        self.A = self.A.get()
        self.perm = self.perm.get()
        self.A_reordered = self.A[self.perm][:, self.perm]
        print(self.A_reordered.nnz)
        plt.figure(figsize=(6, 6))
        plt.spy(self.A_reordered, markersize=2)

        plt.savefig("A_perm_cudss_amd.png", dpi=1200, bbox_inches="tight")
        plt.close()
        return scipy_linalg.splu(self.A_reordered, permc_spec='NATURAL').solve

    @measure_time
    def solve(self) -> cp.ndarray:
        b = self.P + self.C.dot(self.t0)/self.step

        with nvtx.annotate("solve", color="green"):
            result: cp.ndarray = self.solve_factorized(b[self.perm])[cp.argsort(self.perm)]
        cp.cuda.get_current_stream().synchronize() # Only for profiling

        self.t0 = result
        return result

    @measure_time
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

        lib_path = "/usr/lib/x86_64-linux-gnu/libcudss/12/libcudss_mtlayer_gomp.so.0.7.1"
        cudss.set_threading_layer(self._handle, lib_path)

        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)

        # Setting reordering algorithm to AMD
        reordering_alg_c = ctypes.c_int32(cudss.AlgType.ALG_3.value)
        reordering_alg_ptr = ctypes.addressof(reordering_alg_c)

        cudss.config_set(
            self._config,
            cudss.ConfigParam.REORDERING_ALG,
            reordering_alg_ptr,
            ctypes.sizeof(reordering_alg_c)
        )

        self.A_cudss = cudss.matrix_create_csr(
            nrows=self.A.shape[0],
            ncols=self.A.shape[1],
            nnz=self.A.nnz,
            row_start=self.A.indptr.data.ptr,
            row_end=0,
            col_indices=self.A.indices.data.ptr,
            values=self.A.data.data.ptr,
            index_type=nvmath.CudaDataType.CUDA_R_32I,
            value_type=nvmath.CudaDataType.CUDA_R_64F,
            mtype=cudss.MatrixType.SPD,
            mview=cudss.MatrixViewType.FULL,
            index_base=cudss.IndexBase.ZERO,
        )
        placeholder = cp.zeros((self.dim, 1), dtype=cp.float64)
        self.x_cudss = cudss.matrix_create_dn(
            nrows=self.dim,
            ncols=1,
            ld=self.dim,
            values=placeholder.data.ptr,
            value_type=nvmath.CudaDataType.CUDA_R_64F,
            layout=cudss.Layout.COL_MAJOR,
        )
        self.b_cudss = cudss.matrix_create_dn(
            nrows=self.dim,
            ncols=1,
            ld=self.dim,
            values=placeholder.data.ptr,
            value_type=nvmath.CudaDataType.CUDA_R_64F,
            layout=cudss.Layout.COL_MAJOR,
        )
        self.P = self.P.get()
        self.C = self.C.get()
        self.t0 = self.t0.get()

    @measure_time
    def _analyze(self) -> None:
        with nvtx.annotate("cudss_analysis", color="yellow"):
            cudss.execute(self._handle, cudss.Phase.ANALYSIS, self._config, self._data, self.A_cudss, self.x_cudss, self.b_cudss)
            cp.cuda.get_current_stream().synchronize() # Only for profiling

    @measure_time
    def _factorize(self) -> None:
        with nvtx.annotate("cudss_factorization", color="red"):
            cudss.execute(self._handle, cudss.Phase.FACTORIZATION, self._config, self._data, self.A_cudss, self.x_cudss, self.b_cudss)
            cp.cuda.get_current_stream().synchronize() # Only for profiling

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