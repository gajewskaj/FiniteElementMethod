import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import nvmath
from nvmath.bindings import cudss as cudss
import nvtx
import ctypes

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.soe.system_of_equations_cupy import SystemOfEquationsCuPy

class SystemOfEquationsCuDSS(SystemOfEquationsCuPy):
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

        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)
        # self.A = cupy_sparse.tril(self.A).tocsr()

        # Setting reordering algorithm to AMD
        # reordering_alg_c = ctypes.c_int32(cudss.AlgType.ALG_3.value)
        # reordering_alg_ptr = ctypes.addressof(reordering_alg_c)

        # cudss.config_set(
        #     self._config,
        #     cudss.ConfigParam.REORDERING_ALG,
        #     reordering_alg_ptr,
        #     ctypes.sizeof(reordering_alg_c)
        # )

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
            mtype=cudss.MatrixType.GENERAL,
            mview=cudss.MatrixViewType.FULL,
            # mview=cudss.MatrixViewType.LOWER,
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