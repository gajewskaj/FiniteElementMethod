import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import nvmath
from nvmath.bindings import cudss as cudss
import ctypes
import os

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.soe.cudss.system_of_equations_cudss import SystemOfEquationsCuDSS

class SystemOfEquationsCuDSSV4(SystemOfEquationsCuDSS):
    @measure_time
    def prepare_data(self) -> None:
        super().prepare_data()

        # Threading layer - for analysis
        lib_path = "/usr/lib/x86_64-linux-gnu/libcudss/12/libcudss_mtlayer_gomp.so.0"
        cudss.set_threading_layer(self._handle, lib_path)

        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)

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