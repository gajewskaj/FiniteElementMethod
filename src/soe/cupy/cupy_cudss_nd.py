import cupy as cp
import cupyx.scipy.sparse.linalg as cupy_linalg
from nvmath.bindings import cudss as cudss
import ctypes
import numpy as np
import torch

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.soe.cudss.cudss_v4 import SystemOfEquationsCuDSSV4

class SystemOfEquationsCuPyND(SystemOfEquationsCuDSSV4):
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

        self.A_reordered = self.A[self.perm][:, self.perm]
        return cupy_linalg.splu(self.A_reordered, permc_spec='NATURAL').solve

    @measure_time
    def solve(self) -> cp.ndarray:
        b = self.P + self.C.dot(self.t0)/self.step

        torch.cuda.nvtx.range_push("Solve")
        torch.cuda.cudart().cudaProfilerStart()
        result: cp.ndarray = self.solve_factorized(b[self.perm])[cp.argsort(self.perm)]
        cp.cuda.get_current_stream().synchronize() # Only for profiling
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()

        self.t0 = result
        return result