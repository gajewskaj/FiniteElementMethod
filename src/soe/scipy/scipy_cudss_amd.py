import cupy as cp
import scipy.sparse.linalg as scipy_linalg
from nvmath.bindings import cudss as cudss
import ctypes
import numpy as np

from src.helpers.config import logger
from src.helpers.helpers import measure_time
from src.soe.cudss.cudss_v2 import SystemOfEquationsCuDSSV2

class SystemOfEquationsSciPyAMD(SystemOfEquationsCuDSSV2):
    @measure_time
    def prepare_data(self) -> None:
        super().prepare_data()
        self.P = self.P.get()
        self.C = self.C.get()
        self.t0 = self.t0.get()
 
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
        return scipy_linalg.splu(self.A_reordered, permc_spec='NATURAL').solve

    @measure_time
    def solve(self) -> np.ndarray:
        b = self.P + self.C.dot(self.t0)/self.step
        result: np.ndarray = self.solve_factorized(b[self.perm])[cp.argsort(self.perm)]
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