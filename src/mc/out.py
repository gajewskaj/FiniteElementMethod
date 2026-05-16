import numpy as np
import cupy as cp

from src.helpers.config import Settings
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh

class OutMatrices():
    def __init__(self, mesh: Mesh):
        matrix_size = len(mesh.elements_id) * Settings.MatricesCalculation.DOF * Settings.MatricesCalculation.DOF
        self.H_val_out = np.zeros(matrix_size, dtype=np.float64)
        self.H_row_out = np.zeros(matrix_size, dtype=np.int32)
        self.H_col_out = np.zeros(matrix_size, dtype=np.int32)
        self.C_val_out = np.zeros(matrix_size, dtype=np.float64)
        self.C_row_out = np.zeros(matrix_size, dtype=np.int32)
        self.C_col_out = np.zeros(matrix_size, dtype=np.int32)
        self.Hbc_val_out = np.zeros(matrix_size, dtype=np.float64)
        self.Hbc_row_out = np.zeros(matrix_size, dtype=np.int32)
        self.Hbc_col_out = np.zeros(matrix_size, dtype=np.int32)
        self.P_out = np.zeros(len(mesh.nodes_id), dtype=np.float64)

    def to_numpy(self) -> None:
        if isinstance(self.P_out, np.ndarray): return
        self.H_val_out = cp.asnumpy(self.H_val_out)
        self.H_row_out = cp.asnumpy(self.H_row_out)
        self.H_col_out = cp.asnumpy(self.H_col_out)
        self.C_val_out = cp.asnumpy(self.C_val_out)
        self.C_row_out = cp.asnumpy(self.C_row_out)
        self.C_col_out = cp.asnumpy(self.C_col_out)
        self.Hbc_val_out = cp.asnumpy(self.Hbc_val_out)
        self.Hbc_row_out = cp.asnumpy(self.Hbc_row_out)
        self.Hbc_col_out = cp.asnumpy(self.Hbc_col_out)
        self.P_out = cp.asnumpy(self.P_out)

    def to_cupy(self) -> None:
        if isinstance(self.P_out, cp.ndarray): return
        self.H_val_out = cp.asarray(self.H_val_out, dtype=cp.float64)
        self.H_row_out = cp.asarray(self.H_row_out, dtype=cp.int32)
        self.H_col_out = cp.asarray(self.H_col_out, dtype=cp.int32)
        self.C_val_out = cp.asarray(self.C_val_out, dtype=cp.float64)
        self.C_row_out = cp.asarray(self.C_row_out, dtype=cp.int32)
        self.C_col_out = cp.asarray(self.C_col_out, dtype=cp.int32)
        self.Hbc_val_out = cp.asarray(self.Hbc_val_out, dtype=cp.float64)
        self.Hbc_row_out = cp.asarray(self.Hbc_row_out, dtype=cp.int32)
        self.Hbc_col_out = cp.asarray(self.Hbc_col_out, dtype=cp.int32)
        self.P_out = cp.asarray(self.P_out, dtype=cp.float64)

