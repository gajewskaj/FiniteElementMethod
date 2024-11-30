from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as cpu_sparse
from src.grid.grid import Grid

class SystemOfEquations(ABC):
    @abstractmethod
    def __init__(self, grid: Grid):
        pass

    @abstractmethod
    def _assemble(self) -> tuple:
        pass

    @abstractmethod
    def solve(self) -> np.ndarray:
        pass