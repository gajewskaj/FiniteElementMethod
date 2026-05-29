from abc import ABC, abstractmethod

import numpy as np
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices

class SystemOfEquations(ABC):
    @abstractmethod
    def __init__(self, mesh: Mesh, out_mat: OutMatrices) -> None:
        self.dim: int = len(mesh.nodes_id)
        self.step: float = mesh.global_data.simulation_step_time
        self.mesh = mesh
        self.out_mat = out_mat

    @abstractmethod
    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def factorize(self) -> callable:
        pass

    @abstractmethod
    def solve(self) -> np.ndarray:
        pass

    @abstractmethod
    def simulate(self) -> tuple[list[float], list[np.ndarray[float]]]:
        pass