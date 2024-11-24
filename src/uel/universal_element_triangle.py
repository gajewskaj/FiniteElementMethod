import numpy as np

from src.uel.gaussian_quadrature import GaussianQuadrature

NUM_OF_SHAPE_FUNCTIONS = 3
NUM_OF_SURFACES = NUM_OF_SHAPE_FUNCTIONS

# Shape functions
N = [
    lambda xi, eta: np.float32(-0.5 * (xi+eta)), # N1
    lambda xi, eta: np.float32(0.5 * (1+xi)),    # N2
    lambda xi, eta: np.float32(0.5 * (1+eta))    # N3
]

class UniversalElementTriangle():
    def __init__(self, n: int = 3):
        self.n = n if n in [3] else 3
        self.gaussian_quadrature = GaussianQuadrature(2)
        self.xi = np.array([-2/3, 1/3, -2/3], dtype=np.float32)
        self.eta = np.array([-2/3, -2/3, 1/3], dtype=np.float32)
        self.weights = np.array([2/3, 2/3, 2/3], dtype=np.float32)

        self.dN_dxi = np.array([[-0.5, 0.5, 0] for _ in range(self.n)], dtype=np.float32)
        self.dN_deta = np.array([[-0.5, 0, 0.5] for _ in range(self.n)], dtype=np.float32)
        self.N = np.empty((self.n, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
        self.surfaces = np.empty((NUM_OF_SURFACES, self.gaussian_quadrature.n, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
        self._calculate_shape_functions_and_derivatives()
        self._calculate_shape_functions_for_surfaces()

    def _calculate_shape_functions_and_derivatives(self) -> None:
        for i in range(self.n):
            xi = self.xi[i]
            eta = self.eta[i]
            for j in range(len(N)):
                self.N[i][j] = N[j](xi, eta)

    def _calculate_shape_functions_for_surfaces(self) -> None:
        for i in range(NUM_OF_SURFACES):
            if i == 0:
                xi_list = np.array(self.gaussian_quadrature.points)
                eta_list = np.full(self.gaussian_quadrature.n, -1)
            elif i == 1:
                xi_list = np.array(self.gaussian_quadrature.points)
                eta_list = -xi_list
            elif i == 2:
                xi_list = np.full(self.gaussian_quadrature.n, -1)
                eta_list = np.array(self.gaussian_quadrature.points)
            for j in range(self.gaussian_quadrature.n):
                for k in range(NUM_OF_SHAPE_FUNCTIONS):
                    self.surfaces[i][j][k] = N[k](xi_list[j], eta_list[j])