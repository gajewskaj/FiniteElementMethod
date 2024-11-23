from math import sqrt
import numpy as np

from src.lmc.gaussian_quadrature import GaussianQuadrature

NUM_OF_SHAPE_FUNCTIONS = 4
NUM_OF_SURFACES = NUM_OF_SHAPE_FUNCTIONS

# Shape functions
N = [
    lambda xi, eta: np.float32(0.25 * (1-xi) * (1-eta)), # N1
    lambda xi, eta: np.float32(0.25 * (1+xi) * (1-eta)), # N2
    lambda xi, eta: np.float32(0.25 * (1+xi) * (1+eta)), # N3
    lambda xi, eta: np.float32(0.25 * (1-xi) * (1+eta))  # N4
]

# Derivatives of shape functions
dN_dxi = [
    lambda eta: np.float32(-0.25 * (1-eta)), # dN1/dxi
    lambda eta: np.float32(0.25 * (1-eta)),  # dN2/dxi
    lambda eta: np.float32(0.25 * (1+eta)),  # dN3/dxi
    lambda eta: np.float32(-0.25 * (1+eta))  # dN4/dxi
]

dN_deta = [
    lambda xi: np.float32(-0.25 * (1-xi)), # dN1/deta
    lambda xi: np.float32(-0.25 * (1+xi)), # dN2/deta
    lambda xi: np.float32(0.25 * (1+xi)),  # dN3/deta
    lambda xi: np.float32(0.25 * (1-xi))   # dN4/deta
]

class UniversalElement():
    def __init__(self, n: int):
        self.n = n if n in [1, 4, 9, 16, 25] else 4
        self.gaussian_quadrature = GaussianQuadrature(sqrt(n))
        self.xi = np.empty(n, dtype=np.float32)
        self.eta = np.empty(n, dtype=np.float32)
        self.weights = np.empty(n, dtype=np.float32)

        self.dN_dxi = np.empty((self.n, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
        self.dN_deta = np.empty((self.n, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
        self.N = np.empty((self.n, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
        self.surfaces = np.empty((NUM_OF_SURFACES, self.gaussian_quadrature.n, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)

        self._calculate_integration_points(self.gaussian_quadrature.n,
                                           self.gaussian_quadrature.points,
                                           self.gaussian_quadrature.weights)
        self._calculate_shape_functions_and_derivatives()
        self._calculate_shape_functions_for_surfaces()

    def _calculate_integration_points(self, n_1d: int,
                                      points_1d: np.ndarray[np.float32],
                                      weights_1d: np.ndarray[np.float32]) -> None:
        for i in range(self.n):
            self.xi[i] = points_1d[i % n_1d]
            self.eta[i] = points_1d[i // n_1d]
            self.weights[i] = weights_1d[i % n_1d] * weights_1d[i // n_1d]

    def _calculate_shape_functions_and_derivatives(self) -> None:
        for i in range(self.n):
            xi = self.xi[i]
            eta = self.eta[i]
            for j in range(len(N)):
                self.N[i][j] = N[j](xi, eta)
                self.dN_dxi[i][j] = dN_dxi[j](eta)
                self.dN_deta[i][j] = dN_deta[j](xi)

    def _calculate_shape_functions_for_surfaces(self) -> None:
        for i in range(NUM_OF_SURFACES):
            if i % 2 == 0:
                xi_list = np.array(self.gaussian_quadrature.points)
                eta_list = np.full(self.gaussian_quadrature.n, i-1)
            else:
                eta_list = np.array(self.gaussian_quadrature.points)
                xi_list = np.full(self.gaussian_quadrature.n, 2-i)
            for j in range(self.gaussian_quadrature.n):
                for k in range(NUM_OF_SHAPE_FUNCTIONS):
                    self.surfaces[i][j][k] = N[k](xi_list[j], eta_list[j])