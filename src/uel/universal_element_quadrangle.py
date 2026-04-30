from math import sqrt
import numpy as np

from src.uel.gaussian_quadrature import GaussianQuadrature

DOF = 4

# Shape functions
N = [
    lambda xi, eta: 0.25 * (1-xi) * (1-eta), # N1
    lambda xi, eta: 0.25 * (1+xi) * (1-eta), # N2
    lambda xi, eta: 0.25 * (1+xi) * (1+eta), # N3
    lambda xi, eta: 0.25 * (1-xi) * (1+eta)  # N4
]

# Derivatives of shape functions
dN_dxi = [
    lambda eta: -0.25 * (1-eta), # dN1/dxi
    lambda eta: 0.25 * (1-eta),  # dN2/dxi
    lambda eta: 0.25 * (1+eta),  # dN3/dxi
    lambda eta: -0.25 * (1+eta)  # dN4/dxi
]

dN_deta = [
    lambda xi: -0.25 * (1-xi), # dN1/deta
    lambda xi: -0.25 * (1+xi), # dN2/deta
    lambda xi: 0.25 * (1+xi),  # dN3/deta
    lambda xi: 0.25 * (1-xi)   # dN4/deta
]

class UniversalElementQuadrangle():
    def __init__(self, n: int):
        self.n = n
        self.quadrature_1d = GaussianQuadrature(sqrt(n))
        self.xi = np.empty(n)
        self.eta = np.empty(n)
        self.weights = np.empty(n)

        self.dN_dxi = np.empty((self.n, DOF))
        self.dN_deta = np.empty((self.n, DOF))
        self.N = np.empty((self.n, DOF))
        self.surfaces = np.empty((DOF, self.quadrature_1d.n, DOF))

        self._init_integration_points_and_weights()
        self._fill_shape_functions_and_derivatives()
        self._fill_shape_functions_for_surfaces()

    def _init_integration_points_and_weights(self) -> None:
        for i in range(self.n):
            self.xi[i] = self.quadrature_1d.points[i % self.quadrature_1d.n]
            self.eta[i] = self.quadrature_1d.points[i // self.quadrature_1d.n]
            self.weights[i] = self.quadrature_1d.weights[i % self.quadrature_1d.n] * self.quadrature_1d.weights[i // self.quadrature_1d.n]

    def _fill_shape_functions_and_derivatives(self) -> None:
        for i in range(self.n):
            for j in range(len(N)):
                self.N[i, j] = N[j](self.xi[i], self.eta[i])
                self.dN_dxi[i, j] = dN_dxi[j](self.eta[i])
                self.dN_deta[i, j] = dN_deta[j](self.xi[i])

    def _fill_shape_functions_for_surfaces(self) -> None:
        for i in range(DOF):
            if i % 2 == 0:
                xi_list = np.array(self.quadrature_1d.points)
                eta_list = np.full(self.quadrature_1d.n, i-1)
            else:
                eta_list = np.array(self.quadrature_1d.points)
                xi_list = np.full(self.quadrature_1d.n, 2-i)
            for j in range(self.quadrature_1d.n):
                for k in range(DOF):
                    self.surfaces[i, j, k] = N[k](xi_list[j], eta_list[j])