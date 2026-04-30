import numpy as np

from src.helpers.config import logger
from src.uel.gaussian_quadrature import GaussianQuadrature

DOF = 3

# Shape functions
N = [
    lambda xi, eta: -0.5 * (xi+eta), # N1
    lambda xi, eta: 0.5 * (1+xi),    # N2
    lambda xi, eta: 0.5 * (1+eta)    # N3
]

class UniversalElementTriangle():
    def __init__(self, n: int):
        self.n = n
        self.gaussian_quadrature_traingle_mapping = {1: 1, 3: 2, 4: 3, 6: 4, 7: 5}
        self.quadrature_1d = GaussianQuadrature(self.gaussian_quadrature_traingle_mapping[n])
        self.xi = np.empty(n)
        self.eta = np.empty(n)
        self.weights = np.empty(n)

        self.dN_dxi = np.array([[-0.5, 0.5, 0] for _ in range(self.n)], dtype=float)
        self.dN_deta = np.array([[-0.5, 0, 0.5] for _ in range(self.n)], dtype=float)
        self.N = np.empty((self.n, DOF))
        self.surfaces = np.empty((DOF, self.quadrature_1d.n, DOF))

        self._init_integration_points_and_weights()
        self._fill_shape_functions()
        self._fill_shape_functions_for_surfaces()

    def _init_integration_points_and_weights(self) -> None:
        match self.n:
            case 1:
                self.xi = np.array([-1/3], dtype=float)
                self.eta = np.array([-1/3], dtype=float)
                self.weights = np.array([2], dtype=float)
            case 3:
                self.xi = np.array([-2/3, 1/3, -2/3], dtype=float)
                self.eta = np.array([-2/3, -2/3, 1/3], dtype=float)
                self.weights = np.array([2/3, 2/3, 2/3], dtype=float)
            case 4:
                self.xi = np.array([-1/3,
                                    -0.6,
                                    -0.6,
                                    0.2], dtype=float)
                self.eta = np.array([-1/3,
                                    -0.6,
                                    0.2,
                                    -0.6], dtype=float)
                self.weights = np.array([-1.125,
                                         1.041666666666667,
                                         1.041666666666667,
                                         1.041666666666667], dtype=float)
            case 6:
                self.xi = np.array([-0.108103018168070,
                                    -0.108103018168070,
                                    -0.783793963663860,
                                    -0.816847572980458,
                                    -0.816847572980458,
                                    0.633695145960918], dtype=float)
                self.eta = np.array([-0.108103018168070,
                                    -0.78379396363860,
                                    -0.108103018168070,
                                    -0.816847572980458,
                                    0.633695145960918,
                                    -0.816847572980458], dtype=float)
                self.weights = np.array([0.446763179356022,
                                         0.446763179356022,
                                         0.446763179356022,
                                         0.219903487310644,
                                         0.219903487310644,
                                         0.219903487310644], dtype=float)
            case 7:
                self.xi = np.array([-1/3,
                                    -0.059715871789770,
                                    -0.059715871789770,
                                    -0.880568256420460,
                                    -0.797426985353088,
                                    -0.797426985353088,
                                    0.594853970706174], dtype=float)
                self.eta = np.array([-1/3,
                                    -0.059715871789770,
                                    -0.880568256420460,
                                    -0.059715871789770,
                                    -0.797426985353088,
                                    0.594853970706174,
                                    -0.797426985353088], dtype=float)
                self.weights = np.array([0.45,
                                         0.264788305577012,
                                         0.264788305577012,
                                         0.264788305577012,
                                         0.251878361089654,
                                         0.251878361089654,
                                         0.251878361089654], dtype=float)
            case _:
                err_msg: str = "Number of integration points for a triangle element must be 1, 3, 4, 6 or 7."
                logger.error(err_msg)
                raise RuntimeError(err_msg)

    def _fill_shape_functions(self) -> None:
        for i in range(self.n):
            for j in range(len(N)):
                self.N[i, j] = N[j](self.xi[i], self.eta[i])

    def _fill_shape_functions_for_surfaces(self) -> None:
        for i in range(DOF):
            if i == 0:
                xi_list = np.array(self.quadrature_1d.points)
                eta_list = np.full(self.quadrature_1d.n, -1)
            elif i == 1:
                xi_list = np.array(self.quadrature_1d.points)
                eta_list = -xi_list
            elif i == 2:
                xi_list = np.full(self.quadrature_1d.n, -1)
                eta_list = np.array(self.quadrature_1d.points)
            for j in range(self.quadrature_1d.n):
                for k in range(DOF):
                    self.surfaces[i, j, k] = N[k](xi_list[j], eta_list[j])