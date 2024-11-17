from math import sqrt
import numpy as np

from src.helpers import config

class GaussianQuadrature:
    """
    A class for numerical integration using Gaussian Quadrature.

    Attributes:
        fun (callable): Function to integrate.
        n (int): Number of integration points.
        points (np.ndarray): List of integration points, xi.
        weights (np.ndarray): Weights for integration points, wi.
    """
    def __init__(self, n: int, fun: callable = None):
        self.fun: callable = fun
        self.n: np.int8 = np.int8(n)
        self.points, self.weights = self._init_points_weights()

    def _init_points_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        """
        Initializes arrays containing integration points and their weights.
        """
        match self.n:
            case 1:
                points = np.array([0], dtype=np.float32)
                weights = np.array([2], dtype=np.float32)
            case 2:
                points = np.array([-sqrt(1/3), sqrt(1/3)], dtype=np.float32)
                weights = np.array([1, 1], dtype=np.float32)
            case 3:
                points = np.array([-sqrt(3/5),
                                   0,
                                   sqrt(3/5)], dtype=np.float32)
                weights = np.array([5/9,
                                    8/9,
                                    5/9], dtype=np.float32)
            case 4:
                points = np.array([-sqrt(3/7 + 2/7*sqrt(6/5)),
                                   -sqrt(3/7 - 2/7*sqrt(6/5)),
                                   sqrt(3/7 - 2/7*sqrt(6/5)),
                                   sqrt(3/7 + 2/7*sqrt(6/5))], dtype=np.float32)
                weights = np.array([(18 - sqrt(30))/36,
                                    (18 + sqrt(30))/36,
                                    (18 + sqrt(30))/36,
                                    (18 - sqrt(30))/36], dtype=np.float32)
            case 5:
                points = np.array([-(1/3)*sqrt(5 + 2*sqrt(10/7)),
                                   -(1/3)*sqrt(5 - 2*sqrt(10/7)),
                                   0,
                                   (1/3)*sqrt(5 - 2*sqrt(10/7)),
                                   (1/3)*sqrt(5 + 2*sqrt(10/7))], dtype=np.float32)
                weights = np.array([(322 - 13*sqrt(70))/900,
                                    (322 + 13*sqrt(70))/900,
                                    128/225,
                                    (322 + 13*sqrt(70))/900,
                                    (322 - 13*sqrt(70))/900], dtype=np.float32)
            case _:
                err_msg: str = "Number of nodes in numerical integration must be an integer in range (1;5)."
                config.logger.error(err_msg)
                raise RuntimeError
        return points, weights

    def calculate_1d(self) -> np.float32:
        """
        Calculates integral for function of 1 variable f(x).

        Returns:
            float: The calculated integral.
        """
        result = np.float32(0.0)
        for i in range (0, self.n):
            x = self.points[i]
            result += self.weights[i] * self.fun(x)
        return np.float32(result)

    def calculate_2d(self) -> np.float32:
        """
        Calculates integral for function of 2 variables f(x, y).

        Returns:
            float: The calculated integral.
        """
        result = np.float32(0.0)
        for i in range (0, self.n):
            x = self.points[i]
            for j in range (self.n):
                y = self.points[j]
                result += self.weights[i] * self.weights[j] * self.fun(x, y)
        return np.float32(result)