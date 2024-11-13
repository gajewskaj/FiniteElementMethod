from math import sqrt
import numpy as np

from . import config

class GaussianQuadrature:
    """
    A class for numerical integration using Gaussian Quadrature.

    Attributes:
        - fun (callable): Function to integrate.
        - n (int): Number of integration points.
        - points (np.ndarray): List of integration points, xi.
        - weights (np.ndarray): Weights for integration points, wi.
    """
    def __init__(self, n: int, fun: callable = None):
        self.fun: callable = fun
        self.n: int = n
        self.points: np.ndarray[float]
        self.weights: np.ndarray[float]
        self._init_points_weights()

    def _init_points_weights(self) -> tuple[np.ndarray]:
        """
        Initializes arrays containing integration points and their weights.
        """
        match self.n:
            case 1:
                self.points = np.array([0])
                self.weights = np.array([2])
            case 2:
                self.points = np.array([-sqrt(1/3), sqrt(1/3)])
                self.weights = np.array([1, 1])
            case 3:
                self.points = np.array([-sqrt(3/5),
                                   0,
                                   sqrt(3/5)])
                self.weights = np.array([5/9,
                                    8/9,
                                    5/9])
            case 4:
                self.points = np.array([-sqrt(3/7 + 2/7*sqrt(6/5)),
                                   -sqrt(3/7 - 2/7*sqrt(6/5)),
                                   sqrt(3/7 - 2/7*sqrt(6/5)),
                                   sqrt(3/7 + 2/7*sqrt(6/5))])
                self.weights = np.array([(18 - sqrt(30))/36,
                                    (18 + sqrt(30))/36,
                                    (18 + sqrt(30))/36,
                                    (18 - sqrt(30))/36])
            case 5:
                self.points = np.array([-(1/3)*sqrt(5 + 2*sqrt(10/7)),
                                   -(1/3)*sqrt(5 - 2*sqrt(10/7)),
                                   0,
                                   (1/3)*sqrt(5 - 2*sqrt(10/7)),
                                   (1/3)*sqrt(5 + 2*sqrt(10/7))])
                self.weights = np.array([(322 - 13*sqrt(70))/900,
                                    (322 + 13*sqrt(70))/900,
                                    128/225,
                                    (322 + 13*sqrt(70))/900,
                                    (322 - 13*sqrt(70))/900])
            case _:
                err_msg: str = "Number of nodes in numerical integration must be an integer in range (1;5)."
                config.logger.error(err_msg)
                raise RuntimeError

    def calculate_1d(self) -> float:
        """
        Calculates integral for function of 1 variable f(x).

        Returns:
            float: The calculated integral.
        """
        result = 0.0
        for i in range (0, self.n):
            x = self.points[i]
            result += self.weights[i] * self.fun(x)
        return result

    def calculate_2d(self) -> float:
        """
        Calculates integral for function of 2 variables f(x, y).

        Returns:
            float: The calculated integral.
        """
        result = 0.0
        for i in range (0, self.n):
            x = self.points[i]
            for j in range (self.n):
                y = self.points[j]
                result += self.weights[i] * self.weights[j] * self.fun(x, y)
        return result