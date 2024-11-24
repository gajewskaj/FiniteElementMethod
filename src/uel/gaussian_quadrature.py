from math import sqrt
import numpy as np

from src.helpers import config

class GaussianQuadrature:
    def __init__(self, n: int):
        self.n: int = int(n)
        self.points, self.weights = self._init_points_weights()

    def _init_points_weights(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        match self.n:
            case 1:
                points = np.array([0])
                weights = np.array([2])
            case 2:
                points = np.array([-sqrt(1/3), sqrt(1/3)])
                weights = np.array([1, 1])
            case 3:
                points = np.array([-sqrt(3/5),
                                   0,
                                   sqrt(3/5)])
                weights = np.array([5/9,
                                    8/9,
                                    5/9])
            case 4:
                points = np.array([-sqrt(3/7 + 2/7*sqrt(6/5)),
                                   -sqrt(3/7 - 2/7*sqrt(6/5)),
                                   sqrt(3/7 - 2/7*sqrt(6/5)),
                                   sqrt(3/7 + 2/7*sqrt(6/5))])
                weights = np.array([(18 - sqrt(30))/36,
                                    (18 + sqrt(30))/36,
                                    (18 + sqrt(30))/36,
                                    (18 - sqrt(30))/36])
            case 5:
                points = np.array([-(1/3)*sqrt(5 + 2*sqrt(10/7)),
                                   -(1/3)*sqrt(5 - 2*sqrt(10/7)),
                                   0,
                                   (1/3)*sqrt(5 - 2*sqrt(10/7)),
                                   (1/3)*sqrt(5 + 2*sqrt(10/7))])
                weights = np.array([(322 - 13*sqrt(70))/900,
                                    (322 + 13*sqrt(70))/900,
                                    128/225,
                                    (322 + 13*sqrt(70))/900,
                                    (322 - 13*sqrt(70))/900])
            case _:
                err_msg: str = "Number of integration points in numerical integration must be an integer in range (1;5)."
                config.logger.error(err_msg)
                raise RuntimeError
        return points, weights