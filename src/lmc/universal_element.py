import numpy as np

from src.lmc.numerical_integration import GaussianQuadrature

class UniversalElement(GaussianQuadrature):
    """
    Contains all the calculations that are universal for every 4-node element.

    Attributes:
        dNdKsiTab (np.ndarray): Table of dN/dKsi results for N1, N2, N3, N4 in integration points (n^2x4).
        dNdEtaTab (np.ndarray): Table of dN/dEta results for N1, N2, N3, N4 in integration points (n^2x4).
        NTab (np.ndarray): Table of N(ksi, eta) values for N1, N2, N3, N4 in integration points (nx4).
        surfaces (list[Surface]): List of Surface type elements, necessary for calculations that take border conditions into account.
    """
    def __init__(self, n: int):
        """
        Initializes the UniversalElement with the given number of integration points.

        Args:
            n (int): Number of integration points.
        """
        super().__init__(n)
        self.dn_dksi_tab = np.empty((self.n*self.n, 4), dtype=np.float32)
        self.dn_deta_tab = np.empty((self.n*self.n, 4), dtype=np.float32)
        self.n_tab = np.empty((self.n*self.n, 4), dtype=np.float32)
        self.surfaces: list[Surface] = [
            Surface(n), # down
            Surface(n), # right
            Surface(n), # up
            Surface(n)  # left
        ]
        self._fill_dn_dksi_dn_deta_tabs()
        self._fill_surface_tab()

    def _fill_dn_dksi_dn_deta_tabs(self) -> None:
        """
        Calculates dN/dKsi and dN/dEta for N1, N2, N3, N4 in integration points.
        Results are stored in tables.
        """
        for j in range(self.n*self.n):
            ksi = self.points[j%self.n]
            eta = self.points[j//self.n]
            for i in range (0, 4):
                self.dn_dksi_tab[j][i] = dn_dksi_fun_tab[i](eta)
                self.dn_deta_tab[j][i] = dn_deta_fun_tab[i](ksi)
                self.n_tab[j][i] = n_fun_tab[i](ksi, eta)

    def _fill_surface_tab(self):
        """
        Calculates integration points coordinates for surface calculations.
        """
        for i in range(0, len(self.surfaces)):
            if i % 2 == 0:
                ksi_list = np.array(self.points)
                eta_list = np.full(self.n, i-1)
            else:
                eta_list = np.array(self.points)
                ksi_list = np.full(self.n, 2-i)
            self.surfaces[i].fill_N(ksi_list, eta_list)

class Surface():
    """
    Describes the surface of the universal element.

    Attributes:
        n (int): Number of integration points.
        N (np.ndarray): Table of N(ksi, eta) for N1, N2, N3, N4 and for each integration point on the surface.
    """
    def __init__(self, n: int):
        """
        Initializes the Surface with the given number of integration points.

        Args:
            n (int): Number of integration points.
        """
        self.n = n
        self.N = np.zeros((self.n, 4), dtype=np.float32)

    def fill_N(self, ksi_list: np.ndarray[np.float32], eta_list: np.ndarray[np.float32]):
        """
        Calculates N(ksi, eta) for N1, N2, N3, N4 and for each integration point on the surface.
        Results are stored in the table.

        Args:
            ksi_list (np.ndarray[float]): List of ksi values for integration points.
            eta_list (np.ndarray[float]): List of eta values for integration points.
        """
        for j in range(0, len(ksi_list)):
            for i in range(0, 4):
                self.N[j][i] = n_fun_tab[i](ksi_list[j], eta_list[j])

def N1(ksi: np.float32, eta: np.float32) -> np.float32:
    """
    Shape function N1.

    Args:
        ksi (float): Ksi coordinate.
        eta (float): Eta coordinate.

    Returns:
        float: Value of N1 at (ksi, eta).
    """
    return np.float32((1/4) * (1-ksi) * (1-eta))

def N2(ksi: np.float32, eta: np.float32) -> np.float32:
    """
    Shape function N2.

    Args:
        ksi (float): Ksi coordinate.
        eta (float): Eta coordinate.

    Returns:
        float: Value of N2 at (ksi, eta).
    """
    return np.float32((1/4) * (1+ksi) * (1-eta))

def N3(ksi: np.float32, eta: np.float32) -> np.float32:
    """
    Shape function N3.

    Args:
        ksi (float): Ksi coordinate.
        eta (float): Eta coordinate.

    Returns:
        float: Value of N3 at (ksi, eta).
    """
    return np.float32((1/4) * (1+ksi) * (1+eta))

def N4(ksi: np.float32, eta: np.float32) -> np.float32:
    """
    Shape function N4.

    Args:
        ksi (float): Ksi coordinate.
        eta (float): Eta coordinate.

    Returns:
        float: Value of N4 at (ksi, eta).
    """
    return np.float32((1/4) * (1-ksi) * (1+eta))

def dN1Ksi(eta: np.float32) -> np.float32:
    """
    Derivative of N1 with respect to Ksi.

    Args:
        eta (float): Eta coordinate.

    Returns:
        float: Value of dN1/dKsi at (eta).
    """
    return np.float32(-(1/4) * (1-eta))

def dN2Ksi(eta: np.float32) -> np.float32:
    """
    Derivative of N2 with respect to Ksi.

    Args:
        eta (float): Eta coordinate.

    Returns:
        float: Value of dN2/dKsi at (eta).
    """
    return np.float32((1/4) * (1-eta))

def dN3Ksi(eta: np.float32) -> np.float32:
    """
    Derivative of N3 with respect to Ksi.

    Args:
        eta (float): Eta coordinate.

    Returns:
        float: Value of dN3/dKsi at (eta).
    """
    return np.float32((1/4) * (1+eta))

def dN4Ksi(eta: np.float32) -> np.float32:
    """
    Derivative of N4 with respect to Ksi.

    Args:
        eta (float): Eta coordinate.

    Returns:
        float: Value of dN4/dKsi at (eta).
    """
    return np.float32(-(1/4) * (1+eta))

def dN1Eta(ksi: np.float32) -> np.float32:
    """
    Derivative of N1 with respect to Eta.

    Args:
        ksi (float): Ksi coordinate.

    Returns:
        float: Value of dN1/dEta at (ksi).
    """
    return np.float32(-(1/4) * (1-ksi))

def dN2Eta(ksi: np.float32) -> np.float32:
    """
    Derivative of N2 with respect to Eta.

    Args:
        ksi (float): Ksi coordinate.

    Returns:
        float: Value of dN2/dEta at (ksi).
    """
    return np.float32(-(1/4) * (1+ksi))

def dN3Eta(ksi: np.float32) -> np.float32:
    """
    Derivative of N3 with respect to Eta.

    Args:
        ksi (float): Ksi coordinate.

    Returns:
        float: Value of dN3/dEta at (ksi).
    """
    return np.float32((1/4) * (1+ksi))

def dN4Eta(ksi: np.float32) -> np.float32:
    """
    Derivative of N4 with respect to Eta.

    Args:
        ksi (float): Ksi coordinate.

    Returns:
        float: Value of dN4/dEta at (ksi).
    """
    return np.float32((1/4) * (1-ksi))

dn_dksi_fun_tab = [
    dN1Ksi,
    dN2Ksi,
    dN3Ksi,
    dN4Ksi
]

dn_deta_fun_tab = [
    dN1Eta,
    dN2Eta,
    dN3Eta,
    dN4Eta
]

n_fun_tab = [
    N1,
    N2,
    N3,
    N4,
]