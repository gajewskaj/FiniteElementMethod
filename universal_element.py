from common import *
from numerical_integration import GaussianQuadrature
from math import *
import numpy as np

class UniversalElement(GaussianQuadrature):
    """
    Contains all the calculations that are universal for every 4-node element.

    dNdKsiTab:      table of dN/dKsi results for N1, N2, N3, N4 in integration points (n^2x4)
    dNdEtaTab:      table of dN/dEta results for N1, N2, N3, N4 in integration points (n^2x4)
    NTab:           table of N(ksi, eta) values for N1, N2, N3, N4 in integration points (nx4)
    surfaces:       list of Surface type elements, necessary for calculations that take border conditions into account
    """
    def __init__(self, n):
        super().__init__(n)
        self.dn_dksi_tab = np.empty((self.n*self.n, 4), dtype=float)
        self.dn_deta_tab = np.empty((self.n*self.n, 4), dtype=float)
        self.n_tab = np.empty((self.n*self.n, 4), dtype=float)
        self.surfaces: list[Surface] = [
            Surface(n), # down
            Surface(n), # right
            Surface(n), # up
            Surface(n)  # left
        ]
        self._fill_dn_dksi_dn_deta_tabs()
        self._fill_surface_tab()

    def _fill_dn_dksi_dn_deta_tabs(self):
        """
        Calculates dN/dKsi and dN/dEta for N1, N2, N3, N4 in integration points. Result is stored in tables.
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
        Calculates integration points coords for surface calculations.
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
    n:      number of integration points
    N:      table of N(ksi, eta) for N1, N2, N3, N4 and for each integration point on the surface
    """
    def __init__(self, n: int):
        self.n = n
        self.N = np.zeros((self.n, 4))

    def fill_N(self, ksi_list: np.ndarray[float], eta_list: np.ndarray[float]):
        """
        Calculates N(ksi, eta) for N1, N2, N3, N4 and for each integration point on the surface
        Results are stored in the table
        """
        for j in range(0, len(ksi_list)):
            for i in range(0, 4):
                self.N[j][i] = n_fun_tab[i](ksi_list[j], eta_list[j])

def N1(ksi: float, eta: float) -> float:
    return (1/4) * (1-ksi) * (1-eta)

def N2(ksi: float, eta: float) -> float:
    return (1/4) * (1+ksi) * (1-eta)

def N3(ksi: float, eta: float) -> float:
    return (1/4) * (1+ksi) * (1+eta)

def N4(ksi: float, eta: float) -> float:
    return (1/4) * (1-ksi) * (1+eta)

def dN1Ksi(eta: float) -> float:
    return -(1/4) * (1-eta)

def dN2Ksi(eta: float) -> float:
    return (1/4) * (1-eta)

def dN3Ksi(eta: float) -> float:
    return (1/4) * (1+eta)

def dN4Ksi(eta: float) -> float:
    return -(1/4) * (1+eta)

def dN1Eta(ksi: float) -> float:
    return -(1/4) * (1-ksi)

def dN2Eta(ksi: float) -> float:
    return -(1/4) * (1+ksi)

def dN3Eta(ksi: float) -> float:
    return (1/4) * (1+ksi)

def dN4Eta(ksi: float) -> float:
    return (1/4) * (1-ksi)

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