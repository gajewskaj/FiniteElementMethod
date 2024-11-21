from math import sqrt
import numpy as np

from src.helpers.helpers import measure_time
from src.grid.grid import Grid, Element, Node
from src.lmc.universal_element import UniversalElement, Surface

NODES_PER_ELEMENT = 4

@measure_time
def calculate_local_matrices(n: int, grid: Grid) -> None:
    """
    Calculates H, C, Hbc matrices and P vector for each element in the grid.
    The output is stored in the Element class.

    Args:
        n (int): Number of integration points.
        grid (Grid): The grid containing elements and nodes.
    """
    u_el = UniversalElement(n)
    for element in grid.elements:
        element: Element
        nodes = [grid.nodes[element.node_ids[0] - 1],
                grid.nodes[element.node_ids[1] - 1],
                grid.nodes[element.node_ids[2] - 1],
                grid.nodes[element.node_ids[3] - 1]]
        x_coords, y_coords = _fill_x_y_coords(nodes)
        _calculate_H_C_for_element(u_el, element.H, element.C,
                                    x_coords, y_coords,
                                    grid.global_data.conductivity, grid.global_data.density, grid.global_data.specific_heat)
        _calculate_Hbc_P_for_element(u_el, element.Hbc, element.P,
                                 nodes,
                                 grid.global_data.alfa, grid.global_data.tot)

def _calculate_H_C_for_element(u_el: UniversalElement, H: np.ndarray[np.float32], C: np.ndarray[np.float32],
                           x_coords: np.ndarray[np.float32], y_coords: np.ndarray[np.float32],
                           c: np.float32, d: np.float32, sh: np.float32):
    dx_dksi_tab = np.empty(u_el.n*u_el.n)
    dx_deta_tab = np.empty(u_el.n*u_el.n)
    dy_dksi_tab = np.empty(u_el.n*u_el.n)
    dy_deta_tab = np.empty(u_el.n*u_el.n)
    det_tab = np.empty(u_el.n*u_el.n)
    dn_dx_tab = np.empty((u_el.n*u_el.n, NODES_PER_ELEMENT))
    dn_dy_tab = np.empty((u_el.n*u_el.n, NODES_PER_ELEMENT))
    for i in range(u_el.n*u_el.n):
        _fill_x_y_ksi_eta_tabs(i, x_coords, y_coords,
                               u_el.dn_dksi_tab, u_el.dn_deta_tab,
                               dx_dksi_tab, dx_deta_tab,
                               dy_dksi_tab, dy_deta_tab)
        _dn_dx_dn_dy(i, dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab,
                     u_el.dn_dksi_tab, u_el.dn_deta_tab,
                     det_tab, dn_dx_tab, dn_dy_tab)
        _calculate_H_C(i, H, C,
                       u_el.n, u_el.weights, u_el.n_tab,
                       det_tab, dn_dx_tab, dn_dy_tab,
                       c, d, sh)

def _calculate_Hbc_P_for_element(u_el: UniversalElement, Hbc: np.ndarray[np.float32], P: np.ndarray[np.float32],
                                 nodes: list[Node],
                                 alfa: np.float32, tot: np.float32):
    _calculate_for_surface(Hbc, P, u_el.n, u_el.weights, u_el.surfaces[0], (nodes[0], nodes[1]), alfa, tot)
    _calculate_for_surface(Hbc, P, u_el.n, u_el.weights, u_el.surfaces[1], (nodes[1], nodes[2]), alfa, tot)
    _calculate_for_surface(Hbc, P, u_el.n, u_el.weights, u_el.surfaces[2], (nodes[2], nodes[3]), alfa, tot)
    _calculate_for_surface(Hbc, P, u_el.n, u_el.weights, u_el.surfaces[3], (nodes[3], nodes[0]), alfa, tot)

def _fill_x_y_coords(nodes: np.ndarray[Node]) -> tuple[np.ndarray, np.ndarray]:
    """
    Fills lists storing x and y coordinates of nodes belonging to the element.

    Args:
        nodes (np.ndarray[Node]): Array of nodes for the element.

    Returns:
        tuple: Lists of x and y coordinates.
    """
    x_coords = np.ndarray(NODES_PER_ELEMENT, dtype=np.float32)
    y_coords = np.ndarray(NODES_PER_ELEMENT, dtype=np.float32)
    for i in range(NODES_PER_ELEMENT):
        x_coords[i] = nodes[i].x
        y_coords[i] = nodes[i].y
    return x_coords, y_coords

def _fill_x_y_ksi_eta_tabs(i: int, x_coords: list[float], y_coords: list[float],
                           dn_dksi_tab: np.ndarray[np.float32], dn_deta_tab: np.ndarray[np.float32],
                           dx_dksi_tab: np.ndarray[np.float32], dx_deta_tab: np.ndarray[np.float32],
                           dy_dksi_tab: np.ndarray[np.float32], dy_deta_tab: np.ndarray[np.float32]):
    """
    Calculates dx/dksi, dx/deta, dy/dksi, dy/deta for every integration point and returns 4 tables with output.

    Args:
        x_coords (list[float]): List of x coordinates.
        y_coords (list[float]): List of y coordinates.
        u_el (UniversalElement): Universal element containing shape functions and derivatives.

    Returns:
        tuple: Lists of dx/dksi, dx/deta, dy/dksi, dy/deta.
    """
    dx_dksi_tab[i] = _interpolate(dn_dksi_tab[i][0], dn_dksi_tab[i][1], dn_dksi_tab[i][2], dn_dksi_tab[i][3], x_coords)
    dx_deta_tab[i] = _interpolate(dn_deta_tab[i][0], dn_deta_tab[i][1], dn_deta_tab[i][2], dn_deta_tab[i][3], x_coords)
    dy_dksi_tab[i] = _interpolate(dn_dksi_tab[i][0], dn_dksi_tab[i][1], dn_dksi_tab[i][2], dn_dksi_tab[i][3], y_coords)
    dy_deta_tab[i] = _interpolate(dn_deta_tab[i][0], dn_deta_tab[i][1], dn_deta_tab[i][2], dn_deta_tab[i][3], y_coords)

def _dn_dx_dn_dy(i: int, dx_dksi_tab: list, dx_deta_tab: list, dy_dksi_tab: list, dy_deta_tab: list,
                 dn_dksi_tab: np.ndarray[np.float32], dn_deta_tab: np.ndarray[np.float32],
                 det_tab: np.ndarray[np.float32], dn_dx_tab: np.ndarray[np.float32], dn_dy_tab: np.ndarray[np.float32]):
    """
    Fills dN/dx and dN/dy tables and calculates Jacobian and det[J].

    Args:
        dx_dksi_tab (list): List of dx/dksi values.
        dx_deta_tab (list): List of dx/deta values.
        dy_dksi_tab (list): List of dy/dksi values.
        dy_deta_tab (list): List of dy/deta values.
        u_el (UniversalElement): Universal element containing shape functions and derivatives.

    Returns:
        tuple: Lists of dN/dx, dN/dy and det[J] values.
    """
    mxJ = np.array([[dx_dksi_tab[i], dy_dksi_tab[i]],
                        [dx_deta_tab[i], dy_deta_tab[i]]])
    detJ = np.linalg.det(mxJ)
    det_tab[i] = detJ
    mx1 = np.array([[dy_deta_tab[i], -dy_dksi_tab[i]],
                        [-dx_deta_tab[i], dx_dksi_tab[i]]])
    for k in range (0, NODES_PER_ELEMENT):
        mx2 = np.array([[dn_dksi_tab[i][k]],[dn_deta_tab[i][k]]])
        # (1/detJ)*mx1 = mxJ^(-1)
        mxOutput = np.matmul(((1/detJ)*mx1), mx2)
        dn_dx_tab[i][k] = mxOutput[0][0]
        dn_dy_tab[i][k] = mxOutput[1][0]

def _calculate_H_C(i: int, H: np.ndarray[np.float32], C: np.ndarray[np.float32],
                   n, weights, n_tab,
                   det_tab: np.ndarray[np.float32], dn_dx_tab: np.ndarray[np.float32], dn_dy_tab: np.ndarray[np.float32],
                   c: np.float32, d: np.float32, sh: np.float32):
    """
    Calculates H, C matrices for each integration point. Returns list of matrices.

    Args:
        dn_dx_tab (list): List of dN/dx values.
        dn_dy_tab (list): List of dN/dy values.
        NTab (list): List of shape function values.
        det_tab (list): List of det[J] values.
        n (int): Number of integration points.
        c (int): Conductivity.
        d (int): Density.
        sH (int): Specific heat.

    Returns:
        tuple: Lists of H and C matrices for each integration point.
    """
    mxDNdX = np.array([[dn_dx_tab[i][0]],
                        [dn_dx_tab[i][1]],
                        [dn_dx_tab[i][2]],
                        [dn_dx_tab[i][3]]])
    mxDNdY = np.array([[dn_dy_tab[i][0]],
                        [dn_dy_tab[i][1]],
                        [dn_dy_tab[i][2]],
                        [dn_dy_tab[i][3]]])
    mxN = np.array([[n_tab[i][0]],
                        [n_tab[i][1]],
                        [n_tab[i][2]],
                        [n_tab[i][3]]])
    ipMxH = c*(np.matmul(mxDNdX, mxDNdX.transpose()) + np.matmul(mxDNdY, mxDNdY.transpose()))*det_tab[i]
    ipMxC = sh*d*(np.matmul(mxN, mxN.transpose()))*det_tab[i]

    H += ipMxH*(weights[i//n])*(weights[i%n])
    C += ipMxC*(weights[i//n])*(weights[i%n])

def _calculate_for_surface(Hbc: np.ndarray[np.float32], P: np.ndarray[np.float32],
                           n: int, weights: np.ndarray[np.float32], surface: Surface,
                           nodes: np.ndarray[Node],
                           alfa, tot) -> tuple:
    """
    Calculates Hbc matrix and P vector for the given surface of the element.

    Args:
        surface (Surface): Surface of the element.
        nodes (tuple[Node]): Nodes of the surface.
        weights (np.ndarray[float]): Weights for integration points.
        n (int): Number of integration points.
        alfa (int): Heat transfer coefficient.
        tot (int): Ambient temperature.

    Returns:
        tuple: Hbc matrix and P vector.
    """
    Hbc_surf = np.zeros((NODES_PER_ELEMENT, NODES_PER_ELEMENT))
    P_surf = np.zeros((NODES_PER_ELEMENT, 1))
    if nodes[0].BC == 0 or nodes[1].BC == 0:
        return
    L = sqrt(pow(nodes[0].x-nodes[1].x, 2)+pow(nodes[0].y-nodes[1].y, 2))
    detJ = L/2
    for i in range(n):
        mx = np.array([[surface.N[i][0]],
                        [surface.N[i][1]],
                        [surface.N[i][2]],
                        [surface.N[i][3]]])
        Hbc_surf += np.matmul(mx, mx.transpose())*weights[i]
        P_surf += mx*weights[i]
    Hbc += Hbc_surf*alfa*detJ
    P += P_surf*alfa*tot*detJ

def _interpolate(dN1: np.float32, dN2: np.float32, dN3: np.float32, dN4: np.float32, var: list[np.float32]) -> np.float32:
    """
    Returns dx/deta, dx/dksi, dy/deta or dy/dksi depending on given arguments.

    Args:
        dN1 (float): Derivative of shape function N1.
        dN2 (float): Derivative of shape function N2.
        dN3 (float): Derivative of shape function N3.
        dN4 (float): Derivative of shape function N4.
        var (list[float]): List of variable values.

    Returns:
        float: Interpolated value.
    """
    return dN1*var[0] + dN2*var[1] + dN3*var[2] + dN4*var[3]