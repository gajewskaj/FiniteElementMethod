from math import sqrt
import numpy as np

from src.helpers import config
from src.grid.grid import Grid, GlobalData, Element, Node
from src.lmc.universal_element import UniversalElement, Surface

def calculate_local_matrices(n: int, grid: Grid) -> None:
    """
    Calculates H, C, Hbc matrices and P vector for each element in the grid.
    The output is stored in the Element class.

    Args:
        n (int): Number of integration points.
        grid (Grid): The grid containing elements and nodes.
    """
    config.logger.info(f"Calculating local matrices for {len(grid.elements)} elements.")
    u_el = UniversalElement(n)
    for element in grid.elements:
        element: Element
        element.H, element.C, element.Hbc, element.P = _calculate_for_element(
            [grid.nodes[element.node_ids[0] - 1],
                grid.nodes[element.node_ids[1] - 1],
                grid.nodes[element.node_ids[2] - 1],
                grid.nodes[element.node_ids[3] - 1]],
            u_el, grid.global_data)

def _calculate_for_element(nodes: np.ndarray[Node], u_el: UniversalElement, gl_data: GlobalData) -> tuple:
    """
    Calculates H, C, Hbc matrices and P vector for the element.

    Args:
        nodes (np.ndarray[Node]): Array of nodes for the element.
        u_el (UniversalElement): Universal element containing shape functions and derivatives.
        gl_data (GlobalData): Global data containing material properties.

    Returns:
        tuple: H, C, Hbc matrices and P vector.
    """
    x_coords, y_coords = _fill_x_y_coords(nodes)
    #config.logger.debug(f"X coordinates: {x_coords}")
    #config.logger.debug(f"Y coordinates: {y_coords}")
    dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab = _fill_x_y_ksi_eta_tabs(x_coords, y_coords, u_el)
    #config.logger.debug(f"dx/dksi: {dx_dksi_tab}")
    #config.logger.debug(f"dx/deta: {dx_deta_tab}")
    #config.logger.debug(f"dy/dksi: {dy_dksi_tab}")
    #config.logger.debug(f"dy/deta: {dy_deta_tab}")
    dn_dx_tab, dn_dy_tab, det_tab = _dn_dx_dn_dy(dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab, u_el)
    #config.logger.debug(f"dN/dx: {dn_dx_tab}")
    #config.logger.debug(f"dN/dy: {dn_dy_tab}")
    #config.logger.debug(f"det[J]: {det_tab}")
    ip_h_matrices, ip_c_matrices = _calculate_for_integration_points(dn_dx_tab, dn_dy_tab, u_el.n_tab, det_tab, u_el.n, gl_data.conductivity, gl_data.density, gl_data.specific_heat)
    #config.logger.debug(f"H matrices: {ip_h_matrices}")
    #config.logger.debug(f"C matrices: {ip_c_matrices}")
    H = np.zeros((4, 4))
    C = np.zeros((4, 4))
    Hbc = np.zeros((4, 4))
    P = np.zeros((4, 1))

    for i in range(0, u_el.n*u_el.n):
        H += ip_h_matrices[i]*(u_el.weights[i//u_el.n])*(u_el.weights[i%u_el.n])
        C += ip_c_matrices[i]*(u_el.weights[i//u_el.n])*(u_el.weights[i%u_el.n])

    tempList = []
    tempList.append(_calculate_for_surface(u_el.surfaces[0], (nodes[0], nodes[1]), u_el.weights, u_el.n, gl_data.alfa, gl_data.tot))
    tempList.append(_calculate_for_surface(u_el.surfaces[1], (nodes[1], nodes[2]), u_el.weights, u_el.n, gl_data.alfa, gl_data.tot))
    tempList.append(_calculate_for_surface(u_el.surfaces[2], (nodes[2], nodes[3]), u_el.weights, u_el.n, gl_data.alfa, gl_data.tot))
    tempList.append(_calculate_for_surface(u_el.surfaces[3], (nodes[3], nodes[0]), u_el.weights, u_el.n, gl_data.alfa, gl_data.tot))
    for surfaceHbc, surfaceP in tempList:
        Hbc+=surfaceHbc
        P+=surfaceP

    #config.logger.debug(f"H matrix: {H}")
    #config.logger.debug(f"C matrix: {C}")
    #config.logger.debug(f"Hbc matrix: {Hbc}")
    #config.logger.debug(f"P vector: {P}")
    return H, C, Hbc, P

def _fill_x_y_coords(nodes: np.ndarray[Node]) -> tuple[list[float]]:
    """
    Fills lists storing x and y coordinates of nodes belonging to the element.

    Args:
        nodes (np.ndarray[Node]): Array of nodes for the element.

    Returns:
        tuple: Lists of x and y coordinates.
    """
    x_coords = []; y_coords = []
    for i in range(0, 4):
        x_coords.append(nodes[i].x)
        y_coords.append(nodes[i].y)
    return x_coords, y_coords

def _fill_x_y_ksi_eta_tabs(x_coords: list[float], y_coords: list[float], u_el: UniversalElement) -> tuple[list[float]]:
    """
    Calculates dx/dksi, dx/deta, dy/dksi, dy/deta for every integration point and returns 4 tables with output.

    Args:
        x_coords (list[float]): List of x coordinates.
        y_coords (list[float]): List of y coordinates.
        u_el (UniversalElement): Universal element containing shape functions and derivatives.

    Returns:
        tuple: Lists of dx/dksi, dx/deta, dy/dksi, dy/deta.
    """
    dx_dksi_tab = []; dx_deta_tab = []; dy_dksi_tab = []; dy_deta_tab = []
    for i in range(0, u_el.n*u_el.n):
        dx_dksi_tab.append(_interpolate(u_el.dn_dksi_tab[i][0], u_el.dn_dksi_tab[i][1], u_el.dn_dksi_tab[i][2], u_el.dn_dksi_tab[i][3], x_coords))
        dx_deta_tab.append(_interpolate(u_el.dn_deta_tab[i][0], u_el.dn_deta_tab[i][1], u_el.dn_deta_tab[i][2], u_el.dn_deta_tab[i][3], x_coords))
        dy_dksi_tab.append(_interpolate(u_el.dn_dksi_tab[i][0], u_el.dn_dksi_tab[i][1], u_el.dn_dksi_tab[i][2], u_el.dn_dksi_tab[i][3], y_coords))
        dy_deta_tab.append(_interpolate(u_el.dn_deta_tab[i][0], u_el.dn_deta_tab[i][1], u_el.dn_deta_tab[i][2], u_el.dn_deta_tab[i][3], y_coords))
    return dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab

def _dn_dx_dn_dy(dx_dksi_tab: list, dx_deta_tab: list, dy_dksi_tab: list, dy_deta_tab: list, u_el: UniversalElement) -> tuple[list[float]]:
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
    def initialize_dNdXdNdY(n: int) -> tuple[list[list]]:
        """
        Initializes empty tables for dN/dx and dN/dy calculations.

        Args:
            n (int): Number of integration points.

        Returns:
            tuple: Empty tables for dN/dx and dN/dy.
        """
        dn_dx_tab = []; dn_dy_tab = []
        for j in range(n*n):
            dn_dx_tab.append([])
            dn_dy_tab.append([])
            for i in range (0, 4):
                dn_dx_tab[j].append(0)
                dn_dy_tab[j].append(0)
        return dn_dx_tab, dn_dy_tab

    dn_dx_tab, dn_dy_tab = initialize_dNdXdNdY(u_el.n)
    det_tab = []
    for j in range(u_el.n*u_el.n):
        mxJ = np.array([[dx_dksi_tab[j], dy_dksi_tab[j]],
                            [dx_deta_tab[j], dy_deta_tab[j]]])
        detJ = np.linalg.det(mxJ)
        det_tab.append(detJ)
        mx1 = np.array([[dy_deta_tab[j], -dy_dksi_tab[j]],
                            [-dx_deta_tab[j], dx_dksi_tab[j]]])
        for i in range (0, 4):
            mx2 = np.array([[u_el.dn_dksi_tab[j][i]],[u_el.dn_deta_tab[j][i]]])
            # (1/detJ)*mx1 = mxJ^(-1)
            mxOutput = np.matmul(((1/detJ)*mx1), mx2)
            dn_dx_tab[j][i] = mxOutput[0][0]
            dn_dy_tab[j][i] = mxOutput[1][0]
    return dn_dx_tab, dn_dy_tab, det_tab

def _calculate_for_integration_points(dn_dx_tab: list, dn_dy_tab: list, NTab: list, det_tab: list, n: int, c: int, d: int, sH: int) -> tuple[np.ndarray]:
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
    mxHTab = []
    mxCTab = []
    for i in range(0, n*n):
        mxDNdX = np.array([[dn_dx_tab[i][0]],
                            [dn_dx_tab[i][1]],
                            [dn_dx_tab[i][2]],
                            [dn_dx_tab[i][3]]])
        mxDNdY = np.array([[dn_dy_tab[i][0]],
                            [dn_dy_tab[i][1]],
                            [dn_dy_tab[i][2]],
                            [dn_dy_tab[i][3]]])
        mxN = np.array([[NTab[i][0]],
                            [NTab[i][1]],
                            [NTab[i][2]],
                            [NTab[i][3]]])
        ipMxH = c*(np.matmul(mxDNdX, mxDNdX.transpose()) + np.matmul(mxDNdY, mxDNdY.transpose()))*det_tab[i]
        ipMxC = sH*d*(np.matmul(mxN, mxN.transpose()))*det_tab[i]
        mxHTab.append(ipMxH)
        mxCTab.append(ipMxC)
    return mxHTab, mxCTab

def _calculate_for_surface(surface: Surface, nodes: tuple[Node], weights: np.ndarray[float], n: int, alfa: int, tot: int) -> tuple:
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
    Hbc = np.zeros((4, 4))
    P = np.zeros((4, 1))
    if nodes[0].BC == 0 or nodes[1].BC == 0:
        return Hbc, P
    L = sqrt(pow(nodes[0].x-nodes[1].x, 2)+pow(nodes[0].y-nodes[1].y, 2))
    detJ = L/2
    for i in range(0, n):
        mx = np.array([[surface.N[i][0]],
                        [surface.N[i][1]],
                        [surface.N[i][2]],
                        [surface.N[i][3]]])
        Hbc += np.matmul(mx, mx.transpose())*weights[i]
        P += mx*weights[i]
    Hbc = Hbc*alfa*detJ
    P = P*alfa*tot*detJ
    return Hbc, P

def _interpolate(dN1: float, dN2: float, dN3: float, dN4: float, var: list[float]) -> float:
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