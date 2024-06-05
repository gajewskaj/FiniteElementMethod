from common import *
from universal_element import UniversalElement, Surface
from grid import Grid, GlobalData, Node
from math import *
import numpy as np

class LocalMatricesCalculation:
    """
    Abstract class for calculating of H, C, Hbc matrices and P vector for each element of the given grid.
    """
    def __init__(self):
        raise FiniteElementMethodException('LocalMatricesCalculation is an abstract class, you cannot create an instance of this class.')

    @staticmethod
    def calculate(n: int, grid: Grid) -> None:
        """
        Calculates H, C, Hbc matrices and P vector for each element in the grid, output is stored in Element class.
        """
        u_el = UniversalElement(n)
        for element in grid.elements:
            element.H, element.C, element.Hbc, element.P = LocalMatricesCalculation._calculate_for_element([grid.nodes[element.node_ids[0] - 1],
                                      grid.nodes[element.node_ids[1] - 1],
                                      grid.nodes[element.node_ids[2] - 1],
                                      grid.nodes[element.node_ids[3] - 1]],
                                      u_el, grid.global_data)
            #print(f'H:\n{element.H}\nC:{element.C}\nHbc:\n{element.Hbc}\nP:\n{element.P}')

    @staticmethod
    def _calculate_for_element(nodes: list[Node], u_el: UniversalElement, gl_data: GlobalData) -> None:
        """
        Calculates H, C, Hbc marices and P vector for the element.
        """
        x_coords, y_coords = LocalMatricesCalculation._fill_x_y_coords(nodes)
        dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab = LocalMatricesCalculation._fill_x_y_ksi_eta_tabs(x_coords, y_coords, u_el)
        dn_dx_tab, dn_dy_tab, det_tab = LocalMatricesCalculation._dn_dx_dn_dy(dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab, u_el)
        ip_h_matrices, ip_c_matrices = LocalMatricesCalculation._calculate_for_integration_points(dn_dx_tab, dn_dy_tab, u_el.n_tab, det_tab, u_el.n, gl_data.conductivity, gl_data.density, gl_data.specific_heat)

        H = np.zeros((4, 4))
        C = np.zeros((4, 4))
        Hbc = np.zeros((4, 4))
        P = np.zeros((4, 1))

        for i in range(0, u_el.n*u_el.n):
            H += ip_h_matrices[i]*(u_el.weights[i//u_el.n])*(u_el.weights[i%u_el.n])
            C += ip_c_matrices[i]*(u_el.weights[i//u_el.n])*(u_el.weights[i%u_el.n])

        tempList = []
        tempList.append(LocalMatricesCalculation._calculate_for_surface(u_el.surfaces[0], (nodes[0], nodes[1]), u_el.weights, u_el.n, gl_data.alfa, gl_data.tot))
        tempList.append(LocalMatricesCalculation._calculate_for_surface(u_el.surfaces[1], (nodes[1], nodes[2]), u_el.weights, u_el.n, gl_data.alfa, gl_data.tot))
        tempList.append(LocalMatricesCalculation._calculate_for_surface(u_el.surfaces[2], (nodes[2], nodes[3]), u_el.weights, u_el.n, gl_data.alfa, gl_data.tot))
        tempList.append(LocalMatricesCalculation._calculate_for_surface(u_el.surfaces[3], (nodes[3], nodes[0]), u_el.weights, u_el.n, gl_data.alfa, gl_data.tot))
        for surfaceHbc, surfaceP in tempList:
            Hbc+=surfaceHbc
            P+=surfaceP
        return H, C, Hbc, P

    @staticmethod
    def _fill_x_y_coords(nodes: list[Node]) -> tuple[list[float]]:
        """
        Fills lists storing x and y coords of nodes belonging to the element.
        """
        x_coords = []; y_coords = []
        for i in range(0, 4):
            x_coords.append(nodes[i].x)
            y_coords.append(nodes[i].y)
        #print(f'x_coords: {x_coords}\ny_coords: {y_coords}')
        return x_coords, y_coords

    @staticmethod
    def _fill_x_y_ksi_eta_tabs(x_coords: list[float], y_coords: list[float], u_el: UniversalElement) -> tuple[list[float]]:
        """
        Calculates dx/dksi, dx/deta, dy/dksi, dy/deta for every integration point and returns 4 tables with output.
        """
        dx_dksi_tab = []; dx_deta_tab = []; dy_dksi_tab = []; dy_deta_tab = []
        for i in range(0, u_el.n*u_el.n):
            dx_dksi_tab.append(LocalMatricesCalculation._interpolate(u_el.dn_dksi_tab[i][0], u_el.dn_dksi_tab[i][1], u_el.dn_dksi_tab[i][2], u_el.dn_dksi_tab[i][3], x_coords))
            dx_deta_tab.append(LocalMatricesCalculation._interpolate(u_el.dn_deta_tab[i][0], u_el.dn_deta_tab[i][1], u_el.dn_deta_tab[i][2], u_el.dn_deta_tab[i][3], x_coords))
            dy_dksi_tab.append(LocalMatricesCalculation._interpolate(u_el.dn_dksi_tab[i][0], u_el.dn_dksi_tab[i][1], u_el.dn_dksi_tab[i][2], u_el.dn_dksi_tab[i][3], y_coords))
            dy_deta_tab.append(LocalMatricesCalculation._interpolate(u_el.dn_deta_tab[i][0], u_el.dn_deta_tab[i][1], u_el.dn_deta_tab[i][2], u_el.dn_deta_tab[i][3], y_coords))
        #print(f'dx_dksi_tab: {dx_dksi_tab}\ndx_deta_tab: {dx_deta_tab}\ndy_dksi_tab: {dy_dksi_tab}\ndy_deta_tab: {dy_deta_tab}')
        return dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab

    @staticmethod
    def _dn_dx_dn_dy(dx_dksi_tab: list, dx_deta_tab: list, dy_dksi_tab: list, dy_deta_tab: list, u_el: UniversalElement) -> tuple[list[float]]:
        """
        Fills dN/dx and dN/dy tables and calculates Jacobian and det[J].

        mxJ: Jacobian matrix
        detJ: Jacobian determinant
        """
        def initialize_dNdXdNdY(n: int) -> tuple[list[list]]:
            """
            Initializes empty tables for dN/dx and dN/dy calculations.
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
            #print(f'Jacobian matrix:\n{mxJ}')
            detJ = np.linalg.det(mxJ)
            #print(f'Jacobian determinant:\n{detJ}')
            det_tab.append(detJ)
            mx1 = np.array([[dy_deta_tab[j], -dy_dksi_tab[j]],
                             [-dx_deta_tab[j], dx_dksi_tab[j]]])
            for i in range (0, 4):
                mx2 = np.array([[u_el.dn_dksi_tab[j][i]],[u_el.dn_deta_tab[j][i]]])
                # (1/detJ)*mx1 = mxJ^(-1)
                mxOutput = np.matmul(((1/detJ)*mx1), mx2)
                dn_dx_tab[j][i] = mxOutput[0][0]
                dn_dy_tab[j][i] = mxOutput[1][0]
        #print('dn_dx_tab:')
        #print2dTab(dn_dx_tab)
        #print('dn_dy_tab:')
        #print2dTab(dn_dy_tab)
        return dn_dx_tab, dn_dy_tab, det_tab

    @staticmethod
    def _calculate_for_integration_points(dn_dx_tab: list, dn_dy_tab: list, NTab: list, det_tab: list, n: int, c: int, d: int, sH: int) -> tuple[np.ndarray]:
        """
        Calculates H, C matrices for each integration point. Returns list of matrices.
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
            #print(f'IP {i+1}:\n{ipMxH}')
            mxHTab.append(ipMxH)
            mxCTab.append(ipMxC)
        return mxHTab, mxCTab

    @staticmethod
    def _calculate_for_surface(surface: Surface, nodes: tuple[Node], weights: list[float], n: int, alfa: int, tot: int) -> tuple:
        """
        Calculates Hbc matrix and P vector for the given surface of the element.
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

    @staticmethod
    def _interpolate(dN1: float, dN2: float, dN3: float, dN4: float, var: list[float]) -> float:
        """
        Returns dx/deta, dx/dksi, dy/deta or dy/dksi depending on given arguments.
        """
        return dN1*var[0] + dN2*var[1] + dN3*var[2] + dN4*var[3]