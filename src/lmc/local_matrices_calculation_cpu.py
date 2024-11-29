from math import sqrt

import numpy as np

from src.helpers.helpers import measure_time
from src.grid.grid import Grid, Element
from src.uel.universal_element import u_el, NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SURFACES

@measure_time
def calculate_local_matrices(grid: Grid) -> None:
    x_coords = np.empty((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS))
    y_coords = np.empty((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS))
    bc = np.empty((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS))
    for i, element in enumerate(grid.elements):
        element: Element
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            x_coords[i][j] = grid.nodes[element.node_ids[j] - 1].x
            y_coords[i][j] = grid.nodes[element.node_ids[j] - 1].y
            bc[i][j] = grid.nodes[element.node_ids[j] - 1].BC
    for i, element in enumerate(grid.elements):
        element: Element
        _calculate_Hbc_P_for_element(x_coords[i], y_coords[i], bc[i],
                                     grid.global_data.alpha, grid.global_data.ambient_temp,
                                     u_el.quadrature_1d.n, u_el.quadrature_1d.weights, u_el.surfaces,
                                     element.Hbc, element.P)
        _calculate_H_C_for_element(x_coords[i], y_coords[i],
                                   u_el.n, u_el.weights, u_el.N,
                                   u_el.dN_dxi, u_el.dN_deta,
                                   grid.global_data.conductivity, grid.global_data.density, grid.global_data.specific_heat,
                                   element.H, element.C)

def _calculate_H_C_for_element(x_coords: np.ndarray, y_coords: np.ndarray,
                               n: int, weights: np.ndarray, N: np.ndarray,
                               dN_dxi: np.ndarray, dN_deta: np.ndarray,
                               c: float, d: float, sh: float,
                               H: np.ndarray, C: np.ndarray) -> None:
    for i in range(n):
        dN_dx = np.empty(NUM_OF_SHAPE_FUNCTIONS)
        dN_dy = np.empty(NUM_OF_SHAPE_FUNCTIONS)
        dx_dxi = _interpolate(dN_dxi[i], x_coords)
        dx_deta = _interpolate(dN_deta[i], x_coords)
        dy_dxi = _interpolate(dN_dxi[i], y_coords)
        dy_deta = _interpolate(dN_deta[i], y_coords)
        jacobian_det = _calculate_jacobian_and_global_shape_derivatives(dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                                        dN_dxi[i], dN_deta[i],
                                                                        dN_dx, dN_dy)
        _calculate_H_C_for_integration_point(weights[i], N[i],
                                             jacobian_det, dN_dx, dN_dy,
                                             c, d, sh, H, C)

def _interpolate(dN: np.ndarray, var: np.ndarray) -> float:
    return sum([dN*var[i] for i, dN in enumerate(dN)])

def _calculate_jacobian_and_global_shape_derivatives(dx_dxi: float, dx_deta: float, dy_dxi: float, dy_deta: float,
                                                     dN_dxi: np.ndarray, dN_deta: np.ndarray,
                                                     dN_dx: np.ndarray, dN_dy: np.ndarray) -> float:
    jacobian = np.array([[dx_dxi, dy_dxi],
                         [dx_deta, dy_deta]])
    jacobian_det = jacobian[0][0]*jacobian[1][1] - jacobian[0][1]*jacobian[1][0]
    jacobian_inv = np.array([[dy_deta, -dy_dxi],
                             [-dx_deta, dx_dxi]])/jacobian_det
    for i in range(NUM_OF_SHAPE_FUNCTIONS):
        local_shape_function_derivatives = np.array([[dN_dxi[i]],
                                                     [dN_deta[i]]])
        global_shape_function_derivatives = np.matmul(jacobian_inv, local_shape_function_derivatives)
        dN_dx[i] = global_shape_function_derivatives[0][0]
        dN_dy[i] = global_shape_function_derivatives[1][0]
    return jacobian_det

def _calculate_H_C_for_integration_point(weight: float, N: np.ndarray,
                                         jacobian_det: float, dN_dx: np.ndarray, dN_dy: np.ndarray,
                                         c: float, d: float, sh: float, H: np.ndarray, C: np.ndarray) -> None:
    dN_dx = np.array(dN_dx).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
    dN_dy = np.array(dN_dy).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
    N = np.array(N).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)

    H += c*(np.matmul(dN_dx, dN_dx.transpose()) + np.matmul(dN_dy, dN_dy.transpose()))*jacobian_det*weight
    C += sh*d*(np.matmul(N, N.transpose()))*jacobian_det*weight

def _calculate_Hbc_P_for_element(node_x_coords: np.ndarray, node_y_coords: np.ndarray, node_bc: np.ndarray,
                                 alpha: float, ambient_temp: float,
                                 n: int, weights: np.ndarray, surfaces: np.ndarray,
                                 Hbc: np.ndarray, P: np.ndarray) -> None:
    for i in range(NUM_OF_SURFACES):
        _calculate_for_surface(n, weights, surfaces[i],
                               node_x_coords[i], node_y_coords[i], node_bc[i],
                               node_x_coords[(i+1)%NUM_OF_SHAPE_FUNCTIONS], node_y_coords[(i+1)%NUM_OF_SHAPE_FUNCTIONS], node_bc[(i+1)%NUM_OF_SHAPE_FUNCTIONS],
                               alpha, ambient_temp,
                               Hbc, P)

def _calculate_for_surface(n: int, weights: np.ndarray, surface: np.ndarray,
                           node1_x: float, node1_y: float, node1_bc: int,
                           node2_x: float, node2_y: float, node2_bc: int,
                           alpha: float, ambient_temp: float,
                           Hbc: np.ndarray, P: np.ndarray) -> None:
    Hbc_surf = np.zeros((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS))
    P_surf = np.zeros((NUM_OF_SHAPE_FUNCTIONS, 1))
    if node1_bc == 0 or node2_bc == 0:
        return
    L = sqrt((node2_x - node1_x)**2 + (node2_y - node1_y)**2)
    jacobian_det = L/2
    for i in range(n):
        N = np.array(surface[i]).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
        Hbc_surf += np.matmul(N, N.transpose())*weights[i]
        P_surf += N*weights[i]
    Hbc += Hbc_surf*alpha*jacobian_det
    P += P_surf*alpha*ambient_temp*jacobian_det