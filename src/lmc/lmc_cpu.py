from math import sqrt

import numpy as np

from src.helpers.helpers import measure_time
from src.helpers import config
from src.grid.grid import Grid, Element, Node
from src.universal_element.universal_element import universal_element, NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SURFACES

@measure_time
def calculate_local_matrices(n: int, grid: Grid) -> None:
    u_el = universal_element
    for element in grid.elements:
        element: Element
        nodes = [grid.nodes[node_id - 1] for node_id in element.node_ids]
        x_coords, y_coords = _fill_x_y_coords(nodes)
        _calculate_H_C_for_element(element.H, element.C,
                                   x_coords, y_coords,
                                   grid.global_data.conductivity, grid.global_data.density, grid.global_data.specific_heat,
                                   u_el.n, u_el.weights, u_el.N,
                                   u_el.dN_dxi, u_el.dN_deta)
        _calculate_Hbc_P_for_element(element.Hbc, element.P,
                                    nodes,
                                    grid.global_data.alpha, grid.global_data.ambient_temp,
                                    u_el.gaussian_quadrature.n, u_el.gaussian_quadrature.weights, u_el.surfaces)

def _fill_x_y_coords(nodes: list[Node]) -> tuple[np.ndarray, np.ndarray]:
    x_coords = np.ndarray(NUM_OF_SHAPE_FUNCTIONS, dtype=np.float32)
    y_coords = np.ndarray(NUM_OF_SHAPE_FUNCTIONS, dtype=np.float32)
    for i in range(NUM_OF_SHAPE_FUNCTIONS):
        x_coords[i] = nodes[i].x
        y_coords[i] = nodes[i].y
    return x_coords, y_coords

def _calculate_H_C_for_element(H: np.ndarray[np.float32], C: np.ndarray[np.float32],
                               x_coords: np.ndarray[np.float32], y_coords: np.ndarray[np.float32],
                               c: np.float32, d: np.float32, sh: np.float32,
                               n: int, weights: np.ndarray[np.float32], N: np.ndarray[np.float32],
                               dN_dxi: np.ndarray[np.float32], dN_deta: np.ndarray[np.float32]):
    for i in range(n):
        dx_dxi = _interpolate(dN_dxi[i], x_coords)
        dx_deta = _interpolate(dN_deta[i], x_coords)
        dy_dxi = _interpolate(dN_dxi[i], y_coords)
        dy_deta = _interpolate(dN_deta[i], y_coords)
        jacobian_det, dN_dx, dN_dy = _calculate_jacobian_and_global_shape_derivatives(dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                                                      dN_dxi[i], dN_deta[i])
        _calculate_H_C_for_integration_point(H, C,
                                             weights[i], N[i],
                                             jacobian_det, dN_dx, dN_dy,
                                             c, d, sh)

def _interpolate(dN: np.ndarray[np.float32], var: list[np.float32]) -> np.float32:
    return sum([dN*var[i] for i, dN in enumerate(dN)])

def _calculate_jacobian_and_global_shape_derivatives(dx_dxi: float, dx_deta: float, dy_dxi: float, dy_deta: float,
                                                    dN_dxi: np.ndarray[np.float32], dN_deta: np.ndarray[np.float32]) -> tuple:
    dN_dx = np.empty(NUM_OF_SHAPE_FUNCTIONS)
    dN_dy = np.empty(NUM_OF_SHAPE_FUNCTIONS)
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
    return jacobian_det, dN_dx, dN_dy

def _calculate_H_C_for_integration_point(H: np.ndarray[np.float32], C: np.ndarray[np.float32],
                                         weight: float, N: np.ndarray[np.float32],
                                         jacobian_det: np.ndarray[np.float32], dN_dx: np.ndarray[np.float32], dN_dy: np.ndarray[np.float32],
                                         c: np.float32, d: np.float32, sh: np.float32):
    dN_dx = np.array(dN_dx).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
    dN_dy = np.array(dN_dy).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
    N = np.array(N).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)

    H += c*(np.matmul(dN_dx, dN_dx.transpose()) + np.matmul(dN_dy, dN_dy.transpose()))*jacobian_det*weight
    C += sh*d*(np.matmul(N, N.transpose()))*jacobian_det*weight

def _calculate_Hbc_P_for_element(Hbc: np.ndarray[np.float32], P: np.ndarray[np.float32],
                                 nodes: list[Node],
                                 alpha: np.float32, ambient_temp: np.float32,
                                 n: int, weights: np.ndarray[np.float32], surfaces: np.ndarray[np.float32]):
    for i in range(NUM_OF_SURFACES):
        _calculate_for_surface(Hbc, P, n, weights, surfaces[i], (nodes[i], nodes[(i + 1) % NUM_OF_SURFACES]), alpha, ambient_temp, i)

def _calculate_for_surface(Hbc: np.ndarray[np.float32], P: np.ndarray[np.float32],
                           n: int, weights: np.ndarray[np.float32], surface: np.ndarray,
                           nodes: np.ndarray[Node],
                           alpha, ambient_temp,
                           surf_num) -> tuple:
    Hbc_surf = np.zeros((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS))
    P_surf = np.zeros((NUM_OF_SHAPE_FUNCTIONS, 1))
    if nodes[0].BC == 0 or nodes[1].BC == 0:
        return
    L = sqrt(pow(nodes[0].x-nodes[1].x, 2)+pow(nodes[0].y-nodes[1].y, 2))
    jacobian_det = L/2
    for i in range(n):
        N = np.array(surface[i]).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
        if config.element_type == "triangle" and surf_num == 1:
            Hbc_surf += np.matmul(N, N.transpose())*weights[i]*sqrt(2)
            P_surf += N*weights[i]*sqrt(2)
        else:
            Hbc_surf += np.matmul(N, N.transpose())*weights[i]
            P_surf += N*weights[i]
    Hbc += Hbc_surf*alpha*jacobian_det
    P += P_surf*alpha*ambient_temp*jacobian_det