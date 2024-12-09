from math import sqrt

from numba import jit
import numpy as np

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid
from src.uel.universal_element import u_el

NUM_OF_SHAPE_FUNCTIONS = config.num_of_shape_functions
NUM_OF_SURFACES = NUM_OF_SHAPE_FUNCTIONS

@measure_time
def calculate_and_assemble_matrices(grid: Grid) -> None:
    x_coords = np.empty((len(grid.elements_id), NUM_OF_SHAPE_FUNCTIONS))
    y_coords = np.empty((len(grid.elements_id), NUM_OF_SHAPE_FUNCTIONS))
    bc = np.empty((len(grid.elements_id), NUM_OF_SHAPE_FUNCTIONS), dtype=np.int64)
    for i in range(len(grid.elements_id)):
        H = np.zeros((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS))
        C = np.zeros((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS))
        Hbc = np.zeros((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS))
        P = np.zeros(NUM_OF_SHAPE_FUNCTIONS)
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            x_coords[i, j] = grid.nodes_x[grid.elements_node_ids[i, j] - 1]
            y_coords[i, j] = grid.nodes_y[grid.elements_node_ids[i, j] - 1]
            bc[i, j] = grid.nodes_bc[grid.elements_node_ids[i, j] - 1]
        _calculate_H_C_for_element(x_coords[i], y_coords[i],
                                   u_el.n, u_el.weights, u_el.N,
                                   u_el.dN_dxi, u_el.dN_deta,
                                   grid.global_data.conductivity, grid.global_data.density, grid.global_data.specific_heat,
                                   H, C)
        _calculate_Hbc_P_for_element(x_coords[i], y_coords[i], bc[i],
                                     u_el.quadrature_1d.n, u_el.quadrature_1d.weights, u_el.surfaces,
                                     grid.global_data.alpha, grid.global_data.ambient_temp,
                                     Hbc, P)
        # Assembly
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            grid.global_P[grid.elements_node_ids[i, j] - 1] += P[j]
            for k in range(NUM_OF_SHAPE_FUNCTIONS):
                grid.global_H_values[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = H[j, k]
                grid.global_H_row[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = grid.elements_node_ids[i, j] - 1
                grid.global_H_col[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = grid.elements_node_ids[i, k] - 1
                grid.global_C_values[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = C[j, k]
                grid.global_C_row[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = grid.elements_node_ids[i, j] - 1
                grid.global_C_col[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = grid.elements_node_ids[i, k] - 1
                grid.global_Hbc_values[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = Hbc[j, k]
                grid.global_Hbc_row[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = grid.elements_node_ids[i, j] - 1
                grid.global_Hbc_col[i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k] = grid.elements_node_ids[i, k] - 1
    grid.global_P = grid.global_P.reshape(-1, 1)

@jit('float64(float64[:], float64[:], float64, float64, float64, float64, float64[:], float64[:])', nopython=True)
def _calculate_jacobian_and_global_shape_derivatives(dN_dxi, dN_deta,
                                                     dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                     dN_dx, dN_dy):
    jacobian = np.array([[dx_dxi, dy_dxi], [dx_deta, dy_deta]])
    jacobian_det = np.linalg.det(jacobian)
    jacobian_inv = np.linalg.inv(jacobian)
    for k in range(NUM_OF_SHAPE_FUNCTIONS):
        global_shape_function_derivatives = np.dot(jacobian_inv, np.array([dN_dxi[k], dN_deta[k]]))
        dN_dx[k] = global_shape_function_derivatives[0]
        dN_dy[k] = global_shape_function_derivatives[1]
    return jacobian_det

@jit('float64(float64[:], float64[:])')
def _interpolate(dN, var):
    return sum([dN[i]*var[i] for i in range(len(dN))])

@jit('void(float64, float64[:], float64, float64[:], float64[:], float64, float64, float64, float64[:,:], float64[:,:])', nopython=True)
def _calculate_H_C_for_integration_point(weight, N,
                                         jacobian_det, dN_dx, dN_dy,
                                         c, d, sh,
                                         H, C):
    dN_dx = np.ascontiguousarray(dN_dx).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
    dN_dy = np.ascontiguousarray(dN_dy).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
    N = np.ascontiguousarray(N).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
    H += c*(np.dot(dN_dx, dN_dx.transpose()) + np.dot(dN_dy, dN_dy.transpose()))*jacobian_det*weight
    C += sh*d*(np.dot(N, N.transpose()))*jacobian_det*weight

@jit('void(float64[:], float64[:], int64, float64[:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, float64[:,:], float64[:,:])', nopython=True)
def _calculate_H_C_for_element(x_coords, y_coords,
                               n, weights, N,
                               dN_dxi, dN_deta,
                               c, d, sh,
                               H, C):
    for i in range(n):
        dN_dx = np.empty(NUM_OF_SHAPE_FUNCTIONS)
        dN_dy = np.empty(NUM_OF_SHAPE_FUNCTIONS)
        dx_dxi = _interpolate(dN_dxi[i], x_coords)
        dx_deta = _interpolate(dN_deta[i], x_coords)
        dy_dxi = _interpolate(dN_dxi[i], y_coords)
        dy_deta = _interpolate(dN_deta[i], y_coords)
        jacobian_det = _calculate_jacobian_and_global_shape_derivatives(dN_dxi[i], dN_deta[i],
                                                                        dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                                        dN_dx, dN_dy)
        _calculate_H_C_for_integration_point(weights[i], N[i],
                                             jacobian_det, dN_dx, dN_dy,
                                             c, d, sh, H, C)

@jit('void(int64, float64[:], float64[:,:], float64, float64, int64, float64, float64, int64, float64, float64, float64[:,:], float64[:])', nopython=True)
def _calculate_for_surface(n, weights, surface,
                           node1_x, node1_y, node1_bc,
                           node2_x, node2_y, node2_bc,
                           alpha, ambient_temp,
                           Hbc, P):
    if node1_bc == 0 or node2_bc == 0:
        return
    L = sqrt((node2_x - node1_x)**2 + (node2_y - node1_y)**2)
    jacobian_det = L / 2
    for i in range(n):
        N = np.ascontiguousarray(surface[i]).reshape(NUM_OF_SHAPE_FUNCTIONS, 1)
        Hbc += np.dot(N, N.transpose())*weights[i]*alpha*jacobian_det
        P += N*weights[i]*alpha*ambient_temp*jacobian_det

@jit('void(float64[:], float64[:], int64[:], int64, float64[:], float64[:,:,:], float64, float64, float64[:,:], float64[:])', nopython=True)
def _calculate_Hbc_P_for_element(x_coords, y_coords, bc,
                                 n, weights, surfaces,
                                 alpha, ambient_temp,
                                 Hbc, P):
    for i in range(NUM_OF_SURFACES):
        _calculate_for_surface(n, weights, surfaces[i],
                               x_coords[i], y_coords[i], bc[i],
                               x_coords[(i+1)%NUM_OF_SHAPE_FUNCTIONS], y_coords[(i+1)%NUM_OF_SHAPE_FUNCTIONS], bc[(i+1)%NUM_OF_SHAPE_FUNCTIONS],
                               alpha, ambient_temp,
                               Hbc, P)