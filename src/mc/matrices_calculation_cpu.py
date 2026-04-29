from math import sqrt

from numba import jit
import numpy as np

from src.helpers.config import Settings
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh
from src.uel.universal_element import u_el

NUM_DOF = Settings.MatricesCalculation.DOF
NUM_SURFACES = NUM_DOF

@measure_time
def calculate_and_assemble_matrices(grid: Mesh) -> None:
    for i in range(len(grid.elements_id)):
        _calculate_H_C_for_element(i, grid.nodes_x, grid.nodes_y, grid.elements_node_ids,
                                   u_el.n, u_el.weights, u_el.N,
                                   u_el.dN_dxi, u_el.dN_deta,
                                   grid.global_data.conductivity, grid.global_data.density, grid.global_data.specific_heat,
                                   grid.H_val, grid.H_row, grid.H_col,
                                   grid.C_val, grid.C_row, grid.C_col)
        _calculate_Hbc_P_for_element(i, grid.nodes_x, grid.nodes_y, grid.nodes_bc, grid.elements_node_ids,
                                     u_el.quadrature_1d.n, u_el.quadrature_1d.weights, u_el.surfaces,
                                     grid.global_data.alpha, grid.global_data.ambient_temp,
                                     grid.Hbc_val, grid.Hbc_row, grid.Hbc_col, grid.P)

@jit('float64(float64[:], float64[:], float64, float64, float64, float64, float64[:], float64[:])')
def _calculate_jacobian_and_global_shape_derivatives(dN_dxi, dN_deta,
                                                     dx_dxi, dx_deta,
                                                     dy_dxi, dy_deta,
                                                     dN_dx, dN_dy):
    jacobian = np.array([[dx_dxi, dy_dxi], [dx_deta, dy_deta]])
    jacobian_det = np.linalg.det(jacobian)
    jacobian_inv = np.linalg.inv(jacobian)
    for i in range(NUM_DOF):
        global_shape_function_derivatives = np.dot(jacobian_inv,
            np.array([dN_dxi[i], dN_deta[i]]))
        dN_dx[i] = global_shape_function_derivatives[0]
        dN_dy[i] = global_shape_function_derivatives[1]
    return jacobian_det

@jit('float64(float64[:], float64[:])')
def _interpolate(dN, var):
    return sum([dN[i]*var[i] for i in range(len(dN))])

@jit('void(float64, float64[:], float64, float64[:], float64[:], float64, float64, float64, float64[:,:], float64[:,:])')
def _calculate_H_C_for_integration_point(weight, N,
                                         jacobian_det, dN_dx, dN_dy,
                                         c, d, sh,
                                         H, C):
    dN_dx = np.ascontiguousarray(dN_dx).reshape(NUM_DOF, 1)
    dN_dy = np.ascontiguousarray(dN_dy).reshape(NUM_DOF, 1)
    N = np.ascontiguousarray(N).reshape(NUM_DOF, 1)
    H += c*(np.dot(dN_dx, dN_dx.transpose()) +
            np.dot(dN_dy, dN_dy.transpose()))*jacobian_det*weight
    C += sh*d*(np.dot(N, N.transpose()))*jacobian_det*weight

@jit('void(int64, float64[:], float64[:], int64[:,:], int64, float64[:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, float64[:], int64[:], int64[:], float64[:], int64[:], int64[:])')
def _calculate_H_C_for_element(i, nodes_x, nodes_y, elements_node_ids,
                               n, weights, N,
                               dN_dxi, dN_deta,
                               c, d, sh,
                               H_val, H_row, H_col,
                               C_val, C_row, C_col):
    H = np.zeros((NUM_DOF, NUM_DOF))
    C = np.zeros((NUM_DOF, NUM_DOF))
    x_coords = np.zeros(NUM_DOF)
    y_coords = np.zeros(NUM_DOF)
    for j in range(NUM_DOF):
        idx = elements_node_ids[i, j] - 1
        x_coords[j] = nodes_x[idx]
        y_coords[j] = nodes_y[idx]
    for j in range(n):
        dN_dx = np.empty(NUM_DOF)
        dN_dy = np.empty(NUM_DOF)
        dx_dxi = _interpolate(dN_dxi[j], x_coords)
        dx_deta = _interpolate(dN_deta[j], x_coords)
        dy_dxi = _interpolate(dN_dxi[j], y_coords)
        dy_deta = _interpolate(dN_deta[j], y_coords)
        jacobian_det = _calculate_jacobian_and_global_shape_derivatives(dN_dxi[j], dN_deta[j],
                                                                        dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                                        dN_dx, dN_dy)
        _calculate_H_C_for_integration_point(weights[j], N[j],
                                             jacobian_det, dN_dx, dN_dy,
                                             c, d, sh, H, C)
    # Assembly
    for j in range(NUM_DOF):
        for k in range(NUM_DOF):
            idx = i*NUM_DOF*NUM_DOF+j*NUM_DOF+k
            H_val[idx] = H[j, k]
            H_row[idx] = elements_node_ids[i, j] - 1
            H_col[idx] = elements_node_ids[i, k] - 1
            C_val[idx] = C[j, k]
            C_row[idx] = elements_node_ids[i, j] - 1
            C_col[idx] = elements_node_ids[i, k] - 1

@jit('void(int64, float64[:], float64[:,:], float64, float64, int64, float64, float64, int64, float64, float64, float64[:,:], float64[:])')
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
        N = np.ascontiguousarray(surface[i]).reshape(NUM_DOF, 1)
        Hbc += np.dot(N, N.transpose())*weights[i]*alpha*jacobian_det
        P += N*weights[i]*alpha*ambient_temp*jacobian_det

@jit('void(int64, float64[:], float64[:], int64[:], int64[:,:], int64, float64[:], float64[:,:,:], float64, float64, float64[:], int64[:], int64[:], float64[:])')
def _calculate_Hbc_P_for_element(i, nodes_x, nodes_y, nodes_bc, elements_node_ids,
                                 n, weights, surfaces,
                                 alpha, ambient_temp,
                                 Hbc_val, Hbc_row, Hbc_col,
                                 P_global):
    x_coords = np.zeros(NUM_DOF)
    y_coords = np.zeros(NUM_DOF)
    bc =np.zeros(NUM_DOF)
    for j in range(NUM_DOF):
        idx = elements_node_ids[i, j] - 1
        x_coords[j] = nodes_x[idx]
        y_coords[j] = nodes_y[idx]
        bc[j] = nodes_bc[idx]
    Hbc = np.zeros((NUM_DOF, NUM_DOF))
    P = np.zeros(NUM_DOF)
    for j in range(NUM_SURFACES):
        _calculate_for_surface(n, weights, surfaces[j],
                               x_coords[j], y_coords[j], bc[j],
                               x_coords[(j+1)%NUM_DOF], y_coords[(j+1)%NUM_DOF], bc[(j+1)%NUM_DOF],
                               alpha, ambient_temp,
                               Hbc, P)
    # Assembly
    for j in range(NUM_DOF):
        P_global[elements_node_ids[i, j] - 1] += P[j]
        for k in range(NUM_DOF):
            idx = i*NUM_DOF*NUM_DOF+j*NUM_DOF+k
            Hbc_val[idx] = Hbc[j, k]
            Hbc_row[idx] = elements_node_ids[i, j] - 1
            Hbc_col[idx] = elements_node_ids[i, k] - 1