from math import sqrt

from numba import njit, prange, set_num_threads
import numpy as np

from src.helpers.config import Settings
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices
from src.uel.universal_element import u_el

DOF = Settings.MatricesCalculation.DOF

set_num_threads(2)

@measure_time
def calculate_and_assemble_matrices(mesh: Mesh, out_mat: OutMatrices) -> None:
    calculate_H_C(mesh.nodes_x, mesh.nodes_y, mesh.elements_node_ids, mesh.elements_material_ids,
                                u_el.n, u_el.weights, u_el.N,
                                u_el.dN_dxi, u_el.dN_deta,
                                mesh.global_data.materials,
                                out_mat.H_val_out, out_mat.H_row_out, out_mat.H_col_out,
                                out_mat.C_val_out, out_mat.C_row_out, out_mat.C_col_out)
    calculate_Hbc_P(mesh.nodes_x, mesh.nodes_y, mesh.nodes_bc, mesh.elements_node_ids, mesh.elements_material_ids,
                                    u_el.quadrature_1d.n, u_el.quadrature_1d.weights, u_el.surfaces,
                                    mesh.global_data.materials, mesh.global_data.ambient_temp,
                                    out_mat.Hbc_val_out, out_mat.Hbc_row_out, out_mat.Hbc_col_out, out_mat.P_out)

@njit('float64(float64[:], float64[:], float64, float64, float64, float64, float64[:], float64[:])')
def _calculate_jacobian_and_global_shape_derivatives(dN_dxi, dN_deta,
                                                     dx_dxi, dx_deta,
                                                     dy_dxi, dy_deta,
                                                     dN_dx, dN_dy):
    jacobian = np.array([[dx_dxi, dy_dxi], [dx_deta, dy_deta]])
    jacobian_det = np.linalg.det(jacobian)
    jacobian_inv = np.linalg.inv(jacobian)
    for i in range(DOF):
        global_shape_function_derivatives = np.dot(jacobian_inv,
            np.array([dN_dxi[i], dN_deta[i]]))
        dN_dx[i] = global_shape_function_derivatives[0]
        dN_dy[i] = global_shape_function_derivatives[1]
    return jacobian_det

@njit('float64(float64[:], float64[:])')
def _interpolate(dN, var):
    return sum([dN[i]*var[i] for i in range(len(dN))])

@njit('void(float64, float64[:], float64, float64[:], float64[:], float64, float64, float64, float64[:,:], float64[:,:])')
def _calculate_H_C_for_integration_point(weight, N,
                                         jacobian_det, dN_dx, dN_dy,
                                         c, d, sh,
                                         H, C):
    dN_dx = np.ascontiguousarray(dN_dx).reshape(DOF, 1)
    dN_dy = np.ascontiguousarray(dN_dy).reshape(DOF, 1)
    N = np.ascontiguousarray(N).reshape(DOF, 1)
    H += c*(np.dot(dN_dx, dN_dx.transpose()) +
            np.dot(dN_dy, dN_dy.transpose()))*jacobian_det*weight
    C += sh*d*(np.dot(N, N.transpose()))*jacobian_det*weight

@njit('void(float64[:], float64[:], int32[:,:], int32[:], int32, float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], int32[:], int32[:], float64[:], int32[:], int32[:])', parallel=True)
def calculate_H_C(nodes_x, nodes_y, elements_node_ids, elements_material_ids,
                               n, weights, N,
                               dN_dxi, dN_deta,
                               materials,
                               H_val, H_row, H_col,
                               C_val, C_row, C_col):
    num_el = len(elements_material_ids)
    H = np.zeros((num_el, DOF, DOF))
    C = np.zeros((num_el, DOF, DOF))
    x_coords = np.zeros((num_el, DOF))
    y_coords = np.zeros((num_el, DOF))
    dN_dx = np.empty((num_el, DOF))
    dN_dy = np.empty((num_el, DOF))
    for i in prange(num_el):
        mat_id: int = elements_material_ids[i] - 1
        c: np.float64 = materials[mat_id, 1]
        d: np.float64 = materials[mat_id, 2]
        sh: np.float64 = materials[mat_id, 3]
        for j in range(DOF):
            idx = elements_node_ids[i, j] - 1
            x_coords[i, j] = nodes_x[idx]
            y_coords[i, j] = nodes_y[idx]
        for j in range(n):
            dx_dxi = _interpolate(dN_dxi[j], x_coords[i])
            dx_deta = _interpolate(dN_deta[j], x_coords[i])
            dy_dxi = _interpolate(dN_dxi[j], y_coords[i])
            dy_deta = _interpolate(dN_deta[j], y_coords[i])
            jacobian_det = _calculate_jacobian_and_global_shape_derivatives(dN_dxi[j], dN_deta[j],
                                                                            dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                                            dN_dx[i], dN_dy[i])
            _calculate_H_C_for_integration_point(weights[j], N[j],
                                                jacobian_det, dN_dx[i], dN_dy[i],
                                                c, d, sh, H[i], C[i])
        # Assembly
        for j in range(DOF):
            for k in range(DOF):
                idx = i*DOF*DOF+j*DOF+k
                H_val[idx] = H[i, j, k]
                H_row[idx] = elements_node_ids[i, j] - 1
                H_col[idx] = elements_node_ids[i, k] - 1
                C_val[idx] = C[i, j, k]
                C_row[idx] = elements_node_ids[i, j] - 1
                C_col[idx] = elements_node_ids[i, k] - 1

@njit('void(int32, float64[:], float64[:,:], float64, float64, int32, float64, float64, int32, float64, float64, float64[:,:], float64[:])')
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
        N = np.ascontiguousarray(surface[i]).reshape(DOF, 1)
        Hbc += np.dot(N, N.transpose())*weights[i]*alpha*jacobian_det
        P += N*weights[i]*alpha*ambient_temp*jacobian_det

@njit('void(float64[:], float64[:], int32[:], int32[:,:], int32[:], int32, float64[:], float64[:,:,:], float64[:,:], float64, float64[:], int32[:], int32[:], float64[:])', parallel=True)
def calculate_Hbc_P(nodes_x, nodes_y, nodes_bc, elements_node_ids, elements_material_ids,
                                 n, weights, surfaces,
                                 materials, ambient_temp,
                                 Hbc_val, Hbc_row, Hbc_col,
                                 P_global):
    num_el = len(elements_material_ids)
    x_coords = np.zeros((num_el, DOF))
    y_coords = np.zeros((num_el, DOF))
    bc = np.zeros((num_el, DOF))
    Hbc = np.zeros((num_el, DOF, DOF))
    P = np.zeros((num_el, DOF))
    for i in prange(num_el):
        alpha: np.float64 = materials[elements_material_ids[i] - 1, 0]
        for j in range(DOF):
            idx = elements_node_ids[i, j] - 1
            x_coords[i, j] = nodes_x[idx]
            y_coords[i, j] = nodes_y[idx]
            bc[i, j] = nodes_bc[idx]
        for j in range(DOF):
            _calculate_for_surface(n, weights, surfaces[j],
                                x_coords[i, j], y_coords[i, j], bc[i, j],
                                x_coords[i, (j+1)%DOF], y_coords[i, (j+1)%DOF], bc[i, (j+1)%DOF],
                                alpha, ambient_temp,
                                Hbc[i], P[i])
        # Assembly
        for j in range(DOF):
            P_global[elements_node_ids[i, j] - 1] += P[i, j]
            for k in range(DOF):
                idx = i*DOF*DOF+j*DOF+k
                Hbc_val[idx] = Hbc[i, j, k]
                Hbc_row[idx] = elements_node_ids[i, j] - 1
                Hbc_col[idx] = elements_node_ids[i, k] - 1