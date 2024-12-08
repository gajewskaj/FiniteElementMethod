from math import sqrt

from numba import cuda
import numpy as np

from src.helpers.helpers import measure_time
from src.grid.grid import Grid
from src.uel.universal_element import u_el, NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SURFACES

@measure_time
def calculate_local_matrices(grid: Grid) -> None:
    stream_H_C = cuda.stream()
    stream_Hbc_P = cuda.stream()
    # Allocate memory on GPU
    # Universal element properties
    weights_cuda = cuda.to_device(u_el.weights, stream=stream_H_C)
    surface_weights_cuda = cuda.to_device(u_el.quadrature_1d.weights, stream=stream_Hbc_P)
    dN_dxi_cuda = cuda.to_device(u_el.dN_dxi, stream=stream_H_C)
    dN_deta_cuda = cuda.to_device(u_el.dN_deta, stream=stream_H_C)
    N_cuda = cuda.to_device(u_el.N, stream=stream_H_C)
    surfaces_cuda = cuda.to_device(u_el.surfaces, stream=stream_Hbc_P)
    # Grid elements
    nodes_x_cuda = cuda.to_device(grid.nodes_x)
    nodes_y_cuda = cuda.to_device(grid.nodes_y)
    nodes_bc_cuda = cuda.to_device(grid.nodes_bc)
    elements_node_ids_cuda = cuda.to_device(grid.elements_node_ids)
    # Local matrices
    H_matrices_cuda = cuda.to_device(grid.elements_H, stream=stream_H_C)
    C_matrices_cuda = cuda.to_device(grid.elements_C, stream=stream_H_C)
    Hbc_matrices_cuda = cuda.to_device(grid.elements_Hbc, stream=stream_Hbc_P)
    P_vectors_cuda = cuda.to_device(grid.elements_P, stream=stream_Hbc_P)
    # Parallel calculations on GPU
    threads_per_block = 128
    blocks_per_grid = (len(grid.elements_id) + threads_per_block - 1) // threads_per_block
    _calculate_H_C_for_element[blocks_per_grid, threads_per_block, stream_H_C](nodes_x_cuda, nodes_y_cuda, elements_node_ids_cuda,
                                                                               u_el.n, weights_cuda, N_cuda,
                                                                               dN_dxi_cuda, dN_deta_cuda,
                                                                               grid.global_data.conductivity, grid.global_data.density, grid.global_data.specific_heat,
                                                                               H_matrices_cuda, C_matrices_cuda)
    _calculate_Hbc_P_for_element[blocks_per_grid, threads_per_block, stream_Hbc_P](nodes_x_cuda, nodes_y_cuda, nodes_bc_cuda, elements_node_ids_cuda,
                                                                                   u_el.quadrature_1d.n, surface_weights_cuda, surfaces_cuda,
                                                                                   grid.global_data.alpha, grid.global_data.ambient_temp,
                                                                                   Hbc_matrices_cuda, P_vectors_cuda)
    stream_H_C.synchronize()
    stream_Hbc_P.synchronize()
    # Retrieve results from GPU
    grid.elements_H = H_matrices_cuda.copy_to_host()
    grid.elements_C = C_matrices_cuda.copy_to_host()
    grid.elements_Hbc = Hbc_matrices_cuda.copy_to_host()
    grid.elements_P = P_vectors_cuda.copy_to_host()

@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:])', device=True)
def _sum_matrices(A, B, C):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] + B[i, j]

@cuda.jit('void(float64[:], float64[:], float64[:,:])', device=True)
def _multiply_vectors(A, B, C):
    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = A[i] * B[j]

@cuda.jit('void(float64[:,:], float64, float64[:,:])', device=True)
def _multiply_matrix_by_scalar(A, scalar, B):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = A[i, j] * scalar

@cuda.jit('float64(float64[:], float64[:], float64, float64, float64, float64, float64[:], float64[:])', device=True)
def _calculate_jacobian_and_global_shape_derivatives(dN_dxi, dN_deta,
                                                     dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                     dN_dx, dN_dy):
    jacobian_00 = dx_dxi
    jacobian_01 = dy_dxi
    jacobian_10 = dx_deta
    jacobian_11 = dy_deta
    jacobian_det = jacobian_00 * jacobian_11 - jacobian_01 * jacobian_10
    jacobian_inv_00 = jacobian_11 / jacobian_det
    jacobian_inv_01 = -jacobian_01 / jacobian_det
    jacobian_inv_10 = -jacobian_10 / jacobian_det
    jacobian_inv_11 = jacobian_00 / jacobian_det
    for k in range(NUM_OF_SHAPE_FUNCTIONS):
        dN_dx[k] = jacobian_inv_00 * dN_dxi[k] + jacobian_inv_01 * dN_deta[k]
        dN_dy[k] = jacobian_inv_10 * dN_dxi[k] + jacobian_inv_11 * dN_deta[k]
    return jacobian_det

@cuda.jit('float64(float64[:], float64[:])', device=True)
def _interpolate(dN, var):
    result = 0.0
    for i in range(NUM_OF_SHAPE_FUNCTIONS):
        result += dN[i] * var[i]
    return result

@cuda.jit('void(float64, float64[:], float64, float64[:], float64[:], float64, float64, float64, float64[:,:], float64[:,:])', device=True)
def _calculate_H_C_for_integration_point(weight, N,
                                         jacobian_det, dN_dx, dN_dy,
                                         c, d, sh,
                                         H, C):
    H_ip_matrix = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float64)
    C_ip_matrix = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float64)
    dN_dx_multiplied = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float64)
    dN_dy_multiplied = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float64)

    _multiply_vectors(dN_dx, dN_dx, dN_dx_multiplied)
    _multiply_vectors(dN_dy, dN_dy, dN_dy_multiplied)
    _sum_matrices(dN_dx_multiplied, dN_dy_multiplied, H_ip_matrix)
    _multiply_matrix_by_scalar(H_ip_matrix, c*jacobian_det*weight, H_ip_matrix)

    _multiply_vectors(N, N, C_ip_matrix)
    _multiply_matrix_by_scalar(C_ip_matrix, sh*d*jacobian_det*weight, C_ip_matrix)

    _sum_matrices(H, H_ip_matrix, H)
    _sum_matrices(C, C_ip_matrix, C)

@cuda.jit('void(float64[:], float64[:], int64[:,:], int64, float64[:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, float64[:,:,:], float64[:,:,:])')
def _calculate_H_C_for_element(nodes_x, nodes_y, elements_node_ids,
                               n, weights, N,
                               dN_dxi, dN_deta,
                               c, d, sh,
                               H_matrices, C_matrices):
    i = cuda.grid(1)
    if i < H_matrices.shape[0]:
        x_coords = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        y_coords = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            x_coords[j] = nodes_x[elements_node_ids[i, j] - 1]
            y_coords[j] = nodes_y[elements_node_ids[i, j] - 1]
        H = H_matrices[i]
        C = C_matrices[i]
        for j in range(n):
            dN_dx = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
            dN_dy = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
            dx_dxi = _interpolate(dN_dxi[j], x_coords)
            dx_deta = _interpolate(dN_deta[j], x_coords)
            dy_dxi = _interpolate(dN_dxi[j], y_coords)
            dy_deta = _interpolate(dN_deta[j], y_coords)
            jacobian_det = _calculate_jacobian_and_global_shape_derivatives(dN_dxi[j], dN_deta[j],
                                                                            dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                                            dN_dx, dN_dy)
            _calculate_H_C_for_integration_point(weights[j], N[j],
                                                 jacobian_det, dN_dx, dN_dy,
                                                 c, d, sh,
                                                 H, C)

@cuda.jit('void(int64, float64[:], float64[:,:], float64, float64, int64, float64, float64, int64, float64, float64, float64[:,:], float64[:])', device=True)
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
        Hbc_ip = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float64)
        N = surface[i]
        for k in range(NUM_OF_SHAPE_FUNCTIONS):
            P[k] += N[k] * weights[i] * alpha * ambient_temp * jacobian_det
        _multiply_vectors(N, N, Hbc_ip)
        _multiply_matrix_by_scalar(Hbc_ip, weights[i] * alpha * jacobian_det, Hbc_ip)
        _sum_matrices(Hbc_ip, Hbc, Hbc)

@cuda.jit('void(float64[:], float64[:], int64[:], int64[:,:], int64, float64[:], float64[:,:,:], float64, float64, float64[:,:,:], float64[:,:])')
def _calculate_Hbc_P_for_element(nodes_x, nodes_y, nodes_bc, elements_node_ids,
                                 n, weights, surfaces,
                                 alpha, ambient_temp,
                                 Hbc_matrices, P_vectors):
    i = cuda.grid(1)
    if i < Hbc_matrices.shape[0]:
        x_coords = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        y_coords = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        bc = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.int64)
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            x_coords[j] = nodes_x[elements_node_ids[i, j] - 1]
            y_coords[j] = nodes_y[elements_node_ids[i, j] - 1]
            bc[j] = nodes_bc[elements_node_ids[i, j] - 1]
        for j in range(NUM_OF_SURFACES):
            _calculate_for_surface(n, weights, surfaces[j],
                                   x_coords[j], y_coords[j], bc[j],
                                   x_coords[(j+1)%NUM_OF_SHAPE_FUNCTIONS], y_coords[(j+1)%NUM_OF_SHAPE_FUNCTIONS], bc[(j+1)%NUM_OF_SHAPE_FUNCTIONS],
                                   alpha, ambient_temp,
                                   Hbc_matrices[i], P_vectors[i])