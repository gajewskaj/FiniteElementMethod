from math import sqrt

from numba import cuda
import numpy as np

from src.helpers import config
from src.helpers.helpers import measure_time
from src.grid.grid import Grid
from src.uel.universal_element import u_el

NUM_OF_SHAPE_FUNCTIONS = config.num_of_shape_functions
NUM_OF_SURFACES = NUM_OF_SHAPE_FUNCTIONS

@measure_time
def calculate_and_assemble_matrices(grid: Grid) -> None:
    stream_H_C = cuda.stream()
    stream_Hbc_P = cuda.stream()
    data_H_C, data_Hbc_P = _send_to_GPU(stream_H_C, stream_Hbc_P, grid)
    threads_per_block = 512
    blocks_per_grid = (len(grid.elements_id) + threads_per_block - 1) // threads_per_block
    _calculate(data_H_C, data_Hbc_P, stream_H_C, stream_Hbc_P, threads_per_block, blocks_per_grid)
    _retrieve_from_GPU(data_H_C, data_Hbc_P, grid)

@measure_time
def _calculate(data_H_C: tuple, data_Hbc_P: tuple, stream_H_C: cuda.stream, stream_Hbc_P: cuda.stream, threads_per_block: int, blocks_per_grid: int) -> tuple:
    _calculate_H_C_for_element[blocks_per_grid, threads_per_block, stream_H_C](*data_H_C)
    _calculate_Hbc_P_for_element[blocks_per_grid, threads_per_block, stream_Hbc_P](*data_Hbc_P)
    stream_H_C.synchronize()
    stream_Hbc_P.synchronize()

@measure_time
def _send_to_GPU(stream_H_C: cuda.stream, stream_Hbc_P: cuda.stream, grid: Grid) -> tuple:
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
    global_H_values_cuda = cuda.to_device(grid.global_H_values, stream=stream_H_C)
    global_H_row_cuda = cuda.to_device(grid.global_H_row, stream=stream_H_C)
    global_H_col_cuda = cuda.to_device(grid.global_H_col, stream=stream_H_C)
    global_C_values_cuda = cuda.to_device(grid.global_C_values, stream=stream_H_C)
    global_C_row_cuda = cuda.to_device(grid.global_C_row, stream=stream_H_C)
    global_C_col_cuda = cuda.to_device(grid.global_C_col, stream=stream_H_C)
    global_Hbc_values_cuda = cuda.to_device(grid.global_Hbc_values, stream=stream_Hbc_P)
    global_Hbc_row_cuda = cuda.to_device(grid.global_Hbc_row, stream=stream_Hbc_P)
    global_Hbc_col_cuda = cuda.to_device(grid.global_Hbc_col, stream=stream_Hbc_P)
    global_P_cuda = cuda.to_device(grid.global_P, stream=stream_Hbc_P)
    return (
        nodes_x_cuda, nodes_y_cuda, elements_node_ids_cuda,
        u_el.n, weights_cuda, N_cuda,
        dN_dxi_cuda, dN_deta_cuda,
        grid.global_data.conductivity,
        grid.global_data.density,
        grid.global_data.specific_heat,
        global_H_values_cuda, global_H_row_cuda, global_H_col_cuda,
        global_C_values_cuda, global_C_row_cuda, global_C_col_cuda
        ), (
        nodes_x_cuda, nodes_y_cuda, nodes_bc_cuda, elements_node_ids_cuda,
        u_el.quadrature_1d.n, surface_weights_cuda, surfaces_cuda,
        grid.global_data.alpha,
        grid.global_data.ambient_temp,
        global_Hbc_values_cuda, global_Hbc_row_cuda, global_Hbc_col_cuda,
        global_P_cuda
        )

@measure_time
def _retrieve_from_GPU(data_H_C: tuple, data_Hbc_P: tuple, grid: Grid) -> None:
    grid.global_H_values = data_H_C[-6].copy_to_host().astype(np.float64)
    grid.global_H_row = data_H_C[-5].copy_to_host()
    grid.global_H_col = data_H_C[-4].copy_to_host()
    grid.global_C_values = data_H_C[-3].copy_to_host().astype(np.float64)
    grid.global_C_row = data_H_C[-2].copy_to_host()
    grid.global_C_col = data_H_C[-1].copy_to_host()
    grid.global_Hbc_values = data_Hbc_P[-4].copy_to_host().astype(np.float64)
    grid.global_Hbc_row = data_Hbc_P[-3].copy_to_host()
    grid.global_Hbc_col = data_Hbc_P[-2].copy_to_host()
    grid.global_P = data_Hbc_P[-1].copy_to_host().reshape(-1, 1)

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

@cuda.jit('void(float64[:], float64[:], int64[:,:], int64, float64[:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, float64[:], int64[:], int64[:], float64[:], int64[:], int64[:])')
def _calculate_H_C_for_element(nodes_x, nodes_y, elements_node_ids,
                               n, weights, N,
                               dN_dxi, dN_deta,
                               c, d, sh,
                               global_H_values, global_H_row, global_H_col,
                               global_C_values, global_C_row, global_C_col):
    i = cuda.grid(1)
    if i < len(global_H_values)//NUM_OF_SHAPE_FUNCTIONS//NUM_OF_SHAPE_FUNCTIONS:
        x_coords = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        y_coords = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        H = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float64)
        C = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float64)
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            x_coords[j] = nodes_x[elements_node_ids[i, j] - 1]
            y_coords[j] = nodes_y[elements_node_ids[i, j] - 1]
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
        # Assembly
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            for k in range(NUM_OF_SHAPE_FUNCTIONS):
                cuda.atomic.add(global_H_values, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, H[j, k])
                cuda.atomic.add(global_H_row, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, elements_node_ids[i, j] - 1)
                cuda.atomic.add(global_H_col, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, elements_node_ids[i, k] - 1)
                cuda.atomic.add(global_C_values, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, C[j, k])
                cuda.atomic.add(global_C_row, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, elements_node_ids[i, j] - 1)
                cuda.atomic.add(global_C_col, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, elements_node_ids[i, k] - 1)

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

@cuda.jit('void(float64[:], float64[:], int64[:], int64[:,:], int64, float64[:], float64[:,:,:], float64, float64, float64[:], int64[:], int64[:], float64[:])')
def _calculate_Hbc_P_for_element(nodes_x, nodes_y, nodes_bc, elements_node_ids,
                                 n, weights, surfaces,
                                 alpha, ambient_temp,
                                 global_Hbc_values, global_Hbc_row, global_Hbc_col,
                                 global_P):
    i = cuda.grid(1)
    if i < len(global_Hbc_values)//NUM_OF_SHAPE_FUNCTIONS//NUM_OF_SHAPE_FUNCTIONS:
        x_coords = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        y_coords = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        bc = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.int64)
        P = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float64)
        Hbc = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float64)
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            x_coords[j] = nodes_x[elements_node_ids[i, j] - 1]
            y_coords[j] = nodes_y[elements_node_ids[i, j] - 1]
            bc[j] = nodes_bc[elements_node_ids[i, j] - 1]
        for j in range(NUM_OF_SURFACES):
            _calculate_for_surface(n, weights, surfaces[j],
                                   x_coords[j], y_coords[j], bc[j],
                                   x_coords[(j+1)%NUM_OF_SHAPE_FUNCTIONS], y_coords[(j+1)%NUM_OF_SHAPE_FUNCTIONS], bc[(j+1)%NUM_OF_SHAPE_FUNCTIONS],
                                   alpha, ambient_temp,
                                   Hbc, P)
        # Assembly
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            cuda.atomic.add(global_P, elements_node_ids[i, j] - 1, P[j])
            for k in range(NUM_OF_SHAPE_FUNCTIONS):
                cuda.atomic.add(global_Hbc_values, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, Hbc[j, k])
                cuda.atomic.add(global_Hbc_row, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, elements_node_ids[i, j] - 1)
                cuda.atomic.add(global_Hbc_col, i*NUM_OF_SHAPE_FUNCTIONS*NUM_OF_SHAPE_FUNCTIONS + j*NUM_OF_SHAPE_FUNCTIONS + k, elements_node_ids[i, k] - 1)