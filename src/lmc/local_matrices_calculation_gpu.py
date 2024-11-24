from math import sqrt

from numba import cuda
import numpy as np

from src.helpers.helpers import measure_time
from src.grid.grid import Grid, Element
from src.uel.universal_element import universal_element, NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SURFACES

@measure_time
def calculate_local_matrices(grid: Grid) -> None:
    u_el = universal_element
    # Allocate memory on GPU
    # Universal element properties
    weights_cuda = cuda.to_device(u_el.weights)
    surface_weights_cuda = cuda.to_device(u_el.gaussian_quadrature.weights)
    dN_dxi_cuda = cuda.to_device(u_el.dN_dxi)
    dN_deta_cuda = cuda.to_device(u_el.dN_deta)
    N_cuda = cuda.to_device(u_el.N)
    surfaces_cuda = cuda.to_device(u_el.surfaces)
    # Grid elements
    node_x_coords = np.empty((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
    node_y_coords = np.empty((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
    node_bc = np.empty((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS), dtype=np.int32)
    for i, element in enumerate(grid.elements):
        element: Element
        for j in range(NUM_OF_SHAPE_FUNCTIONS):
            node_x_coords[i][j] = grid.nodes[element.node_ids[j] - 1].x
            node_y_coords[i][j] = grid.nodes[element.node_ids[j] - 1].y
            node_bc[i][j] = grid.nodes[element.node_ids[j] - 1].BC
    node_x_coords_cuda = cuda.to_device(node_x_coords)
    node_y_coords_cuda = cuda.to_device(node_y_coords)
    node_bc_cuda = cuda.to_device(node_bc)
    # Local matrices
    H_matrices = np.zeros((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
    Hbc_matrices = np.zeros((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
    P_vectors = np.zeros((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
    C_matrices = np.zeros((len(grid.elements), NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), dtype=np.float32)
    H_matrices_cuda = cuda.to_device(H_matrices)
    Hbc_matrices_cuda = cuda.to_device(Hbc_matrices)
    P_vectors_cuda = cuda.to_device(P_vectors)
    C_matrices_cuda = cuda.to_device(C_matrices)
    # Parallel calculations on GPU
    threads_per_block = 512
    blocks_per_grid = (len(grid.elements) + threads_per_block - 1) // threads_per_block
    stream_Hbc_P = cuda.stream()
    stream_H_C = cuda.stream()
    _calculate_Hbc_P_for_element[blocks_per_grid, threads_per_block, stream_Hbc_P](node_x_coords_cuda, node_y_coords_cuda, node_bc_cuda,
                                                                                   u_el.gaussian_quadrature.n, surface_weights_cuda, surfaces_cuda,
                                                                                   grid.global_data.alpha, grid.global_data.ambient_temp,
                                                                                   Hbc_matrices_cuda, P_vectors_cuda)
    _calculate_H_C_for_element[blocks_per_grid, threads_per_block, stream_H_C](node_x_coords_cuda, node_y_coords_cuda,
                                                                               u_el.n, weights_cuda, N_cuda,
                                                                               dN_dxi_cuda, dN_deta_cuda,
                                                                               grid.global_data.conductivity, grid.global_data.density, grid.global_data.specific_heat,
                                                                               H_matrices_cuda, C_matrices_cuda)
    # Retrieve results from GPU
    stream_Hbc_P.synchronize()
    Hbc_matrices = Hbc_matrices_cuda.copy_to_host()
    P_vectors = P_vectors_cuda.copy_to_host()

    stream_H_C.synchronize()
    H_matrices = H_matrices_cuda.copy_to_host()
    C_matrices = C_matrices_cuda.copy_to_host()

    _save_to_elements(grid, H_matrices, C_matrices, Hbc_matrices, P_vectors)

def _save_to_elements(grid, H_matrices, C_matrices, Hbc_matrices, P_vectors):
    for i, element in enumerate(grid.elements):
        element: Element
        element.H = H_matrices[i]
        element.C = C_matrices[i]
        element.Hbc = Hbc_matrices[i]
        element.P = P_vectors[i]

@cuda.jit
def _calculate_H_C_for_element(node_x_coords, node_y_coords,
                               n: int, weights, N,
                               dN_dxi, dN_deta,
                               c: np.float32, d: np.float32, sh: np.float32,
                               H_matrices, C_matrices) -> None:
    i = cuda.grid(1)
    if i < H_matrices.shape[0]:
        _x_coords = node_x_coords[i]
        _y_coords = node_y_coords[i]
        H = H_matrices[i]
        C = C_matrices[i]
        for j in range(n):
            dN_dx = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float32)
            dN_dy = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float32)
            dx_dxi = _interpolate(dN_dxi[j], _x_coords)
            dx_deta = _interpolate(dN_deta[j], _x_coords)
            dy_dxi = _interpolate(dN_dxi[j], _y_coords)
            dy_deta = _interpolate(dN_deta[j], _y_coords)
            jacobian_det = _calculate_jacobian_and_global_shape_derivatives(dN_dxi[j], dN_deta[j],
                        dx_dxi, dx_deta, dy_dxi, dy_deta,
                        dN_dx, dN_dy)
            _calculate_H_C_for_integration_point(weights[j], N[j],
                           jacobian_det, dN_dx, dN_dy,
                           c, d, sh,
                           H, C)

@cuda.jit(device=True)
def _calculate_jacobian_and_global_shape_derivatives(dN_dxi, dN_deta,
                 dx_dxi, dx_deta, dy_dxi, dy_deta,
                 dN_dx, dN_dy) -> float:
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

@cuda.jit(device=True)
def _interpolate(dN, var) -> np.float32:
    result = 0
    for i in range(NUM_OF_SHAPE_FUNCTIONS):
        result += dN[i] * var[i]
    return result

@cuda.jit(device=True)
def _calculate_H_C_for_integration_point(weight, N,
                   jacobian_det, dN_dx, dN_dy,
                   c: np.float32, d: np.float32, sh: np.float32, H, C) -> None:
    H_ip_matrix = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float32)
    C_ip_matrix = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float32)
    dN_dx_multiplied = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float32)
    dN_dy_multiplied = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float32)

    _multiply_matrices(dN_dx, dN_dx, dN_dx_multiplied)
    _multiply_matrices(dN_dy, dN_dy, dN_dy_multiplied)
    _sum_matrices(dN_dx_multiplied, dN_dy_multiplied, H_ip_matrix)
    _multiply_matrix_by_scalar(H_ip_matrix, c*jacobian_det*weight, H_ip_matrix)

    _multiply_matrices(N, N, C_ip_matrix)
    _multiply_matrix_by_scalar(C_ip_matrix, sh*d*jacobian_det*weight, C_ip_matrix)

    _sum_matrices(H, H_ip_matrix, H)
    _sum_matrices(C, C_ip_matrix, C)

@cuda.jit
def _calculate_Hbc_P_for_element(node_x_coords, node_y_coords, node_bc,
                                 n: int, weights: np.ndarray[np.float32], surfaces: np.ndarray[np.float32],
                                 alpha: np.float32, ambient_temp: np.float32,
                                 Hbc_matrices, P_vectors) -> None:
    i = cuda.grid(1)
    if i < Hbc_matrices.shape[0]:
        for j in range(NUM_OF_SURFACES):
            _calculate_for_surface(n, weights, surfaces[j],
                                   node_x_coords[i][j], node_y_coords[i][j], node_bc[i][j],
                                   node_x_coords[i][(j+1)%NUM_OF_SHAPE_FUNCTIONS], node_y_coords[i][(j+1)%NUM_OF_SHAPE_FUNCTIONS], node_bc[i][(j+1)%NUM_OF_SHAPE_FUNCTIONS],
                                   alpha, ambient_temp, Hbc_matrices[i], P_vectors[i])

@cuda.jit(device=True)
def _calculate_for_surface(n: np.int32, weights: np.ndarray[np.float32], surface: np.ndarray[np.float32],
                           node1_x: np.float32, node1_y: np.float32, node1_bc: np.int32,
                           node2_x: np.float32, node2_y: np.float32, node2_bc: np.int32,
                           alpha: np.float32, ambient_temp: np.float32, Hbc_matrix, P_vector: np.ndarray) -> None:
    if node1_bc == 0 or node2_bc == 0:
        return
    L = sqrt((node2_x - node1_x)**2 + (node2_y - node1_y)**2)
    jacobian_det = L/2
    for j in range(n):
        mx = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float32)
        mx_P = cuda.local.array(NUM_OF_SHAPE_FUNCTIONS, np.float32)
        mx_H = cuda.local.array((NUM_OF_SHAPE_FUNCTIONS, NUM_OF_SHAPE_FUNCTIONS), np.float32)
        for k in range(NUM_OF_SHAPE_FUNCTIONS):
            mx[k] = surface[j][k]
            mx_P[k] = mx[k]*weights[j]*alpha*ambient_temp*jacobian_det
        for k in range(NUM_OF_SHAPE_FUNCTIONS):
            P_vector[k] += mx_P[k]
        _multiply_matrices(mx, mx, mx_H)
        _multiply_matrix_by_scalar(mx_H, weights[j]*alpha*jacobian_det, mx_H)
        _sum_matrices(mx_H, Hbc_matrix, Hbc_matrix)

@cuda.jit(device=True)
def _sum_matrices(A, B, C) -> None:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i][j] = A[i][j] + B[i][j]

@cuda.jit(device=True)
def _multiply_matrices(A: cuda.local.array, B: cuda.local.array, C: cuda.local.array) -> None:
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            C[i][j] = A[i] * B[j]

@cuda.jit(device=True)
def _multiply_matrix_by_scalar(A, scalar: np.float32, B) -> None:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i][j] = A[i][j] * scalar