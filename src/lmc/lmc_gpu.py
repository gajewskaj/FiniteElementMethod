from math import sqrt

from numba import cuda
import numpy as np

from src.helpers.helpers import measure_time
from src.grid.grid import Grid, Element
from src.lmc.universal_element import UniversalElement

NODES_PER_ELEMENT = 4

@measure_time
def calculate_local_matrices(n: int, grid: Grid) -> None:
    u_el = UniversalElement(n)
    # Allocate memory on GPU
    # Universal element properties
    weights = cuda.to_device(u_el.weights)
    dn_dxi_tab = cuda.to_device(u_el.dN_dxi)
    dn_deta_tab = cuda.to_device(u_el.dN_deta)
    n_tab = cuda.to_device(u_el.N)
    suraces_N_tab = np.empty((NODES_PER_ELEMENT, n, NODES_PER_ELEMENT), dtype=np.float32)
    for i, surface in enumerate(u_el.surfaces):
        suraces_N_tab[i] = np.array(surface.N, dtype=np.float32)
    surfaces_N_tab_cuda = cuda.to_device(suraces_N_tab)
    # Grid elements
    node_x_coords = np.empty((len(grid.elements), NODES_PER_ELEMENT), dtype=np.float32)
    node_y_coords = np.empty((len(grid.elements), NODES_PER_ELEMENT), dtype=np.float32)
    node_bc = np.empty((len(grid.elements), NODES_PER_ELEMENT), dtype=np.int32)
    for i, element in enumerate(grid.elements):
        element: Element
        for j in range(NODES_PER_ELEMENT):
            node_x_coords[i][j] = grid.nodes[element.node_ids[j] - 1].x
            node_y_coords[i][j] = grid.nodes[element.node_ids[j] - 1].y
            node_bc[i][j] = grid.nodes[element.node_ids[j] - 1].BC
    node_x_coords_cuda = cuda.to_device(node_x_coords)
    node_y_coords_cuda = cuda.to_device(node_y_coords)
    node_bc_cuda = cuda.to_device(node_bc)
    # Local matrices
    H_matrices = np.zeros((len(grid.elements), NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=np.float32)
    Hbc_matrices = np.zeros((len(grid.elements), NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=np.float32)
    P_vectors = np.zeros((len(grid.elements), NODES_PER_ELEMENT), dtype=np.float32)
    C_matrices = np.zeros((len(grid.elements), NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=np.float32)
    H_matrices_cuda = cuda.to_device(H_matrices)
    Hbc_matrices_cuda = cuda.to_device(Hbc_matrices)
    P_vectors_cuda = cuda.to_device(P_vectors)
    C_matrices_cuda = cuda.to_device(C_matrices)
    # Parallel calculations on GPU
    threads_per_block = 512
    blocks_per_grid = (len(grid.elements) + threads_per_block - 1) // threads_per_block
    stream_Hbc_P = cuda.stream()
    stream_H_C = cuda.stream()
    _calculate_Hbc_P_for_element[blocks_per_grid, threads_per_block, stream_Hbc_P](n, weights, surfaces_N_tab_cuda,
                                                                     node_x_coords_cuda, node_y_coords_cuda, node_bc_cuda,
                                                                     Hbc_matrices_cuda, P_vectors_cuda,
                                                                     grid.global_data.alpha, grid.global_data.ambient_temp)
    _calculate_H_C_for_element[blocks_per_grid, threads_per_block, stream_H_C](n, weights, n_tab,
                                                                               dn_dxi_tab, dn_deta_tab,
                                                                   node_x_coords_cuda, node_y_coords_cuda,
                                                                   H_matrices_cuda, C_matrices_cuda,
                                                                   grid.global_data.conductivity, grid.global_data.density, grid.global_data.specific_heat)
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
def _calculate_H_C_for_element(n: int, weights, n_tab,
                               dn_dxi_tabs, dn_deta_tabs,
                               node_x_coords, node_y_coords,
                               H_matrices, C_matrices,
                               c: np.float32, d: np.float32, sh: np.float32) -> None:
    i = cuda.grid(1)
    if i < H_matrices.shape[0]:
        _x_coords = node_x_coords[i]
        _y_coords = node_y_coords[i]
        H = H_matrices[i]
        C = C_matrices[i]

        for j in range(n*n):
            dn_dx_tab = cuda.local.array(NODES_PER_ELEMENT, np.float32)
            dn_dy_tab = cuda.local.array(NODES_PER_ELEMENT, np.float32)
            dx_dxi = _interpolate(dn_dxi_tabs[j][0], dn_dxi_tabs[j][1], dn_dxi_tabs[j][2], dn_dxi_tabs[j][3], _x_coords)
            dx_deta = _interpolate(dn_deta_tabs[j][0], dn_deta_tabs[j][1], dn_deta_tabs[j][2], dn_deta_tabs[j][3], _x_coords)
            dy_dxi = _interpolate(dn_dxi_tabs[j][0], dn_dxi_tabs[j][1], dn_dxi_tabs[j][2], dn_dxi_tabs[j][3], _y_coords)
            dy_deta = _interpolate(dn_deta_tabs[j][0], dn_deta_tabs[j][1], dn_deta_tabs[j][2], dn_deta_tabs[j][3], _y_coords)
            detJ = _dn_dx_dn_dy(dn_dxi_tabs[j], dn_deta_tabs[j],
                        dx_dxi, dx_deta, dy_dxi, dy_deta,
                        dn_dx_tab, dn_dy_tab)
            _calculate_H_C(H, C,
                           weights[j//n], weights[j%n], n_tab[j],
                           detJ, dn_dx_tab, dn_dy_tab,
                           c, d, sh)

@cuda.jit(device=True)
def _dn_dx_dn_dy(dn_dxi_tab, dn_deta_tab,
                 dx_dxi, dx_deta, dy_dxi, dy_deta,
                 dn_dx_tab, dn_dy_tab) -> float:
    jacobian_00 = dx_dxi
    jacobian_01 = dy_dxi
    jacobian_10 = dx_deta
    jacobian_11 = dy_deta
    jacobian_det = jacobian_00 * jacobian_11 - jacobian_01 * jacobian_10
    jacobian_inv_00 = jacobian_11 / jacobian_det
    jacobian_inv_01 = -jacobian_01 / jacobian_det
    jacobian_inv_10 = -jacobian_10 / jacobian_det
    jacobian_inv_11 = jacobian_00 / jacobian_det
    for k in range(NODES_PER_ELEMENT):
        dn_dx_tab[k] = jacobian_inv_00 * dn_dxi_tab[k] + jacobian_inv_01 * dn_deta_tab[k]
        dn_dy_tab[k] = jacobian_inv_10 * dn_dxi_tab[k] + jacobian_inv_11 * dn_deta_tab[k]
    return jacobian_det

@cuda.jit(device=True)
def _interpolate(dN1: np.float32,
                 dN2: np.float32,
                 dN3: np.float32,
                 dN4: np.float32,
                 var) -> np.float32:
    return np.float32(dN1*var[0] + dN2*var[1] + dN3*var[2] + dN4*var[3])

@cuda.jit(device=True)
def _calculate_H_C(H, C,
                   weight1, weight2, n_tab,
                   detJ, dn_dx_tab, dn_dy_tab,
                   c: np.float32, d: np.float32, sh: np.float32) -> None:
    H_ip_matrix = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
    C_ip_matrix = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
    dN_dx_vc = cuda.local.array(NODES_PER_ELEMENT, np.float32)
    dN_dx_vc_multiplied = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
    dN_dy_vc = cuda.local.array(NODES_PER_ELEMENT, np.float32)
    dN_dy_vc_multiplied = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
    N_vc = cuda.local.array(NODES_PER_ELEMENT, np.float32)
    for k in range(NODES_PER_ELEMENT):
        dN_dx_vc[k] = dn_dx_tab[k]
        dN_dy_vc[k] = dn_dy_tab[k]
        N_vc[k] = n_tab[k]
    _multiply_matrices(dN_dx_vc, dN_dx_vc, dN_dx_vc_multiplied)
    _multiply_matrices(dN_dy_vc, dN_dy_vc, dN_dy_vc_multiplied)
    _sum_matrices(dN_dx_vc_multiplied, dN_dy_vc_multiplied, H_ip_matrix)
    _multiply_matrix_by_scalar(H_ip_matrix, c*detJ, H_ip_matrix)

    _multiply_matrices(N_vc, N_vc, C_ip_matrix)
    _multiply_matrix_by_scalar(C_ip_matrix, sh*d*detJ, C_ip_matrix)

    _multiply_matrix_by_scalar(H_ip_matrix, weight1*weight2, H_ip_matrix)
    _sum_matrices(H, H_ip_matrix, H)
    _multiply_matrix_by_scalar(C_ip_matrix, weight1*weight2, C_ip_matrix)
    _sum_matrices(C, C_ip_matrix, C)

@cuda.jit
def _calculate_Hbc_P_for_element(n: int, weights, surfaces_N_tab,
                               node_x_coords, node_y_coords, node_bc,
                               Hbc_matrices, P_vectors,
                               alpha: np.float32, ambient_temp: np.float32) -> None:
    i = cuda.grid(1)
    if i < Hbc_matrices.shape[0]:
        _calculate_for_surface(n, weights, surfaces_N_tab[0],
                               node_x_coords[i][0], node_y_coords[i][0], node_bc[i][0],
                               node_x_coords[i][1], node_y_coords[i][1], node_bc[i][1],
                               Hbc_matrices[i], P_vectors[i],
                               alpha, ambient_temp)
        _calculate_for_surface(n, weights, surfaces_N_tab[1],
                               node_x_coords[i][1], node_y_coords[i][1], node_bc[i][1],
                               node_x_coords[i][2], node_y_coords[i][2], node_bc[i][2],
                               Hbc_matrices[i], P_vectors[i],
                               alpha, ambient_temp)
        _calculate_for_surface(n, weights, surfaces_N_tab[2],
                               node_x_coords[i][2], node_y_coords[i][2], node_bc[i][2],
                               node_x_coords[i][3], node_y_coords[i][3], node_bc[i][3],
                               Hbc_matrices[i], P_vectors[i],
                               alpha, ambient_temp)
        _calculate_for_surface(n, weights, surfaces_N_tab[3],
                               node_x_coords[i][3], node_y_coords[i][3], node_bc[i][3],
                               node_x_coords[i][0], node_y_coords[i][0], node_bc[i][0],
                               Hbc_matrices[i], P_vectors[i],
                               alpha, ambient_temp)

@cuda.jit(device=True)
def _calculate_for_surface(n: np.int32, weights, surface_N_tab,
                           node1_x: np.float32, node1_y: np.float32, node1_bc: np.int32,
                           node2_x: np.float32, node2_y: np.float32, node2_bc: np.int32,
                           Hbc_matrix, P_vector: np.ndarray,
                           alpha: np.float32, ambient_temp: np.float32) -> None:
    if node1_bc == 0 or node2_bc == 0:
        return
    L = sqrt((node2_x - node1_x)**2 + (node2_y - node1_y)**2)
    detJ = L/2
    for j in range(n):
        mx = cuda.local.array(NODES_PER_ELEMENT, np.float32)
        mx_P = cuda.local.array(NODES_PER_ELEMENT, np.float32)
        mx_H = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
        for k in range(NODES_PER_ELEMENT):
            mx[k] = surface_N_tab[j][k]
            mx_P[k] = mx[k]*weights[j]*alpha*ambient_temp*detJ
        for k in range(NODES_PER_ELEMENT):
            P_vector[k] += mx_P[k]
        _multiply_matrices(mx, mx, mx_H)
        _multiply_matrix_by_scalar(mx_H, weights[j]*alpha*detJ, mx_H)
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