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
    dn_dksi_tab = cuda.to_device(u_el.dn_dksi_tab)
    dn_deta_tab = cuda.to_device(u_el.dn_deta_tab)
    n_tab = cuda.to_device(u_el.n_tab)
    suraces_N_tab = np.empty((NODES_PER_ELEMENT, n, NODES_PER_ELEMENT), dtype=np.float32)
    for i, surface in enumerate(u_el.surfaces):
        suraces_N_tab[i] = np.array(surface.N, dtype=np.float32)
    surfaces_N_tab_cuda = cuda.to_device(suraces_N_tab)
    # Temporary arrays
    dx_dksi_tabs_cuda = cuda.to_device(np.empty((len(grid.elements), u_el.n*u_el.n), dtype=np.float32))
    dx_deta_tabs_cuda = cuda.to_device(np.empty((len(grid.elements), u_el.n*u_el.n), dtype=np.float32))
    dy_dksi_tabs_cuda = cuda.to_device(np.empty((len(grid.elements), u_el.n*u_el.n), dtype=np.float32))
    dy_deta_tabs_cuda = cuda.to_device(np.empty((len(grid.elements), u_el.n*u_el.n), dtype=np.float32))
    det_tab_cuda = cuda.to_device(np.empty((len(grid.elements), u_el.n*u_el.n), dtype=np.float32))
    dn_dx_tab_cuda = cuda.to_device(np.empty((len(grid.elements), u_el.n*u_el.n, NODES_PER_ELEMENT), dtype=np.float32))
    dn_dy_tab_cuda = cuda.to_device(np.empty((len(grid.elements), u_el.n*u_el.n, NODES_PER_ELEMENT), dtype=np.float32))
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
                                                                     grid.global_data.alfa, grid.global_data.tot)
    _calculate_H_C_for_element[blocks_per_grid, threads_per_block, stream_H_C](n, weights, n_tab,
                                                                   dn_dksi_tab, dn_deta_tab,
                                                                   node_x_coords_cuda, node_y_coords_cuda,
                                                                   dx_dksi_tabs_cuda, dx_deta_tabs_cuda, dy_dksi_tabs_cuda, dy_deta_tabs_cuda,
                                                                   det_tab_cuda, dn_dx_tab_cuda, dn_dy_tab_cuda,
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
                               dn_dksi_tab, dn_deta_tab,
                               node_x_coords, node_y_coords,
                               dx_dksi_tabs, dx_deta_tabs, dy_dksi_tabs, dy_deta_tabs,
                               det_tab, dn_dx_tab, dn_dy_tab,
                               H_matrices, C_matrices,
                               c: np.float32, d: np.float32, sh: np.float32) -> None:
    i = cuda.grid(1)
    if i < H_matrices.shape[0]:
        for j in range(n*n):
            _fill_x_y_ksi_eta_tabs(j, dn_dksi_tab, dn_deta_tab,
                                node_x_coords[i], node_y_coords[i],
                                dx_dksi_tabs[i], dx_deta_tabs[i], dy_dksi_tabs[i], dy_deta_tabs[i])
            _dn_dx_dn_dy(j, dn_dksi_tab, dn_deta_tab,
                        dx_dksi_tabs[i], dx_deta_tabs[i], dy_dksi_tabs[i], dy_deta_tabs[i],
                        det_tab[i], dn_dx_tab[i], dn_dy_tab[i])
            _calculate_H_C(j, n, weights, n_tab,
                        det_tab[i], dn_dx_tab[i], dn_dy_tab[i],
                        H_matrices[i], C_matrices[i],
                        c, d, sh)

@cuda.jit(device=True)
def _dn_dx_dn_dy(j: int, dn_dksi_tab, dn_deta_tab,
                 dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab,
                 det_tab, dn_dx_tab, dn_dy_tab) -> None:
    mxJ_00 = dx_dksi_tab[j]
    mxJ_01 = dy_dksi_tab[j]
    mxJ_10 = dx_deta_tab[j]
    mxJ_11 = dy_deta_tab[j]
    det_J = mxJ_00 * mxJ_11 - mxJ_01 * mxJ_10
    det_tab[j] = det_J
    invJ_00 = mxJ_11 / det_J
    invJ_01 = -mxJ_01 / det_J
    invJ_10 = -mxJ_10 / det_J
    invJ_11 = mxJ_00 / det_J
    for k in range(NODES_PER_ELEMENT):
        dn_dx_tab[j][k] = invJ_00 * dn_dksi_tab[j][k] + invJ_01 * dn_deta_tab[j][k]
        dn_dy_tab[j][k] = invJ_10 * dn_dksi_tab[j][k] + invJ_11 * dn_deta_tab[j][k]

@cuda.jit(device=True)
def _fill_x_y_ksi_eta_tabs(j: np.int32, dn_dksi_tab, dn_deta_tab,
                           node_x_coords, node_y_coords,
                           dx_dksi_tab, dx_deta_tab, dy_dksi_tab, dy_deta_tab) -> None:
    dx_dksi_tab[j] = _interpolate(dn_dksi_tab[j][0], dn_dksi_tab[j][1], dn_dksi_tab[j][2], dn_dksi_tab[j][3], node_x_coords)
    dx_deta_tab[j] = _interpolate(dn_deta_tab[j][0], dn_deta_tab[j][1], dn_deta_tab[j][2], dn_deta_tab[j][3], node_x_coords)
    dy_dksi_tab[j] = _interpolate(dn_dksi_tab[j][0], dn_dksi_tab[j][1], dn_dksi_tab[j][2], dn_dksi_tab[j][3], node_y_coords)
    dy_deta_tab[j] = _interpolate(dn_deta_tab[j][0], dn_deta_tab[j][1], dn_deta_tab[j][2], dn_deta_tab[j][3], node_y_coords)

@cuda.jit(device=True)
def _interpolate(dN1: np.float32,
                 dN2: np.float32,
                 dN3: np.float32,
                 dN4: np.float32,
                 var) -> np.float32:
    return np.float32(dN1*var[0] + dN2*var[1] + dN3*var[2] + dN4*var[3])

@cuda.jit(device=True)
def _calculate_H_C(j: int, n: int, weights, n_tab,
                   det_tab, dn_dx_tab, dn_dy_tab,
                   H_matrix, C_matrix,
                   c: np.float32, d: np.float32, sh: np.float32) -> None:
    H_ip_matrix = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
    C_ip_matrix = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
    dN_dx_vc = cuda.local.array(NODES_PER_ELEMENT, np.float32)
    dN_dx_vc_multiplied = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
    dN_dy_vc = cuda.local.array(NODES_PER_ELEMENT, np.float32)
    dN_dy_vc_multiplied = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), np.float32)
    N_vc = cuda.local.array(NODES_PER_ELEMENT, np.float32)
    for k in range(NODES_PER_ELEMENT):
        dN_dx_vc[k] = dn_dx_tab[j][k]
        dN_dy_vc[k] = dn_dy_tab[j][k]
        N_vc[k] = n_tab[j][k]
    _multiply_matrices(dN_dx_vc, dN_dx_vc, dN_dx_vc_multiplied)
    _multiply_matrices(dN_dy_vc, dN_dy_vc, dN_dy_vc_multiplied)
    _sum_matrices(dN_dx_vc_multiplied, dN_dy_vc_multiplied, H_ip_matrix)
    _multiply_matrix_by_scalar(H_ip_matrix, c*det_tab[j], H_ip_matrix)

    _multiply_matrices(N_vc, N_vc, C_ip_matrix)
    _multiply_matrix_by_scalar(C_ip_matrix, sh*d*det_tab[j], C_ip_matrix)

    _multiply_matrix_by_scalar(H_ip_matrix, weights[j//n]*weights[j%n], H_ip_matrix)
    _sum_matrices(H_matrix, H_ip_matrix, H_matrix)
    _multiply_matrix_by_scalar(C_ip_matrix, weights[j//n]*weights[j%n], C_ip_matrix)
    _sum_matrices(C_matrix, C_ip_matrix, C_matrix)

@cuda.jit
def _calculate_Hbc_P_for_element(n: int, weights, surfaces_N_tab,
                               node_x_coords, node_y_coords, node_bc,
                               Hbc_matrices, P_vectors,
                               alfa: np.float32, tot: np.float32) -> None:
    i = cuda.grid(1)
    if i < Hbc_matrices.shape[0]:
        _calculate_for_surface(n, weights, surfaces_N_tab[0],
                               node_x_coords[i][0], node_y_coords[i][0], node_bc[i][0],
                               node_x_coords[i][1], node_y_coords[i][1], node_bc[i][1],
                               Hbc_matrices[i], P_vectors[i],
                               alfa, tot)
        _calculate_for_surface(n, weights, surfaces_N_tab[1],
                               node_x_coords[i][1], node_y_coords[i][1], node_bc[i][1],
                               node_x_coords[i][2], node_y_coords[i][2], node_bc[i][2],
                               Hbc_matrices[i], P_vectors[i],
                               alfa, tot)
        _calculate_for_surface(n, weights, surfaces_N_tab[2],
                               node_x_coords[i][2], node_y_coords[i][2], node_bc[i][2],
                               node_x_coords[i][3], node_y_coords[i][3], node_bc[i][3],
                               Hbc_matrices[i], P_vectors[i],
                               alfa, tot)
        _calculate_for_surface(n, weights, surfaces_N_tab[3],
                               node_x_coords[i][3], node_y_coords[i][3], node_bc[i][3],
                               node_x_coords[i][0], node_y_coords[i][0], node_bc[i][0],
                               Hbc_matrices[i], P_vectors[i],
                               alfa, tot)

@cuda.jit(device=True)
def _calculate_for_surface(n: np.int32, weights, surface_N_tab,
                           node1_x: np.float32, node1_y: np.float32, node1_bc: np.int32,
                           node2_x: np.float32, node2_y: np.float32, node2_bc: np.int32,
                           Hbc_matrix, P_vector: np.ndarray,
                           alfa: np.float32, tot: np.float32) -> None:
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
            mx_P[k] = mx[k]*weights[j]*alfa*tot*detJ
        for k in range(NODES_PER_ELEMENT):
            P_vector[k] += mx_P[k]
        _multiply_matrices(mx, mx, mx_H)
        _multiply_matrix_by_scalar(mx_H, weights[j]*alfa*detJ, mx_H)
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



