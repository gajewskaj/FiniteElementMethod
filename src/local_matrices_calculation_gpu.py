from math import sqrt

import cupy as cp

import numpy as np
from numba import cuda

from . import config
from .grid import Grid, GlobalData, Element, Node
from .universal_element import UniversalElement, Surface

NODES_PER_ELEMENT = 4

def calculate_local_matrices(n: int, grid: Grid) -> None:
    config.logger.info(f"Calculating local matrices for {len(grid.elements)} elements.")
    u_el = UniversalElement(n)
    # Allocate memory on GPU
    # Universal element properties
    n = cp.int32(u_el.n)
    weights = cp.array(u_el.weights, dtype=cp.float32)
    dn_dksi_tab = cp.array(u_el.dn_dksi_tab, dtype=cp.float32)
    dn_deta_tab = cp.array(u_el.dn_deta_tab, dtype=cp.float32)
    n_tab = cp.array(u_el.n_tab, dtype=cp.float32)
    suraces_N_tab = cp.empty((NODES_PER_ELEMENT, n, NODES_PER_ELEMENT), dtype=cp.float32)
    for i, surface in enumerate(u_el.surfaces):
        suraces_N_tab[i] = cp.array(surface.N, dtype=cp.float32)
    # Temporary arrays
    dx_dksi_tabs = cp.empty((len(grid.elements), u_el.n*u_el.n), dtype=cp.float32)
    dx_deta_tabs = cp.empty((len(grid.elements), u_el.n*u_el.n), dtype=cp.float32)
    dy_dksi_tabs = cp.empty((len(grid.elements), u_el.n*u_el.n), dtype=cp.float32)
    dy_deta_tabs = cp.empty((len(grid.elements), u_el.n*u_el.n), dtype=cp.float32)
    dn_dx_tab = cp.empty((len(grid.elements), u_el.n*u_el.n, NODES_PER_ELEMENT), dtype=cp.float32)
    dn_dy_tab = cp.empty((len(grid.elements), u_el.n*u_el.n, NODES_PER_ELEMENT), dtype=cp.float32)
    det_tab = cp.empty((len(grid.elements), u_el.n*u_el.n), dtype=cp.float32)
    H_ip_matrices = cp.empty((len(grid.elements), u_el.n*u_el.n, NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=cp.float32)
    Hbc_ip_matrices = cp.empty((len(grid.elements), u_el.n*u_el.n, NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=cp.float32)
    P_ip_vectors = cp.empty((len(grid.elements), u_el.n*u_el.n, NODES_PER_ELEMENT), dtype=cp.float32)
    C_ip_matrices = cp.empty((len(grid.elements), u_el.n*u_el.n, NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=cp.float32)
    # Grid elements
    element_ids = cp.empty(len(grid.elements), dtype=cp.int32)
    node_x_coords = cp.empty((len(grid.elements), NODES_PER_ELEMENT), dtype=cp.float32)
    node_y_coords = cp.empty((len(grid.elements), NODES_PER_ELEMENT), dtype=cp.float32)
    node_bc = cp.empty((len(grid.elements), NODES_PER_ELEMENT), dtype=cp.int32)
    # Global data
    c = cp.float32(grid.global_data.conductivity)
    d = cp.float32(grid.global_data.density)
    sh = cp.float32(grid.global_data.specific_heat)
    alfa = cp.float32(grid.global_data.alfa)
    tot = cp.float32(grid.global_data.tot)
    # Local matrices
    H_matrices = cp.zeros((len(grid.elements), NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=cp.float32)
    Hbc_matrices = cp.zeros((len(grid.elements), NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=cp.float32)
    P_vectors = cp.zeros((len(grid.elements), NODES_PER_ELEMENT), dtype=cp.float32)
    C_matrices = cp.zeros((len(grid.elements), NODES_PER_ELEMENT, NODES_PER_ELEMENT), dtype=cp.float32)
    # Fill arrays with data
    for i, element in enumerate(grid.elements):
        element: Element
        element_ids[i] = element.id
        for j in range(NODES_PER_ELEMENT):
            node_x_coords[i][j] = grid.nodes[element.node_ids[j] - 1].x
            node_y_coords[i][j] = grid.nodes[element.node_ids[j] - 1].y
            node_bc[i][j] = grid.nodes[element.node_ids[j] - 1].BC
    # config.logger.debug(element_ids)
    # config.logger.debug(node_x_coords)
    # config.logger.debug(node_y_coords)
    # config.logger.debug(node_bc)

    # Parallel calculations on GPU
    threads_per_block = 128
    blocks_per_grid = (len(grid.elements) + threads_per_block - 1) // threads_per_block
    _calculate_for_element[blocks_per_grid, threads_per_block](n, weights, n_tab, suraces_N_tab,
                                                               dn_dksi_tab, dn_deta_tab,
                                                               element_ids,
                                                               node_x_coords, node_y_coords, node_bc,
                                                               H_matrices, Hbc_matrices, P_vectors, C_matrices,
                                                               dx_dksi_tabs, dx_deta_tabs, dy_dksi_tabs, dy_deta_tabs,
                                                               dn_dx_tab, dn_dy_tab, det_tab,
                                                               H_ip_matrices, Hbc_ip_matrices, P_ip_vectors, C_ip_matrices,
                                                               c, d, sh, alfa, tot)
    cuda.synchronize()
    # Retrieve results from GPU
    # element_ids = element_ids.get()
    # dx_dksi_tabs = dx_dksi_tabs.get()
    # dx_deta_tabs = dx_deta_tabs.get()
    # dy_dksi_tabs = dy_dksi_tabs.get()
    # dy_deta_tabs = dy_deta_tabs.get()
    # config.logger.debug(f"Element {element_ids[0]}:")
    # config.logger.debug(f"dx/dksi:\n{dx_dksi_tabs[0]}")
    # config.logger.debug(f"dx/deta:\n{dx_deta_tabs[0]}")
    # config.logger.debug(f"dy/dksi:\n{dy_dksi_tabs[0]}")
    # config.logger.debug(f"dy/deta:\n{dy_deta_tabs[0]}")
    # dn_dx_tab = dn_dx_tab.get()
    # dn_dy_tab = dn_dy_tab.get()
    # det_tab = det_tab.get()
    # config.logger.debug(f"dn/dx:\n{dn_dx_tab[0]}")
    # config.logger.debug(f"dn/dy:\n{dn_dy_tab[0]}")
    # config.logger.debug(f"det:\n{det_tab[0]}")
    # H_ip_matrices = H_ip_matrices.get()
    # C_ip_matrices = C_ip_matrices.get()
    # config.logger.debug(f"H_ip:\n{H_ip_matrices[0]}")
    # config.logger.debug(f"C_ip:\n{C_ip_matrices[0]}")
    H_matrices = H_matrices.get()
    C_matrices = C_matrices.get()
    Hbc_matrices = Hbc_matrices.get()
    P_vectors = P_vectors.get()
    # config.logger.debug(f"H:\n{H_matrices[0]}")
    # config.logger.debug(f"C:\n{C_matrices[0]}")
    # config.logger.debug(f"Hbc:\n{Hbc_matrices[0]}")
    # config.logger.debug(f"P:\n{P_vectors[0]}")
    _save_to_elements(grid, H_matrices, C_matrices, Hbc_matrices, P_vectors)

def _save_to_elements(grid, H_matrices, C_matrices, Hbc_matrices, P_vectors):
    for i, element in enumerate(grid.elements):
        element: Element
        element.H = H_matrices[i]
        element.C = C_matrices[i]
        element.Hbc = Hbc_matrices[i]
        element.P = P_vectors[i]

@cuda.jit
def _calculate_for_element(n: cp.int32, weights: cp.ndarray, n_tab: cp.ndarray, surfaces_N_tab: cp.ndarray,
                           dn_dksi_tab: cp.ndarray, dn_deta_tab: cp.ndarray,
                           element_ids: cp.ndarray,
                           node_x_coords: cp.ndarray, node_y_coords: cp.ndarray, node_bc: cp.ndarray,
                           H_matrices: cp.ndarray, Hbc_matrices: cp.ndarray, P_vectors: cp.ndarray, C_matrices: cp.ndarray,
                           dx_dksi_tabs: cp.ndarray, dx_deta_tabs: cp.ndarray, dy_dksi_tabs: cp.ndarray, dy_deta_tabs: cp.ndarray,
                           dn_dx_tab: cp.ndarray, dn_dy_tab: cp.ndarray, det_tab: cp.ndarray,
                           H_ip_matrices: cp.ndarray, Hbc_ip_matrices: cp.ndarray, P_ip_vectors: cp.ndarray, C_ip_matrices: cp.ndarray,
                           c: cp.float32, d: cp.float32, sh: cp.float32, alfa: cp.float32, tot: cp.float32) -> None:
    i = cuda.grid(1)
    if i < element_ids.size:
        _fill_x_y_ksi_eta_tabs(n, dn_dksi_tab, dn_deta_tab,
                               node_x_coords[i], node_y_coords[i],
                               dx_dksi_tabs[i], dx_deta_tabs[i], dy_dksi_tabs[i], dy_deta_tabs[i])
        _dn_dx_dn_dy(n, dn_dksi_tab, dn_deta_tab,
                     dx_dksi_tabs[i], dx_deta_tabs[i], dy_dksi_tabs[i], dy_deta_tabs[i],
                     dn_dx_tab[i], dn_dy_tab[i], det_tab[i])
        _calculate_for_integration_points(n, n_tab,
                                          dn_dx_tab[i], dn_dy_tab[i], det_tab[i],
                                          H_ip_matrices[i], C_ip_matrices[i],
                                          c, d, sh)
        for j in range(n*n):
            _multiply_matrix_by_scalar(H_ip_matrices[i][j], weights[j//n]*weights[j%n])
            _add_matrices(H_matrices[i], H_ip_matrices[i][j]) # Stores result in H_matrices[i]
            _multiply_matrix_by_scalar(C_ip_matrices[i][j], weights[j//n]*weights[j%n])
            _add_matrices(C_matrices[i], C_ip_matrices[i][j]) # Stores result in C_matrices[i]

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
def _calculate_for_surface(n: cp.int32, weights: cp.ndarray, surface_N_tab: cp.ndarray,
                           node1_x: cp.float32, node1_y: cp.float32, node1_bc: cp.int32,
                           node2_x: cp.float32, node2_y: cp.float32, node2_bc: cp.int32,
                           Hbc_matrix: cp.ndarray, P_vector: np.ndarray,
                           alfa: cp.float32, tot: cp.float32) -> None:
    if node1_bc == 0 or node2_bc == 0:
        return
    L = sqrt((node2_x - node1_x)**2 + (node2_y - node1_y)**2)
    detJ = L/2
    for j in range(n):
        mx = cuda.local.array(NODES_PER_ELEMENT, cp.float32)
        mx_P = cuda.local.array(NODES_PER_ELEMENT, cp.float32)
        mx_H = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), cp.float32)
        for k in range(NODES_PER_ELEMENT):
            mx[k] = surface_N_tab[j][k]
            mx_P[k] = mx[k]*weights[j]*alfa*tot*detJ
        for k in range(NODES_PER_ELEMENT):
            P_vector[k] += mx_P[k]
        _multiply_cuda_local_arrays(mx, mx, mx_H)
        _multiply_matrix_by_scalar(mx_H, weights[j]*alfa*detJ)
        _add_cuda_local_arrays(mx_H, Hbc_matrix, Hbc_matrix)

@cuda.jit(device=True)
def _calculate_for_integration_points(n: cp.int32, n_tab: cp.ndarray,
                                      dn_dx_tab: cp.ndarray, dn_dy_tab: cp.ndarray, det_tab: cp.ndarray,
                                      H_ip_matrix: cp.ndarray, C_ip_matrix: cp.ndarray,
                                      c: cp.float32, d: cp.float32, sh: cp.float32) -> None:
    for j in range(n*n):
        dN_dx_vc = cuda.local.array(NODES_PER_ELEMENT, cp.float32)
        dN_dx_vc_multiplied = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), cp.float32)
        dN_dy_vc = cuda.local.array(NODES_PER_ELEMENT, cp.float32)
        dN_dy_vc_multiplied = cuda.local.array((NODES_PER_ELEMENT, NODES_PER_ELEMENT), cp.float32)
        N_vc = cuda.local.array(NODES_PER_ELEMENT, cp.float32)
        for k in range(NODES_PER_ELEMENT):
            dN_dx_vc[k] = dn_dx_tab[j][k]
            dN_dy_vc[k] = dn_dy_tab[j][k]
            N_vc[k] = n_tab[j][k]
        _multiply_cuda_local_arrays(dN_dx_vc, dN_dx_vc, dN_dx_vc_multiplied)
        _multiply_cuda_local_arrays(dN_dy_vc, dN_dy_vc, dN_dy_vc_multiplied)
        _add_cuda_local_arrays(dN_dx_vc_multiplied, dN_dy_vc_multiplied, H_ip_matrix[j])
        _multiply_matrix_by_scalar(H_ip_matrix[j], c*det_tab[j])

        _multiply_cuda_local_arrays(N_vc, N_vc, C_ip_matrix[j])
        _multiply_matrix_by_scalar(C_ip_matrix[j], sh*d*det_tab[j])

@cuda.jit(device=True)
def _multiply_cuda_local_arrays(A: cuda.local.array, B: cuda.local.array, C: cuda.local.array) -> None:
    for i in range(4):
        for j in range(4):
            C[i][j] = A[i] * B[j]

@cuda.jit(device=True)
def _multiply_matrix_by_scalar(A: cp.ndarray, scalar: cp.float32) -> None:
    for j in range(A.shape[0]):
        for k in range(A.shape[1]):
            A[j][k] *= scalar

@cuda.jit(device=True)
def _add_matrices(A: cp.ndarray, B: cp.ndarray) -> None:
    for j in range(A.shape[0]):
        for k in range(A.shape[1]):
            A[j][k] += B[j][k]

@cuda.jit(device=True)
def _add_cuda_local_arrays(A: cuda.local.array, B: cuda.local.array, C: cp.ndarray) -> None:
    for i in range(4):
        for j in range(4):
            C[i][j] = A[i][j] + B[i][j]

@cuda.jit(device=True)
def _dn_dx_dn_dy(n: cp.int32, dn_dksi_tab: cp.ndarray, dn_deta_tab: cp.ndarray,
                 dx_dksi_tab: cp.ndarray, dx_deta_tab: cp.ndarray, dy_dksi_tab: cp.ndarray, dy_deta_tab: cp.ndarray,
                 dn_dx_tab: cp.ndarray, dn_dy_tab: cp.ndarray, det_tab: cp.ndarray) -> None:
    for j in range(n*n):
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
            dn_dx_tab[j, k] = invJ_00 * dn_dksi_tab[j, k] + invJ_01 * dn_deta_tab[j, k]
            dn_dy_tab[j, k] = invJ_10 * dn_dksi_tab[j, k] + invJ_11 * dn_deta_tab[j, k]

@cuda.jit(device=True)
def _fill_x_y_ksi_eta_tabs(n: cp.int32, dn_dksi_tab: cp.ndarray, dn_deta_tab: cp.ndarray,
                           node_x_coords: cp.ndarray, node_y_coords: cp.ndarray,
                           dx_dksi_tab: cp.ndarray, dx_deta_tab: cp.ndarray, dy_dksi_tab: cp.ndarray, dy_deta_tab: cp.ndarray) -> None:
    for j in range(n*n):
        dx_dksi_tab[j] = _interpolate(dn_dksi_tab[j][0], dn_dksi_tab[j][1], dn_dksi_tab[j][2], dn_dksi_tab[j][3], node_x_coords)
        dx_deta_tab[j] = _interpolate(dn_deta_tab[j][0], dn_deta_tab[j][1], dn_deta_tab[j][2], dn_deta_tab[j][3], node_x_coords)
        dy_dksi_tab[j] = _interpolate(dn_dksi_tab[j][0], dn_dksi_tab[j][1], dn_dksi_tab[j][2], dn_dksi_tab[j][3], node_y_coords)
        dy_deta_tab[j] = _interpolate(dn_deta_tab[j][0], dn_deta_tab[j][1], dn_deta_tab[j][2], dn_deta_tab[j][3], node_y_coords)

@cuda.jit(device=True)
def _interpolate(dN1: cp.float32,
                 dN2: cp.float32,
                 dN3: cp.float32,
                 dN4: cp.float32,
                 var: cp.ndarray) -> cp.float32:
    return cp.float32(dN1*var[0] + dN2*var[1] + dN3*var[2] + dN4*var[3])