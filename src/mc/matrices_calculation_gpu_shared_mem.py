from math import sqrt

from numba import cuda
import numpy as np

from src.helpers.config import Settings
from src.helpers.helpers import measure_time
from src.mesh.mesh import Mesh
from src.uel.universal_element import u_el

DOF = Settings.MatricesCalculation.DOF
TPB = Settings.MatricesCalculation.TPB
MAX_MATERIALS = Settings.MatricesCalculation.MAX_MATERIALS
MATERIAL_QUAL = Settings.MatricesCalculation.MATERIAL_QUAL
MAX_IP = Settings.MatricesCalculation.MAX_IP

@measure_time
def calculate_and_assemble_matrices(grid: Mesh) -> None:
    stream_H_C = cuda.stream()
    stream_Hbc_P = cuda.stream()
    data_H_C, data_Hbc_P = _send_to_gpu(stream_H_C, stream_Hbc_P, grid)
    blocks_per_grid = (len(grid.elements_id) + TPB - 1) // TPB
    _calculate(data_H_C, data_Hbc_P, stream_H_C, stream_Hbc_P, blocks_per_grid)
    _retrieve_from_gpu(data_H_C, data_Hbc_P, grid)

@measure_time
def _calculate(data_H_C: tuple, data_Hbc_P: tuple,
               stream_H_C: cuda.stream, stream_Hbc_P: cuda.stream,
               blocks_per_grid: int) -> tuple:
    _calculate_H_C_for_element[blocks_per_grid, TPB, stream_H_C](*data_H_C)
    _calculate_Hbc_P_for_element[blocks_per_grid, TPB, stream_Hbc_P](*data_Hbc_P)
    stream_H_C.synchronize()
    stream_Hbc_P.synchronize()

@measure_time
def _send_to_gpu(stream_H_C: cuda.stream, stream_Hbc_P: cuda.stream, grid: Mesh) -> tuple:
    # Materials
    materials_cuda = cuda.to_device(grid.global_data.materials)
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
    elements_material_ids_cuda = cuda.to_device(grid.elements_material_ids)
    # Local matrices
    global_H_values_cuda = cuda.to_device(grid.H_val, stream=stream_H_C)
    global_H_row_cuda = cuda.to_device(grid.H_row, stream=stream_H_C)
    global_H_col_cuda = cuda.to_device(grid.H_col, stream=stream_H_C)
    global_C_values_cuda = cuda.to_device(grid.C_val, stream=stream_H_C)
    global_C_row_cuda = cuda.to_device(grid.C_row, stream=stream_H_C)
    global_C_col_cuda = cuda.to_device(grid.C_col, stream=stream_H_C)
    global_Hbc_values_cuda = cuda.to_device(grid.Hbc_val, stream=stream_Hbc_P)
    global_Hbc_row_cuda = cuda.to_device(grid.Hbc_row, stream=stream_Hbc_P)
    global_Hbc_col_cuda = cuda.to_device(grid.Hbc_col, stream=stream_Hbc_P)
    global_P_cuda = cuda.to_device(grid.P, stream=stream_Hbc_P)
    return (
        nodes_x_cuda, nodes_y_cuda, elements_node_ids_cuda, elements_material_ids_cuda,
        u_el.n, weights_cuda, N_cuda,
        dN_dxi_cuda, dN_deta_cuda,
        materials_cuda,
        global_H_values_cuda, global_H_row_cuda, global_H_col_cuda,
        global_C_values_cuda, global_C_row_cuda, global_C_col_cuda
        ), (
        nodes_x_cuda, nodes_y_cuda, nodes_bc_cuda, elements_node_ids_cuda, elements_material_ids_cuda,
        u_el.quadrature_1d.n, surface_weights_cuda, surfaces_cuda,
        materials_cuda,
        grid.global_data.ambient_temp,
        global_Hbc_values_cuda, global_Hbc_row_cuda, global_Hbc_col_cuda,
        global_P_cuda
        )

@measure_time
def _retrieve_from_gpu(data_H_C: tuple, data_Hbc_P: tuple, grid: Mesh) -> None:
    grid.H_val = data_H_C[-6].copy_to_host().astype(np.float64)
    grid.H_row = data_H_C[-5].copy_to_host()
    grid.H_col = data_H_C[-4].copy_to_host()
    grid.C_val = data_H_C[-3].copy_to_host().astype(np.float64)
    grid.C_row = data_H_C[-2].copy_to_host()
    grid.C_col = data_H_C[-1].copy_to_host()
    grid.Hbc_val = data_Hbc_P[-4].copy_to_host().astype(np.float64)
    grid.Hbc_row = data_Hbc_P[-3].copy_to_host()
    grid.Hbc_col = data_Hbc_P[-2].copy_to_host()
    grid.P = data_Hbc_P[-1].copy_to_host()

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
    for i in range(DOF):
        dN_dx[i] = jacobian_inv_00 * dN_dxi[i] + jacobian_inv_01 * dN_deta[i]
        dN_dy[i] = jacobian_inv_10 * dN_dxi[i] + jacobian_inv_11 * dN_deta[i]
    return jacobian_det

@cuda.jit('float64(float64[:], float64[:])', device=True)
def _interpolate(dN, var):
    result = 0.0
    for i in range(DOF):
        result += dN[i] * var[i]
    return result

@cuda.jit('void(float64, float64[:], float64, float64[:], float64[:], float64, float64, float64, float64[:,:], float64[:,:])', device=True)
def _calculate_H_C_for_integration_point(weight, N,
                                         jacobian_det, dN_dx, dN_dy,
                                         c, d, sh,
                                         H, C):
    factor_H = c * jacobian_det * weight
    factor_C = sh * d * jacobian_det * weight
    for i in range(DOF):
        for j in range(DOF):
            H[i, j] += (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]) * factor_H
            C[i, j] += N[i] * N[j] * factor_C

@cuda.jit('void(float64[:], float64[:], int64[:,:], int64[:], int64, float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], int64[:], int64[:], float64[:], int64[:], int64[:])')
def _calculate_H_C_for_element(nodes_x, nodes_y, elements_node_ids, elements_material_ids,
                               n, weights, N,
                               dN_dxi, dN_deta,
                               materials,
                               H_val, H_row, H_col,
                               C_val, C_row, C_col):
    materials_shared = cuda.shared.array((MAX_MATERIALS, MATERIAL_QUAL), np.float64)
    weights_shared = cuda.shared.array(MAX_IP, np.float64)
    local_idx = cuda.threadIdx.x

    if local_idx == 0:
        for material_idx in range(len(materials)):
            for k in range(MATERIAL_QUAL):
                materials_shared[material_idx, k] = materials[material_idx, k]

    if local_idx < len(weights):
        weights_shared[local_idx] = weights[local_idx]
    cuda.syncthreads()

    i = cuda.grid(1)
    if i >= len(H_val)//DOF//DOF:
        return
    x_coords = cuda.local.array(DOF, np.float64)
    y_coords = cuda.local.array(DOF, np.float64)
    H = cuda.local.array((DOF, DOF), np.float64)
    C = cuda.local.array((DOF, DOF), np.float64)
    dN_dx = cuda.local.array(DOF, np.float64)
    dN_dy = cuda.local.array(DOF, np.float64)
    c = materials_shared[elements_material_ids[i] - 1, 1]
    d = materials_shared[elements_material_ids[i] - 1, 2]
    sh= materials_shared[elements_material_ids[i] - 1, 3]
    for j in range(DOF):
        x_coords[j] = nodes_x[elements_node_ids[i, j] - 1]
        y_coords[j] = nodes_y[elements_node_ids[i, j] - 1]
    for j in range(n):
        dx_dxi = _interpolate(dN_dxi[j], x_coords)
        dx_deta = _interpolate(dN_deta[j], x_coords)
        dy_dxi = _interpolate(dN_dxi[j], y_coords)
        dy_deta = _interpolate(dN_deta[j], y_coords)
        jacobian_det = _calculate_jacobian_and_global_shape_derivatives(dN_dxi[j], dN_deta[j],
                                                                        dx_dxi, dx_deta, dy_dxi, dy_deta,
                                                                        dN_dx, dN_dy)
        _calculate_H_C_for_integration_point(weights_shared[j], N[j],
                                                jacobian_det, dN_dx, dN_dy,
                                                c, d, sh,
                                                H, C)
    # Assembly
    for j in range(DOF):
        for k in range(DOF):
            idx = i*DOF*DOF+j*DOF+k
            H_val[idx] = H[j, k]
            H_row[idx] = elements_node_ids[i, j] - 1
            H_col[idx] = elements_node_ids[i, k] - 1
            C_val[idx] = C[j, k]
            C_row[idx] = elements_node_ids[i, j] - 1
            C_col[idx] = elements_node_ids[i, k] - 1

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
        N = surface[i]
        factor = weights[i] * alpha * jacobian_det
        for k in range(DOF):
            P[k] += N[k] * factor * ambient_temp
            for j in range(DOF):
                Hbc[k, j] += N[k] * N[j] * factor

@cuda.jit('void(float64[:], float64[:], int64[:], int64[:,:], int64[:], int64, float64[:], float64[:,:,:], float64[:,:], float64, float64[:], int64[:], int64[:], float64[:])')
def _calculate_Hbc_P_for_element(nodes_x, nodes_y, nodes_bc, elements_node_ids, elements_material_ids,
                                 n, weights, surfaces,
                                 materials, ambient_temp,
                                 Hbc_val, Hbc_row, Hbc_col,
                                 P_global):
    materials_shared = cuda.shared.array((MAX_MATERIALS, 4), np.float64)
    surface_weights_shared = cuda.shared.array(MAX_IP, np.float64)
    local_idx = cuda.threadIdx.x

    if local_idx == 0:
        for material_idx in range(len(materials)):
            for k in range(4):
                materials_shared[material_idx, k] = materials[material_idx, k]

    if local_idx < len(weights):
        surface_weights_shared[local_idx] = weights[local_idx]
    cuda.syncthreads()

    i = cuda.grid(1)
    if i >= len(Hbc_val)//DOF//DOF:
        return
    x_coords = cuda.local.array(DOF, np.float64)
    y_coords = cuda.local.array(DOF, np.float64)
    bc = cuda.local.array(DOF, np.int64)
    P = cuda.local.array(DOF, np.float64)
    Hbc = cuda.local.array((DOF, DOF), np.float64)
    alpha = materials_shared[elements_material_ids[i] - 1, 0]
    for j in range(DOF):
        x_coords[j] = nodes_x[elements_node_ids[i, j] - 1]
        y_coords[j] = nodes_y[elements_node_ids[i, j] - 1]
        bc[j] = nodes_bc[elements_node_ids[i, j] - 1]
    for j in range(DOF):
        _calculate_for_surface(n, surface_weights_shared, surfaces[j],
                                x_coords[j], y_coords[j], bc[j],
                                x_coords[(j+1)%DOF], y_coords[(j+1)%DOF], bc[(j+1)%DOF],
                                alpha, ambient_temp,
                                Hbc, P)
    # Assembly
    for j in range(DOF):
        cuda.atomic.add(P_global, elements_node_ids[i, j] - 1, P[j])
        for k in range(DOF):
            idx = i*DOF*DOF+j*DOF+k
            Hbc_val[idx] = Hbc[j, k]
            Hbc_row[idx] = elements_node_ids[i, j] - 1
            Hbc_col[idx] = elements_node_ids[i, k] - 1