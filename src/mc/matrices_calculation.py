from src.helpers.config import logger, Settings
from src.mesh.mesh import Mesh
from src.mc.out import OutMatrices

def calculate_and_assemble_matrices(mesh: Mesh, out_mat: OutMatrices) -> None:
    if Settings.MatricesCalculation.use_gpu:
        logger.info(f"Calculating and assembling matrices using GPU.")
        from src.mc.matrices_calculation_gpu import calculate_and_assemble_matrices
    else:
        logger.info(f"Calculating and assembling matrices using CPU.")
        from src.mc.matrices_calculation_cpu import calculate_and_assemble_matrices
    calculate_and_assemble_matrices(mesh, out_mat)