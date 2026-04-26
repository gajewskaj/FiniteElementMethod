from src.helpers.config import logger, Settings

if Settings.MatricesCalculation.num_of_shape_functions == 3:
    from src.uel.universal_element_triangle import UniversalElementTriangle
    u_el = UniversalElementTriangle(3)
elif Settings.MatricesCalculation.num_of_shape_functions == 4:
    from src.uel.universal_element_quadrangle import UniversalElementQuadrangle
    u_el = UniversalElementQuadrangle(4)
else:
    err_msg: str = "Invalid element type."
    logger.error(err_msg)
    raise RuntimeError(err_msg)