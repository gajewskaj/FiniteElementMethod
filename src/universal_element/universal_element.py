from src.helpers import config

if config.element_type == "triangle":
    from src.universal_element.universal_element_triangle import UniversalElementTriangle
    NUM_OF_SHAPE_FUNCTIONS = 3
    NUM_OF_SURFACES = NUM_OF_SHAPE_FUNCTIONS
    universal_element = UniversalElementTriangle(3)
elif config.element_type == "quadrangle":
    from src.universal_element.universal_element_quadrangle import UniversalElementQuadrangle
    NUM_OF_SHAPE_FUNCTIONS = 4
    NUM_OF_SURFACES = NUM_OF_SHAPE_FUNCTIONS
    universal_element = UniversalElementQuadrangle(4)
else:
    err_msg: str = "Invalid element type."
    config.logger.error(err_msg)
    raise RuntimeError(err_msg)