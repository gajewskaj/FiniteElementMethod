import argparse
import os
import numpy as np

from src.helpers.helpers import set_algorithms, create_or_clear_directory, init_logging
from src.helpers.config import Settings, MAIN_LOGGER_NAME
import src.helpers.config as config

def parse_arguments() -> tuple[str, str, bool]:
    parser = argparse.ArgumentParser(description="Finite Element Method Simulation")
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join(Settings.input_path, "100x100_quad.msh"),
        help="Path to the input mesh file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(Settings.input_path, "global_data.json"),
        help="Path to the input data file",
    )
    parser.add_argument(
        "--mc",
        type=str,
        default="gpu",
        help="Processing unit for calculating matrices",
        choices=["cpu", "gpu"]
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="cudss",
        help="Solver to use for the simulation",
        choices=["cudss", "cupy", "scipy"]
    )
    args = parser.parse_args()
    return args.mesh, args.data, args.mc, args.solver

def run() -> None:
    try:
        mesh_filepath, data_filepath, mc, solver = parse_arguments()
        # In output directory, create a subdirectory with the name of the input mesh file
        output_dir_path = create_or_clear_directory(os.path.join(Settings.output_path,
                                                                 os.path.basename(mesh_filepath).split(".")[0]))
        config.logger = init_logging(logger_name=MAIN_LOGGER_NAME, log_dirpath=output_dir_path)
        set_algorithms(mc, solver)

        from src.mesh.mesh import Mesh
        mesh = Mesh(mesh_filepath, data_filepath)

        from src.mc.matrices_calculation import calculate_and_assemble_matrices
        from src.soe.temperature_simulation import simulate

        calculate_and_assemble_matrices(mesh)
        times, temperatures = simulate(mesh)

        config.logger.info(f"Time        Min temp        Max temp")
        for i in range(len(temperatures)):
            config.logger.info(f"{(times[i]):<12}{round(np.min(temperatures[i]), 6):<16}{round(np.max(temperatures[i]), 6):<16}")

        # Generate .vtk files for ParaView
        from src.helpers.vtk_generator import generate_vtk_files
        generate_vtk_files(output_dir_path, mesh, temperatures)
    except Exception as e:
        config.logger.error(f"Script execution failed due to an error:\n{e}", exc_info=True)

if __name__ == "__main__":
    run()