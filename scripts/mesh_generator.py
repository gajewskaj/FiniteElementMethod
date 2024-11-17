import argparse
import os
import sys
import gmsh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.helpers import config

def parse_arguments() -> tuple[int, int, str]:
    parser = argparse.ArgumentParser(description="Mesh generator")
    parser.add_argument("--x-elem", type=int, default=100,
                        help="Number of x elements")
    parser.add_argument("--y-elem", type=int, default=100,
                        help="Number of y elements")
    args = parser.parse_args()
    return args.x_elem, args.y_elem

# Define the size of the mesh
num_elements_x, num_elements_y = parse_arguments()
filename = f"{num_elements_x}x{num_elements_y}.msh"

min_x = 0
max_x = 0.100000001
min_y = -0.0949999988
max_y = 0.00499999989

# Initialize Gmsh
gmsh.initialize()

# Create a new model
gmsh.model.add(filename.strip(".msh"))

# Define the corner points of the quadrilateral
p1 = gmsh.model.geo.addPoint(min_x, min_y, 0)      # Bottom-left corner (0, 0)
p2 = gmsh.model.geo.addPoint(max_x, min_y, 0)  # Bottom-right corner (1, 0)
p3 = gmsh.model.geo.addPoint(max_x, max_y, 0)  # Top-right corner (1, 1)
p4 = gmsh.model.geo.addPoint(min_x, max_y, 0)      # Top-left corner (0, 1)

# Create the edges of the quadrilateral
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

# Create a curve loop and a plane surface for the quadrilateral
curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surface = gmsh.model.geo.addPlaneSurface([curve_loop])

# Apply a structured quadrilateral mesh using a transfinite algorithm
gmsh.model.geo.mesh.setTransfiniteCurve(l1, num_elements_x + 1)  # Number of divisions along x
gmsh.model.geo.mesh.setTransfiniteCurve(l3, num_elements_x + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, num_elements_y + 1)  # Number of divisions along y
gmsh.model.geo.mesh.setTransfiniteCurve(l4, num_elements_y + 1)

# Set the transfinite meshing for the surface
gmsh.model.geo.mesh.setTransfiniteSurface(surface)
gmsh.model.geo.mesh.setRecombine(2, surface)  # Set recombination to create quadrilaterals

# Synchronize the model
gmsh.model.geo.synchronize()

# Generate the mesh
gmsh.model.mesh.generate(2)  # 2D mesh

# Save the mesh to a file
gmsh.write(os.path.join(config.input_path, filename))

# Finalize Gmsh
gmsh.finalize()
