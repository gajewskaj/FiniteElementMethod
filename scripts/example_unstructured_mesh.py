import argparse
import os
import sys
import gmsh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.helpers.config import logger, Settings

filename = "unstructured.msh"

min_x = 0
max_x = 0.100000001
min_y = -0.0949999988
max_y = 0.00499999989

gmsh.initialize()

gmsh.model.add(filename.strip(".msh"))

p1 = gmsh.model.geo.addPoint(min_x, min_y, 0)
p2 = gmsh.model.geo.addPoint(max_x, min_y, 0)
p3 = gmsh.model.geo.addPoint(max_x, max_y, 0)
p4 = gmsh.model.geo.addPoint(min_x, max_y, 0)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surface = gmsh.model.geo.addPlaneSurface([curve_loop])

# Synchronize geometry to be able to work with the mesh
gmsh.model.geo.synchronize()

# Set mesh size near the boundary points
gmsh.model.mesh.field.add("MathEval", 1)
gmsh.model.mesh.field.setString(1, "F", "0.02")  # Defining mesh size
gmsh.model.mesh.field.setAsBackgroundMesh(1)

# Generate the mesh without transfinite or recombination (unstructured)
gmsh.model.mesh.generate(2)

gmsh.fltk.run()

gmsh.write(os.path.join(Settings.input_path, filename))

gmsh.finalize()
