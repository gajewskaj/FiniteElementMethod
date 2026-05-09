import argparse
import os
import sys
import gmsh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.helpers.config import Settings

def parse_arguments() -> tuple[int, int]:
    parser = argparse.ArgumentParser(description="Mesh generator")
    parser.add_argument("--x-elem", type=int, default=100,
                        help="Number of x elements")
    parser.add_argument("--y-elem", type=int, default=100,
                        help="Number of y elements")
    args = parser.parse_args()
    return args.x_elem, args.y_elem

num_elements_x, num_elements_y = parse_arguments()
filename = f"{num_elements_x}x{num_elements_y}_quad.msh"

min_x = 0
max_x = 0.1
min_y = -0.095
max_y = 0.005

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

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

gmsh.model.geo.mesh.setTransfiniteCurve(l1, num_elements_x + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, num_elements_x + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, num_elements_y + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, num_elements_y + 1)

gmsh.model.geo.mesh.setTransfiniteSurface(surface)
gmsh.model.geo.mesh.setRecombine(2, surface)

gmsh.model.geo.synchronize()

pg = gmsh.model.addPhysicalGroup(2, [surface])
gmsh.model.setPhysicalName(2, pg, "material")

borders: list[int] = [l1, l2, l3, l4]
pg_bc = gmsh.model.addPhysicalGroup(1, borders)
gmsh.model.setPhysicalName(1, pg_bc, "bc")

gmsh.model.mesh.generate(2)

# gmsh.fltk.run()

gmsh.write(os.path.join(Settings.input_path, filename))

gmsh.finalize()
