import gmsh
import sys
import math

gmsh.initialize()

gmsh.model.add("circle_mesh")


radius = 0.1
lc = 0.05

p1 = gmsh.model.geo.addPoint(0, radius, 0, lc)
p2 = gmsh.model.geo.addPoint(radius, 0, 0, lc)
p3 = gmsh.model.geo.addPoint(0, -radius, 0, lc)
p4 = gmsh.model.geo.addPoint(-radius, 0, 0, lc)

center = gmsh.model.geo.addPoint(0, 0, 0, lc)

arc1 = gmsh.model.geo.addCircleArc(p1, center, p2)
arc2 = gmsh.model.geo.addCircleArc(p2, center, p3)
arc3 = gmsh.model.geo.addCircleArc(p3, center, p4)
arc4 = gmsh.model.geo.addCircleArc(p4, center, p1)

loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])

surface = gmsh.model.geo.addPlaneSurface([loop])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

gmsh.write("circle_mesh.msh")
gmsh.finalize()
