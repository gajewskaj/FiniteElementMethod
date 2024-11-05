import sys
import gmsh

def create_mesh():
    gmsh.initialize()
    gmsh.model.add("example_grid")

    nodes = [
        (1, 0.100000001, 0.00499999989, 0.0),
        (2, 0.0666666701, 0.00499999989, 0.0),
        (3, 0.0333333351, 0.00499999989, 0.0),
        (4, 0.0, 0.00499999989, 0.0),
        (5, 0.100000001, -0.0283333343, 0.0),
        (6, 0.0666666701, -0.0283333343, 0.0),
        (7, 0.0333333351, -0.0283333343, 0.0),
        (8, 0.0, -0.0283333343, 0.0),
        (9, 0.100000001, -0.0616666675, 0.0),
        (10, 0.0666666701, -0.0616666675, 0.0),
        (11, 0.0333333351, -0.0616666675, 0.0),
        (12, 0.0, -0.0616666675, 0.0),
        (13, 0.100000001, -0.0949999988, 0.0),
        (14, 0.0666666701, -0.0949999988, 0.0),
        (15, 0.0333333351, -0.0949999988, 0.0),
        (16, 0.0, -0.0949999988, 0.0)
    ]

    for node in nodes:
        gmsh.model.geo.addPoint(node[1], node[2], node[3], tag=node[0])
        print(f"Added node {node[0]} at ({node[1]}, {node[2]}, {node[3]})")

    gmsh.model.geo.synchronize()

    elements = [
        (1, [1, 2, 6, 5]),
        (2, [2, 3, 7, 6]),
        (3, [3, 4, 8, 7]),
        (4, [5, 6, 10, 9]),
        (5, [6, 7, 11, 10]),
        (6, [7, 8, 12, 11]),
        (7, [9, 10, 14, 13]),
        (8, [10, 11, 15, 14]),
        (9, [11, 12, 16, 15])
    ]

    for element in elements:
        element_id, node_ids = element
        lines = []
        for i in range(4):
            start_node = node_ids[i]
            end_node = node_ids[(i + 1) % 4]
            line_tag = gmsh.model.geo.addLine(start_node, end_node)
            lines.append(line_tag)
        curve_loop = gmsh.model.geo.addCurveLoop(lines)
        gmsh.model.geo.addPlaneSurface([curve_loop], tag=element_id)

    boundary_conditions = [1, 2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 16]
    gmsh.model.addPhysicalGroup(0, boundary_conditions, tag=1)
    gmsh.model.setPhysicalName(0, 1, "Boundary")

    all_nodes = set(node[0] for node in nodes)
    non_boundary_nodes = list(all_nodes - set(boundary_conditions))

    gmsh.model.addPhysicalGroup(0, non_boundary_nodes, tag=2)
    gmsh.model.setPhysicalName(0, 2, "NonBoundary")

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)

    gmsh.write("example_grid.msh")

    if 'close' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == "__main__":
    create_mesh()