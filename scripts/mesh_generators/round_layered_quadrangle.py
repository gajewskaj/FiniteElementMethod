import argparse
import gmsh
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.helpers.config import Settings

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--r-inner", type=float, default=0.15)
    parser.add_argument("--r-mid", type=float, default=0.20)
    parser.add_argument("--r-outer", type=float, default=0.25)

    parser.add_argument("--theta-elem", type=int, default=60, help="Number of elements along the full circumference (must be divisible by 4)",)
    parser.add_argument("--radial-inner-elem", type=int, default=3, help="Number of radial elements between r_inner and r_mid")
    parser.add_argument("--radial-outer-elem", type=int, default=3, help="Number of radial elements between r_mid and r_outer")
    return parser.parse_args()

def _pt(radius: float, angle_rad: float) -> tuple[float, float]:
    return radius * math.cos(angle_rad), radius * math.sin(angle_rad)

def _require_radii(r_inner: float, r_mid: float, r_outer: float) -> None:
    if not (r_inner < r_mid < r_outer):
        raise ValueError("Require r_inner < r_mid < r_outer")

def _require_theta_divisible_by_4(theta_elem: int) -> None:
    if theta_elem % 4 != 0:
        raise ValueError("--theta-elem must be divisible by 4 (for 4 quadrants)")

def _add_points_on_radius(radius: float, angles: list[float]) -> list[int]:
    return [gmsh.model.geo.addPoint(*_pt(radius, angle), 0) for angle in angles]

def _add_circle_arcs(points: list[int], center: int) -> list[int]:
    return [gmsh.model.geo.addCircleArc(points[i], center, points[(i + 1) % 4]) for i in range(4)]

def main() -> int:
    args = parse_arguments()
    _require_radii(args.r_inner, args.r_mid, args.r_outer)
    _require_theta_divisible_by_4(args.theta_elem)

    n_theta_quarter = args.theta_elem // 4
    filename = (
        f"round_layered_quad.msh"
    )
    out_path = os.path.join(Settings.input_path, filename)

    model_name = os.path.splitext(os.path.basename(out_path))[0]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add(model_name)

        center = gmsh.model.geo.addPoint(0, 0, 0)
        radii = [args.r_inner, args.r_mid, args.r_outer]
        core_radius = args.r_inner * 0.25

        # Angles for 4 quadrants: 0, 90, 180, 270 deg
        angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]

        # Points for each radius at each quadrant angle.
        pts_r1, pts_r2, pts_r3 = [
            _add_points_on_radius(radius, angles) for radius in radii
        ]

        # Small inner diamond so the center can also be meshed with quads.
        pts_core = _add_points_on_radius(core_radius, angles)

        # Circle arcs for each radius, per quadrant.
        arcs_r1, arcs_r2, arcs_r3 = [
            _add_circle_arcs(points, center) for points in (pts_r1, pts_r2, pts_r3)
        ]

        # Radial lines (per quadrant angle).
        radial_r1_r2 = [gmsh.model.geo.addLine(pts_r1[i], pts_r2[i]) for i in range(4)]
        radial_r2_r3 = [gmsh.model.geo.addLine(pts_r2[i], pts_r3[i]) for i in range(4)]

        radial_core_r1 = [gmsh.model.geo.addLine(pts_core[i], pts_r1[i]) for i in range(4)]

        core_edges = [gmsh.model.geo.addLine(pts_core[i], pts_core[(i + 1) % 4]) for i in range(4)]

        core_surfaces: list[int] = []
        inner_ring_surfaces: list[int] = []
        outer_ring_surfaces: list[int] = []

        # Create a central quad and 4 quad sectors around it.
        loop_core = gmsh.model.geo.addCurveLoop(core_edges)
        core_surfaces.append(gmsh.model.geo.addPlaneSurface([loop_core]))

        for i in range(4):
            j = (i + 1) % 4

            loop_center = gmsh.model.geo.addCurveLoop([
                radial_core_r1[i],
                arcs_r1[i],
                -radial_core_r1[j],
                -core_edges[i],
            ])
            core_surfaces.append(gmsh.model.geo.addPlaneSurface([loop_center]))

        # Create 4 transfinite quad patches for each ring.
        for i in range(4):
            j = (i + 1) % 4

            # Inner ring patch between r_inner and r_mid.
            loop_inner = gmsh.model.geo.addCurveLoop([
                arcs_r2[i],
                -radial_r1_r2[j],
                -arcs_r1[i],
                radial_r1_r2[i],
            ])
            s_inner = gmsh.model.geo.addPlaneSurface([loop_inner])
            inner_ring_surfaces.append(s_inner)

            # Outer ring patch between r_mid and r_outer.
            loop_outer = gmsh.model.geo.addCurveLoop([
                arcs_r3[i],
                -radial_r2_r3[j],
                -arcs_r2[i],
                radial_r2_r3[i],
            ])
            s_outer = gmsh.model.geo.addPlaneSurface([loop_outer])
            outer_ring_surfaces.append(s_outer)

        # Transfinite constraints.
        for arc in arcs_r1 + arcs_r2 + arcs_r3:
            gmsh.model.geo.mesh.setTransfiniteCurve(arc, n_theta_quarter + 1)
        for line in radial_r1_r2:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, args.radial_inner_elem + 1)
        for line in radial_r2_r3:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, args.radial_outer_elem + 1)
        for line in radial_core_r1:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, args.radial_inner_elem + 1)
        for line in core_edges:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, args.radial_inner_elem + 1)

        for s in core_surfaces + inner_ring_surfaces + outer_ring_surfaces:
            gmsh.model.geo.mesh.setTransfiniteSurface(s)
            gmsh.model.geo.mesh.setRecombine(2, s)

        gmsh.model.geo.synchronize()

        # Physical groups (materials and boundary conditions).
        pg_inner = gmsh.model.addPhysicalGroup(2, core_surfaces)
        gmsh.model.setPhysicalName(2, pg_inner, "copper")

        pg_mid = gmsh.model.addPhysicalGroup(2, inner_ring_surfaces)
        gmsh.model.setPhysicalName(2, pg_mid, "styrofoam")

        pg_outer = gmsh.model.addPhysicalGroup(2, outer_ring_surfaces)
        gmsh.model.setPhysicalName(2, pg_outer, "steel")

        # Boundary curves
        pg_bc = gmsh.model.addPhysicalGroup(1, arcs_r3)
        gmsh.model.setPhysicalName(1, pg_bc, "bc")

        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.generate(2)

        gmsh.fltk.run()

        gmsh.write(out_path)
        return 0
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    raise SystemExit(main())