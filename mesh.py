import gmsh


def progressive_mesh(
    e: float,
    progression: float = 0.9,
) -> None:
    gmsh.initialize()
    gmsh.model.add("ProgressiveMesh")
    geom = gmsh.model.geo

    p1 = geom.addPoint(0.0, 0.0, 0.0)
    p2 = geom.addPoint(e, 0.0, 0.0)
    p3 = geom.addPoint(e, 1e-3, 0.0)
    p4 = geom.addPoint(0.0, 1e-3, 0.0)

    l1 = geom.addLine(p1, p2)
    l2 = geom.addLine(p2, p3)
    l3 = geom.addLine(p4, p3)
    l4 = geom.addLine(p4, p1)

    cl = geom.addCurveLoop([l1, l2, -l3, l4])
    pl = geom.addPlaneSurface([cl])

    geom.mesh.setTransfiniteCurve(l1, 50, "Progression", coef=progression)
    geom.mesh.setTransfiniteCurve(l3, 50, "Progression", coef=progression)
    geom.mesh.setTransfiniteCurve(l2, 2)
    geom.mesh.setTransfiniteCurve(l4, 2)
    geom.mesh.setTransfiniteSurface(pl)
    geom.mesh.setRecombine(2, 1)

    geom.synchronize()
    gmsh.model.addPhysicalGroup(2, [pl])
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
    gmsh.finalize
