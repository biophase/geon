"""Installed-wheel/frozen-executable smoke test; exercises both OCCT kernels."""
import sys


def check_ifc():
    import ifcopenshell
    import ifcopenshell.geom
    model = ifcopenshell.file(schema="IFC4")
    origin = model.createIfcCartesianPoint((0., 0., 0.))
    placement = model.createIfcAxis2Placement3D(origin)
    solid = model.createIfcBlock(placement, 1., 1., 1.)
    shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), solid)
    assert len(shape.verts) == 24


def check_geon():
    from geon._native import surface_geometry as native
    assert native.occt_version == '7.9.3'
    a = native.Surface(dict(kind='plane', origin=[0., 0., 0.], normal=[0., 0., 1.],
                            u=[1., 0., 0.], bounds=[-1., -1., 1., 1.]))
    b = native.Surface(dict(kind='plane', origin=[0., 0., 0.], normal=[1., 0., 0.],
                            u=[0., 1., 0.], bounds=[-1., -1., 1., 1.]))
    curves = native.intersections(a, b, .05)
    assert len(curves) == 1 and len(curves[0]) > 2
    assert abs(curves[0][:, [0, 2]]).max() < 1e-8


if __name__ == '__main__':
    checks = [check_geon, check_ifc]
    if '--ifc-first' in sys.argv:
        checks.reverse()
    for check in checks:
        check()
    # Exercise both again after both native libraries have loaded.
    check_geon()
    check_ifc()
    print('OCCT 7.9.3 and IfcOpenShell geometry smoke test passed')
