"""Subprocess-only native interoperability check, in both import orders."""
import sys
import numpy as np

if '--ifc-first' in sys.argv:
    import ifcopenshell
    from geon._native import surface_geometry as geometry
else:
    from geon._native import surface_geometry as geometry
    import ifcopenshell

model = ifcopenshell.file(schema='IFC4')
model.create_entity('IfcCartesianPoint', Coordinates=(0., 0., 0.))
spec = dict(kind='plane', bounds=[-2., -2., 2., 2.], origin=[0., 0., 0.],
            normal=[0., 0., 1.], u=[1., 0., 0.])
a = geometry.Surface(spec)
b = geometry.Surface(dict(spec, normal=[0., 1., 0.]))
curves = geometry.intersections(a, b, .1)
assert curves and any(len(curve) >= 2 for curve in curves)
for curve in curves:
    np.testing.assert_allclose(np.asarray(curve)[:, 1:], 0., atol=1e-6)
projected = geometry.closest_points(a, np.array([[.5, .5, 1.]]))
np.testing.assert_allclose(projected, [[.5, .5, 0.]], atol=1e-6)
assert len(model.by_type('IfcCartesianPoint')) == 1
print('smoke test passed')
