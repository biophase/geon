from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box
from scipy.spatial.transform import Rotation

from geon.algorithms.reconstruction.core import Config, Patch, reconstruct, _edge_points
from geon.algorithms.reconstruction.surfaces import BSplineHeight, Plane, fit_bspline
from geon.algorithms.reconstruction.primitives import Cylinder, raw_intersection_curves
from geon.algorithms.reconstruction.geometry_backend import triple_points, closest_points


def cloud():
    x, y = np.meshgrid(np.linspace(-1, 1, 31), np.linspace(-.7, .7, 29))
    x, y = x.ravel(), y.ravel()
    return np.column_stack((x, y, .12*x*x+.07*y*y+.025*x*y))


def config(**kwargs):
    return Config(**dict(dict(fit_cylinders=False, max_plane_rmse=.01, alpha_radius=.09,
                         join_coplanar=False, stitch_mesh_boundaries=False), **kwargs))


@pytest.mark.parametrize("rotated", [False, True])
def test_gently_curved_pca_local_patch_and_mesh(rotated):
    points = cloud()
    rotation = Rotation.from_euler('xyz', [0, 88, 23], degrees=True).as_matrix() if rotated else np.eye(3)
    offset = np.array([1e6, -2e6, 3e6]) if rotated else np.zeros(3)
    points = points@rotation.T+offset
    result = reconstruct(points, np.zeros(len(points), dtype=int), config=config())
    assert result.patches[0].kind == 'bspline', result.diagnostics
    assert result.patches[0].rmse < .01
    assert result.faces.size > 0
    p = result.patches[0]
    np.testing.assert_allclose(p.lift(p.project(result.vertices)), result.vertices, atol=1e-8)
    tri = result.vertices[result.faces]
    normals = p.normal_at(p.project(tri.mean(axis=1)))
    assert np.all(np.einsum('ij,ij->i', np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]), normals) > 0)
    assert np.linalg.norm(p.lift(p.project(tri.mean(axis=1)))-tri.mean(axis=1), axis=1).max() <= .0051


def test_adequate_plane_never_attempts_spline(monkeypatch):
    import geon.algorithms.reconstruction.core as core
    def unexpected(*args):
        raise AssertionError('Spline fit should not run')
    monkeypatch.setattr(core, 'fit_bspline', unexpected)
    points = cloud()
    result = reconstruct(points, np.zeros(len(points), dtype=int), config=config(max_plane_rmse=.5))
    assert result.patches[0].kind == 'plane'


def test_bounded_training_preserves_dense_surface_and_full_validation(monkeypatch):
    import geon.algorithms.reconstruction.surfaces as surfaces
    rng = np.random.default_rng(83)
    xy = rng.uniform([-1., -.7], [1., .7], (24000, 2))
    z = .12*xy[:, 0]**2 + .07*xy[:, 1]**2 + rng.normal(0, .001, len(xy))
    points = np.column_stack((xy, z))
    args = (points, np.zeros(3), np.eye(3)[:, :2], np.array([0., 0., 1.]), config())
    sampled, why = fit_bspline(*args)
    assert sampled is not None, why
    original_sample = surfaces._fit_sample
    monkeypatch.setattr(surfaces, '_fit_sample', lambda strata: original_sample(strata, len(strata)))
    full, why = fit_bspline(*args)
    assert full is not None, why
    assert abs(sampled.rmse-full.rmse) < .0002
    assert np.max(np.linalg.norm(sampled.lift(xy)-full.lift(xy), axis=1)) < .001


def test_fit_sample_bound_and_weights():
    from geon.algorithms.reconstruction.surfaces import _fit_sample
    strata = np.arange(50000) % 128
    rows, weights = _fit_sample(strata)
    assert len(rows) == 128*128
    assert len(np.unique(rows)) == len(rows)
    np.testing.assert_allclose(np.bincount(strata[rows], weights=weights), np.bincount(strata))
    again, _ = _fit_sample(strata)
    np.testing.assert_array_equal(rows, again)


@pytest.mark.parametrize('case', ['noise', 'multilayer', 'rank', 'slope'])
def test_bad_spline_support_is_rejected(case):
    points = cloud()
    if case == 'noise':
        points[:,2] = np.random.default_rng(42).normal(0,.1,len(points))
    elif case == 'multilayer':
        points = np.vstack((points, points+[0,0,.3]))
    elif case == 'rank':
        points[:,1] = 0
    else:
        points[:,2] *= 20
    model, reason = fit_bspline(points, np.zeros(3), np.eye(3)[:,:2], np.array([0,0,1.]), config())
    assert model is None, reason


def model():
    coef = np.array([1.,-1/3,-1/3,1.])
    return BSplineHeight(np.zeros(3), np.array([0.,0,1]), np.eye(3)[:,:2],
                        np.array([-1.,-1,1,1]), .12*coef[:,None]+.07*coef[None,:])


def patch(surface, label=0):
    if surface.kind == 'bspline':
        bounds = surface.bounds
    elif surface.kind == 'cylinder':
        bounds = [0.,-.3,2*np.pi*surface.radius,.3]
    else:
        bounds = [-1.2,-1.2,1.2,1.2]
    uv = np.array([[bounds[0],bounds[1]], [bounds[2],bounds[1]], [bounds[0],bounds[3]], [bounds[2],bounds[3]]])
    return SimpleNamespace(id=label, surface=surface, footprint=box(*bounds), points=surface.lift(uv))


@pytest.mark.parametrize('other_kind', ['plane', 'cylinder', 'bspline'])
def test_spline_intersection_pairs_are_on_both_surfaces(other_kind):
    a = model()
    if other_kind == 'plane':
        b = Plane(np.array([0.,0,.06]), np.array([0.,0,1]), np.eye(3)[:,:2])
    elif other_kind == 'cylinder':
        b = Cylinder(np.zeros(3), np.array([0.,0,1]), np.array([1.,0,0]), np.array([0.,1,0]), .7,0,2*np.pi,True,0)
    else:
        b = model()
        b.heights = b.heights+.1*np.linspace(-1,1,4)[:,None]
    curves = raw_intersection_curves(patch(a), patch(b,1), .025, .001)
    assert curves
    for curve in curves:
        assert np.isfinite(curve).all()
        np.testing.assert_allclose(a.lift(a.project(curve)), curve, atol=1e-5)
        if b.kind == 'bspline':
            np.testing.assert_allclose(b.lift(b.project(curve)), curve, atol=1e-5)
        else:
            assert np.abs(b.residual(curve)).max() < 1e-6


def test_three_surface_corner_and_geometric_projection():
    s = model()
    xplane = Plane(np.zeros(3), np.array([1.,0,0]), np.eye(3)[:,1:])
    yplane = Plane(np.zeros(3), np.array([0.,1,0]), np.array([[1.,0],[0,0],[0,-1]]))
    corners = triple_points([patch(s), patch(xplane,1), patch(yplane,2)], .1)
    assert len(corners) == 1
    np.testing.assert_allclose(corners[0], 0, atol=1e-7)
    points = s.lift(np.array([[.4,.2],[-.2,.3]])) + [0,0,.02]
    projected = closest_points(s, points)
    np.testing.assert_allclose(s.lift(s.project(projected)), projected, atol=1e-7)
    assert np.linalg.norm(projected-points,axis=1).max() <= .020001


def test_tangent_plane_does_not_invent_curve():
    a = model()
    b = Plane(np.zeros(3),np.array([0.,0,1]),np.eye(3)[:,:2])
    curves = raw_intersection_curves(patch(a),patch(b,1),.025)
    assert not curves or all(np.ptp(c,axis=0).max()<1e-4 for c in curves)


def test_tangent_cylinder_plane_and_coincident_surface_diagnostics():
    cylinder = Cylinder(np.zeros(3), np.array([0.,0,1]), np.array([1.,0,0]),
                        np.array([0.,1,0]), .7, 0, 2*np.pi, True, 0)
    plane = Plane(np.array([.7,0,0]), np.array([1.,0,0]), np.eye(3)[:,1:])
    messages = []
    assert not raw_intersection_curves(patch(cylinder), patch(plane,1), .025, diagnostic=messages.append)
    assert any('tangent' in text for text in messages)
    messages.clear()
    spline = model()
    assert not raw_intersection_curves(patch(spline), patch(spline,1), .025, diagnostic=messages.append)
    assert any('coincident' in text or 'overlap' in text for text in messages)


def test_spline_hole_survives():
    points = cloud()
    keep = np.linalg.norm(points[:,:2],axis=1) > .25
    result = reconstruct(points[keep],np.zeros(keep.sum(),dtype=int),config=config(alpha_radius=.08))
    assert result.patches[0].kind == 'bspline'
    assert len(result.patches[0].polygon.interiors) == 1


def test_border_recovery_cannot_extrapolate_past_spline_halo():
    from geon.algorithms.reconstruction.core import _batch_boundary
    surface = model()
    footprint = box(-1,-1,1,1)
    p = Patch(0, None, surface.center, surface.normal, surface.basis, cloud(),
              footprint, footprint, 0., spline=surface)
    def boundary(x):
        xyz = surface.lift(np.array([[x,-1.],[x,1.]]))
        return _batch_boundary(p, [dict(points=xyz, endpoints=xyz, patch_ids=(0,1))],
                               config(edge_tolerance=.2))
    assert boundary(.98) is not None
    assert boundary(1.05) is None


@pytest.mark.parametrize('rotated', [False, True])
def test_long_curved_web_uses_alternate_pca_chart_and_multiple_spans(rotated):
    # The bend varies much more than the web height, so the smallest PCA
    # eigenvector points up rather than across the side of the girder.
    u,v = np.meshgrid(np.linspace(-18,18,181),np.linspace(-.2,.2,13))
    points = np.column_stack((u.ravel(),(30-np.sqrt(30**2-u*u)).ravel(),v.ravel()))
    if rotated:
        points = points@Rotation.from_euler('xyz',[61,17,32],degrees=True).as_matrix().T + [1e6,-2e6,3e6]
    result = reconstruct(points,np.zeros(len(points),dtype=int),config=config(
        max_plane_rmse=.05,max_bspline_rmse=.005,edge_tolerance=.04,alpha_radius=.15,
        curve_tolerance=.003,bridge_remaining_gaps=False))
    assert len(result.faces) > 0, result.diagnostics
    surface = result.patches[0].spline
    assert surface is not None and max(surface.heights.shape) > 4
    assert surface.rmse < .005
    uv_triangles=surface.project(result.vertices[result.faces])
    assert np.max(np.ptp(uv_triangles,axis=1)) <= .5+1e-8
    assert np.bincount(result.faces.ravel()).max() <= 24
    # Native geometry and the Python chart must represent exactly the same
    # multiple-span surface, including the Greville-based linear coordinates.
    on_surface = surface.lift(surface.project(points[::17]))
    np.testing.assert_allclose(closest_points(surface,on_surface),on_surface,atol=1e-7)


def test_multiple_span_spline_plane_intersection():
    s = model()
    # An exact knot insertion preserves this surface while exercising multiple
    # spans and non-uniform Greville abscissae in the native descriptor.
    from scipy.interpolate import BSpline
    spline = BSpline(s.knots_u,s.heights,3).insert_knot(.3).insert_knot(.7)
    s = BSplineHeight(s.center,s.normal,s.basis,s.bounds,spline.c,knots_u=spline.t)
    plane = Plane(np.array([0.,0,.06]),np.array([0.,0,1]),np.eye(3)[:,:2])
    curves = raw_intersection_curves(patch(s),patch(plane,1),.025,.001)
    assert curves
    for curve in curves:
        np.testing.assert_allclose(s.lift(s.project(curve)),curve,atol=1e-6)
        assert np.abs(plane.residual(curve)).max() < 1e-6


def test_supported_seam_splits_at_narrow_deep_scan_notch():
    from shapely.geometry import Polygon
    from geon.algorithms.reconstruction.core import _supported_boundary_pieces, _batch_boundary
    footprint = Polygon([(0,0),(2,0),(2,.6),(2.08,.6),(2.08,0),(4,0),(4,1),(0,1)])
    a = Patch(0,None,np.zeros(3),np.array([0.,0,1]),np.eye(3)[:,:2],
              np.zeros((3,3)),footprint,footprint,0.)
    wall = box(0,0,4,1)
    b = Patch(1,None,np.zeros(3),np.array([0.,-1,0]),np.array([[1.,0],[0,0],[0,1.]]),
              np.zeros((3,3)),wall,wall,0.)
    cfg = config(edge_tolerance=.1,support_step=.02,min_edge_length=.05)
    parts = _supported_boundary_pieces(a,b,np.array([[0.,0,0],[4.,0,0]]),cfg)
    assert len(parts) == 2
    assert parts[0][-1,0] < 2 and parts[1][0,0] > 2.08
    edges=[dict(points=p,endpoints=p[[0,-1]],patch_ids=(0,1)) for p in parts]
    for patch_ in (a,b):
        assert _batch_boundary(patch_,edges,cfg) is not None


def test_grid_mesh_covers_footprint_and_preserves_hole():
    import shapely
    from geon.algorithms.reconstruction.meshing import grid_triangles
    polygon=box(-2,-1,2,1).difference(box(-.3,-.4,.4,.2))
    triangles=shapely.polygons(np.asarray(grid_triangles([polygon],.23)))
    assert abs(shapely.area(triangles).sum()-polygon.area)<1e-10
    assert np.all(shapely.covers(polygon,triangles))


def test_spline_preview_is_trimmed_and_has_control_net():
    import shapely
    from geon.algorithms.reconstruction.preview import spline_preview
    points=cloud()
    keep=np.linalg.norm(points[:,:2],axis=1)>.25
    result=reconstruct(points[keep],np.zeros(keep.sum(),dtype=int),config=config(alpha_radius=.08))
    overlays=spline_preview(result)
    p=result.patches[0]
    overlay=overlays[0]
    assert overlay['control_points'].shape==(p.spline.heights.size,3)
    assert all(len(c)==2 for c in overlay['control_net'])
    assert len(overlay['control_net'])>sum(p.spline.heights.shape)
    for line in overlay['grid']:
        np.testing.assert_allclose(p.lift(p.project(line)),line,atol=1e-9)
        assert np.all(shapely.covers(p.polygon.buffer(1e-8),shapely.points(p.project(line))))


def test_spline_plane_shared_seam_is_conforming():
    x,y=np.meshgrid(np.linspace(-1,1,31), np.linspace(.03,.7,24))
    x,y=x.ravel(),y.ravel()
    curved=np.column_stack((x,y,.12*x*x+.07*y*y))
    x,z=np.meshgrid(np.linspace(-1,1,31),np.linspace(-.6,0,24))
    x,z=x.ravel(),z.ravel()
    wall=np.column_stack((x,np.zeros(len(x)),.12*x*x+z-.03))
    points=np.vstack((curved,wall));labels=np.repeat([0,1],[len(curved),len(wall)])
    result=reconstruct(points,labels,config=config(edge_tolerance=.12,adjacency_radius=.12,alpha_radius=.09))
    assert result.patches[0].kind=='bspline'
    assert result.patches[1].kind=='plane'
    assert result.shared_edges, result.diagnostics
    for edge in result.shared_edges:
        curve=_edge_points(edge)
        indices=[np.argmin(np.linalg.norm(result.vertices-p,axis=1)) for p in curve]
        for label in edge['patch_ids']:
            faces=result.faces[result.face_patch_ids==label]
            for a,b in zip(indices,indices[1:]):
                assert np.any(np.isin(faces,[a,b]).sum(axis=1)==2)
