from copy import deepcopy
import numpy as np
import shapely
import pytest
from shapely.geometry import Polygon, box

from geon.algorithms.reconstruction.core import Config, Patch
from geon.algorithms.reconstruction.clean_joints import (
    clean_footprints, planar_border_proposals, coordinate_endpoints, reconstruct_joints,
)


def patch(label,polygon,height=0.):
    uv=np.asarray(polygon.exterior.coords)
    xyz=np.column_stack((uv,np.full(len(uv),height)))
    return Patch(label,None,np.array([0.,0.,height]),np.array([0.,0.,1.]),
                 np.array([[1.,0.],[0.,1.],[0.,0.]]),xyz,polygon,polygon,0.,source_point_count=100)


def test_clean_preserves_holes_components_and_deviation():
    polygon=Polygon([(0,0),(.5,.01),(1,0),(1,1),(.5,.99),(0,1)],
                    holes=[[(.3,.3),(.7,.3),(.7,.7),(.3,.7)]])
    p=patch(1,polygon);clean_footprints({1:p},Config(simplify_tolerance=.025),lambda _:None)
    assert p.polygon.is_valid
    assert len(p.polygon.interiors)==1
    assert shapely.hausdorff_distance(polygon.boundary,p.polygon.boundary,densify=.25)<=.025


def test_complete_planar_chain_instead_of_sample_fragments():
    a=patch(1,box(0,0,1,10));b=patch(2,box(1.03,0,2,10))
    proposals=planar_border_proposals(a,b,Config())
    assert len(proposals)==1
    assert len(proposals[0]['points'])==2
    assert np.isclose(np.linalg.norm(np.diff(proposals[0]['points'],axis=0)),10)
    assert proposals[0]['kind']=='bridge'


def test_duplicate_overlapping_faces_are_not_joined():
    a=patch(1,box(0,0,1,10));b=patch(2,box(0,0,1,10))
    assert planar_border_proposals(a,b,Config())==[]


def test_clean_planar_transaction_preserves_both_faces():
    a=patch(1,box(0,0,1,10));b=patch(2,box(1.03,0,2,10))
    shared,bridges=reconstruct_joints({1:a,2:b},[(1,2)],Config(),[],lambda _:None)
    assert len(bridges)==1 and not shared
    assert a.polygon.is_valid and b.polygon.is_valid
    assert len(a.locked)==len(b.locked)==1


def test_boundary_interval_cannot_receive_two_transition_strips():
    a=patch(1,box(0,0,1,10))
    b=patch(2,box(1.03,0,2,10))
    c=patch(3,box(1.04,0,2,10))
    shared,bridges=reconstruct_joints({1:a,2:b,3:c},[(1,2),(1,3)],
                                      Config(),[],lambda _:None)
    assert not shared and len(bridges)==1
    assert len(a.locked)==1


@pytest.mark.parametrize('simplification', [0., .1])
def test_endpoint_groups_do_not_chain_or_move_off_analytic_planes(simplification):
    a=patch(1,box(-1,-1,2,2));b=patch(2,box(-1,-1,2,2))
    # Provide dense support so this tests grouping, not observed support refusal.
    a.points=b.points=np.array([[0.,0.,0.],[.07,0.,0.],[.14,0.,0.]])
    # Boundary support near y=-1; all targets remain within the corner radius.
    a.footprint=b.footprint=box(-1,0,2,2)
    curves=[np.array([[x,0.,0.],[x,1.,0.]]) for x in (0.,.07,.14)]
    proposals=[dict(patch_ids=(1,2),points=c.copy(),endpoints=c.copy()) for c in curves]
    coordinate_endpoints(proposals,{1:a,2:b},Config(simplify_tolerance=simplification,edge_tolerance=.1))
    np.testing.assert_allclose(proposals[0]['points'][0],proposals[1]['points'][0])
    assert np.linalg.norm(proposals[0]['points'][0]-curves[0][0])<=.1
    assert not np.allclose(proposals[0]['points'][0],proposals[2]['points'][0])
    assert all(np.all(e['points'][:,2]==0) for e in proposals)


def test_analytic_joint_does_not_extend_past_clean_corners():
    from geon.algorithms.reconstruction.clean_joints import _clip_planar_joint
    a=patch(1,box(0,0,1,1));b=patch(2,box(1,0,2,1))
    curve=np.array([[1.,-.2,0.],[1.,1.2,0.]])
    result=_clip_planar_joint(curve,a,b,Config(edge_tolerance=.25),{})
    assert len(result)==1
    np.testing.assert_allclose(result[0],[[1,0,0],[1,1,0]],atol=1e-12)


def test_crossing_plane_bridge_has_a_zero_width_junction_node():
    a=patch(1,box(0,0,1,10));b=patch(2,box(1.03,0,2,10),height=-.05)
    b.normal=np.array([0.,-.01,1.]);b.normal/=np.linalg.norm(b.normal)
    v=np.array([0.,1.,.01]);v/=np.linalg.norm(v)
    b.basis=np.column_stack(([1.,0.,0.],v))
    b.points=b.lift(np.asarray(b.polygon.exterior.coords))
    proposals=planar_border_proposals(a,b,Config(coplanar_plane_tolerance=.2))
    assert len(proposals)==1
    assert len(proposals[0]['points'])==3
    np.testing.assert_allclose(proposals[0]['side_points'][1][1],proposals[0]['side_points'][2][1],atol=1e-9)


def test_joint_budget_uses_original_border_not_already_moved_border():
    from geon.algorithms.reconstruction.clean_joints import _observed_curve_bound
    p=patch(1,box(-.04,0,1,1));p._clean_observed_footprint=box(0,0,1,1)
    cfg=Config(edge_tolerance=.05,support_step=.01)
    assert _observed_curve_bound(p,np.array([[-.04,.2],[-.04,.8]]),cfg,{})
    assert not _observed_curve_bound(p,np.array([[-.08,.2],[-.08,.8]]),cfg,{})


def test_observed_border_fallback_retains_long_tilted_plane_join():
    # A long, narrow transition from the bridge dataset: projecting both
    # observed borders onto their average line folds the strip near an end.
    # The observed sides still form a supported, non-crossing transition.
    specs = [
        ([1.6127901943, 5.3329049955, 1.1234362174],
         [[.721054835648, -.692660714832], [.692848613781, .721028777521],
          [.006381408158, -.018402068214]],
         [[-.497457553895, -15.07222935], [-.48811633995, 12.037346084],
          [.488768644728, 12.1143274], [.521104090678, -14.335768006],
          [.367456588046, -15.008722093]]),
        ([.9878935074, 4.5488004796, 1.0877899939],
         [[.720422126159, -.690811853976], [.690226561592, .722807892036],
          [.067670184084, -.018103414532]],
         [[-.531421305964, -13.369498935], [-.502230846873, 12.006962954],
          [.473867988034, 12.171830014], [.49858717856, -14.866405603],
          [-.393113662363, -15.056784919]]),
    ]
    patches = {}
    for label, (center, basis, ring) in enumerate(specs, 1):
        p = patch(label, Polygon(ring))
        p.center, p.basis = np.array(center), np.array(basis)
        p.normal = np.cross(p.basis[:, 0], p.basis[:, 1])
        p.normal /= np.linalg.norm(p.normal)
        p.points = p.lift(np.asarray(p.polygon.exterior.coords))
        patches[label] = p
    cfg = Config(simplify_tolerance=.12, edge_tolerance=.25,
                 coplanar_boundary_tolerance=.3, coplanar_plane_tolerance=.3)
    shared, bridges = reconstruct_joints(patches, [(1, 2)], cfg, [], lambda _: None)
    assert not shared and len(bridges) == 1
    edge = bridges[0]
    assert np.linalg.norm(np.diff(edge['points'], axis=0), axis=1).sum() > 26
    for label in patches:
        np.testing.assert_allclose(edge['side_points'][label], edge['observed_side_points'][label])
        assert patches[label].polygon.is_valid
