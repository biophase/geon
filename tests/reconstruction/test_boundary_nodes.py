import numpy as np
import pytest
import shapely
from scipy.spatial import Delaunay
from geon.algorithms.reconstruction.core import alpha_footprint
from geon.algorithms.reconstruction.meshing import insert_boundary_nodes
from geon.algorithms.reconstruction.timing import ProgressTimer

@pytest.mark.parametrize('seed',range(4))
def test_alpha_matches_retained_triangle_union_with_holes_and_islands(seed):
    rng=np.random.default_rng(seed)
    p=rng.uniform(-2,2,(1200,2))
    r=np.linalg.norm(p,axis=1)
    p=p[((r>.65)&(r<1.7)) | (r<.2)]
    p=np.vstack((p, rng.uniform([3,0],[3.6,.6],(90,2))))
    radius=.15
    q=np.unique(p,axis=0);tri=q[Delaunay(q).simplices]
    a,b=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    twice_area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    radii=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*np.linalg.norm(a-b,axis=1)/(2*twice_area)
    reference=shapely.union_all(shapely.polygons(tri[radii<=radius]))
    actual=alpha_footprint(p,radius)
    assert actual.symmetric_difference(reference).area<1e-10
    assert len(shapely.get_parts(actual))==len(shapely.get_parts(reference))


def test_insert_boundary_nodes_preserves_holes_components_and_adjacent_seams():
    outer=shapely.box(0,0,4,2).difference(shapely.box(1,.5,2,1.5))
    polygon=shapely.MultiPolygon([outer,shapely.box(6,0,7,1)])
    curves=[np.array([[0,0],[.5,0],[2,0]]),np.array([[2,0],[3,0],[4,0]]),
            np.array([[1,.5],[1,.7],[1,1.5]]),np.array([[6,0],[6.3,0],[7,0]])]
    result=insert_boundary_nodes(polygon,curves+curves[:1])
    assert result.equals(polygon)
    assert len(result.geoms)==2 and len(result.geoms[0].interiors)==1
    coordinates=shapely.get_coordinates(result)
    for point in np.concatenate(curves):
        assert np.min(np.linalg.norm(coordinates-point,axis=1))<1e-12
    with pytest.raises(ValueError,match='not on the accepted'):
        insert_boundary_nodes(polygon,[np.array([[.5,.02],[2,.02]])])


def test_progress_timer_reports_elapsed_delta_and_preserves_worker_timestamps():
    ticks=iter([10.,11.,13.5,20.])
    timer=ProgressTimer(clock=lambda:next(ticks))
    assert timer.format('Fitting')=='[elapsed 1.000s; delta 1.000s] Fitting'
    entry=timer.format('Meshing')
    assert entry=='[elapsed 3.500s; delta 2.500s] Meshing'
    assert timer.format(entry)==entry
    assert timer.message(entry)=='Meshing'
