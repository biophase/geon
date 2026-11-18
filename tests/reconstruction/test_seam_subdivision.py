import numpy as np
from shapely.geometry import Polygon
from types import SimpleNamespace

from geon.algorithms.reconstruction.core import _has_curves, _synchronize_existing_seam_nodes
from geon.algorithms.reconstruction.meshing import insert_boundary_nodes


def test_existing_seam_can_be_subdivided_but_not_moved_or_gapped():
    seam=np.array([[0.,0.],[2.,0.]])
    polygon=Polygon([(0,0),(1,0),(2,0),(2,1),(0,1)])
    assert _has_curves(polygon,[seam])
    assert not _has_curves(Polygon([(0,0),(1,.01),(2,0),(2,1),(0,1)]),[seam])


def test_junction_node_is_shared_by_both_seam_sides():
    patches={}
    for label,ring in [(1,[(0,0),(1,0),(2,0),(2,1),(0,1)]),
                       (2,[(0,0),(2,0),(2,-1),(0,-1)])]:
        patches[label]=SimpleNamespace(id=label,kind='plane',polygon=Polygon(ring),
            cylinder=None,project=lambda points:np.asarray(points)[...,:2])
    points=np.array([[0.,0.,0.],[2.,0.,0.]])
    edge=dict(patch_ids=(1,2),points=points,endpoints=points)
    _synchronize_existing_seam_nodes([edge],patches)
    np.testing.assert_allclose(edge['points'],[[0,0,0],[1,0,0],[2,0,0]])
    for label,patch in patches.items():
        geometry=insert_boundary_nodes(patch.polygon,[edge['mesh_uv'][label]])
        assert any(np.allclose(p,[1,0]) for p in geometry.exterior.coords)
