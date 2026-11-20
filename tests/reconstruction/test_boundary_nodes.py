import numpy as np
import pytest
from shapely.geometry import Polygon

from geon.algorithms.reconstruction.meshing import insert_boundary_nodes


@pytest.mark.parametrize('gap', [5e-9, 5e-16])
def test_existing_vertex_is_not_inserted_into_nearby_hairpin_edge(gap):
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, -1),
                       (gap, .5), (gap, -.5)])
    assert polygon.is_valid
    result = insert_boundary_nodes(polygon, [np.array([[0., 0.], [0., .25]])])
    assert result.is_valid
    assert result.equals(polygon)
    assert list(result.exterior.coords).count((0., 0.)) == 2
    assert (0., .25) in result.exterior.coords


def test_actual_shared_vertex_on_adjacent_edges_remains_unchanged():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = insert_boundary_nodes(polygon, [np.array([[1., 0.], [1., .5]])])
    assert result.is_valid and result.equals(polygon)
    assert (1., .5) in result.exterior.coords
