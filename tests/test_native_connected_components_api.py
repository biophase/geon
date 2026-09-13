import numpy as np
import pytest

from geon._native import region_growing
from geon.algorithms.connected_components import segment_connected_components


def test_connected_components_labels_separated_clusters_deterministically():
    coords = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [1.00, 0.00, 0.00],
            [1.01, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    labels, stats = region_growing.connected_components(coords, epsilon=0.05)

    np.testing.assert_array_equal(labels, np.asarray([0, 0, 1, 1], dtype=np.int32))
    assert stats["num_points"] == 4
    assert stats["num_components"] == 2


def test_connected_components_empty_cloud():
    labels, stats = segment_connected_components(
        np.empty((0, 3), dtype=np.float32), epsilon=0.05
    )
    assert labels.shape == (0,)
    assert stats["num_components"] == 0


def test_connected_components_handles_zero_extent_axes():
    coords = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
        dtype=np.float32,
    )
    labels, stats = segment_connected_components(coords, epsilon=0.05)

    np.testing.assert_array_equal(labels, np.zeros(3, dtype=np.int32))
    assert stats["num_components"] == 1


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.nan])
def test_connected_components_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        segment_connected_components(np.zeros((1, 3), dtype=np.float32), epsilon=epsilon)


def test_connected_components_rejects_invalid_coords():
    with pytest.raises(ValueError):
        segment_connected_components(np.zeros((3, 2), dtype=np.float32), epsilon=0.1)
