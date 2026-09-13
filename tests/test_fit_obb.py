import numpy as np
import pytest

from geon.data.boundingbox import BoundingBox
from geon.algorithms.fit_obb import fit_obb


def _box_points(dimensions=(6.0, 3.0, 1.0), yaw=0.4):
    local = np.asarray(
        [[x, y, z] for x in (-3.0, 3.0) for y in (-1.5, 1.5) for z in (0.0, 1.0)],
        dtype=np.float64,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    return local @ rotation.T + np.asarray((4.0, -2.0, 5.0))


def test_fit_obb_pca_recovers_rotated_box():
    result = fit_obb(_box_points(), "pca", trim=0.0)

    assert isinstance(result, BoundingBox)
    np.testing.assert_allclose(result.dimensions, (6.0, 3.0, 1.0), atol=1e-8)
    np.testing.assert_allclose(np.sort(result.corners(), axis=0).mean(axis=0), (4, -2, 5.5))
    assert result.attributes == {"fit_method": "pca", "trim": 0.0}


def test_fit_obb_z_locked_keeps_vertical_axis():
    result = fit_obb(_box_points(), "pca_z_locked", trim=0.0)

    np.testing.assert_allclose(result.axes()[2], (0.0, 0.0, 1.0), atol=1e-10)
    assert result.center_bottom_xyz[2] == pytest.approx(5.0)
    assert result.height == pytest.approx(1.0)


def test_fit_obb_trims_projection_outlier():
    core = np.column_stack((np.linspace(-1.0, 1.0, 100), np.zeros(100), np.zeros(100)))
    points = np.vstack((core, (1000.0, 0.0, 0.0)))

    untrimmed = fit_obb(points, "pca_z_locked", trim=0.0)
    trimmed = fit_obb(points, "pca_z_locked", trim=0.01)

    assert untrimmed.width > 900.0
    assert trimmed.width < 3.0


@pytest.mark.parametrize(
    ("points", "method", "trim"),
    [
        (np.empty((0, 3)), "pca", 0.01),
        (np.zeros((3, 2)), "pca", 0.01),
        (np.asarray([[np.nan, 0.0, 0.0]]), "pca", 0.01),
        (np.zeros((1, 3)), "unknown", 0.01),
        (np.zeros((1, 3)), "pca", 0.5),
    ],
)
def test_fit_obb_rejects_invalid_input(points, method, trim):
    with pytest.raises(ValueError):
        fit_obb(points, method, trim)
