import numpy as np
import pytest
import vtk
from PyQt6.QtWidgets import QApplication
from vtk.util import numpy_support as ns

from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.scene import Scene
from geon.settings import Preferences, coarse_points_stride
from geon.ui.preferences_dialog import PreferencesDialog


@pytest.mark.parametrize('value,percent,stride', [
    (.01, .01, 10000), (1, 1, 100), (10, 10, 10), (50, 50, 2),
    (0, .01, 10000), (99, 50, 2), ('bad', 1, 100),
    (float('nan'), 1, 100), (float('inf'), 1, 100), (30, 30, 3),
])
def test_percentage_persistence_and_bounds(tmp_path, value, percent, stride):
    prefs = Preferences(path=tmp_path / 'prefs.toml', coarse_points_percent=value)
    prefs.save()
    loaded = Preferences.load(prefs.path)
    assert loaded.coarse_points_percent == percent
    assert coarse_points_stride(loaded.coarse_points_percent) == stride


def test_legacy_and_invalid_preference_file(tmp_path):
    path = tmp_path / 'prefs.toml'
    for text, expected in [('', 1), ('coarse_points_percent = -10', .01),
                           ('coarse_points_percent = inf', 1),
                           ('coarse_points_percent = "bad"', 1)]:
        path.write_text(text)
        assert Preferences.load(path).coarse_points_percent == expected


@pytest.fixture(scope='session')
def coarse_preferences_app():
    return QApplication.instance() or QApplication([])


def test_percentage_widget_applies_only_on_accept(tmp_path, coarse_preferences_app):
    prefs = Preferences(path=tmp_path / 'prefs.toml')
    dialog = PreferencesDialog(prefs)
    spin = dialog.coarse_points_input
    assert spin.minimum() == .01
    assert spin.maximum() == 50
    assert spin.suffix().strip() == '%'
    spin.setValue(12.5)
    dialog.reject()
    assert prefs.coarse_points_percent == 1
    dialog.apply()
    assert prefs.coarse_points_percent == 12.5
    dialog.close()


def test_percentage_rebuilds_existing_and_new_clouds_with_masks_and_colors(coarse_preferences_app):
    count = 201
    points = np.column_stack((np.arange(count), np.zeros(count), np.zeros(count))).astype(np.float32)
    data = PointCloudData(points)
    colors = np.column_stack((np.arange(count), np.zeros(count), np.zeros(count))).astype(np.uint8)
    data.add_field('color', colors, FieldType.COLOR)
    data.add_field('background', colors[:, ::-1].copy(), FieldType.COLOR)
    scene = Scene(vtk.vtkRenderer(), coarse_points_percent=10)
    layer = scene.add_data(data)
    layer.update()
    layer.set_background_field_name('background')
    layer.set_foreground_mix(.3)
    mask = np.arange(count) % 3 == 0
    layer.set_visibility_mask(mask)
    layer.active_selection = np.array([0, 10, 100], dtype=np.int32)
    plane = vtk.vtkPlane()
    layer.set_clipping_planes([plane])
    actor = layer._main_actor
    try:
        for percentage in [50, .01, 10, 1]:
            scene.set_coarse_points_percent(percentage)
            stride = coarse_points_stride(percentage)
            poly = layer._poly_coarse
            assert layer._main_actor is actor
            assert scene.layer_for_prop(actor) is layer
            assert layer._mapper_coarse.GetNumberOfClippingPlanes() == 1
            assert layer.foreground_mix == .3
            assert layer._poly.GetNumberOfPoints() == count
            np.testing.assert_array_equal(ns.vtk_to_numpy(poly.GetPoints().GetData()), points[::stride])
            np.testing.assert_array_equal(ns.vtk_to_numpy(poly.GetPointData().GetArray('geonVisible')), mask[::stride])
            selected = np.zeros(count, dtype=np.uint8)
            selected[[0, 10, 100]] = 1
            np.testing.assert_array_equal(ns.vtk_to_numpy(poly.GetPointData().GetArray('geonSelected')), selected[::stride])
            np.testing.assert_allclose(ns.vtk_to_numpy(poly.GetPointData().GetArray('geonForegroundInput')), colors[::stride], atol=1e-6)
            np.testing.assert_allclose(ns.vtk_to_numpy(poly.GetPointData().GetArray('geonBackgroundInput')), colors[::stride, ::-1], atol=1e-6)
            assert poly.GetNumberOfVerts() == len(points[::stride])
            actor.SetMapper(layer._mapper_coarse)
            if poly.GetNumberOfPoints() > 1:
                assert layer.data_index_from_picked_id(1) == stride
                assert layer.world_xyz_from_picked_id(1) == tuple(points[stride])
        scene.set_coarse_points_percent(25)
        other = scene.add_data(PointCloudData(points.copy()))
        assert other._coarse_stride == 4
        assert other._poly_coarse.GetNumberOfPoints() == 51
        assert layer._coarse_stride == 4
    finally:
        scene.clear(delete_data=False)
