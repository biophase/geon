from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import h5py
import numpy as np
import pytest
import vtk
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QComboBox, QMenu
from PyQt6.QtCore import QThreadPool, Qt
from PyQt6.QtCore import QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QContextMenuEvent
from geon.ui.gradient_editor import GradientEditor
from geon.data.definitions import (ColorMap, ChannelMap, VectorColorMap, vector_preset,
                                  sample_map, resolve_limits)
from geon.data.pointcloud import PointCloudData, FieldBase, FieldType
from geon.data.document import Document
from geon.io.dataset import Dataset, DocumentReference
from geon.io.color_maps import scan_maps, save_map, resolve_conflicts
from geon.rendering.pointcloud import PointCloudLayer
from geon.rendering.color_mapping import default_map
from vtk.util import numpy_support as ns
from geon.ui.color_histogram import calculate_histogram
from geon.ui.color_channels import ColorChannels


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_relative_limits_and_validation():
    mapping = ColorMap.get_cmap('gray')
    mapping.color_definitions = np.array([[0., 0., 0.], [1., 1., 1.]])
    assert resolve_limits(mapping, (10, 100)) == (10, 100)
    mapping.min = 20
    assert resolve_limits(mapping, (10, 100)) == (20, 100)
    np.testing.assert_allclose(sample_map(mapping, [0, .5, 1]), [[0]*3, [.5]*3, [1]*3])
    mapping.color_positions = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        mapping.validate()


def test_gradient_stop_add_delete_and_endpoints(app):
    mapping = ColorMap('test')
    editor = GradientEditor(mapping)
    editor.resize(300, 60)
    editor.show()
    app.processEvents()
    QTest.mouseDClick(editor, Qt.MouseButton.LeftButton, pos=QPoint(150, 20))
    assert len(mapping.color_positions) == 3
    np.testing.assert_allclose(mapping.color_definitions[1], [.5,.5,0])
    QTest.keyClick(editor, Qt.Key.Key_Delete)
    assert len(mapping.color_positions) == 2
    editor.select(0)
    editor.remove_stop()
    editor.select(1)
    editor.remove_stop()
    np.testing.assert_equal(mapping.color_positions, [0,1])
    editor.close()


def test_gradient_stop_context_menu(app, monkeypatch):
    mapping = ChannelMap(np.array([0., .5, 1.]), np.array([0., .5, 1.]))
    editor = GradientEditor(mapping, 0)
    editor.resize(300, 46)
    def execute(menu, *_):
        assert [a.text() for a in menu.actions()] == ['Edit value', 'Remove']
        assert menu.actions()[1].isEnabled()
        menu.actions()[1].trigger()
    monkeypatch.setattr(QMenu, 'exec', execute)
    editor.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(150,40)))
    np.testing.assert_equal(mapping.color_positions, [0,1])
    def endpoint(menu, *_):
        assert menu.actions()[0].isEnabled()
        assert not menu.actions()[1].isEnabled()
    monkeypatch.setattr(QMenu, 'exec', endpoint)
    editor.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(12,40)))
    editor.close()


def test_legacy_and_vector_serialization(tmp_path):
    with h5py.File(tmp_path/'maps.h5', 'w') as f:
        group = f.create_group('field')
        old = group.create_dataset('ColorMap', data=[[10,0,0,0],[100,1,1,1]])
        old.attrs['name'] = '温度'
        mapping = ColorMap.load_h5py(old)
        assert (mapping.min, mapping.max) == (10, 100)
        np.testing.assert_array_equal(mapping.color_positions, [0,1])
        mapping.save_h5py(group)
        assert 'ColorMap' not in group
        vector = vector_preset(-1, 1)
        vector.green.min = None
        vector.save_h5py(group)
        assert ColorMap.load_h5py(group['color_map']).signature() == vector.signature()
        mapping.save_h5py(group)
        assert ColorMap.load_h5py(group['color_map']).signature() == mapping.signature()


def test_field_component_bindings_round_trip(tmp_path):
    field = FieldBase('vector', np.ones((3,4)), FieldType.VECTOR, vector_preset(0,1), 3, (2,0,1))
    with h5py.File(tmp_path/'bindings.h5', 'w') as handle:
        group = field.save_hdf5(handle)
        loaded = FieldBase.from_hdf5_fieldgroup(group)
        assert loaded.scalar_component == 3
        assert loaded.rgb_components == (2,0,1)
        assert loaded.color_map.signature() == field.color_map.signature()


def test_histogram_visible_shared_bins_and_nonfinite():
    values = [np.array([0., 1., 2., np.nan]), np.array([2., 1., 0., np.inf])]
    edges, counts = calculate_histogram(values, np.array([True, False, True, True]))
    assert edges[0] == 0 and edges[-1] == 2
    assert all(c.sum() == 2 for c in counts)
    assert counts[0][0] == counts[1][0] == 1
    labels, counts = calculate_histogram([np.array([-1, 3, 3, 9])], categorical=True)
    np.testing.assert_equal(labels, [-1,3,9])
    np.testing.assert_equal(counts[0], [1,2,1])


def test_metadata_only_index_save_and_conflicts(tmp_path, monkeypatch):
    dataset = Dataset()
    for index in range(2):
        cloud = PointCloudData(np.zeros((4,3), dtype=np.float32))
        cloud.add_field('s', np.arange(4.).reshape(-1,1), FieldType.SCALAR)
        cloud.get_fields('s')[0].color_map = ColorMap.get_cmap('gray')
        doc = Document(f'doc{index}')
        doc.add_data(cloud)
        path = tmp_path/f'{index}.h5'
        doc.save_hdf5(path)
        dataset._doc_refs.append(DocumentReference(doc.name, str(path)))
        if index == 0:
            active, field = cloud, cloud.get_fields('s')[0]
            dataset._loaded_docs[doc.name] = doc
    getitem = h5py.Dataset.__getitem__
    def no_large_reads(self, key):
        assert not self.name.endswith(('/points', '/data'))
        return getitem(self, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', no_large_reads)
    original = deepcopy(field.color_map)
    changed = deepcopy(original)
    changed.max = 50
    assert len(scan_maps(dataset)) == 2
    assert len(save_map(dataset, original, changed, active, field)) == 2
    assert all(r.mapping.max == 50 for r in scan_maps(dataset))
    field.color_map.max = 100
    reports = resolve_conflicts(dataset)
    assert len(reports) == 1
    assert len({r.mapping.name for r in scan_maps(dataset)}) == 2


def make_layer():
    data = PointCloudData(np.zeros((4,3), dtype=np.float32))
    data.add_field('v', np.array([[0.,0.,1.], [1.,0.,0.], [.5,.5,.5], [0.,1.,0.]]), FieldType.NORMAL)
    layer = PointCloudLayer(data)
    layer.attach(vtk.vtkRenderer())
    layer.update()
    return layer


def test_gpu_preview_keeps_input_and_geometry_buffers(app):
    layer = make_layer()
    field = layer.active_field
    assert isinstance(default_map(field), VectorColorMap)
    arrays = [layer._poly.GetPoints(), layer._poly.GetVerts(),
              layer._poly.GetPointData().GetArray('geonForegroundInput')]
    before = [a.GetMTime() for a in arrays]
    mapping = vector_preset(-1,1)
    mapping.red.output_values[:] = [.3,.7]
    layer.preview_color_map(field, mapping)
    assert [a.GetMTime() for a in arrays] == before
    assert field.color_map is None
    layer.detach()


def test_panel_draft_revert_and_hidden_histogram(app):
    window = QMainWindow()
    window.dataset_manager = SimpleNamespace(_dataset=None)
    window.viewer = SimpleNamespace(rerender=Mock())
    panel = ColorChannels(window)
    layer = make_layer()
    panel.activate(layer)
    assert not panel.timer.isActive()
    panel.draft.red.output_values[:] = [.2,.8]
    panel.changed()
    assert layer.active_field.color_map is None
    assert panel.dirty
    panel.revert()
    assert not panel.dirty
    np.testing.assert_equal(panel.draft.red.output_values, [0,1])
    panel.histogram_ready(panel.generation-1, np.arange(3), [np.ones(2)])
    assert panel.hist_data is None
    panel.activate(None)
    layer.detach()
    window.close()


def test_unchanged_controls_do_not_refresh_or_prompt(app, monkeypatch):
    window = QMainWindow()
    window.dataset_manager = SimpleNamespace(_dataset=None)
    window.viewer = SimpleNamespace(rerender=Mock())
    panel = ColorChannels(window)
    layer = make_layer()
    panel.activate(layer)
    generation = panel.generation
    combo = panel.controls.findChildren(QComboBox)[0]
    combo.lineEdit().editingFinished.emit()  # Losing focus to the dropdown.
    combo.activated.emit(combo.currentIndex())
    panel.limits_changed()
    panel.choose_map(0)
    panel.changed()
    assert panel.generation == generation
    assert not panel.dirty
    window.viewer.rerender.assert_not_called()
    panel.bind(0, 1)
    assert panel.dirty
    panel.bind(0, 0)
    assert not panel.dirty
    panel.draft.red.output_values[:] = [.2,.8]
    panel.changed()
    assert panel.dirty
    panel.draft.red.output_values[:] = [0,1]
    panel.changed()
    assert not panel.dirty
    prompt = Mock(side_effect=AssertionError('Unexpected save prompt'))
    monkeypatch.setattr(QMessageBox, 'question', prompt)
    assert panel.allow_switch()
    assert panel.font().bold() and not panel.body.font().bold()
    assert panel.plot.maximumHeight() == 260
    panel.activate(None)
    layer.detach()
    window.close()


def test_histogram_view_reset_and_axis_controls(app, monkeypatch):
    window = QMainWindow()
    window.dataset_manager = SimpleNamespace(_dataset=None)
    window.viewer = SimpleNamespace(rerender=Mock())
    panel = ColorChannels(window)
    layer = make_layer()
    panel.activate(layer)
    monkeypatch.setattr(panel, 'isVisible', lambda: True)
    auto_range = Mock()
    monkeypatch.setattr(panel.plot, 'autoRange', auto_range)
    layer.set_visibility_mask(np.array([True,False,True,False]))
    panel.histogram_ready(panel.generation, np.linspace(0,1,3), [np.ones(2)]*3)
    auto_range.assert_called_once()
    panel.draft.red.output_values[:] = [.2,.8]
    panel.changed()
    auto_range.assert_called_once()  # Recoloring should preserve manual zoom.
    for axis_name in ('bottom', 'left'):
        panel.axis_actions[axis_name, 'numbers'].setChecked(False)
        assert not panel.plot.getAxis(axis_name).style['showValues']
        panel.axis_actions[axis_name, 'label'].setChecked(False)
        assert not panel.plot.getAxis(axis_name).label.isVisible()
    panel.revert()
    panel.activate(None)
    panel.timer.stop()
    layer.detach()
    window.close()


def test_dirty_switch_cancel_revert_and_save_failure(app, monkeypatch):
    window = QMainWindow()
    window.dataset_manager = SimpleNamespace(_dataset=None)
    window.viewer = SimpleNamespace(rerender=Mock())
    panel = ColorChannels(window)
    layer = make_layer()
    panel.activate(layer)
    panel.draft.red.output_values[:] = [.2,.8]
    panel.changed()
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Cancel)
    assert not panel.activate(None)
    assert panel.layer is layer and panel.dirty
    window.dataset_manager._dataset = object()
    monkeypatch.setattr('geon.ui.color_channels.save_map', Mock(side_effect=OSError('read-only')))
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[-1]))
    assert not panel.save()
    assert panel.dirty and errors == ['read-only']
    assert layer.active_field.color_map is None
    window.dataset_manager._dataset = None
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Discard)
    assert panel.activate(None)
    assert layer.active_field.color_map is None
    layer.detach()
    window.close()


def test_panel_visible_histogram_and_save(app):
    window = QMainWindow()
    window.dataset_manager = SimpleNamespace(_dataset=None)
    window.viewer = SimpleNamespace(rerender=Mock())
    panel = ColorChannels(window)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, panel)
    layer = make_layer()
    panel.activate(layer)
    window.show()
    panel.show()
    app.processEvents()
    panel.refresh_histogram()
    QThreadPool.globalInstance().waitForDone()
    app.processEvents()
    assert panel.hist_data is not None
    assert all(c.sum() == 4 for c in panel.hist_data[1])
    panel.draft.red.output_values[:] = [.2,.8]
    panel.changed()
    assert panel.save()
    assert isinstance(layer.active_field.color_map, VectorColorMap)
    np.testing.assert_allclose(layer.active_field.color_map.red.output_values, [.2,.8])
    panel.hide()
    panel.timer.stop()
    panel.activate(None)
    layer.detach()
    window.close()


@pytest.mark.parametrize('kind, values, expected', [
    (FieldType.NORMAL, [0., 0., 1.], [128, 128, 255]),
    (FieldType.COLOR, [10., 20., 30.], [10, 20, 30]),
    (FieldType.COLOR, [.2, .4, .6], [51, 102, 153]),
    (FieldType.NORMAL, [np.nan, 0., 1.], [153, 153, 153]),
])
def test_default_gpu_colors(app, kind, values, expected):
    data = PointCloudData(np.zeros((1,3), dtype=np.float32))
    data.add_field('field', np.array([values]), kind)
    layer = PointCloudLayer(data)
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(60,60)
    window.AddRenderer(renderer)
    layer.attach(renderer)
    layer.update()
    layer.set_point_size(15)
    renderer.ResetCamera()
    try:
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.ReadFrontBufferOff()
        capture.Update()
        pixel = ns.vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(60,60,-1)[30,30,:3]
        np.testing.assert_allclose(pixel, expected, atol=1)
        if np.isfinite(values).all():
            draft = deepcopy(default_map(layer.active_field))
            draft.red.output_values[:] = 0
            layer.preview_color_map(layer.active_field, draft)
            window.Render()
            capture.Modified()
            capture.Update()
            pixel = ns.vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(60,60,-1)[30,30,:3]
            np.testing.assert_allclose(pixel, [0, *expected[1:]], atol=1)
    finally:
        layer.detach()
        window.Finalize()
