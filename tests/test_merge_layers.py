from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import vtk
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QInputDialog, QMenu

from geon.data.pointcloud import PointCloudData, FieldType, SemanticSchema, SemanticClass
from geon.data.mesh import MeshData, MeshLevel
from geon.data.boundingbox import BoundingBox, BoundingBoxData
from geon.data.cellcomplex import CellComplexData, VertexCell
from geon.data.camera import CameraData
from geon.data.document import Document
from geon.rendering.pointcloud import PointCloudLayer
from geon.rendering.mesh import MeshLayer
from geon.rendering.scene import Scene
from geon.tools.command_manager import CommandManager
from geon.tools.merge_layers import MergeLayersCommand


@pytest.fixture(scope='module', autouse=True)
def application():
    # Rendering helpers create timers even in the non-dialog merge tests.
    app = QApplication.instance() or QApplication([])
    yield app


def cloud(x=0):
    return PointCloudData(np.array([[x, 0., 0.], [x+1, 0., 0.]]))


def test_point_fields_union_and_independence():
    a, b = cloud(), cloud(4)
    for obj in (a, b):
        obj.add_field('ids', np.array([5, 9]), FieldType.INSTANCE)
    a.add_field('value', np.array([[1], [2]]), FieldType.SCALAR)
    b.add_field('value', np.ones((2, 3)), FieldType.VECTOR)
    merged = PointCloudData.merge([a, b], name='Together')
    assert merged.name == 'Together'
    np.testing.assert_array_equal(merged.points, np.concatenate([a.points, b.points]))
    fields = {f.name: f for f in merged.get_fields()}
    np.testing.assert_array_equal(fields['ids'].data.ravel(), [0, 1, 2, 3])
    assert np.isnan(fields['value'].data[2:]).all()
    assert np.isnan(fields['value_001'].data[:2]).all()
    merged.points[:] = 99
    fields['ids'].data[:] = 99
    assert a.points[0, 0] == 0
    assert a.get_fields(names=['ids'])[0].data.ravel()[0] == 5


def test_semantic_schema_alignment():
    a, b = cloud(), cloud()
    for obj, label in ((a, 'wall'), (b, 'roof')):
        schema = SemanticSchema('classes', [SemanticClass(1, label, (1, 2, 3))])
        obj.add_field('classes', np.array([1, 1]), FieldType.SEMANTIC, schema=schema)
    field = PointCloudData.merge([a, b]).get_fields()[0]
    assert field.data[0] != field.data[2]
    assert {c.name for c in field.schema.semantic_classes} == {'wall', 'roof'}


def test_instance_renumbering_sparse_repeated_missing_and_negative():
    labels = np.array([900000, -1, 5, 900000, -2])
    inputs = [PointCloudData(np.zeros((len(labels), 3))) for _ in range(3)]
    for obj in inputs[:2]:
        obj.add_field('ids', labels.copy(), FieldType.INSTANCE)
    merged = PointCloudData.merge(inputs)
    actual = merged.get_fields()[0].data.ravel()
    np.testing.assert_array_equal(actual, [1, -1, 0, 1, -2, 3, -1, 2, 3, -2,
                                           -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(inputs[0].get_fields()[0].data.ravel(), labels)
    np.testing.assert_array_equal(PointCloudData.merge(inputs).get_fields()[0].data.ravel(), actual)


@pytest.mark.parametrize('level', list(MeshLevel))
def test_mesh_instance_fields_keep_inputs_separate(level):
    meshes = [MeshData([[0, 0, z], [1, 0, z], [0, 1, z]], [[0, 1, 2]]) for z in (0, 1)]
    for mesh in meshes:
        mesh.add_field('ids', np.full(mesh.element_count(level), 5), FieldType.INSTANCE, level=level)
    merged = MeshData.merge(meshes)
    np.testing.assert_array_equal(merged.get_fields(level=level)[0].data.ravel(),
                                  np.repeat([0, 1], meshes[0].element_count(level)))


def test_disjoint_mesh_topology_and_all_field_levels():
    meshes = [MeshData([[0, 0, z], [1, 0, z], [0, 1, z]], [[0, 1, 2]],
                       face_object_ids=np.array([7])) for z in (0, 2)]
    for i, mesh in enumerate(meshes):
        for level in MeshLevel:
            mesh.add_field('value', np.full(mesh.element_count(level), i), FieldType.SCALAR, level=level)
    merged = MeshData.merge(meshes)
    np.testing.assert_array_equal(merged.faces, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(merged.face_object_ids, [0, 1])
    np.testing.assert_array_equal(merged.edges[3:], meshes[1].edges + 3)
    for level in MeshLevel:
        values = merged.get_fields(level=level)[0].data.ravel()
        np.testing.assert_array_equal(values, np.repeat([0, 1], meshes[0].element_count(level)))


def test_boxes_and_cells_do_not_alias_or_collide():
    boxes = [BoundingBoxData([BoundingBox(id='same')]) for _ in range(2)]
    merged = BoundingBoxData.merge(boxes)
    assert len({b.id for b in merged.boxes}) == 2
    merged.boxes[0].width = 10
    assert boxes[0].boxes[0].width == 1
    graphs = []
    for _ in range(2):
        graph = CellComplexData([VertexCell((0, 0, 0), id='a'), VertexCell((1, 0, 0), id='b')])
        graph.build_edge('a', 'b')
        graphs.append(graph)
    merged = CellComplexData.merge(graphs)
    assert len(merged.vertices) == 4
    assert len({c.id for c in merged.get_cells()}) == len(merged.get_cells())
    assert all(set(e.boundary) <= {v.id for v in merged.vertices} for e in merged.edges)


def test_validation_and_name_roundtrip(tmp_path):
    a, b = cloud(), cloud()
    with pytest.raises(ValueError): PointCloudData.merge([a])
    with pytest.raises(ValueError): PointCloudData.merge([a, a])
    with pytest.raises(ValueError): PointCloudData.merge([a, CameraData()])
    with pytest.raises(ValueError): CameraData.merge([CameraData(), CameraData()])
    merged = PointCloudData.merge([a, b], name='Chosen name')
    doc = Document(); doc.add_data(merged)
    path = tmp_path / 'merged.h5'; doc.save_hdf5(path)
    loaded = Document.load_hdf5(path).scene_items[merged.id]
    assert loaded.name == 'Chosen name'
    scene = Scene(vtk.vtkRenderer()); scene.set_document(Document.load_hdf5(path))
    assert scene.layers[merged.id]._browser_name == 'Chosen name'


def test_command_replaces_at_first_position_and_undo_redo():
    scene = Scene(vtk.vtkRenderer())
    layers = []
    for i in range(3):
        data = cloud(i); scene.doc.add_data(data); layers.append(scene.add_data(data))
    scene.active_layer_id = layers[2].id
    merged = PointCloudLayer.merge([layers[0], layers[2]], name='Combined')
    commands = CommandManager()
    commands.do(MergeLayersCommand(scene, [layers[0], layers[2]], merged, Mock()))
    assert list(scene.layers) == [merged.id, layers[1].id]
    assert set(scene.doc.scene_items) == set(scene.layers)
    commands.undo()
    assert list(scene.layers) == [l.id for l in layers]
    assert scene.active_layer is layers[2]
    commands.redo()
    assert scene.active_layer is merged


def test_browser_merge_and_cancel(monkeypatch):
    from geon.ui.scene_manager import SceneManager
    app = QApplication.instance() or QApplication([])
    window = QMainWindow(); viewer = QWidget(window); viewer.rerender = Mock()
    window._mark_active_doc_modified = Mock()
    commands = CommandManager()
    controller = SimpleNamespace(ctx=SimpleNamespace(), deactivate_tool=Mock(), do_global=commands.do)
    manager = SceneManager(viewer, controller, window)
    scene = Scene(vtk.vtkRenderer()); manager._scene = scene
    for i in range(2):
        data = cloud(i); scene.doc.add_data(data); scene.add_data(data)
    manager.populate_tree()
    for i in range(2): manager.tree.topLevelItem(i).setSelected(True)
    assert len(manager.merge_candidates()) == 2
    menus = []
    monkeypatch.setattr(QMenu, 'exec', lambda menu, *a: menus.append([a.text() for a in menu.actions()]))
    manager._on_tree_context_menu(manager.tree.visualItemRect(manager.tree.topLevelItem(0)).center())
    assert menus and 'Merge' in menus[-1]
    original = list(scene.layers)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **kw: ('', False))
    manager.merge_selected_layers()
    assert list(scene.layers) == original
    default_names = []
    def pick(*args, **kwargs):
        default_names.append(kwargs['text'])
        return 'Merged clouds', True
    monkeypatch.setattr(QInputDialog, 'getText', pick)
    manager.merge_selected_layers()
    assert len(scene.layers) == 1
    assert scene.active_layer._browser_name == 'Merged clouds'
    assert default_names
    window._mark_active_doc_modified.assert_called()
    commands.undo()
    assert list(scene.layers) == original
    from geon.ui.menu_bar import MenuBar
    bar = MenuBar(window)
    triggered = Mock()
    bar.mergeLayersRequested.connect(triggered)
    bar.merge_layers_action.setEnabled(True)
    bar.merge_layers_action.trigger()
    triggered.assert_called_once()
    manager.close(); window.close()


def test_layer_eligibility_and_display_settings():
    from geon.rendering.boundingbox import BoundingBoxLayer
    from geon.rendering.cellcomplex import CellComplexLayer
    from geon.rendering.camera import CameraLayer
    a, b = PointCloudLayer(cloud()), PointCloudLayer(cloud())
    assert not PointCloudLayer.can_merge([a])
    assert not PointCloudLayer.can_merge([a, MeshLayer(MeshData([], []))])
    assert not CameraLayer.can_merge([CameraLayer(CameraData()), CameraLayer(CameraData())])
    a.point_size = 7
    assert PointCloudLayer.merge([a, b]).point_size == 7
    for cls, factory in ((BoundingBoxLayer, BoundingBoxData), (CellComplexLayer, CellComplexData),
                         (MeshLayer, lambda: MeshData([], []))):
        merged = cls.merge([cls(factory()), cls(factory())], name='Combined')
        assert merged._browser_name == 'Combined'


def test_attach_failure_preserves_sources(monkeypatch):
    scene = Scene(vtk.vtkRenderer())
    layers = []
    for _ in range(2):
        data = cloud(); scene.doc.add_data(data); layers.append(scene.add_data(data))
    merged = PointCloudLayer.merge(layers)
    monkeypatch.setattr(merged, 'attach', Mock(side_effect=RuntimeError('Rendering failed')))
    with pytest.raises(RuntimeError, match='Rendering failed'):
        MergeLayersCommand(scene, layers, merged, Mock()).execute()
    assert list(scene.layers) == [l.id for l in layers]
    assert set(scene.doc.scene_items) == set(scene.layers)
