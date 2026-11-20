"""Display names may change or repeat; data references must remain stable."""
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import h5py
import numpy as np
import pytest
import vtk
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget

from geon.data.boundingbox import BoundingBoxData
from geon.data.camera import CameraData
from geon.data.cellcomplex import CellComplexData, PointCloudRef, VertexCell
from geon.data.document import Document
from geon.data.mesh import MeshData
from geon.data.pointcloud import FieldType, PointCloudData
from geon.rendering.layer_registry import LAYER_REGISTRY
from geon.rendering.scene import Scene, DuplicateLayerIdError
from geon.ui.cellcomplex_dialogs import CellComplexAttributesDialog
from geon.ui.scene_manager import SceneManager


def cloud():
    return PointCloudData(np.zeros((3, 3)))


FACTORIES = [cloud, lambda: MeshData([], []), BoundingBoxData, CellComplexData, CameraData]


@pytest.fixture(scope="module", autouse=True)
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.mark.parametrize("factory", FACTORIES)
def test_names_are_independent_of_uuid_and_survive_layer_creation(factory):
    data = factory()
    original_id = data.id
    assert UUID(original_id).version == 4
    assert data.name.startswith(data.get_short_type_id() + "_")
    data.name = "Bridge / south side"
    layer = LAYER_REGISTRY.create_layer_for(data)
    assert layer.data.name == "Bridge / south side"
    assert layer.id == original_id
    assert not hasattr(layer, "_browser_name")
    data.name = "Renamed"
    assert layer.data.name == "Renamed"
    assert layer.id == original_id


@pytest.mark.parametrize("factory", FACTORIES)
def test_standalone_hdf5_preserves_identity_and_name(tmp_path, factory):
    data = factory()
    data.name = "Bridge / Träger 12"
    with h5py.File(tmp_path / "object.h5", "w") as handle:
        group = handle.create_group("arbitrary-container")
        data.save_hdf5(group)
        loaded = type(data).load_hdf5(group)
    assert loaded.id == data.id
    assert loaded.name == data.name


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("has_id_attribute", [False, True])
def test_legacy_hdf5_keeps_reference_and_uses_id_as_missing_name(tmp_path, factory, has_id_attribute):
    data = factory()
    with h5py.File(tmp_path / "legacy.h5", "w") as handle:
        group = handle.create_group("legacy-object")
        data.save_hdf5(group)
        del group.attrs["name"]
        if has_id_attribute:
            group.attrs["id"] = b"legacy-object"
        else:
            del group.attrs["id"]
        loaded = type(data).load_hdf5(group)
    assert loaded.id == "legacy-object"
    assert loaded.name == "legacy-object"


def test_duplicate_names_and_rename_do_not_change_document_references(tmp_path):
    data, other = cloud(), cloud()
    data.name = other.name = "Same name"
    data.add_field("patches", np.array([7, 7, 8]), FieldType.INSTANCE)
    ref = PointCloudRef(ref_id=data.id, field_name="patches", instance_id=7)
    complex_data = CellComplexData([VertexCell((0, 0, 0), geometry_refs=[ref])])
    doc = Document()
    for obj in (data, other, complex_data):
        doc.add_data(obj)
    original_id = data.id
    data.name = "Renamed / cloud"
    path = tmp_path / "scene.h5"
    doc.save_hdf5(path)
    loaded = Document.load_hdf5(path)
    assert set(loaded.scene_items) == set(doc.scene_items)
    assert loaded.scene_items[original_id].name == data.name
    loaded_ref = loaded.scene_items[complex_data.id].vertices[0].geometry_refs[0]
    assert loaded_ref.ref_id == original_id
    assert loaded.scene_items[loaded_ref.ref_id].get_fields(names=[loaded_ref.field_name])


def test_scene_browser_and_reference_table_display_names_but_resolve_ids():
    scene = Scene(vtk.vtkRenderer())
    data, other = cloud(), cloud()
    data.name = other.name = "Shared display name"
    for obj in (data, other):
        scene.doc.add_data(obj)
        scene.add_data(obj)
    layer = scene.get_layer(layer_id=data.id)
    assert layer.data is data
    with pytest.raises(DuplicateLayerIdError):
        scene.add_layer(layer)
    scene.active_layer_id = data.id
    window = QMainWindow()
    viewer = QWidget(window)
    viewer.rerender = Mock()
    controller = SimpleNamespace(ctx=SimpleNamespace(), deactivate_tool=Mock())
    manager = SceneManager(viewer, controller, window)
    manager._scene = scene
    manager.populate_tree()
    assert manager.tree.topLevelItem(0).text(0) == data.name
    assert manager.tree.topLevelItem(1).text(0) == other.name
    data.name = "Renamed"
    manager.populate_tree()
    assert manager.tree.topLevelItem(0).text(0) == "Renamed"
    assert scene.active_layer is layer
    ref = PointCloudRef(ref_id=data.id, field_name="patches", instance_id=7)
    vertex = VertexCell((0, 0, 0), geometry_refs=[ref])
    complex_layer = LAYER_REGISTRY.create_layer_for(CellComplexData([vertex]))
    dialog = CellComplexAttributesDialog(complex_layer, vertex, SimpleNamespace(scene=scene))
    table = dialog._make_references_tab()
    assert table.item(0, 1).text() == "Renamed"
    assert table.item(0, 1).toolTip() == data.id
    assert dialog._layer_for_ref(ref.ref_id) is layer
    dialog.close()
    manager.close()
    window.close()


def test_extraction_and_merge_keep_names_separate_from_fresh_ids():
    data, other = cloud(), cloud()
    data.name = "Source"
    layers = [LAYER_REGISTRY.create_layer_for(obj) for obj in (data, other)]
    layers[0].active_selection = np.array([0, 1])
    extracted = layers[0].extract_selection()
    merged = type(layers[0]).merge(layers, name="Combined")
    assert extracted.data.name == "Source (selection)"
    assert merged.data.name == "Combined"
    assert len({data.id, other.id, extracted.id, merged.id}) == 4
    assert data.name == "Source"
