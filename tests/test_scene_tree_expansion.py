from types import SimpleNamespace
from unittest.mock import Mock

import vtk
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget

from geon.data.mesh import MeshData
from geon.rendering.scene import Scene
from geon.ui.scene_manager import SceneManager


def test_tree_expansion_survives_refresh_rename_and_scene_switch():
    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    viewer = QWidget(window)
    viewer.rerender = Mock()
    manager = SceneManager(viewer, SimpleNamespace(ctx=None), window)
    scenes = [Scene(vtk.vtkRenderer()), Scene(vtk.vtkRenderer())]
    for scene in scenes:
        data = MeshData([], [], name='Same name')
        # Legacy documents may reuse IDs; their UI state must remain separate.
        data.id = 'MD_0001'
        scene.doc.add_data(data)
        scene.add_data(data)
    manager._scene = scenes[0]
    manager.populate_tree()
    root = manager.tree.topLevelItem(0)
    assert root.isExpanded()
    root.setExpanded(False)
    scenes[0].active_layer.data.name = 'Renamed'
    manager.populate_tree()
    assert not manager.tree.topLevelItem(0).isExpanded()
    manager._scene = scenes[1]
    manager.populate_tree()
    assert manager.tree.topLevelItem(0).isExpanded()
    manager._scene = scenes[0]
    manager.populate_tree()
    assert not manager.tree.topLevelItem(0).isExpanded()
    manager.tree.expandAll()
    manager.populate_tree()
    assert manager.tree.topLevelItem(0).isExpanded()
    manager.tree.collapseAll()
    manager.populate_tree()
    assert not manager.tree.topLevelItem(0).isExpanded()
    manager.close()
    window.close()
