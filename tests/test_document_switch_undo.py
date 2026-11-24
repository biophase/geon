from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from geon.tools.command_manager import CommandManager
from geon.tools.controller import ToolController
from geon.ui.scene_manager import SceneManager
from geon.ui.dataset_manager import DatasetManager
from geon.io.dataset import RefLoadedState


@pytest.mark.parametrize('cancelled', [False, True])
def test_scene_replacement_clears_global_and_tool_history(monkeypatch, cancelled):
    global_cm = CommandManager([Mock()], [Mock()])
    local_cm = CommandManager([Mock()], [Mock()])
    controller = SimpleNamespace(
        _global_command_manager=global_cm,
        _active_tool=SimpleNamespace(use_local_cm=True, command_manager=local_cm),
    )
    controller.clear_undo_stacks = lambda: ToolController.clear_undo_stacks(controller)
    panel = SimpleNamespace(activate=Mock(return_value=not cancelled))
    old_scene, new_scene = Mock(), Mock()
    new_scene.active_layer = None
    monkeypatch.setattr('geon.ui.scene_manager.Scene', Mock(return_value=new_scene))
    manager = SimpleNamespace(
        window=lambda: SimpleNamespace(color_channels=panel),
        tool_controller=controller, _scene=old_scene,
        broadcastDeleteScene=Mock(), viewer=Mock(), preferences=None,
        populate_tree=Mock(),
    )
    doc = Mock()
    SceneManager.on_document_loaded(manager, doc)
    for cm in (global_cm, local_cm):
        assert bool(cm.undo_stack) == cancelled
        assert bool(cm.redo_stack) == cancelled
    if cancelled:
        assert manager._scene is old_scene
        old_scene.clear.assert_not_called()
    else:
        assert manager._scene is new_scene
        new_scene.set_document.assert_called_once_with(doc)


def test_selecting_current_document_does_not_replace_scene():
    manager = SimpleNamespace(window=lambda: SimpleNamespace(), _dataset=Mock(),
                              requestSetActiveDocInScene=Mock())
    ref = SimpleNamespace(loadedState=RefLoadedState.ACTIVE, name='Current')
    DatasetManager.set_active_doc(manager, ref)
    manager.requestSetActiveDocInScene.emit.assert_not_called()
    manager._dataset.activate_reference.assert_not_called()
