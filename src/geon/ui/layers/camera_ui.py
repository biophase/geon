from __future__ import annotations

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QMenu, QWidget

from geon.rendering.camera import CameraLayer
from geon.tools.controller import ToolController
from geon.util.resources import resource_path

from .registry import LAYER_UI, LayerUIHooks


def _mark_modified(parent: QWidget) -> None:
    window = parent.window()
    mark_modified = getattr(window, "_mark_active_doc_modified", None)
    if callable(mark_modified):
        mark_modified()
    dataset_manager = getattr(window, "dataset_manager", None)
    if dataset_manager is not None and hasattr(dataset_manager, "populate_tree"):
        dataset_manager.populate_tree()


def _refresh(parent: QWidget, controller: ToolController, layer: CameraLayer) -> None:
    layer.update()
    _mark_modified(parent)
    controller.scene_tree_request_change.emit()
    ctx = controller.ctx
    if ctx is not None:
        ctx.viewer.rerender()


def _menu(layer: CameraLayer, parent: QWidget, controller: ToolController) -> QMenu:
    menu = QMenu(parent)

    act_update = menu.addAction("Update from current")
    act_restore = menu.addAction("Restore")
    act_export = menu.addAction("Export to JSON...")

    def update_from_current() -> None:
        ctx = controller.ctx
        if ctx is None:
            return
        layer.data.update_from_camera(ctx.viewer._renderer.GetActiveCamera())
        _refresh(parent, controller, layer)

    def restore() -> None:
        ctx = controller.ctx
        if ctx is None:
            return
        camera = ctx.viewer._renderer.GetActiveCamera()
        layer.data.apply_to_camera(camera)
        ctx.viewer._renderer.ResetCameraClippingRange()
        camera.SetClippingRange(*layer.data.clipping_range)
        ctx.viewer.rerender()

    def export_to_json() -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            parent,
            "Export Camera Snapshot to JSON",
            f"{layer.data.name}.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            layer.data.save_json(path)
        except Exception as exc:
            QMessageBox.critical(parent, "Export Camera Snapshot", str(exc))

    act_update.triggered.connect(update_from_current)
    act_restore.triggered.connect(restore)
    act_export.triggered.connect(export_to_json)
    return menu


def _text(layer: CameraLayer) -> str:
    return layer.browser_name


def _icon(layer: CameraLayer) -> QIcon:
    return QIcon(resource_path("camera_perspective_toggle.png"))


LAYER_UI.register(
    CameraLayer,
    LayerUIHooks(
        tree_menu=_menu,
        tree_item_text=_text,
        tree_item_icon=_icon,
    ),
)
