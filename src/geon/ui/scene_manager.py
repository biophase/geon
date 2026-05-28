from .common import Dock, ElidedLabel
from .viewer import VTKViewer
from ..data.document import Document
from ..rendering.scene import Scene
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.cellcomplex import CellComplexLayer
from ..rendering.base import BaseLayer
from ..data.pointcloud import PointCloudData, FieldType, SemanticSegmentation, SemanticClass
from ..tools.tool_context import ToolContext
from ..tools.selection import SelectPointsCmd
from ..util.resources import resource_path
from ..util.common import bool_op_index_mask
from ..ui.boolean_dialog import BooleanChoiceDialog
from ..ui.layers import LAYER_UI

from geon.settings import Preferences
import json
from datetime import datetime, timezone
from ..tools.controller import ToolController

from PyQt6.QtWidgets import (QStackedWidget, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QCheckBox, QButtonGroup, QRadioButton, QHeaderView, QMenu,
                             QDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QAction

from typing import Optional, cast
from dataclasses import dataclass, field
import weakref

import numpy as np



import vtk
import traceback

@dataclass
class SemClsHandle:
    layer: PointCloudLayer
    sem_cls: SemanticClass
    field: SemanticSegmentation

class CheckBoxActive(QRadioButton):
    def __init__(self):
        super().__init__()
        

class CheckBoxVisible(QCheckBox):
    def __init__(self):
        super().__init__()


class SceneManager(Dock):
    # signals
    broadcastDeleteScene = pyqtSignal(Scene)
    broadcastActivatedLayer = pyqtSignal(BaseLayer)
    broadcastActivatedPcdField = pyqtSignal(PointCloudLayer)

    def __init__(self, 
                 viewer: VTKViewer, 
                 controller: ToolController,
                 parent=None, 
                 ):
        super().__init__("Scene", parent)
        self._scene : Optional[Scene] =  None
        self.tool_controller: ToolController = controller
        # self._renderer: vtk.vtkRenderer = vtk.vtkRenderer()
        
        self.viewer: VTKViewer = viewer
        
        # the UI stacks two cases: 1) no scene loaded and 2) scene loaded
        self.stack = QStackedWidget()
        self.overlay_label = QLabel("No Scene loaded yet")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("font-size: 16px; color: gray;")
        
        page = QWidget()
        self.tree_layout = QVBoxLayout(page)
        self.scene_label = ElidedLabel("")

        self.tree = QTreeWidget(self)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setIconSize(QSize(12, 12))
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        self.tree_layout.addWidget(self.scene_label)
        self.tree_layout.addWidget(self.tree)
        
        self.stack.addWidget(self.overlay_label)    # index 0
        self.stack.addWidget(page)                  # index 1

        self.setWidget(self.stack)
        self.tree.setHeaderLabels(["Name", "Active", "Visible"])
        header_item = self.tree.headerItem()
        if header_item is not None:
            header_item.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            header_item.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)
        header_view = self.tree.header()
        if header_view is not None:
            header_view.setStretchLastSection(False)
            header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
            header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(1, 40)
        self.tree.setColumnWidth(2, 40)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.preferences: Optional[Preferences] = None

    def on_document_loaded(self, doc: Document):
        if self._scene is not None:
            self.broadcastDeleteScene.emit(self._scene) 
            self._scene.clear(delete_data=False)
        self._scene = Scene(self.viewer._renderer)
        self._scene.set_document(doc)
        
        # reference scene into viewer
        self.viewer.scene = self._scene 
        # reference scene into tool context
        self.tool_controller.ctx = ToolContext(
            scene=self._scene,
            viewer=self.viewer,
            controller = self.tool_controller
            )
        self.viewer.focus_camera_on_scene(self._scene)
        self.populate_tree()
        
        self.viewer.rerender()

    def _center_widget(self, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(widget)
        return container

    def update_tree_visibility(self):
        """
        Show the tree only if a dataset is loaded,
        otherwise show centered overlay text.
        """
        if self._scene is None:
            self.tree.clear()
            self.stack.setCurrentIndex(0)  # show overlay
            self.scene_label.setText("")
            
        else:
            self.stack.setCurrentIndex(1)  # show tree
            self.scene_label.setText(f"{self._scene.doc.name}")
    
    def populate_tree(self):
        self.tree.clear()

        print(f"called populate_tree")
        
        if self._scene is None:
            return
        scene = self._scene
        
        def activate_layer(l:BaseLayer):
            scene.active_layer_id = l.id
            self.broadcastActivatedLayer.emit(l)
        
        layers_btnGroup_active = QButtonGroup(self)
        layers_btnGroup_active.setExclusive(True)
        
        for key, layer in self._scene.layers.items():
            hooks = LAYER_UI.resolve(layer)
            layer_text = hooks.tree_item_text(layer) if hooks.tree_item_text is not None else layer.browser_name
            layer_root = QTreeWidgetItem([layer_text])
            layer_root.setData(0, Qt.ItemDataRole.UserRole, layer)
            if hooks.tree_item_icon is not None:
                icon = hooks.tree_item_icon(layer)
                if isinstance(icon, QIcon) and not icon.isNull():
                    layer_root.setIcon(0, icon)
            self.tree.addTopLevelItem(layer_root)
            
            # activate button
            btn_active = CheckBoxActive()
            if self._scene.active_layer_id is None:
                self._scene.active_layer_id = layer.id
                btn_active.setChecked(True)
                activate_layer(layer)
            elif self._scene.active_layer_id == layer.id:
                btn_active.setChecked(True)
            btn_active.clicked.connect(lambda checked, l=layer: checked and activate_layer(l))
            layers_btnGroup_active.addButton(btn_active)
            
            
            self.tree.setItemWidget(layer_root,1, self._center_widget(btn_active))
            
            
            # visibility button
            btn_visible = CheckBoxVisible()
            btn_visible.setChecked(layer.visible)
            self.tree.setItemWidget(layer_root,2, self._center_widget(btn_visible))
            def update_visibility(visibility: bool, target_layer: BaseLayer):
                layer = target_layer
                layer.set_visible(visibility)
                self.viewer.rerender()
            btn_visible.clicked.connect(lambda checked, l=layer: update_visibility(checked, l))
            
            # populate
            if isinstance (layer, PointCloudLayer):   
                layer_root.setIcon(0, QIcon(resource_path("tree_icon_pointcloud.png")))
                self._populate_point_cloud_layer(layer, layer_root)
            elif isinstance(layer, CellComplexLayer):
                self._populate_cell_complex_layer(layer, layer_root)
            else:
                layer_root.addChild(QTreeWidgetItem([type(layer).__name__]))
        self.tree.expandAll()
        self.update_tree_visibility()

    def _populate_cell_complex_layer(
        self,
        layer: CellComplexLayer,
        layer_root: QTreeWidgetItem,
    ) -> None:
        vertices_item = QTreeWidgetItem([f"Vertices ({layer.vertex_count:,})"])
        edges_item = QTreeWidgetItem([f"Edges ({layer.edge_count:,})"])
        layer_root.addChild(vertices_item)
        layer_root.addChild(edges_item)
            
    def _populate_point_cloud_layer(self, 
                                    layer:PointCloudLayer, 
                                    layer_root: QTreeWidgetItem):

        def set_layer_active_field(scene_manager: SceneManager, layer:PointCloudLayer, field_name: str):
            layer.set_active_field_name(field_name)
            self.broadcastActivatedPcdField.emit(layer)
            scene_manager.viewer.rerender()

        print("called populate point cloud")

        
        # button groups
        fields_group_active = QButtonGroup(self)
        fields_group_active.setExclusive(True)

        fields_group_visible = QButtonGroup(self)
        fields_group_visible.setExclusive(True)

        for field in  layer.data.get_fields():
            field_item = QTreeWidgetItem([field.name])
            field_item.setIcon(0, QIcon(resource_path("tree_icon_field.png")))
            field_item.setData(0, Qt.ItemDataRole.UserRole, field)
            layer_root.addChild(field_item)
            active_box = CheckBoxActive()
            fields_group_active.addButton(active_box)
            
            # handle field items
            if field.field_type == FieldType.SEMANTIC:
                field = cast(SemanticSegmentation, field)
                unique_ids = np.unique(field.data)
                for sem_cls in field.schema.semantic_classes:
                    if sem_cls.id not in unique_ids:
                        continue
                    sem_cls_item = self._make_tree_colored_item(field_item, sem_cls.name, sem_cls.color)
                    sem_cls_item.setData(0, Qt.ItemDataRole.UserRole, SemClsHandle(layer, sem_cls, field))
                    
                    
            
            # set first field to active
            if layer.active_field_name is None:
                set_layer_active_field(self, layer, field.name)
                active_box.setChecked(True)
                
            
            self.tree.setItemWidget(field_item,1, self._center_widget(active_box))
            
            active_box.clicked.connect(
                lambda checked, field_name=field.name: checked 
                and set_layer_active_field(self, layer, field_name) 
                )
            
            
            # activate_btn.clicked.connect(
            #     lambda checked, ref=doc_ref: checked and self.set_active_doc(ref)
            #     ) 
            # self.tree.setItemWidget(field_item,2,CheckBoxVisible()) # TODO: hook up to a visibility / activate method

    def log_tool_event(self, tool, action: str) -> None:
        """
        Append telemetry entry to the active document if enabled in preferences.
        """
        if self.preferences is None or not self.preferences.enable_telemetry:
            return
        if self._scene is None or self._scene.doc is None:
            return
        tool_name = tool.__class__.__name__ if tool is not None else None
        entry = {
            "tool": tool_name,
            "action": action,
            "user": self.preferences.user_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._scene.doc.telemetry.append(json.dumps(entry))
        except Exception:
            pass
    
    def _make_tree_colored_item(
        self,
        parent: QTreeWidgetItem,
        text: str,
        rgb: tuple[int, int, int],
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([""])
        swatch = QLabel()
        swatch.setFixedSize(12,12)
        swatch.setStyleSheet(
            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});"
            "border: 1px solid black;"
        )
        label = QLabel(text)
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(2,0,2,0)
        layout.setSpacing(6)
        layout.addWidget(swatch)
        layout.addWidget(label)
        layout.addStretch(1)

        parent.addChild(item)
        self.tree.setItemWidget(item, 0, w)
        return item
    
    def _on_tree_context_menu(self, pos):
        items = self.tree.selectedItems()
        if not items or self._scene is None:
            return
        
        objs = [item.data(0, Qt.ItemDataRole.UserRole) for item in items]
        ctx = self.tool_controller.ctx
        if ctx is None:
            return
        
        

        sem_cls_handles = [o for o in objs if isinstance(o, SemClsHandle)]
        layers = [o for o in objs if isinstance(o, BaseLayer)]
        
        menu = QMenu(self.tree)
            
        if sem_cls_handles:
            act = None
            if len(sem_cls_handles) == 1:
                act = menu.addAction(f"Select points of '{sem_cls_handles[0].sem_cls.name}'")
            elif len(sem_cls_handles) > 1:
                act = menu.addAction(f"Select points of {len(sem_cls_handles)} classes.")
            act = cast(QAction, act)
            act.triggered.connect(lambda: self._selection_from_sem_cls_handles(sem_cls_handles))

        if len(layers) == 1:
            layer = layers[0]
            hooks = LAYER_UI.resolve(layer)
            if hooks.tree_menu is not None:
                layer_menu = hooks.tree_menu(layer, self.tree, self.tool_controller)
                if not layer_menu.isEmpty():
                    if not menu.isEmpty():
                        menu.addSeparator()
                    for action in layer_menu.actions():
                        menu.addAction(action)
            if not menu.isEmpty():
                menu.addSeparator()
            act_delete_layer = menu.addAction("Delete layer")
            act_delete_layer.triggered.connect(
                lambda _checked=False, l=layer: self._delete_layer(l)
            )
            
        
        viewport = self.tree.viewport()
        if viewport is None:
            return
        if menu.isEmpty():
            return
        menu.exec(viewport.mapToGlobal(pos))

    def _confirm_delete_layer(self, layer: BaseLayer) -> bool:
        msg = QMessageBox(self.tree)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Delete Layer")
        msg.setText(f"Delete layer '{layer.browser_name}'?")
        msg.setInformativeText(
            "This removes the layer from the scene and the document. "
            "This action is irreversible."
        )
        delete_btn = msg.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        return msg.clickedButton() is delete_btn

    def _delete_layer(self, layer: BaseLayer) -> None:
        if self._scene is None:
            return
        if layer.id not in self._scene.layers:
            return
        if not self._confirm_delete_layer(layer):
            return

        self._scene.remove_layer(layer.id, delete_data=True)
        window = self.viewer.window()
        mark_modified = getattr(window, "_mark_active_doc_modified", None)
        if callable(mark_modified):
            mark_modified()
        dataset_manager = getattr(window, "dataset_manager", None)
        if dataset_manager is not None and hasattr(dataset_manager, "populate_tree"):
            dataset_manager.populate_tree()
        self.populate_tree()
        active_layer = self._scene.active_layer
        if active_layer is not None:
            self.broadcastActivatedLayer.emit(active_layer)
        self.viewer.rerender()

    def _selection_from_sem_cls_handles(self, handles:list[SemClsHandle]) -> None:
        if self._scene is None:
            return
        ctx = self.tool_controller.ctx
        if ctx is None:
            return
        
        handles_by_layer: dict[str, list[SemClsHandle]] = {}
        for handle in handles:
            handles_by_layer.setdefault(handle.layer.id, []).append(handle)
        if not handles_by_layer:
            return

        layer_selections: dict[str, np.ndarray] = {}
        needs_merge = False

        for layer in self._scene.layers.values():
            layer = cast(PointCloudLayer, layer)
            layer_sem_handles = handles_by_layer.get(layer.id)
            if not layer_sem_handles:
                continue

            selection = [
                np.flatnonzero(handle.field.data == handle.sem_cls.id)
                for handle in layer_sem_handles
            ]
            indices = (
                np.concatenate(selection) if selection else np.array([], dtype=np.int32)
            )
            layer_selections[layer.id] = indices
            if (
                indices.size > 0
                and layer.active_selection is not None
                and layer.active_selection.size > 0
            ):
                needs_merge = True

        merge_mode = None
        if needs_merge:
            dlg = BooleanChoiceDialog(
                self, message="Choose how to combine with previous selection:"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted or dlg.choice is None:
                return
            merge_mode = dlg.choice

        for layer in self._scene.layers.values():
            layer = cast(PointCloudLayer, layer)
            indices = layer_selections.get(layer.id)
            if indices is None:
                continue
            if (
                merge_mode is not None
                and layer.active_selection is not None
                and layer.active_selection.size > 0
            ):
                indices = bool_op_index_mask(
                    layer.active_selection, indices, merge_mode
                )

            cmd = SelectPointsCmd(
                title="Select class points",
                selection_new=indices,
                layer_ref=weakref.ref(layer),
                ctx_ref=weakref.ref(ctx),
            )
            self.tool_controller.command_manager.do(cmd)
            
        
        


