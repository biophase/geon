from .common import Dock, ElidedLabel
from .viewer import VTKViewer
from ..data.document import Document
from ..rendering.scene import Scene
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.base import BaseLayer
from ..data.pointcloud import PointCloudData
from ..tools.tool_context import ToolContext
from ..tools.controller import ToolController

from PyQt6.QtWidgets import (QStackedWidget, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QCheckBox, QButtonGroup, QRadioButton, QHeaderView)
from PyQt6.QtCore import Qt, pyqtSignal

from typing import Optional, cast



import vtk
import traceback


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
        # focus camera on first layer in scene
        scene_main_layer = self._scene.get_layer()
        if scene_main_layer is not None:
            scene_main_actor = scene_main_layer.actors[0] #FIXME: multiactors support?
            self.viewer.focus_camera_on_actor(scene_main_actor)
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
            layer_root = QTreeWidgetItem([layer.browser_name])
            self.tree.addTopLevelItem(layer_root)
            
            # activate button
            btn_active = CheckBoxActive()
            if self._scene.active_layer_id is None:
                self._scene.active_layer_id = layer.id
                btn_active.setChecked(True)
                activate_layer(layer)
            btn_active.clicked.connect(lambda checked, l=layer: checked and activate_layer(l))
            layers_btnGroup_active.addButton(btn_active)
            
            
            self.tree.setItemWidget(layer_root,1, self._center_widget(btn_active))
            
            
            # visibility button
            btn_visible = CheckBoxVisible()
            btn_visible.setChecked(layer.visible)
            self.tree.setItemWidget(layer_root,2, self._center_widget(btn_visible))
            def update_visibility(visibility: bool):
                layer.set_visible(visibility)
                self.viewer.rerender()
            btn_visible.clicked.connect(lambda checked: update_visibility(checked))
            
            # populate
            if isinstance (layer, PointCloudLayer):   
                self._populate_point_cloud_layer(layer, layer_root)
            else:
                raise NotImplementedError(f"Please implement a `populate` method for type {type(layer)}")
        self.tree.expandAll()
        self.update_tree_visibility()
            
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
            layer_root.addChild(field_item)
            active_box = CheckBoxActive()
            fields_group_active.addButton(active_box)
            
            
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
