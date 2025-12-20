from .common import Dock, ElidedLabel
from .viewer import VTKViewer
from geon.data.document import Document
from geon.rendering.scene import Scene
from geon.rendering.pointcloud import PointCloudLayer
from geon.data.pointcloud import PointCloudData

from PyQt6.QtWidgets import (QStackedWidget, QLabel, QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
                             QCheckBox, QButtonGroup, QRadioButton)
from PyQt6.QtCore import Qt, pyqtSignal

from typing import Optional, cast



import vtk
import traceback


class CheckBoxActive(QRadioButton):
    def __init__(self):
        super().__init__()

class CheckBoxVisible(QRadioButton):
    def __init__(self):
        super().__init__()


class SceneManager(Dock):
    # signals
    broadcastDeleteScene = pyqtSignal(Scene)

    def __init__(self, viewer: VTKViewer, parent=None, ):
        super().__init__("Scene", parent)
        self._scene : Optional[Scene] =  None
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
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)

    def on_document_loaded(self, doc: Document):
        if self._scene is not None:
            self.broadcastDeleteScene.emit(self._scene) # TODO: possible reason for import_ply bug?
            self._scene.clear(delete_data=False)
        self._scene = Scene(self.viewer._renderer)
        self._scene.set_document(doc)
        scene_main_layer = self._scene.get_layer()
        if scene_main_layer is not None:
            scene_main_actor = scene_main_layer.actors[0]
            self.viewer.focus_on_actor(scene_main_actor)
        self.populate_tree()
        
        self.viewer.rerender()

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
        
        for key, layer in self._scene.layers.items():
            if isinstance (layer, PointCloudLayer):   
                self._populate_point_cloud_layer(layer)
            else:
                raise NotImplementedError(f"Please implement a `populate` method for type {type(layer)}")
        self.tree.expandAll()
        self.update_tree_visibility()
            
    def _populate_point_cloud_layer(self, layer:PointCloudLayer):

        def set_layer_active_field(scene_manager: SceneManager, layer:PointCloudLayer, field_name: str):
            layer.set_active_field_name(field_name)
            scene_manager.viewer.rerender()

        print("called populate point cloud")
        pcd_root = QTreeWidgetItem([layer.browser_name])
        self.tree.addTopLevelItem(pcd_root)
        self.tree.setItemWidget(pcd_root,1,CheckBoxActive()) # TODO: hook up to a visibility / activate method
        self.tree.setItemWidget(pcd_root,2,CheckBoxVisible()) # TODO: hook up to a visibility / activate method

        # button groups
        fields_group_active = QButtonGroup(self)
        fields_group_active.setExclusive(True)

        fields_group_visible = QButtonGroup(self)
        fields_group_visible.setExclusive(True)

        for field in  cast(PointCloudData, layer.data).get_fields():
            field_item = QTreeWidgetItem([field.name])
            pcd_root.addChild(field_item)
            active_box = CheckBoxActive()
            fields_group_active.addButton(active_box)
            self.tree.setItemWidget(field_item,1,active_box) # TODO: hook up to a visibility / activate method
            
            active_box.clicked.connect(
                lambda checked, field_name=field.name: checked and set_layer_active_field(self, layer, field_name) 
                )
            
            
            # activate_btn.clicked.connect(
            #     lambda checked, ref=doc_ref: checked and self.set_active_doc(ref)
            #     ) 
            self.tree.setItemWidget(field_item,2,CheckBoxVisible()) # TODO: hook up to a visibility / activate method
