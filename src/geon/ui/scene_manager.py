from .common import Dock, ElidedLabel

from geon.data.document import Document
from geon.rendering.scene import Scene
from geon.rendering.pointcloud import PointCloudLayer, PointCloudData

from PyQt6.QtWidgets import (QStackedWidget, QLabel, QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
                             QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal

from typing import Optional, cast


class CheckBoxActive(QCheckBox):
    def __init__(self):
        super().__init__()

class CheckBoxVisible(QCheckBox):
    def __init__(self):
        super().__init__()


class SceneManager(Dock):
    # signals
    broadcastDeleteScene = pyqtSignal(Scene)

    def __init__(self, parent=None):
        super().__init__("Scene", parent)
        self._scene : Optional[Scene] =  None
        
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
            self.broadcastDeleteScene.emit(self._scene)
            self._scene.clear(delete_data=True)
        self._scene = Scene()
        self._scene.set_document(doc)
        self.populate_tree()

    def update_tree_visibility(self):
        """
        Show the tree only if a dataset is loaded,
        otherwise show centered overlay text.
        """
        if self._scene is None:
            self.tree.clear()
            self.stack.setCurrentIndex(0)  # show overlay
        else:
            self.stack.setCurrentIndex(1)  # show tree
    
    def populate_tree(self):
        self.tree.clear()
        print(f"called populate_tree")
        if self._scene is None:
            return
        
        for key, layer in self._scene.layers.items():
            if isinstance (layer, PointCloudLayer):   
                self.populate_point_cloud_layer(layer)
            else:
                raise NotImplementedError(f"Please implement a `populate` method for type {type(layer)}")
        self.tree.expandAll()
        self.update_tree_visibility()
            
    def populate_point_cloud_layer(self, layer:PointCloudLayer):
        print("called populate point cloud")
        pcd_root = QTreeWidgetItem([layer.browser_name])
        self.tree.addTopLevelItem(pcd_root)
        self.tree.setItemWidget(pcd_root,1,CheckBoxActive()) # TODO: hook up to a visibility / activate method
        self.tree.setItemWidget(pcd_root,2,CheckBoxVisible()) # TODO: hook up to a visibility / activate method
        for field in  cast(PointCloudData, layer.data).get_fields():
            field_item = QTreeWidgetItem([field.name])
            pcd_root.addChild(field_item)
            self.tree.setItemWidget(field_item,1,CheckBoxActive()) # TODO: hook up to a visibility / activate method
            self.tree.setItemWidget(field_item,2,CheckBoxVisible()) # TODO: hook up to a visibility / activate method



            

            
    
