from geon.io.dataset import Dataset, DocumentState
from geon.data.document import Document
from geon.rendering.scene import Scene

from .imports import ImportPLYDialog
from .common import ElidedLabel, Dock


from typing import Optional, cast
import os.path as osp

from PyQt6.QtWidgets import (QLabel, QPushButton, QHBoxLayout, QTreeWidget, QDockWidget, QWidget, 
                             QStackedWidget, QTreeWidgetItem, QFileDialog, QVBoxLayout, QSizePolicy,
                             QButtonGroup, QRadioButton
                             )
from PyQt6.QtCore import Qt, QSize, pyqtSignal

from PyQt6.QtGui import QFontMetrics



class DatasetManager(Dock):
    documentLoaded      = pyqtSignal(Document)
    

    def __init__(self, parent) -> None:
        super().__init__("Datasets", parent)
        self._dataset: Optional[Dataset] = None
        self.create_intermidiate_folder = True

        # overlay widget
        self.stack = QStackedWidget()
        self.overlay_label = QLabel("No Dataset Work Directory set")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("font-size: 16px; color: gray;")

        page = QWidget()
        self.tree_layout = QVBoxLayout(page)
        self.work_dir_label = ElidedLabel("")
        
        self.tree = QTreeWidget(self)
        self.tree_layout.addWidget(self.work_dir_label)
        self.tree_layout.addWidget(self.tree)

        self.stack.addWidget(self.overlay_label)    # index 0
        self.stack.addWidget(page)                  # index 1

        self.setWidget(self.stack)
        self.tree.setHeaderLabels(["","Scene name", "Status", "Path on disk"])
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.tree_button_group = QButtonGroup(self)
        self.tree_button_group.setExclusive(True)
        


    def set_dataset(self, dataset: Optional[Dataset]):
        self._dataset = dataset
        self.update_tree_visibility()
        self.populate_tree()

    def update_tree_visibility(self):
        """
        Show the tree only if a dataset is loaded,
        otherwise show centered overlay text.
        """
        if self._dataset is None:
            self.tree.clear()
            self.stack.setCurrentIndex(0)  # show overlay
        else:
            self.stack.setCurrentIndex(1)  # show tree


    def on_clear_scene(self, scene: Scene) -> None:
        if self._dataset is None:
            return
        for ref in self._dataset.doc_refs:
            if ref.name == scene.doc.name and (
                ref.state is DocumentState.MODIFIED or
                ref.state is DocumentState.UNSAVED
                ):
                if ref.path is None:
                    work_dir=self._dataset.working_dir
                    if work_dir is None:
                        success = self.set_work_dir()
                        if success:
                            work_dir = cast(str, self._dataset.working_dir)
                        else:
                            return
                            
                    ref.path =  osp.join(work_dir, ref.name, f"{ref.name}.h5") if self.create_intermidiate_folder else \
                                osp.join(work_dir,f"{ref.name}.h5")
                    
                scene.doc.save_hdf5(ref.path)
                ref.state = DocumentState.SAVED
        self.populate_tree()
                

            

    def populate_tree(self):     
        self.tree.clear()
        if self._dataset is None:
            return
        
        self._dataset.populate_references()

        for doc_ref in self._dataset._doc_refs:
            item = QTreeWidgetItem(["",doc_ref.name, doc_ref.state.name, doc_ref.path])
            self.tree.addTopLevelItem(item)
            activate_btn = QRadioButton()
            self.tree_button_group.addButton(activate_btn)
            self.tree.setItemWidget(item,0,activate_btn)
        self.tree.expandAll()
        self.update_tree_visibility()

    def set_work_dir(self) -> bool:
        """
        Set a working dir and return `True` for success
        """
        
        dir_path = QFileDialog.getExistingDirectory(self,
                                                    "Select dataset working directory (root)", 
                                                    "", 
                                                    QFileDialog.Option.ShowDirsOnly
                                                    )
        if not dir_path:
            return  False
        
        self.work_dir_label.setText(dir_path)

        dataset = Dataset(dir_path)
        # dataset.populate_references()
        self.set_dataset(dataset)
        if self._dataset is None:
            return False
        if self._dataset._working_dir is None:
            return False
        return True

    def import_doc_from_ply(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PLY File", "", "PLY Files (*.ply)")
        allow_doc_appending = False
        dlg = ImportPLYDialog(
            ply_path=file_path,
            semantic_schemas={},
            color_maps={}, 
            allow_doc_appending=allow_doc_appending, 
            parent=self)
        dlg.exec()
        if dlg.point_cloud is None:
            return
        if self._dataset is None:
            self.set_work_dir()
            if self._dataset is None:
                return
                
        
        # generate candidate name from imported ply name
        name_cand = osp.split(file_path)[-1]
        name = name_cand
        suffix = 0
        while name in self._dataset.doc_ref_names:
            name = f"{name_cand}_{suffix:03}"
            suffix += 1
        doc = Document(name)
        doc.add_data(dlg.point_cloud)
        self._dataset.create_new_reference(doc)
        self.populate_tree()
        print(list(self._dataset._doc_refs))
        self.documentLoaded.emit(doc)

        
