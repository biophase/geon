from geon.io.dataset import Dataset, RefModState, DocumentReference, RefLoadedState
from geon.data.document import Document
from geon.rendering.scene import Scene
from config.common import KNOWN_FILE_EXTENSIONS

from .imports import ImportPLYDialog
from .common import ElidedLabel, Dock


from typing import Optional, cast, Union
import os.path as osp

from PyQt6.QtWidgets import (QLabel, QPushButton, QHBoxLayout, QTreeWidget, QDockWidget, QWidget, 
                             QStackedWidget, QTreeWidgetItem, QFileDialog, QVBoxLayout, QSizePolicy,
                             QButtonGroup, QRadioButton, QMessageBox
                             )
from PyQt6.QtCore import Qt, QSize, pyqtSignal

from PyQt6.QtGui import QFontMetrics



class DatasetManager(Dock):
    requestSetActiveDocInScene      = pyqtSignal(Document)
    

    def __init__(self, parent) -> None:
        super().__init__("Datasets", parent)
        
        # containers
        self._dataset: Optional[Dataset] = None
        self._active_doc_name: Optional[str] = None
        
        # settings
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
        self.tree.setHeaderLabels(["","Scene name", "Modified", "Loaded", "Path on disk"])
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

    def set_active_doc(self, doc_ref: DocumentReference) -> Optional[Document]:
        if self._dataset is None:
            return
        if doc_ref.loadedState == RefLoadedState.ACTIVE:
            print(f"Reference {doc_ref.name} is already active.")
            return
        self._dataset.deactivate_current_ref()
        doc = self._dataset.activate_reference(doc_ref)        
        self.requestSetActiveDocInScene.emit(doc)
        return doc

    def on_clear_scene(self, scene: Scene,ignore_state=False) -> None:
        if self._dataset is None:
            return
        for ref in self._dataset.doc_refs:
            if ref.name == scene.doc.name and (
                ref.modState is RefModState.MODIFIED or
                ignore_state
                ):
                if ref.path is None:
                    work_dir=self._dataset.working_dir
                    if work_dir is None:
                        QMessageBox.warning(self, 
                                            "No working directory set",
                                            f"Please set a working directory to avoid loosing work on {ref.name}!")
                        success = self.set_work_dir()
                        if success:
                            work_dir = cast(str, self._dataset.working_dir)
                        else:
                            # last chance
                            reply = QMessageBox.question(
                                self,
                                "Confirm",
                                "Do you want to continue without saving?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No
                            )

                            if reply == QMessageBox.StandardButton.Yes:
                                return
                            else:
                                self.on_clear_scene(scene, ignore_state)
                                return

                            
                    ref.path =  osp.join(work_dir, ref.name, f"{ref.name}.h5") if self.create_intermidiate_folder else \
                                osp.join(work_dir,f"{ref.name}.h5")
                    
                scene.doc.save_hdf5(ref.path)
                ref.modState = RefModState.SAVED
        self.populate_tree()
                

    def check_dataset_name_duplicates(self):
        if self._dataset is None:
            return
        unique_names = []
        for ref_name in self._dataset.doc_ref_names:
            if ref_name in unique_names:
                raise ValueError(f'Duplicate names detected in dataset:{ref_name}')
            unique_names.append(ref_name)

    


    
    def populate_tree(self):     
        self.tree.clear()
        if self._dataset is None:
            return
        
        self._dataset.populate_references()

        for doc_ref in self._dataset._doc_refs:
            item = QTreeWidgetItem([
                "",
                doc_ref.name, 
                doc_ref.modState.name, 
                doc_ref.loadedState.name, 
                doc_ref.path])
            self.tree.addTopLevelItem(item)
            activate_btn = QRadioButton()
            self.tree_button_group.addButton(activate_btn)
            self.tree.setItemWidget(item,0,activate_btn)
            activate_btn.clicked.connect(
                lambda checked, ref=doc_ref: checked and self.set_active_doc(ref)
                ) 
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
        if self._dataset is None:
            self.set_work_dir()
            if self._dataset is None:
                return
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PLY File", "", "PLY Files (*.ply)")
        allow_doc_appending = False
        dlg = ImportPLYDialog(
            ply_path=file_path,
            semantic_schemas= {s.name : s for s in self._dataset.unique_semantic_schemas},
            color_maps={}, 
            allow_doc_appending=allow_doc_appending, 
            parent=self)
        dlg.exec()
        if dlg.point_cloud is None:
            return
                
        
        # generate candidate name from imported ply name
        name_cand = osp.split(file_path)[-1]
        
        file_name, file_ext = osp.splitext(name_cand)
        if file_ext not in KNOWN_FILE_EXTENSIONS:
            name = name_cand
        else:
            name = file_name

        suffix = 0
        while name in self._dataset.doc_ref_names:
            name = f"{name_cand}_{suffix:03}"
            suffix += 1
        doc = Document(name)
        doc.add_data(dlg.point_cloud)
        
        doc_ref = self._dataset.add_document(doc)
        self.populate_tree()
        self.set_active_doc(doc_ref)
       
        
        

        
