from geon.io.dataset import (Dataset, RefModState, DocumentReference, 
                             RefLoadedState)
from geon.data.document import Document
from geon.rendering.scene import Scene
from geon.config.common import KNOWN_FILE_EXTENSIONS

from .imports import FieldEditorDialog
from .common import ElidedLabel, Dock
from ..data.pointcloud import SemanticSchema

from typing import Optional, cast, Union, Callable
import os.path as osp

from PyQt6.QtWidgets import (QLabel, QPushButton, QHBoxLayout, QTreeWidget, QDockWidget, QWidget, 
                             QStackedWidget, QTreeWidgetItem, QFileDialog, QVBoxLayout, QSizePolicy,
                             QButtonGroup, QRadioButton, QMessageBox, QDialog, QDialogButtonBox,
                             QTextEdit, QProgressDialog, QApplication
                             )
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread, QEventLoop, pyqtSlot

from PyQt6.QtGui import QFontMetrics




class DatasetManager(Dock):
    requestSetActiveDocInScene      = pyqtSignal(Document)
    requestClearUndoStacks          = pyqtSignal()
    

    def __init__(self, parent) -> None:
        super().__init__("Dataset", parent)
        
        # containers
        self._dataset: Optional[Dataset] = None
        self._active_doc_name: Optional[str] = None
        
        # settings
        self.create_intermidiate_folder = False
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
        self.tree.setColumnWidth(0, 20) # selection column can be small
        self.tree.setTreePosition(1) # tree indent in second column
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
        otherwise show overlay.
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

    def save_scene_doc(self, scene: Optional[Scene], ignore_state=False) -> None:
        print(
            "[save_scene_doc] called"
            f" ignore_state={ignore_state}"
            f" scene_name={(scene.doc.name if scene is not None else None)}"
        )
        if scene is None:
            print("[save_scene_doc] no scene; abort")
            return
        if self._dataset is None:
            print("[save_scene_doc] no dataset; abort")
            return
        for ref in self._dataset.doc_refs:
            if ref.name == scene.doc.name and (
                ref.modState is RefModState.MODIFIED or
                ignore_state
                ):
                print(
                    "[save_scene_doc] saving ref"
                    f" name={ref.name} modState={ref.modState.name}"
                    f" loadedState={ref.loadedState.name} path={ref.path}"
                )
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
                                self.save_scene_doc(scene, ignore_state)
                                return

                            
                    ref.path =  osp.join(work_dir, ref.name, f"{ref.name}.h5") if self.create_intermidiate_folder else \
                                osp.join(work_dir,f"{ref.name}.h5")
                    print(f"[save_scene_doc] assigned path={ref.path}")
                    
                try:
                    scene.doc.save_hdf5(ref.path)
                except Exception as e:
                    print(f"[save_scene_doc] save failed for {ref.path}: {e}")
                    raise
                ref.modState = RefModState.SAVED
                print(f"[save_scene_doc] save completed for {ref.name}")
            elif ref.name == scene.doc.name:
                print(
                    "[save_scene_doc] skip save"
                    f" name={ref.name} modState={ref.modState.name}"
                    f" ignore_state={ignore_state}"
                )
        self.populate_tree()
                

    def check_dataset_name_duplicates(self):
        if self._dataset is None:
            return
        unique_names = []
        for ref_name in self._dataset.doc_ref_names:
            if ref_name in unique_names:
                raise ValueError(f'Duplicate names detected in dataset:{ref_name}')
            unique_names.append(ref_name)

    
    def update_semantic_schema (
        self,
        old_schema: SemanticSchema,
        new_schema: SemanticSchema,
        old_2_new_ids: list[tuple[int,int]],
        progress_cb: Optional[Callable[[int, int, str],None]] = None
        ) -> bool:
        """
        returns `True` on successfull remapping
        """
        if self._dataset is None:
            return False
        
        matching_schemas = self._dataset.get_matching_schemas(old_schema)
        
        build_keys = [str(k) for k in matching_schemas.keys()]
        confirm_text = (
            f"You are about to edit the semantic schema {old_schema.name} globally. "
            f"This will affect {len(build_keys)} matching schemas:"
        )

        confirm = QDialog(self)
        confirm.setWindowTitle("Confirm global semantic schema edit")
        vbox = QVBoxLayout(confirm)
        label = QLabel(confirm_text, confirm)
        label.setWordWrap(True)
        vbox.addWidget(label)
        text_box = QTextEdit(confirm)
        text_box.setReadOnly(True)
        text_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        text_box.setText("\n".join(build_keys))
        text_box.setMinimumWidth(360)
        text_box.setMinimumHeight(140)
        vbox.addWidget(text_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=confirm,
        )
        buttons.accepted.connect(confirm.accept)
        buttons.rejected.connect(confirm.reject)
        vbox.addWidget(buttons)
        if confirm.exec() != QDialog.DialogCode.Accepted:
            return False

        if not build_keys:
            return True

        class _SchemaUpdateWorker(QThread):
            progress = pyqtSignal(int, int, str)
            errored = pyqtSignal(str)
            finished_refs = pyqtSignal(list)

            def __init__(self, dataset: Dataset):
                super().__init__()
                self._dataset = dataset
                self._old_schema = old_schema
                self._new_schema = new_schema
                self._old_2_new_ids = old_2_new_ids

            def _progress(self, idx: int, total: int, key: str) -> None:
                self.progress.emit(idx, total, key)

            def run(self) -> None:
                try:
                    refs = self._dataset.update_semantic_schema(
                        self._old_schema,
                        self._new_schema,
                        self._old_2_new_ids,
                        self._progress,
                    )
                    self.finished_refs.emit(refs)
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        progress_dialog = QProgressDialog(
            "Updating semantic schemas...", None, 0, max(1, len(build_keys)), self
        )
        progress_dialog.setWindowTitle("Updating schemas")
        progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setCancelButton(None)

        loop = QEventLoop(self)
        success = {"value": False}

        def _on_progress(idx: int, total: int, key: str) -> None:
            progress_dialog.setMaximum(max(1, total))
            progress_dialog.setValue(idx + 1)
            progress_dialog.setLabelText(f"Processing: {key}")
            QApplication.processEvents()

        def _on_finished(_refs: list[DocumentReference]) -> None:
            progress_dialog.setValue(progress_dialog.maximum())
            progress_dialog.close()
            success["value"] = True
            loop.quit()

        def _on_error(msg: str) -> None:
            progress_dialog.close()
            QMessageBox.critical(self, "Schema update failed", msg)
            loop.quit()

        worker = _SchemaUpdateWorker(self._dataset)
        worker.progress.connect(_on_progress)
        worker.finished_refs.connect(_on_finished)
        worker.errored.connect(_on_error)
        worker.start()
        progress_dialog.show()
        loop.exec()
        worker.wait()

        if success["value"]:
            self.requestClearUndoStacks.emit()
        return success["value"]
        

    
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
            success = self.set_work_dir()
            if self._dataset is None:
                return
            if not success:
                return
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PLY File", "", "PLY Files (*.ply)")
        allow_doc_appending = False
        if file_path is None or file_path=='':
            return
        dlg = FieldEditorDialog(
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
        
    def disable(self) -> None:
        self.tree.setEnabled(False)
    
    def enable(self) -> None:
        self.tree.setEnabled(True)    
       
        
        

        
