from geon.io.dataset import Dataset
from geon.data.document import Document
from .imports import ImportPLYDialog

from typing import Optional
import os.path as osp

from PyQt6.QtWidgets import (QLabel, QPushButton, QHBoxLayout, QTreeWidget, QDockWidget, QWidget, 
                             QStackedWidget, QTreeWidgetItem, QFileDialog, QVBoxLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, pyqtSignal

from PyQt6.QtGui import QFontMetrics

class DockTitleBar(QWidget):
    def __init__(self, dock: QDockWidget):
        super().__init__(dock)
        self.dock = dock
        label = QLabel(dock.windowTitle())
        label_font = label.font()
        label_font.setBold(True)
        label.setFont(label_font)
        detach_btn = QPushButton()
        detach_btn.setText("⧉")
        detach_btn.setToolTip("Detach window")
        detach_btn.setFlat(True)
        detach_btn.setFixedSize(QSize(24,24))
        detach_btn.clicked.connect(lambda: dock.setFloating(True))
        
        layout= QHBoxLayout(self)
        layout.setContentsMargins(4,0,4,0)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(detach_btn)

class Dock(QDockWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setObjectName(title.replace(" ", "_"))
        self.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
            |   QDockWidget.DockWidgetFeature.DockWidgetFloatable
            |   QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setTitleBarWidget(DockTitleBar(self))


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent=None,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle):
        super().__init__("", parent)
        self._full_text = text
        self._elide_mode = mode

        # Let it shrink horizontally instead of forcing min width
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        self._update_elided_text()

    def setText(self, a0: Optional[str]) -> None:
        self._full_text = a0
        self._update_elided_text()

    def fullText(self) -> Optional[str]:
        return self._full_text

    def resizeEvent(self, a0) -> None:
        super().resizeEvent(a0)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        fm = QFontMetrics(self.font())
        width = max(self.width(), 0)
        elided = fm.elidedText(self._full_text, self._elide_mode, width)
        # Call QLabel's setText directly to avoid overwriting _full_text
        super().setText(elided)

class DatasetManager(Dock):
    documentLoaded      = pyqtSignal(Document)
    def __init__(self, parent) -> None:
        super().__init__("Datasets", parent)
        self._dataset: Optional[Dataset] = None

        # --- Create overlay system ---
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
        self.tree.setHeaderLabels(["Scene name", "Status", "Path on disk"])
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)


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


    def populate_tree(self):
        self.tree.clear()
        if self._dataset is None:
            return
        
        self._dataset.update_references()

        for doc_ref in self._dataset._doc_refs:
            item = QTreeWidgetItem([doc_ref.name, doc_ref.state.name, doc_ref.path])
            self.tree.addTopLevelItem(item)
        self.tree.expandAll()
        self.update_tree_visibility()

    def set_work_dir(self) -> None:
        
        dir_path = QFileDialog.getExistingDirectory(self,
                                                    "Select dataset working directory (root)", 
                                                    "", 
                                                    QFileDialog.Option.ShowDirsOnly
                                                    )
        if not dir_path:
            return  
        
        self.work_dir_label.setText(dir_path)

        dataset = Dataset(f"working dir: {dir_path}")
        dataset.update_references()
        self.set_dataset(dataset)

    def import_doc_from_ply(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PLY File", "", "PLY Files (*.ply)")
        dlg = ImportPLYDialog(file_path,{},{},self)
        dlg.exec()
        if dlg.point_cloud is None:
            return
        if self._dataset is None:
            return
        doc = Document(osp.split(file_path)[-1])
        doc.add_data(dlg.point_cloud)
        self._dataset.create_new_reference(doc)
        self.populate_tree()
        print(list(self._dataset._doc_refs))
        self.documentLoaded.emit(doc)

        
