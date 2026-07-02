from typing import cast

from PyQt6.QtWidgets import (QMenuBar, QMenu)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import pyqtSignal


class MenuBar(QMenuBar):
    # signlas
    setWorkdirRequested         = pyqtSignal()
    updateDocumentsRequested    = pyqtSignal()
    importFromRequested         = pyqtSignal()
    importCellComplexTxtRequested = pyqtSignal()
    exportPointCloudPlyRequested = pyqtSignal()
    saveDocRequested            = pyqtSignal()
    renderToFileRequested       = pyqtSignal()
    renameSceneRequested        = pyqtSignal()
    createCameraSnapshotRequested = pyqtSignal()
    importCameraSnapshotJsonRequested = pyqtSignal()
    createEmptyBoundingBoxLayerRequested = pyqtSignal()
    undoRequested               = pyqtSignal()
    redoRequested               = pyqtSignal()
    editPreferencesRequested    = pyqtSignal()
    aboutRequested              = pyqtSignal()
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # dataset menu
        self.dataset_menu = QMenu("D&ataset", self)
        act_set = cast(QAction, self.dataset_menu.addAction("Set working directory"))
        act_set.triggered.connect(self.setWorkdirRequested)
        
        act_update = cast(QAction,self.dataset_menu.addAction("Update documents"))
        act_update.triggered.connect(self.updateDocumentsRequested)

        self.addMenu(self.dataset_menu)
        
        # edit menu
        self.edit_menu = QMenu("&Edit", self)
        act_undo = cast(QAction, self.edit_menu.addAction("&Undo"))
        act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        act_undo.triggered.connect(self.undoRequested)
        
        act_redo = cast(QAction, self.edit_menu.addAction("&Redo"))
        act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        act_redo.triggered.connect(self.redoRequested)        
        self.addMenu(self.edit_menu)
        
        # document menu
        self.doc_menu = QMenu("&Document",self)
        act_save_doc = cast(QAction, self.doc_menu.addAction("&Save"))
        act_save_doc.setShortcut(QKeySequence.StandardKey.Save)
        act_save_doc.triggered.connect(self.saveDocRequested)
        self.scene_menu = cast(QMenu, self.doc_menu.addMenu("Scene"))
        act_rename_scene = cast(QAction, self.scene_menu.addAction("Rename..."))
        act_rename_scene.triggered.connect(self.renameSceneRequested)
        act_camera_snapshot = cast(QAction, self.scene_menu.addAction("Camera snapshot from current"))
        act_camera_snapshot.triggered.connect(self.createCameraSnapshotRequested)
        act_import_camera_snapshot = cast(QAction, self.scene_menu.addAction("Import camera snapshot from JSON..."))
        act_import_camera_snapshot.triggered.connect(self.importCameraSnapshotJsonRequested)

        # display menu
        self.display_menu = QMenu("&Display", self)
        act_render_to_file = cast(QAction, self.display_menu.addAction("Render to File"))
        act_render_to_file.triggered.connect(self.renderToFileRequested)
        self.addMenu(self.display_menu)

        # layer menu
        self.layer_menu = QMenu("&Layer", self)
        bbox_menu = cast(QMenu, self.layer_menu.addMenu("Bounding box"))
        act_empty_bbox = cast(QAction, bbox_menu.addAction("Create empty layer"))
        act_empty_bbox.triggered.connect(self.createEmptyBoundingBoxLayerRequested)
        self.addMenu(self.layer_menu)
        
        # settings menu
        self.settings_menu = QMenu("&Settings", self)
        act_prefs = cast(QAction, self.settings_menu.addAction("Edit preferences"))
        act_prefs.triggered.connect(self.editPreferencesRequested)
        self.addMenu(self.settings_menu)
        
        # self.doc_menu.addAction("&Load")
        self.doc_menu.addSeparator()  
        import_menu = cast(QMenu, self.doc_menu.addMenu("Import document from ..."))
        act_import_from = cast(QAction, import_menu.addAction(".PLY"))
        act_import_cell_txt = cast(QAction, import_menu.addAction(".TXT CellComplex"))
        self.doc_menu.addMenu(import_menu,)
        act_import_from.triggered.connect(self.importFromRequested)
        act_import_cell_txt.triggered.connect(self.importCellComplexTxtRequested)
        export_menu = cast(QMenu, self.doc_menu.addMenu("Export active layer to ..."))
        act_export_ply = cast(QAction, export_menu.addAction(".PLY PointCloud"))
        act_export_ply.triggered.connect(self.exportPointCloudPlyRequested)
        self.addMenu(self.doc_menu)

        # about menu
        self.about_menu = QMenu("&About", self)
        act_about = cast(QAction, self.about_menu.addAction("About geon"))
        act_about.triggered.connect(self.aboutRequested)
        self.addMenu(self.about_menu)



        



        
        
        
