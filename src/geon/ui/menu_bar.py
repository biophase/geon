from typing import cast

from PyQt6.QtWidgets import (QMenuBar, QMenu)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import pyqtSignal


class MenuBar(QMenuBar):
    # signlas
    setWorkdirRequested         = pyqtSignal()
    updateDocumentsRequested    = pyqtSignal()
    importFromRequested         = pyqtSignal()
    saveDocRequested            = pyqtSignal()
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
        
        # settings menu
        self.settings_menu = QMenu("&Settings", self)
        act_prefs = cast(QAction, self.settings_menu.addAction("Edit preferences"))
        act_prefs.triggered.connect(self.editPreferencesRequested)
        self.addMenu(self.settings_menu)
        
        # self.doc_menu.addAction("&Load")
        self.doc_menu.addSeparator()  
        import_menu = cast(QMenu, self.doc_menu.addMenu("Import document from ..."))
        act_import_from = cast(QAction, import_menu.addAction(".PLY"))
        self.doc_menu.addMenu(import_menu,)
        act_import_from.triggered.connect(self.importFromRequested)
        self.doc_menu.addAction("Export document to ...")
        self.addMenu(self.doc_menu)

        # about menu
        self.about_menu = QMenu("&About", self)
        act_about = cast(QAction, self.about_menu.addAction("About geon"))
        act_about.triggered.connect(self.aboutRequested)
        self.addMenu(self.about_menu)



        



        
        
        
