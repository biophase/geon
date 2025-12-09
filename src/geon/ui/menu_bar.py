from typing import cast

from PyQt6.QtWidgets import (QMenuBar, QMenu)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import pyqtSignal


class MenuBar(QMenuBar):
    # signlas
    setWorkdirRequested         = pyqtSignal()
    updateDocumentsRequested    = pyqtSignal()
    importFromRequested         = pyqtSignal()
    def __init__(self, parent):
        super().__init__(parent)
        
        # dataset menu
        
        self.dataset_menu = QMenu("&Dataset", self)
        act_set = cast(QAction, self.dataset_menu.addAction("Set working directory"))
        act_set.triggered.connect(self.setWorkdirRequested)
        
        act_update = cast(QAction,self.dataset_menu.addAction("Update documents"))
        act_update.triggered.connect(self.updateDocumentsRequested)

        self.addMenu(self.dataset_menu)

        

        # document menu
        self.doc_menu = QMenu("&Scene",self)
        self.doc_menu.addAction("&Save")
        self.doc_menu.addAction("&Load")
        self.doc_menu.addSeparator()  
        act_import_from = cast(QAction, self.doc_menu.addAction("Import from"))
        act_import_from.triggered.connect(self.importFromRequested)
        self.doc_menu.addAction("Export to")
        self.addMenu(self.doc_menu)




        



        
        
        
