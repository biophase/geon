from PyQt6.QtWidgets import (QMenuBar, QMenu)

class MenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        
        # dataset menu
        self.dataset_menu = QMenu("&Dataset", self)
        self.dataset_menu.addAction("Set working directory")
        self.dataset_menu.addAction("Update documents")
        self.addMenu(self.dataset_menu)

        # document menu
        self.doc_menu = QMenu("&Scene",self)
        self.doc_menu.addAction("&Save")
        self.doc_menu.addAction("&Load")
        self.doc_menu.addSeparator()
        self.doc_menu.addAction("Import from")
        self.doc_menu.addAction("Export to")
        self.addMenu(self.doc_menu)


        



        
        
        