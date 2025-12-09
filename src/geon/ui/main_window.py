from .tree import Dock, DatasetManager
from .viewer import VTKViewer
from .common_tools import CommonToolsDock
from .menu_bar import MenuBar
from .context_ribbon import ContextRibbon

from geon.io.ply import ply_to_pcd


from geon.rendering.scene import Scene
from PyQt6.QtWidgets import (QMainWindow, QDialog, QFileDialog)
from PyQt6.QtCore import Qt


from plyfile import PlyData
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("geon")
        self.resize(1200,800)

        # content
        self.scene: Scene = Scene()

        # settings
        self.setDockOptions(
                QMainWindow.DockOption.AllowTabbedDocks
            |   QMainWindow.DockOption.AllowNestedDocks
            |   QMainWindow.DockOption.GroupedDragging
        )        
        
        # widget initialization
        self.ribbon = ContextRibbon(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.ribbon)
        self.scene_widget = Dock("Scene", self) # TODO: might need to specialize the tree class
        self.dataset_manager = DatasetManager(self)
        # self.dataset_manager.documentLoaded.connect(self.scene.set_document)
        
        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.menu_bar.setWorkdirRequested.connect(
            self.dataset_manager.set_work_dir)
        self.menu_bar.importFromRequested.connect(
            self.dataset_manager.import_doc_from_ply
        )
        

        
        
        self.viewer = VTKViewer(self)
        self.setCentralWidget(self.viewer)

        self.tool_dock = CommonToolsDock("Tools", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,self.tool_dock)


        
        # initial float widget placement
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dataset_manager)
        # self.tabifyDockWidget(self.scene_widget, self.dataset_widget)


        ... # TODO: implement state, scene, undo manager, ui, tools, etc.   
        

        # temp test ply field dialog

        
        
 