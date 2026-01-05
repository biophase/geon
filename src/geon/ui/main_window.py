from .dataset_manager import Dock, DatasetManager
from .scene_manager import SceneManager
from .viewer import VTKViewer
from .toolbar import CommonToolsDock
from .menu_bar import MenuBar
from .context_ribbon import ContextRibbon


from ..io.ply import ply_to_pcd
from ..tools.controller import ToolController
from ..ui.layers import LAYER_UI


from PyQt6.QtWidgets import (QMainWindow, QDialog, QFileDialog,  QApplication,  
                             QMenu)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence

from typing import cast

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("geon")
        QApplication.setApplicationName("geon")
        self.resize(1200,800)

        # settings
        self.setDockOptions(
                QMainWindow.DockOption.AllowTabbedDocks
            |   QMainWindow.DockOption.AllowNestedDocks
            |   QMainWindow.DockOption.GroupedDragging
        )        
        
        # widget initialization
        self.ribbon = ContextRibbon(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.ribbon)

        self.viewer = VTKViewer(self)
        self.setCentralWidget(self.viewer)
        
        self.tool_controller = ToolController(context_ribbon=self.ribbon)
        self.tool_controller.install_tool_schortcuts(self)
        self.scene_manager = SceneManager(self.viewer, self.tool_controller, self) 
        self.dataset_manager = DatasetManager(self)
        self.menu_bar = MenuBar(self)
        
        
        view_menu = cast(QMenu, self.menu_bar.addMenu("&View"))
        view_menu.addAction(self.scene_manager.toggleViewAction())
        view_menu.addAction(self.dataset_manager.toggleViewAction())

        ###########
        # signals #
        ###########
        
        self.scene_manager.broadcastDeleteScene\
            .connect(self.dataset_manager.save_scene_doc)

        self.dataset_manager.requestSetActiveDocInScene\
            .connect(self.scene_manager.on_document_loaded)
        
        self.menu_bar.setWorkdirRequested\
            .connect(self.dataset_manager.set_work_dir)
        self.menu_bar.importFromRequested\
            .connect(self.dataset_manager.import_doc_from_ply)
        self.menu_bar.saveDocRequested\
            .connect(lambda: self.dataset_manager.save_scene_doc(self.scene_manager._scene, ignore_state=True))
        self.menu_bar.undoRequested\
            .connect(lambda: self.tool_controller.command_manager.undo())
        self.menu_bar.redoRequested\
            .connect(lambda: self.tool_controller.command_manager.redo())
        
        self.tool_controller.tool_activated\
            .connect(lambda w: self.ribbon.set_group(self.tool_controller.active_tool_tooltip, w,'tool'))
        self.tool_controller.tool_activated\
            .connect(lambda _ :self.viewer.on_tool_activation(self.tool_controller.active_tool))
        self.tool_controller.tool_deactivated\
            .connect(lambda :self.viewer.on_tool_deactivation())
            
        self.scene_manager.broadcastActivatedLayer\
            .connect(self._on_layer_activated)
        self.scene_manager.broadcastActivatedPcdField\
            .connect(self._on_layer_activated)

            
        self.tool_controller.layer_internal_sel_changed\
            .connect(self._on_layer_internal_sel_changed)
        # self.tool_controller.tool_activated\
        #     .connect(lambda _: self.viewer.tool_active_frame.show())
        # self.tool_controller.tool_deactivated\
        #     .connect(lambda: self.viewer.tool_active_frame.hide())


        # built-in shortcuts
        pass
        # escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        # escape_shortcut.activated.connect(self.tool_controller.deactivate_tool)
        

        self.tool_dock = CommonToolsDock("Tools", self, self.tool_controller)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,self.tool_dock)
        
        # initial float widget placement
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_manager)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dataset_manager)
        # self.tabifyDockWidget(self.scene_widget, self.dataset_widget)


        ... # TODO: implement state, scene, undo manager, ui, tools, etc.   
        

        # temp test ply field dialog
              
        
        
    def _on_layer_activated(self, layer) -> None:
        hooks = LAYER_UI.resolve(layer)
        if hooks.ribbon_widget is None:
            self.ribbon.clear_group("layer")
            return
        title = getattr(layer, "browser_name", "Layer")
        widget = hooks.ribbon_widget(layer, self.ribbon, self.tool_controller)
        self.ribbon.set_group(title, widget, "layer")
        
    def _on_layer_internal_sel_changed(self, layer) -> None:
        hooks = LAYER_UI.resolve(layer)
        if hooks.ribbon_sel_widget is None:
            self.ribbon.clear_group('selection')
            return
        title = "Active selection"
        widget = hooks.ribbon_sel_widget(layer, self.ribbon, self.tool_controller)
        self.ribbon.set_group(title, widget, 'selection')
 
