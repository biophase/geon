from PyQt6.QtWidgets import (QWidget, QDockWidget, QLabel, QToolButton, QHBoxLayout, QTreeWidget,
                             QVBoxLayout, QGridLayout,QPushButton,)

from config.theme import *

from PyQt6.QtCore import Qt, QSize
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


import vtk

import time


class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, 
                 renderer: vtk.vtkRenderer, 
                 viewer: "VTKViewer"):
        super().__init__()
        self._viewer = viewer
        self._renderer = renderer
        self.camera = renderer.GetActiveCamera()
        self.last_click_time = 0
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.left_button_press_event)
        self.AddObserver(vtk.vtkCommand.RightButtonPressEvent, self.right_button_press_event)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.mouse_move_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelForwardEvent, self.mouse_wheel_forward_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelBackwardEvent, self.mouse_wheel_backward_event)
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_event)


    def left_button_press_event(self, obj, event):
        ... # TODO: add tool hooks
        current_time = time.time()
        self.last_click_time = current_time
        self.OnLeftButtonDown()
    def right_button_press_event(self, obj, event):
        self.OnRightButtonDown()
        
    def mouse_move_event(self, obj, event):
        self.OnMouseMove()
    def mouse_wheel_forward_event(self, obj, event):
        ...
    def mouse_wheel_backward_event(self, obj, event):
        ...
    def key_press_event(self, obj, event):
        pass


class VTKViewer(QWidget):
    """
    VTK Widget
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)
        
        # vtk setup
        self._renderer: vtk.vtkRenderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self._renderer)
        self._renderer.SetBackground(DEFAULT_RENDERER_BACKGROUND)
        self._interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self._interactor.Initialize()
        self._interactor.SetInteractorStyle(InteractorStyle(self._renderer, self))
        self._interactor.Start()

        
        
        # common scene objects        
        self._pivot_point = [0,0,0]
        self._pivot_sphere_source = None
        self.pivot_actor = None
        
        
        self.vtkWidget.GetRenderWindow().Render()
        
        
        
        
    def rerender(self):
        self.vtkWidget.GetRenderWindow().Render()
        
    
    def focus_on_actor(self, actor: vtk.vtkProp):
        b = actor.GetBounds()
        self._renderer.ResetCamera(b)
        self._renderer.ResetCameraClippingRange()
        self.rerender()
        

