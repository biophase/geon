from PyQt6.QtWidgets import (QWidget, QDockWidget, QLabel, QToolButton, QHBoxLayout, QTreeWidget,
                             QVBoxLayout, QGridLayout,QPushButton)

from config.theme import *

from PyQt6.QtCore import Qt, QSize, QTimer

import vtkmodules.qt

import sys
if sys.platform == 'darwin':
    vtkmodules.qt.QVTKRWIBase = "QOpenGLWidget"   

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
        
        # selector setup
        self._selector = vtk.vtkHardwareSelector()
        self._selector.SetRenderer(self._renderer)
        self._selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)
        self._selector_tolerance = 5
        
        
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.left_button_press_event)
        self.AddObserver(vtk.vtkCommand.RightButtonPressEvent, self.right_button_press_event)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.mouse_move_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelForwardEvent, self.mouse_wheel_forward_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelBackwardEvent, self.mouse_wheel_backward_event)
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_event)


    def left_button_press_event(self, obj, event):
        ... # TODO: add tool hooks
        
        
        current_time = time.time()
        if current_time - self.last_click_time < 0.3:
            self.double_click_event()
        else:
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
        # self._interactor.Start()

        
        
        # common scene objects        
        self._pivot_point = (0,0,0)
        self._pivot_sphere_source = None
        self.pivot_actor = None
        
        
        
        # Pivot marker shrink animation 
        self._pivot_shrink_timer = QTimer(self)
        self._pivot_shrink_timer.timeout.connect(self._update_pivot_shrink)
        self._pivot_marker_radius0 = 0.05      
        self._pivot_marker_duration_ms = 800   
        self._pivot_marker_elapsed_ms = 0
        self._pivot_shrink_dt_ms = 16          

        
        
        
        
        self.vtkWidget.GetRenderWindow().Render()
        
        
        
        
    def rerender(self):
        self.vtkWidget.GetRenderWindow().Render()
        
    
    def focus_on_actor(self, actor: vtk.vtkProp):
        b = actor.GetBounds()
        self._renderer.ResetCamera(b)
        self._renderer.ResetCameraClippingRange()
        self.rerender()
        
    def toggle_projection(self):
        camera = self._renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            camera.SetParallelProjection(False)
        else:
            camera.SetParallelProjection(True)
        self._renderer.ResetCamera()
        self.rerender()
        

        
    def set_pivot_point(self, new_pivot: tuple[float, float, float]):
        self._pivot_point = new_pivot

        self._renderer.GetActiveCamera().SetFocalPoint(*new_pivot)
        self._renderer.ResetCameraClippingRange()
        
        self.uodate_pivot_visualization(reset_radius=True)
        self._start_pivot_shrink()
        
        self.rerender()
        
    def uodate_pivot_visualization(self, reset_radius: bool = False):
        if self._pivot_sphere_source is None:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(self._pivot_marker_radius0)
            sphere.SetThetaResolution(16)
            sphere.SetPhiResolution(16)
            sphere.SetCenter(*self._pivot_point)
            self._pivot_sphere_source = sphere
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.5, 0.0 ,0.5)
            actor.GetProperty().SetOpacity(0.5)
            actor.GetProperty().LightingOff()
            
            self.pivot_actor = actor
            self._renderer.AddActor(actor)
            
        else:
            self._pivot_sphere_source.SetCenter(*self._pivot_point)
            if reset_radius:
                self._pivot_sphere_source.SetRadius(self._pivot_marker_radius0)
                
    def _start_pivot_shrink(self):
        self._pivot_marker_elapsed_ms = 0
        if self._pivot_shrink_timer.isActive():
            self._pivot_shrink_timer.stop()
        self._pivot_shrink_timer.start(self._pivot_shrink_dt_ms)
        
    def _update_pivot_shrink(self):
        if self._pivot_sphere_source is None:
            self._pivot_shrink_timer.stop()
            return
        self._pivot_marker_elapsed_ms += self._pivot_shrink_dt_ms
        t = min(1., self._pivot_marker_elapsed_ms / float(self._pivot_marker_duration_ms))
        
        radius = self._pivot_marker_radius0 * (1.0 - t) ** 2
        self._pivot_sphere_source.SetRadius(max(0.0, radius))
        
        if t >= 1.0 or radius <= 1e-6:
            if self.pivot_actor is not None:
                self._renderer.RemoveActor(self.pivot_actor)
            self.pivot_actor = None
            self._pivot_sphere_source = None
            self._pivot_shrink_timer.stop()
            
        self.rerender()
                
        

            
            
            

