from PyQt6.QtGui import QMouseEvent, QResizeEvent
from PyQt6.QtWidgets import (QWidget, QDockWidget, QLabel, QToolButton,QHBoxLayout,
                             QTreeWidget,QVBoxLayout, QGridLayout,QPushButton,
                             QFrame
                             )

from config.theme import *
from .picker import PointPicker

from geon.tools.base import ModeTool, Event, BaseTool

from geon.rendering.base import BaseLayer
from geon.rendering.scene import Scene


from PyQt6.QtCore import Qt, QSize, QTimer

import vtkmodules.qt

import sys
if sys.platform == 'darwin':
    vtkmodules.qt.QVTKRWIBase = "QOpenGLWidget"   

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


import vtk

from typing import Optional, cast
import time
from dataclasses import dataclass        


@dataclass
class PickResult:
    layer: BaseLayer | None
    prop: vtk.vtkProp| None
    element_idx: int | None
    world_xyz: tuple[float,float,float] | None

class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, 
                 renderer: vtk.vtkRenderer, 
                 viewer: "VTKViewer"):
        super().__init__()   
        
        # references
        self._viewer = viewer
        self._renderer = renderer
        self._camera = renderer.GetActiveCamera()
        
        # state
        self.camera_enabled: bool = True
        self.last_click_time = 0
        self.mode_tool: Optional[ModeTool] = None
        
        # default state override
        self.SetMotionFactor(10.0)
        
        # observers
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.left_button_press_event)
        self.AddObserver(vtk.vtkCommand.RightButtonPressEvent, self.right_button_press_event)
        self.AddObserver(vtk.vtkCommand.MiddleButtonPressEvent, self.middle_button_press_event)
        self.AddObserver(vtk.vtkCommand.RightButtonReleaseEvent, self.right_button_release_event)
        self.AddObserver(vtk.vtkCommand.MiddleButtonReleaseEvent, self.middle_button_release_event)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.mouse_move_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelForwardEvent, self.mouse_wheel_forward_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelBackwardEvent, self.mouse_wheel_backward_event)
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_event)
        self.AddObserver(vtk.vtkCommand.KeyReleaseEvent, self.key_release_event)
        
        # silence default charecter presses
        self.AddObserver(vtk.vtkCommand.CharEvent, self.silence_vtk_defaults, 1.0) 
        
        


    @property
    def event_info(self) -> Event:
        interactor = self.GetInteractor()
        event = Event(
            pos = interactor.GetEventPosition(),
            prev_pos = interactor.GetLastEventPosition(),
            shift = bool(interactor.GetShiftKey()),
            ctrl= bool(interactor.GetControlKey()),
            alt= bool(interactor.GetAltKey()),
            key = interactor.GetKeySym(),
        )
        return event
    
    # events
    def left_button_press_event(self, _vtk_obj, _vtk_event):
        
        current_time = time.time()
        if current_time - self.last_click_time < 0.3:
            self.double_click_event()
        else:
            self.last_click_time = current_time
            if self.mode_tool is not None:
                self.mode_tool.left_button_press_hook(self.event_info)
            if self.camera_enabled:
                self.OnLeftButtonDown()
                
    def double_click_event(self):
        if self.mode_tool is not None:
            self.mode_tool.double_click_press_hook(self.event_info)
        if self.camera_enabled:
            self._focus_camera()
        
    def right_button_press_event(self, _vtk_obj, _vtk_event):
        if self.mode_tool is not None:
            self.mode_tool.right_button_press_hook(self.event_info)        
        if self.camera_enabled:
            self.OnRightButtonDown()
        
    def middle_button_press_event(self, _vtk_obj, _vtk_event):
        if self.mode_tool is not None:
            self.mode_tool.middle_button_press_hook(self.event_info)
        if self.camera_enabled:
            self.OnMiddleButtonDown()

    def right_button_release_event(self, _vtk_obj, _vtk_event):
        if self.camera_enabled:
            self.OnRightButtonUp()

    def middle_button_release_event(self, _vtk_obj, _vtk_event):
        if self.camera_enabled:
            self.OnMiddleButtonUp()

    def mouse_move_event(self, _vtk_obj, _vtk_event):
        if self.mode_tool is not None:
            self.mode_tool.mouse_move_event_hook(self.event_info)
        if self.camera_enabled:
            self.OnMouseMove()
            
    def mouse_wheel_forward_event(self, _vtk_obj, _vtk_event):
        if self.mode_tool is not None:
            self.mode_tool.mouse_wheel_forward_hook(self.event_info)
        if self.camera_enabled:
            self.OnMouseWheelForward()

    def mouse_wheel_backward_event(self, _vtk_obj, _vtk_event):
        if self.mode_tool is not None:
            self.mode_tool.mouse_wheel_backward_hook(self.event_info)
        if self.camera_enabled:
            self.OnMouseWheelBackward()
        
    def key_press_event(self, _vtk_obj, _vtk_event):
        # return
        print(f"[Interactor] key press event")
        if self.mode_tool is not None:
            self.mode_tool.key_press_hook(self.event_info)

    def key_release_event(self, _vtk_obj, _vtk_event):
        print(f"[Interactor] key release event")
        if self.mode_tool is not None:
            self.mode_tool.key_release_hook(self.event_info)



            

    
    def _focus_camera(self):
        result = self._viewer.pick()
        if result.world_xyz is None:
            return
        self._viewer.set_pivot_point(result.world_xyz)
        # if self._viewer.scene is None:
        #     return
        # x, y = self.GetInteractor().GetEventPosition()
        # picker = self._viewer.picker
        # result = picker.pick(self._viewer._interactor, x, y)
        # if result is not None:
        #     picked_prop, point_id = result
        #     if picked_prop is not None:
        #         layer = self._viewer.scene.layer_for_prop(picked_prop)
        #         if layer is None:
        #             print(f"DEBUG: no layer found for prop {picked_prop}")
        #             return
        #         world_xyz = layer.world_xyz_from_picked_id(point_id)
        #         self._viewer.set_pivot_point(world_xyz)
        
    #############
    # overrides # 
    #############
    
    
    def OnMiddleButtonDown(self):
        print(f"[OnMiddleButtonDown] called")
        # treat middle press as right press
        super().OnRightButtonDown()

    def OnMiddleButtonUp(self):
        print(f"[OnMiddleButtonUp] called")
        super().OnRightButtonUp()

    def OnRightButtonDown(self):
        print(f"[OnRightButtonDown] called")
        # treat right press as middle press
        super().OnMiddleButtonDown()

    def OnRightButtonUp(self):
        print(f"[OnRightButtonUp] called")
        super().OnMiddleButtonUp()
        
    def silence_vtk_defaults(self, obj, event):
        # override the char handler
        try:
            obj.AbortFlagOn()
        except:
            pass
        return
    
class VTKViewer(QWidget):
    """
    VTK Widget
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.scene: Optional[Scene] = None
        layout.addWidget(self.vtkWidget)
        
        # vtk setup
        self._renderer: vtk.vtkRenderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self._renderer)
        self._renderer.SetBackground(DEFAULT_RENDERER_BACKGROUND)
        self._interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self._interactor.Initialize()
        self._interactor_style = InteractorStyle(self._renderer, self)
        self._interactor.SetInteractorStyle(self._interactor_style)
        self._edl_pass: Optional[vtk.vtkEDLShading] = None
        
        self.picker = PointPicker(self._renderer)
        
       
        # common scene objects        
        self._pivot_point = (0,0,0)
        self._pivot_sphere_source = None
        self.pivot_actor = None
        
        
        
        
        # Pivot marker shrink animation 
        self._pivot_shrink_timer = QTimer(self)
        self._pivot_shrink_timer.timeout.connect(self._update_pivot_shrink)
        self._pivot_marker_radius0 = 1.0      
        self._pivot_marker_duration_ms = 800   
        self._pivot_marker_elapsed_ms = 0
        self._pivot_shrink_dt_ms = 16          
        
        # Qt Settings
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # indicator frame (e.g. at tool activation)
        # self.tool_active_frame = QFrame(self.vtkWidget)

        # self.tool_active_frame.setAutoFillBackground(False)
        # self.tool_active_frame.setStyleSheet(
        #     "QFrame { border: 6px solid rgba(0, 255, 0, 160); }"
        # )
        # self.tool_active_frame.setAttribute(
        #     Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # self.tool_active_frame.hide()


        

        self.vtkWidget.GetRenderWindow().Render()
        

        
        
        
    def rerender(self):
        self.vtkWidget.GetRenderWindow().Render()
        
    def pick(self) -> PickResult:
        interactor = self._interactor_style.GetInteractor()
        x, y = interactor.GetEventPosition()
        result = self.picker.pick(interactor, x, y)
        if result is None or self.scene is None:
            return PickResult(None, None, None, None)
        picked_prop, visible_id = result

        layer = None
        if picked_prop is not None :
            layer = self.scene.layer_for_prop(picked_prop)
            if layer is None:
                print(f'No layer found for prop {picked_prop}')
                return PickResult(None, picked_prop, visible_id, None)
            world_xyz = layer.world_xyz_from_picked_id(visible_id)
            data_id = layer.data_index_from_picked_id(visible_id)
            return PickResult(layer, picked_prop, data_id, world_xyz)
        else:
            return PickResult(None, None, None, None)


        
        
        

    def enable_edl(self) -> None:
        basic_passes = vtk.vtkRenderStepsPass()
        edl_pass = vtk.vtkEDLShading()
        edl_pass.SetDelegatePass(basic_passes)

        gl_renderer = vtk.vtkOpenGLRenderer.SafeDownCast(self._renderer)
        if gl_renderer is None:
            raise TypeError("Renderer must be an OpenGL renderer (vtkOpenGLRenderer).")

        gl_renderer.SetPass(edl_pass)
        gl_renderer.Modified()
        self._edl_pass = edl_pass
        self.rerender()

    def disable_edl(self) -> None:
        gl_renderer = vtk.vtkOpenGLRenderer.SafeDownCast(self._renderer)
        if gl_renderer is None:
            return
        basic_passes = vtk.vtkRenderStepsPass()
        gl_renderer.SetPass(basic_passes)
        gl_renderer.Modified()
        self._edl_pass = None
        self.rerender()

    def toggle_edl(self) -> None:
        if self._edl_pass is None:
            self.enable_edl()
        else:
            self.disable_edl()
        
    def set_camera_enabled(self, enabled: bool) -> None:
        self._interactor_style.camera_enabled = enabled

    def set_camera_sensitivity(self, value: float) -> None:
        self._interactor_style.SetMotionFactor(float(value))
    
    def focus_camera_on_actor(self, actor: vtk.vtkProp):
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
        # self._renderer.ResetCamera()
        self.rerender()
        

        
    def set_pivot_point(self, new_pivot: tuple[float, float, float]):
        self._pivot_point = new_pivot

        self._renderer.GetActiveCamera().SetFocalPoint(*new_pivot)
        self._renderer.ResetCameraClippingRange()
        
        self.update_pivot_visualization(reset_radius=True)
        self._start_pivot_shrink()
        
        self.rerender()

    def on_tool_activation(self, tool: Optional[BaseTool]) -> None:
        if isinstance (tool, ModeTool) or (tool is None):
            self._interactor_style.mode_tool = tool
            if tool is not None:
                if tool.keep_focus:
                    cast(QWidget, self.vtkWidget).grabKeyboard()
        
        # always set focus on viewer    
        cast(QWidget, self.vtkWidget).setFocus(Qt.FocusReason.OtherFocusReason)
        
        
        
            
    def on_tool_deactivation(self) -> None:
        cast(QWidget, self.vtkWidget).releaseKeyboard()
        self._interactor_style.mode_tool = None
        
    def update_pivot_visualization(self, reset_radius: bool = False):
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
        
        radius = self._pivot_marker_radius0 * (1.0 - t) ** 3
        self._pivot_sphere_source.SetRadius(max(0.0, radius))
        
        if t >= 1.0 or radius <= 1e-6:
            if self.pivot_actor is not None:
                self._renderer.RemoveActor(self.pivot_actor)
            self.pivot_actor = None
            self._pivot_sphere_source = None
            self._pivot_shrink_timer.stop()
            
        self.rerender()
        
    # def on_tool_activation(self) -> None:
    #     cast(QWidget, self.vtkWidget).setFocus(Qt.FocusReason.OtherFocusReason)

    
    # def mousePressEvent(self, a0: QMouseEvent | None) -> None:
    #     super().mousePressEvent(a0)
    #     if self._keep_focus:
    #         cast(QWidget, self.vtkWidget).setFocus(Qt.FocusReason.OtherFocusReason)
    
    # def resizeEvent(self, a0: QResizeEvent | None) -> None:
    #     event = a0
    #     super().resizeEvent(event)
    #     self.tool_active_frame.setGeometry(self.rect())
        
                
        

            
            
            
