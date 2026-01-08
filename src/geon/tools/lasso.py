from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,)
from PyQt6.QtCore import QSize
from PyQt6.QtCore import Qt


import geon.core
from ..util.common import bool_op_index_mask
from .base import Event
from .base import ModeTool, ToolZone
from .registry import TOOL_REGISTRY
from .command_manager import Command
from ..data.pointcloud import PointCloudData
from ..rendering.pointcloud import PointCloudLayer
from ..ui.boolean_dialog import BooleanChoiceDialog
from .selection import SelectPointsCmd

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union, cast, Literal, overload
import weakref
from enum import Enum

import numpy as np
import vtk

from geon.util.resources import resource_path


@dataclass
class LassoApplyPolyCmd(Command):
    cand_inds_old: np.ndarray   = field(init=False)
    sd_old: bool                = field(init=False)
    
    cand_inds_new:  np.ndarray
    sd_new: bool
    layer_ref: weakref.ReferenceType[PointCloudLayer]
    tool_ref: weakref.ReferenceType["LassoTool"]
    
    def execute(self) -> None:
        layer = self.layer_ref()
        tool = self.tool_ref()
        if layer is None or tool is None:
            return
        
        # capture state
        self.cand_inds_old = tool.cand_inds
        self.sd_old = tool.selection_dirty
        
        # set state & layer visibility
        tool.poly_points = []
        tool.cand_inds = self.cand_inds_new
        full_mask = np.zeros(tool.points.shape[0], dtype=bool)
        full_mask[self.cand_inds_new] = True
        tool.layer.set_visibility_mask(full_mask)
        tool.selection_dirty = self.sd_new
        tool.ctx.viewer.rerender()
        
        
        
    def undo(self) -> None:
        layer = self.layer_ref()
        tool = self.tool_ref()
        if layer is None or tool is None:
            return
        
        # restore state
        tool.cand_inds = self.cand_inds_old
        tool.selection_dirty = self.sd_old
        
        # layer visibilitys
        tool.poly_points = []
        full_mask = np.zeros(tool.points.shape[0], dtype=bool)
        full_mask[self.cand_inds_old] = True
        layer.set_visibility_mask(full_mask)
        tool.ctx.viewer.rerender()
        


@dataclass
class LassoTool(ModeTool):
    # metadata
    label: ClassVar = 'lasso'
    tooltip: ClassVar = "Lasso segmentation"
    icon_path: ClassVar = resource_path('lasso.png')
    shortcut: ClassVar = 't'
    ui_zones: ClassVar = {ToolZone.SIDEBAR_RIGHT_ESSENTIALS}
    use_local_cm: ClassVar[bool] = True
    show_in_toolbar: ClassVar[bool] = True
    cursor_icon_path : ClassVar = resource_path('lasso.png')
    # cursor_hot: ClassVar = (64, 181) 
    cursor_hot: ClassVar = (8, 20) 
    
    # mode tool meta
    keep_focus: ClassVar[bool] = True
    
    # lasso state
    control_panel: Optional[QWidget] = field(init=False)
    # select_active = True
    select_paused = False
    
    poly_points: list[tuple[int,int]] = field(default_factory=list)
    cand_inds: np.ndarray = field(init=False)
    points: np.ndarray = field(init=False)
    vis_mask_backup: Optional[np.ndarray] = None
    selection_backup: Optional[np.ndarray] = None
    polygon_actor: Optional[vtk.vtkActor2D] = None
    layer: PointCloudLayer = field(init=False) 
    ghost_point: Optional[tuple[int,int]] = None
    selection_conflict_flag: geon.core.Boolean = geon.core.Boolean.OVERWRITE
    selection_dirty: bool = False
        
    # style settings
    POLYGON_COLOR = (1., 0.5, 0)
    POLYGON_LINEWIDTH = 1
    
    
    def __post_init__(self):
        super().__post_init__()
        
        al = self.ctx.scene.active_layer
        if al is None:
            return
        
        
        if type(al) == PointCloudLayer:
            self.layer = al
            data = al.data
            mask = self.layer._visibility_mask
            if mask is None:
                self.cand_inds = np.arange(len(data.points))
            else:
                self.cand_inds = np.nonzero(mask)[0]
                self.vis_mask_backup = mask.copy()
                
            if self.layer.active_selection is not None:
                self.selection_backup = self.layer.active_selection.copy()
            self.points = data.points
            
            
            
        else:
            raise NotImplementedError(f"Lasso is not implemented for {type(al)}")
            return
    

    def _click_event(self, event: Event) -> None:
        if self.select_paused:
           return
        self.poly_points.append(event.pos)
        self._update_polygon_actor()
        
    def _mouse_move_event(self, event:Event) -> None:
        if not self.select_paused and len(self.poly_points) > 0:
            self._update_polygon_actor(ghost_point=event.pos)
            self.ghost_point = event.pos


        
    def _update_polygon_actor(self, ghost_point: Optional[tuple[int,int]]=None
                              ) -> None:
        self._remove_polygon_actor()
        pts = list(self.poly_points)
        if ghost_point is not None:
            pts.append(ghost_point)
        if len(pts) < 2:
            self.ctx.viewer.rerender()
            return
        
        pts.append(pts[0])
        vtk_pts = vtk.vtkPoints()
        polyline = vtk.vtkPolyLine()
        
        polyline.GetPointIds().SetNumberOfIds(len(pts))
        for i, (x, y) in enumerate(pts):
            vtk_pts.InsertNextPoint(x,y,0)
            polyline.GetPointIds().SetId(i,i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_pts)
        polydata.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*self.POLYGON_COLOR)
        actor.GetProperty().SetLineWidth(self.POLYGON_LINEWIDTH)
        self.ctx.viewer._renderer.AddActor(actor)
        self.ctx.viewer.rerender()
        self.polygon_actor = actor
        
        
        
    def _toggle_select_pause(self) -> None:
        self.select_paused = not(self.select_paused)        
        if self.select_paused:
            self.poly_points = []
            self._remove_polygon_actor()
        self.ctx.viewer.set_camera_enabled(self.select_paused)
        self.ctx.viewer.rerender()     
            
    def _remove_polygon_actor(self) -> None:
        if self.polygon_actor is not None:
            self.ctx.viewer._renderer.RemoveActor(self.polygon_actor)
    
    def _sPause(self)->None:
        print(f"called _sPause")
        self._toggle_select_pause()

        
    def _sActivate(self)->None:
        print(f"called _sActivate")
        pass
    
    def _sIn(self) -> None:
        print(f"called _sIn")
        self._sApply(mode='in')
        self._remove_polygon_actor()
        self._toggle_select_pause()
        self.ctx.viewer.rerender()
        
    
    def _sOut(self) -> None:
        print(f"called _sOut")
        self._sApply(mode='out')
        self._remove_polygon_actor()
        self._toggle_select_pause()
        self.ctx.viewer.rerender()
    
    def _sAccept(self) -> None:
        """
        Accept with conflict resolution
        """
        self.request_deactivate()
    
    def _sCancel(self) -> None:
        """
        Throw away mask
        """
        self.selection_dirty = False
        self.request_deactivate()
        
    
    def _sRemoveLastPt(self) -> None:
        print('called remove last')
        if len(self.poly_points) >1:
            self.poly_points.pop(-1)
            self._update_polygon_actor(self.ghost_point)
        self.ctx.viewer.rerender()

    def _sApply (self, mode:Literal['in','out'])-> None:
        """
        points in current polygone are set to be candidates
        other points are hidden until tool deactivation
        """
        self.layer.active_selection = None
        poly_ = self.poly_points 
        if self.ghost_point is not None:
            poly_ += [self.ghost_point]
        if len(poly_) < 3:
            return
        poly = np.asarray(poly_)
        cand_inds = self.cand_inds
        cand_pts = self.points[cand_inds]
        m, width, height = LassoTool._get_cptm(self.ctx.viewer._renderer)
        
        display_coords = self._compute_display_coordinates(cand_pts, m, width, height)
        min_x, max_x = poly[:, 0].min(), poly[:, 0].max()
        min_y, max_y = poly[:, 1].min(), poly[:, 1].max()
        bbox_mask = (
            (display_coords[:, 0] >= min_x) & (display_coords[:, 0] <= max_x) &
            (display_coords[:, 1] >= min_y) & (display_coords[:, 1] <= max_y)
        )
        inds_in_bbox = np.nonzero(bbox_mask)[0]
        filtered_display_coords = display_coords[inds_in_bbox]
        inside_mask = self.points_in_poly_(
            filtered_display_coords, self.poly_points)
        if mode == 'in':
            keep_mask = np.zeros(cand_inds.shape[0], dtype=bool)
            keep_mask[inds_in_bbox] = inside_mask
        elif mode == 'out':
            keep_mask = np.ones(cand_inds.shape[0], dtype=bool)
            keep_mask[inds_in_bbox] = ~inside_mask
        else:
            raise ValueError(f"Unsupported selection mode: {mode}")
        new_cands = cand_inds[keep_mask]
        
        
        cmd = LassoApplyPolyCmd(
            title=f'Lasso {mode}',
            cand_inds_new=new_cands,
            sd_new=True,
            layer_ref=weakref.ref(self.layer),
            tool_ref=weakref.ref(self)
            )
        
        self.command_manager.do(cmd)
        

    @staticmethod
    def _get_cptm( 
                  renderer:vtk.vtkRenderer,
                  
                  ) -> tuple[np.ndarray, int, int]:
        """
        computes the view projection matrix as well as the window size
        """
        # renderer = self.ctx.viewer._renderer
        camera = renderer.GetActiveCamera()
        # window_size = self.ctx.viewer.vtkWidget.GetRenderWindow().GetSize()
        window_size = renderer.GetRenderWindow().GetSize()
        width, height = window_size
        clipping_range = camera.GetClippingRange()
        aspect = renderer.GetTiledAspectRatio()
        matrix = camera.GetCompositeProjectionTransformMatrix(
            aspect, clipping_range[0],clipping_range[1])
        m = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                m[i,j] = matrix.GetElement(i,j)
        return m, width, height



    @staticmethod
    def _compute_display_coordinates( 
                                     points: np.ndarray,
                                     proj_matrix: np.ndarray,
                                     window_width: int,
                                     window_height: int,
                                     ) -> np.ndarray:

        N = points.shape[0]
        pts_hom = np.hstack([points, np.ones((N,1))])
        pts_clip = pts_hom @ proj_matrix.T
        w = pts_clip[:, 3]
        pts_ndc = np.full((N, 3), np.nan, dtype=float)
        valid = w > 0
        pts_ndc[valid] = pts_clip[valid, :3] / w[valid, None]
        display_x = (pts_ndc[:,0] + 1) /2 * window_width
        display_y = (pts_ndc[:,1] + 1) /2 * window_height
        return np.column_stack([display_x, display_y])

    @staticmethod
    def points_in_poly(points:np.ndarray, poly: list[tuple[int,int]]):
        poly_ = np.asarray(poly)
        x = points[:, 0]
        y = points[:, 1]
        poly_x = poly_[:, 0]
        poly_y = poly_[:, 1]
        poly_x_next = np.roll(poly_x, -1)
        poly_y_next = np.roll(poly_y, -1)
        cond = ((poly_y > y[:, None]) != (poly_y_next > y[:, None])) & (
            x[:, None] < (poly_x_next - poly_x) * (y[:, None] - poly_y) /
            (poly_y_next - poly_y + 1e-12) + poly_x
        )
        inside = np.count_nonzero(cond, axis=1) % 2 == 1
        return inside
                
    @staticmethod
    def points_in_poly_(points:np.ndarray, poly: list[tuple[int,int]]) -> np.ndarray:
        """
        memory-reduced points-in-polygon test
        """
        poly_arr = np.asarray(poly)
        if poly_arr.shape[0] < 3:
            return np.zeros(points.shape[0], dtype=bool)
        counts = np.zeros(points.shape[0], dtype=np.uint32)
        x = points[:, 0]
        y = points[:, 1]
        poly_x = poly_arr[:, 0]
        poly_y = poly_arr[:, 1]
        poly_x_next = np.roll(poly_x, -1)
        poly_y_next = np.roll(poly_y, -1)
        for x0, y0, x1, y1 in zip(poly_x, poly_y, poly_x_next, poly_y_next):
            cond = ((y0 > y) != (y1 > y)) & (
                x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
            )
            counts += cond
        inside = (counts & 1) == 1
        return inside
            
        
    #############
    # overrides #
    #############

    def create_context_widget(self, parent: QWidget) -> QWidget | None:
        w = QWidget(parent)
        outer = QHBoxLayout(w)
        outer.setContentsMargins(2, 1, 2, 1)
        outer.setSpacing(2)
        
        button_params = [
            [ # col 1
                ('In', self._sIn),
                ('Out',self._sOut),
            ],
            [ # col 2
                ('Accept', self._sAccept),
                ('Cancel', self._sCancel),            
            ],
            [ # col 3
                ('Pause', self._sPause),
                ('Remove', self._sRemoveLastPt),
            ],
        ]
        

        cols = [QVBoxLayout() for _ in range(len(button_params))]
        
        
        for btn_params, col in zip(button_params,cols):
            col.setAlignment(Qt.AlignmentFlag.AlignTop)
            for btn_param in btn_params:
                text, func = btn_param
                btn = QPushButton()
                btn.setStyleSheet("QPushbutton { font-size: 9px}")
                btn.setFixedHeight(18)
                
                # btn.setStyleSheet("QPushButton { padding: 0 12px; }")
                btn.setText(text)
                # btn.setFixedSize(QSize(48, 24))
                # btn.setFlat(True)
                btn.clicked.connect(func)
                col.addWidget(btn)
            
        for col in cols:
            outer.addLayout(col)

        self.control_panel = w
        return w
        
        

    
    def activate(self) -> None:
        print(f'activated lasso')
        super().activate()
        self.ctx.viewer.set_camera_enabled(False)

        
    def request_deactivate(self) -> None:
        if self.selection_backup is None or\
            self.selection_backup.size == 0 or\
                self.selection_dirty == False:
                self.ctx.controller.deactivate_tool()
                return
        else:
            self._remove_polygon_actor()
            self.ctx.viewer.set_camera_enabled(True)
            dlg = BooleanChoiceDialog(
                message="Choose how to combine with previous selection:")
            dlg.exec()
            if dlg.choice is not None:
                self.selection_conflict_flag = dlg.choice
                self.ctx.controller.deactivate_tool()
            
    def deactivate(self) -> None:

        self._remove_polygon_actor()
        self.ctx.viewer.set_camera_enabled(True)
        self.layer.set_visibility_mask(self.vis_mask_backup)

        if self.selection_dirty:
            if self.selection_backup is not None:
                # resolve conflict with prior selection
                self.cand_inds = bool_op_index_mask(
                    self.selection_backup, 
                    self.cand_inds,
                    self.selection_conflict_flag
                    )
            cmd = SelectPointsCmd(
                title="Lasso selection",
                selection_new=self.cand_inds.copy(),
                layer_ref=weakref.ref(self.layer),
                ctx_ref=weakref.ref(self.ctx),
                selection_old=self.selection_backup
            )

            # finilize command from global command manager
            self.ctx.controller.do_global(cmd)
        self.ctx.viewer.rerender()
        super().deactivate()
    
    def left_button_press_hook(self, event: Event) -> None:
        super().left_button_press_hook(event)
        self._click_event(event)
        
    def mouse_move_event_hook(self, event: Event) -> None:
        super().mouse_move_event_hook(event)
        self._mouse_move_event(event)
        
        
    def key_press_hook(self, event: Event) -> None:
        super().key_press_hook(event)
        print(f'lasso key press event: {event.key=}')
        if event.key is None:
            return
        if event.key.lower() == 'space':
            self._sPause()
        if event.key.lower() == 'i':
            self._sIn()
        if event.key.lower() == 'o':
            self._sOut()
        if event.key.lower() == 'return':
            self._sAccept()
        if event.key.lower() == 'delete':
            self._sRemoveLastPt()
        if event.key.lower() == 'escape':
            self._sCancel()
        return
        
