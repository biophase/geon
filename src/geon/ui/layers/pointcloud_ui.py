from .registry import LAYER_UI, LayerUIHooks

from ...rendering.base import BaseLayer
from ...rendering.pointcloud import (
    PointCloudLayer,
    FieldType,
    SemanticSegmentation,
)
from ...tools.controller import ToolController
from ...tools.selection import DeselectTool
from ...tools.visibility import HideTool, IsolateTool
from ...tools.annotate import AnnotateTool
from ...tools.base import BaseTool
from ..dataset_manager import DatasetManager
from ..semantic_schema_dialog import SemanticSchemaEditDialog


from PyQt6.QtWidgets import (QWidget, QMenu, QHBoxLayout, QVBoxLayout, QLabel,
                             QPushButton, QToolButton, QGridLayout, QDialog,
                             QMessageBox, QSpacerItem, QSizePolicy)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from config.theme import UIStyle
from geon.util.resources import resource_path

from typing import Type, Optional

def get_more_btn(parent: QWidget) -> tuple[QToolButton, QMenu]:
    more_btn = QToolButton(parent)
    more_btn.setFixedHeight(18)
    more_btn.setText("")
    more_btn.setStyleSheet("QToolButton { border: none; background: transparent; }")
    more_menu = QMenu(more_btn)
    more_btn.setMenu(more_menu)
    more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    return more_btn, more_menu


def _get_dataset_manager(parent: QWidget) -> Optional[DatasetManager]:
    window = parent.window()
    if window is None:
        return None
    dataset_manager = getattr(window, "dataset_manager", None)
    if dataset_manager is None:
        return None
    if not hasattr(dataset_manager, "update_semantic_schema"):
        return None
    return dataset_manager


def _collect_schema_names(layer: PointCloudLayer, dataset_manager: Optional[DatasetManager]) -> list[str]:
    names: set[str] = set()
    dataset = None
    if dataset_manager is not None:
        dataset = dataset_manager._dataset
    if dataset is not None:
        names.update(schema.name for schema in dataset.unique_semantic_schemas)
    else:
        fields = layer.data.get_fields(field_type=FieldType.SEMANTIC)
        for field in fields:
            if isinstance(field, SemanticSegmentation):
                names.add(field.schema.name)
    return sorted(names)

def _increase_point_size(
    layer: PointCloudLayer, controller: ToolController, label:QLabel) -> None:
        
        ctx = controller.ctx
        if ctx is None:
            return
        size = layer.increase_point_size()
        label.setText(str(size))
        layer.update()
        ctx.viewer.rerender()
        
def _decrease_point_size(
    layer: PointCloudLayer, controller: ToolController, label:QLabel) -> None:  
      
        ctx = controller.ctx
        if ctx is None:
            return
        size = layer.decrease_point_size()
        label.setText(str(size))
        layer.update()
        
        ctx.viewer.rerender()
        
def _ribbon(
    layer:PointCloudLayer, 
    parent: QWidget, 
    controller: ToolController) -> QWidget: 
    
    w = QWidget(parent)
    outer = QHBoxLayout(w)
    outer.setAlignment(Qt.AlignmentFlag.AlignTop)
    outer.setContentsMargins(2, 1, 2, 1)
    outer.setSpacing(2)
    
    af = layer.active_field
    
    
    
    # col 1 = field general
    col_1 = QGridLayout()
    col_1.setAlignment(Qt.AlignmentFlag.AlignTop)
    
    pt_size_layout = QHBoxLayout()
    pt_size_label = QLabel("Point size: ")
    pt_size_label.setStyleSheet(UIStyle.TYPE_LABEL.value)
    pt_size_value = QLabel(parent)
    pt_size_value.setText(str(layer.point_size))
    pt_size_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    # Reserve space for 2-digit values to avoid layout shifts.
    pt_size_value.setFixedWidth(pt_size_value.fontMetrics().horizontalAdvance("88") + 6)
    pt_size_layout.addWidget(pt_size_label)
    pt_size_layout.addWidget(pt_size_value)
    
    col_1.addLayout(pt_size_layout, 0, 0)
    
    
    btn_layout = QHBoxLayout()
    
    btn_decrease_pt_size = QPushButton(parent)
    btn_decrease_pt_size.setIcon(QIcon(resource_path("minus.png")))
    btn_decrease_pt_size.pressed.connect(
        lambda : _decrease_point_size(layer, controller, pt_size_value))
    btn_layout.addWidget(btn_decrease_pt_size)
    
    btn_increase_pt_size = QPushButton(parent)
    btn_increase_pt_size.setIcon(QIcon(resource_path("plus.png")))
    btn_increase_pt_size.pressed.connect(
        lambda : _increase_point_size(layer, controller, pt_size_value))
    btn_layout.addWidget(btn_increase_pt_size)
    
    col_1.addLayout(btn_layout, 1, 0)
    
    col_1.addItem(QSpacerItem(12, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum), 0, 1, 2, 1)
    
    af_label = QLabel('Active field: ')
    af_label.setStyleSheet(UIStyle.TYPE_LABEL.value)
    if af is not None:
        af_name = QLabel(f'{layer.active_field_name}')
        af_name.setAlignment(Qt.AlignmentFlag.AlignLeft)
        ft_label = QLabel("Field type: ")
        ft_label.setStyleSheet(UIStyle.TYPE_LABEL.value)
        ft_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        ft_name = QLabel(af.field_type.human_name)
        ft_name.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        col_1.addWidget(ft_label, 1, 2, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        col_1.addWidget(ft_name, 1, 3, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        
    else:
        af_name = QLabel("No active field")
        af_name.setStyleSheet(UIStyle.TYPE_LABEL.value)

    col_1.addWidget(af_label, 0, 2)
    col_1.addWidget(af_name, 0, 3)
        
    outer.addLayout(col_1)

    
    # col 2 = field specific info
    col_2 = None
    if af is not None:
        if af.field_type == FieldType.COLOR:
            pass
        if af.field_type == FieldType.INTENSITY:
            pass
        if af.field_type == FieldType.INSTANCE:
            pass
        if af.field_type == FieldType.SCALAR:
            pass
        if af.field_type == FieldType.SEMANTIC:
            assert isinstance(af, SemanticSegmentation)
            col_2 = QVBoxLayout()
            row_21 = QHBoxLayout()
            row_22 = QHBoxLayout()
            schema_label = QLabel("Schema: ")
            schema_label.setStyleSheet(UIStyle.TYPE_LABEL.value)
            schema_name = QLabel(af.schema.name)
            schema_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            schema_name.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            row_21.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            row_21.addWidget(schema_label, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            row_21.addWidget(schema_name, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            edit_btn = QToolButton (w)
            edit_btn.setFixedHeight(18)
            edit_btn.setText("Edit Schema")

            def _on_edit_schema_clicked() -> None:
                if not isinstance(af, SemanticSegmentation):
                    return
                dataset_manager = _get_dataset_manager(parent)
                if dataset_manager is None:
                    return
                taken_names = _collect_schema_names(layer, dataset_manager)
                current_schema = af.schema
                while True:
                    dlg = SemanticSchemaEditDialog(
                        schema=current_schema,
                        required_ids=[],
                        taken_schema_names=taken_names,
                        parent=parent,
                    )
                    if dlg.exec() != QDialog.DialogCode.Accepted or dlg.schema is None:
                        return

                    use_dataset_update = (
                        dlg.update_existing and dataset_manager is not None
                        and getattr(dataset_manager, "_dataset", None) is not None
                    )
                    if use_dataset_update:
                        confirmed = dataset_manager.update_semantic_schema(
                            dlg.old_schema,
                            dlg.schema,
                            dlg.old_2_new_ids,
                        )
                        if not confirmed:
                            current_schema = dlg.schema
                            continue
                    else:
                        if dlg.update_existing:
                            QMessageBox.information(
                                parent,
                                "Dataset unavailable",
                                "Dataset not available. Applying changes only to this field.",
                            )
                        af.remap(dlg.old_2_new_ids)
                        af.schema = dlg.schema
                    schema_name.setText(dlg.schema.name)
                    layer.update()
                    if controller.ctx is not None:
                        controller.ctx.viewer.rerender()
                    break

            edit_btn.clicked.connect(_on_edit_schema_clicked)

            
            more_btn, more_menu = get_more_btn(w)
            # more_menu.addAction("Option 1")
            # more_menu.addAction("Option 2")
            
            row_22.addWidget(edit_btn)
            row_22.addWidget(more_btn, alignment= Qt.AlignmentFlag.AlignRight)
            col_2.addLayout(row_21)
            col_2.addLayout(row_22)
            
            
            
        if af.field_type == FieldType.VECTOR:
            pass # TODO: add choice for which dim to display (meta on layer)
        
        if af.field_type == FieldType.NORMAL:
            pass # TODO: add choice whether to show arrow display
    
    
    if col_2 is not None:
        outer.addSpacing(12)
        outer.addLayout(col_2)
    
    layer.active_field_name
    
    return w

def _ribbon_selection(layer: PointCloudLayer, 
                      parent:QWidget, 
                      controller: ToolController) -> QWidget | None:
    alignment_row = Qt.AlignmentFlag.AlignLeft
    w = QWidget(parent)
    if layer.active_selection is None or\
        layer.active_selection.size == 0:
            return None
    outer = QGridLayout(w)
    outer.setAlignment(Qt.AlignmentFlag.AlignTop)
    outer.setContentsMargins(2, 1, 2, 1)
    outer.setSpacing(2)
    
    row = QHBoxLayout()
    overview_label = QLabel('Size: ')
    overview_label.setStyleSheet(UIStyle.TYPE_LABEL.value)
    row.addWidget(overview_label)
    
    overview_val = QLabel(layer.browser_sel_descr)
    overview_val.setAlignment(alignment_row)
    row.addWidget(overview_val)
    
    outer.addLayout(row, 0, 2, alignment_row)
    
    def format_btn(btn: QPushButton) -> QPushButton:
        # btn.setStyleSheet(
        #     "QPushbutton { font-size: 9px}"
        #     )
        btn.setStyleSheet("QPushButton { text-align: left; }")

        btn.setFixedHeight(18)
        btn.setFixedWidth(100)
        return btn
    
    # tool, pos, alignment
    tool_params :list[tuple[Type[BaseTool],tuple[int,int],Qt.AlignmentFlag]]= [ 
        (DeselectTool,  (0,0), alignment_row),
        (HideTool,      (1,0), alignment_row),
        (IsolateTool,   (0,1), alignment_row),
        (AnnotateTool,  (1,1), alignment_row)
        ]
    
    for param in tool_params:
        tool, pos, alignment = param
        btn = format_btn(QPushButton())
        btn.setIcon(QIcon(tool.icon_path))
        btn.setText(tool.tooltip)
        btn.pressed.connect(
            lambda tid=tool.__name__: controller.activate_tool(tid))
        outer.addWidget(btn, pos[0], pos[1], alignment)
        
    
    # annotate_btn = format_btn(QPushButton())
    # annotate_btn.setIcon(QIcon('resources/annotate.png'))
    # annotate_btn.setText("Annotate")
    # outer.addWidget(annotate_btn, 1, 0, alignment_row)
    
    # hide_btn = format_btn(QPushButton())
    # hide_btn.setIcon(QIcon('resources/hide.png'))
    # hide_btn.setText("Hide")
    # outer.addWidget(annotate_btn, 0, 1, alignment_row)
    
    # isolate_btn = format_btn(QPushButton())
    # isolate_btn.setIcon(QIcon('resources/isolate.png'))
    # isolate_btn.setText("Isolate")
    # outer.addWidget(annotate_btn, 1, 1, alignment_row)
    
    # deselect_tool = 
    
    # deselect_btn.setIcon(QIcon('resources/deselect.png'))
    # deselect_btn.setText("Deselect")
    # outer.addWidget(annotate_btn, 0, 0, alignment_row)
    
    
    more_btn, more_menu = get_more_btn(w)
    outer.addWidget(more_btn, 1, 2, Qt.AlignmentFlag.AlignRight)
    
    
    print(f'called ribbon selection')
    return w
    

def _menu(layer:PointCloudLayer, parent: QWidget, controller: ToolController) -> QMenu: 
    ...
    
def _text(layer:PointCloudLayer)->str: 
    return layer.browser_name

def _icon(layer:PointCloudLayer)->QIcon: 
    ...

LAYER_UI.register(PointCloudLayer,
    LayerUIHooks(
        ribbon_widget = _ribbon,
        ribbon_sel_widget=_ribbon_selection,
        tree_menu = _menu,
        tree_item_text = _text,
        tree_item_icon = _icon        
    ))
