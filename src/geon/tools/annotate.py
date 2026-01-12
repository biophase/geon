from PyQt6.QtWidgets import(
    QWidget, QHBoxLayout, QGridLayout, QCheckBox, QLabel, QComboBox, QLineEdit,
    QPushButton, QCompleter, QStyle, QListView
)
from PyQt6.QtCore import (
    Qt, QSignalBlocker, QObject, pyqtSignal, QSize
)
from PyQt6.QtGui import (
    QColor, QBrush, QPixmap, QIcon, QStandardItemModel, QStandardItem
)

from geon.tools.base import Event

from .base import ModeTool, ToolZone
from .tool_context import ToolContext
from ..rendering.pointcloud import PointCloudLayer
from ..data.pointcloud import (
    SemanticClass, SemanticSegmentation, InstanceSegmentation, FieldType
    )
from .command_manager import Command
from geon.util.resources import resource_path

from dataclasses import dataclass, field
from typing import ClassVar, Optional, cast
import weakref

import numpy as np
from numpy.typing import NDArray

class _AnnotateSignals(QObject):
    requestAddField = pyqtSignal(FieldType)


@dataclass
class AnnotatePointsCmd(Command):
    selection_old: NDArray[np.int32] = field(init=False)
    
    sem_field_name: Optional[str]
    inst_field_name: Optional[str]
    
    sem_inds_old: Optional[NDArray[np.int32]]
    inst_inds_old: Optional[NDArray[np.int32]]
    sem_ind_new: Optional[int]
    
    layer_ref: weakref.ReferenceType[PointCloudLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    
    
    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        sel = layer.active_selection
        if sel is None:
            return
        self.selection_old = sel.copy()
        if self.sem_field_name:
            field = layer.data.get_fields(names=self.sem_field_name)[0]
            if field is not None:
                self.sem_inds_old = field.data[self.selection_old].copy()
                field.data[self.selection_old] = np.full(
                    self.selection_old.shape[0], self.sem_ind_new, dtype = np.int32)
        if self.inst_field_name:
            field = layer.data.get_fields(names=self.inst_field_name)[0]
            if field is not None:
                self.inst_inds_old = field.data[self.selection_old].copy()
                field = cast(InstanceSegmentation, field)
                inst_ind_new = field.get_next_instance_id()
                inst_inds_new = np.full(
                    self.selection_old.shape[0], inst_ind_new, dtype = np.int32)
                field.data[self.selection_old] = inst_inds_new
        layer.update()
        ctx.viewer.rerender()
                
        
    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        selection_old = getattr(self, "selection_old", None)
        if selection_old is None:
            return
        if self.sem_field_name and self.sem_inds_old is not None:
            field = layer.data.get_fields(names=self.sem_field_name)[0]
            if field is not None:
                field.data[selection_old] = self.sem_inds_old
        if self.inst_field_name and self.inst_inds_old is not None:
            field = layer.data.get_fields(names=self.inst_field_name)[0]
            if field is not None:
                field.data[selection_old] = self.inst_inds_old
        layer.update()
        ctx.viewer.rerender()
    

@dataclass
class AnnotateTool(ModeTool):
    # metadata
    label: ClassVar = 'annotate'
    tooltip: ClassVar = "Annotate"
    icon_path: ClassVar = resource_path('annotate.png')
    shortcut: ClassVar = 'a'
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = False
    cursor_icon_path : ClassVar = None
    
    # mode tool meta
    keep_focus: ClassVar[bool] = False
    
    
    # state
    choice_sem_class: Optional[SemanticClass] = None
    
    choice_sem_field: Optional[SemanticSegmentation] = None
    choice_inst_field: Optional[InstanceSegmentation] = None
    
    avail_sem_classes: list[SemanticClass] = field(default_factory=list) 
    avail_sem_fields: list[SemanticSegmentation] = field(default_factory=list) 
    avail_inst_fields: list[InstanceSegmentation] = field(default_factory=list) 
    
    choice_add_semantic: bool = True
    choice_add_instance: bool = False
    
    copy_from_active:bool = False
    accept_possible: bool = False

    _signals: _AnnotateSignals = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._signals = _AnnotateSignals()
        self.requestAddField = self._signals.requestAddField
        self._refresh_available_fields()
        self._refresh_sem_classes()

    def _refresh_available_fields(self) -> None:
        layer = self.ctx.scene.active_layer
        if not isinstance(layer, PointCloudLayer):
            self.avail_sem_fields = []
            self.avail_inst_fields = []
            return
        self.avail_sem_fields = cast(
            list[SemanticSegmentation],
            layer.data.get_fields(field_type=FieldType.SEMANTIC))
        
        self.avail_inst_fields= cast(
            list[InstanceSegmentation],
            layer.data.get_fields(field_type=FieldType.INSTANCE))

    def _refresh_sem_classes(self) -> None:
        self._refresh_available_fields()
        if self.choice_sem_field not in self.avail_sem_fields:
            self.choice_sem_field = self.avail_sem_fields[0] if self.avail_sem_fields else None
                   
        if self.choice_sem_field is not None:
            schema = self.choice_sem_field.schema
            self.avail_sem_classes = list(schema.semantic_classes)
        else:
            self.avail_sem_classes = []


    def activate(self) -> None:
        return super().activate()
    
    def deactivate(self) -> None:
        self.ctx.controller.scene_tree_request_change.emit()
        return super().deactivate()

    def create_context_widget(self, parent: QWidget) -> QWidget | None:
        w = QWidget(parent)
        outer = QGridLayout(w)
        outer.setContentsMargins(2, 1, 2, 1)
        outer.setSpacing(2)
        
        def add_field_item(combo: QComboBox, add_index: int) -> None:
            model = combo.model()
            if model is None:
                return
            item_index = model.index(add_index, 0)
            model.setData(item_index, QBrush(QColor("blue")), Qt.ItemDataRole.ForegroundRole)

        def set_combo_choice(combo: QComboBox,
                             fields: list,
                             choice: object | None,
                             add_index: int) -> None:
            if choice in fields:
                combo.setCurrentIndex(fields.index(choice))
            elif fields:
                combo.setCurrentIndex(0)
            else:
                combo.setCurrentIndex(add_index)

        def emit_add_field(field_type: FieldType) -> None:
            if hasattr(self, "requestAddField"):
                try:
                    self.requestAddField.emit(field_type)
                except AttributeError:
                    pass

        # grid pos 0,0: checkbox, shows and controls self.choice_add_semantic
        sem_checkbox = QCheckBox()
        sem_checkbox.setChecked(self.choice_add_semantic)
        sem_checkbox.toggled.connect(lambda checked: setattr(self, "choice_add_semantic", checked))
        outer.addWidget(sem_checkbox, 0, 0)

        # grid pos 1,0: checkbox, shows and controls self.choice_add_instance
        inst_checkbox = QCheckBox()
        inst_checkbox.setChecked(self.choice_add_instance)
        inst_checkbox.toggled.connect(lambda checked: setattr(self, "choice_add_instance", checked))
        outer.addWidget(inst_checkbox, 1, 0)

        # grid pos 0,1: QLabel("semantic field: ")
        sem_label = QLabel("semantic field: ")
        outer.addWidget(sem_label, 0, 1)

        # grid pos 1,1: QLabel("instance field: ")
        inst_label = QLabel("instance field: ")
        outer.addWidget(inst_label, 1, 1)

        if self.choice_sem_field not in self.avail_sem_fields and self.avail_sem_fields:
            self.choice_sem_field = self.avail_sem_fields[0]

        if self.choice_inst_field not in self.avail_inst_fields and self.avail_inst_fields:
            self.choice_inst_field = self.avail_inst_fields[0]

        # grid pos 0,2: Dropdown for semantic fields
        sem_combo = QComboBox(w)
        for field in self.avail_sem_fields:
            sem_combo.addItem(field.name, field)
        sem_add_index = sem_combo.count()
        sem_combo.addItem("<add field>", None)
        add_field_item(sem_combo, sem_add_index)
        set_combo_choice(sem_combo, self.avail_sem_fields, self.choice_sem_field, sem_add_index)

        def on_sem_changed(index: int) -> None:
            if sem_combo.itemText(index) == "<add field>":
                emit_add_field(FieldType.SEMANTIC)
                with QSignalBlocker(sem_combo):
                    set_combo_choice(
                        sem_combo, self.avail_sem_fields, self.choice_sem_field, sem_add_index
                    )
                return
            self.choice_sem_field = sem_combo.currentData()
            update_sem_class_ui()

        sem_combo.currentIndexChanged.connect(on_sem_changed)
        outer.addWidget(sem_combo, 0, 2)

        # grid pos 1,2: Dropdown for instance fields
        inst_combo = QComboBox(w)
        for field in self.avail_inst_fields:
            inst_combo.addItem(field.name, field)
        inst_add_index = inst_combo.count()
        inst_combo.addItem("<add field>", None)
        add_field_item(inst_combo, inst_add_index)
        set_combo_choice(inst_combo, self.avail_inst_fields, self.choice_inst_field, inst_add_index)

        def on_inst_changed(index: int) -> None:
            if inst_combo.itemText(index) == "<add field>":
                emit_add_field(FieldType.INSTANCE)
                with QSignalBlocker(inst_combo):
                    set_combo_choice(
                        inst_combo, self.avail_inst_fields, self.choice_inst_field, inst_add_index
                    )
                return
            self.choice_inst_field = inst_combo.currentData()

        inst_combo.currentIndexChanged.connect(on_inst_changed)
        outer.addWidget(inst_combo, 1, 2)

        # grid pos 0,3: class input with completer
        sem_class_row = QHBoxLayout()
        sem_class_label = QLabel("class: ")
        sem_class_row.addWidget(sem_class_label)
        sem_class_input = QLineEdit(w)
        if self.choice_sem_class is not None:
            sem_class_input.setPlaceholderText(self.choice_sem_class.name)
        else:
            sem_class_input.setPlaceholderText("Enter class ...")
        def build_sem_class_model(classes: list[SemanticClass]) -> QStandardItemModel:
            model = QStandardItemModel()
            for sem_class in classes:
                item = QStandardItem(sem_class.name)
                r, g, b = sem_class.color
                swatch = QPixmap(12, 12)
                swatch.fill(QColor(r, g, b))
                item.setIcon(QIcon(swatch))
                model.appendRow(item)
            return model

        sem_class_model = build_sem_class_model(self.avail_sem_classes)
        completer = QCompleter(sem_class_model, sem_class_input)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        popup = completer.popup()
        if popup is not None:
            popup = cast(QListView, popup)
            popup.setUniformItemSizes(True)
            popup.setIconSize(QSize(12, 12))
            popup.setMinimumWidth(popup.sizeHintForColumn(0) + 24)
        sem_class_input.setCompleter(completer)

        def update_accept_state(text: str) -> None:
            name = text.strip()
            if not name:
                self.choice_sem_class = None
                self.accept_possible = False
            else:
                match = next(
                    (c for c in self.avail_sem_classes if c.name == name),
                    None
                )
                self.choice_sem_class = match
                self.accept_possible = match is not None
            update_enabled_states()
        def update_sem_class_ui() -> None:
            self._refresh_sem_classes()
            completer.setModel(build_sem_class_model(self.avail_sem_classes))
            popup = completer.popup()
            if popup is not None:
                popup = cast(QListView, popup)
                popup.setUniformItemSizes(True)
                popup.setIconSize(QSize(12, 12))
                popup.setMinimumWidth(popup.sizeHintForColumn(0) + 24)
            if self.choice_sem_class not in self.avail_sem_classes:
                self.choice_sem_class = None
            if self.choice_sem_class is not None:
                sem_class_input.setPlaceholderText(self.choice_sem_class.name)
            else:
                sem_class_input.setPlaceholderText("Enter class ...")
            update_accept_state(sem_class_input.text())

        def on_sem_class_commit() -> None:
            name = sem_class_input.text().strip()
            if not name:
                return
            for sem_class in self.avail_sem_classes:
                if sem_class.name == name:
                    self.choice_sem_class = sem_class
                    update_accept_state(name)
                    return
            self.create_new_sem_class(name)

        sem_class_input.editingFinished.connect(on_sem_class_commit)
        completer.activated.connect(update_accept_state)
        sem_class_input.textEdited.connect(update_accept_state)
        sem_class_row.addWidget(sem_class_input)
        sem_class_dropdown = QPushButton()
        sem_class_dropdown.setFixedSize(20, 20)
        sem_class_dropdown.setText("▼")
        sem_class_dropdown.setStyleSheet("font-size: 18§px;")
        sem_class_dropdown.setToolTip("Select semantic class")

        def show_sem_class_popup() -> None:
            sem_class_input.setFocus(Qt.FocusReason.OtherFocusReason)
            completer.setCompletionPrefix("")
            completer.complete(sem_class_input.rect())

        sem_class_dropdown.clicked.connect(show_sem_class_popup)
        sem_class_row.addWidget(sem_class_dropdown)
        outer.addLayout(sem_class_row, 0, 3)

        # grid pos 1,3: buttons
        inst_row = QHBoxLayout()
        copy_btn = QPushButton("Copy from ...")
        copy_btn.setCheckable(True)
        copy_btn.setChecked(self.copy_from_active)
        copy_btn.toggled.connect(lambda checked: setattr(self, "copy_from_active", checked))
        inst_row.addWidget(copy_btn)
        accept_btn = QPushButton("Accept")
        accept_btn.clicked.connect(self._Accept)
        inst_row.addWidget(accept_btn)
        outer.addLayout(inst_row, 1, 3)

        def update_enabled_states() -> None:
            sem_enabled = sem_checkbox.isChecked()
            inst_enabled = inst_checkbox.isChecked()
            sem_label.setEnabled(sem_enabled)
            sem_combo.setEnabled(sem_enabled)
            sem_class_label.setEnabled(sem_enabled)
            sem_class_input.setEnabled(sem_enabled)
            inst_label.setEnabled(inst_enabled)
            inst_combo.setEnabled(inst_enabled)
            any_enabled = sem_enabled or inst_enabled
            copy_btn.setEnabled(any_enabled)
            accept_btn.setEnabled(any_enabled and self.accept_possible)

        sem_checkbox.toggled.connect(lambda _checked: update_enabled_states())
        inst_checkbox.toggled.connect(lambda _checked: update_enabled_states())
        update_enabled_states()
        update_sem_class_ui()

        return w

    def create_new_sem_class(self, name: str) -> None:
        pass

    def _Accept(self) -> None:
        layer = self.ctx.scene.active_layer
        if not isinstance(layer, PointCloudLayer):
            return
        selection = layer.active_selection
        if selection is None or selection.size == 0:
            return

        sem_field_name = None
        sem_ind_new = None
        if self.choice_add_semantic:
            if self.choice_sem_field is None or self.choice_sem_class is None:
                return
            sem_field_name = self.choice_sem_field.name
            sem_ind_new = self.choice_sem_class.id

        inst_field_name = None
        if self.choice_add_instance:
            if self.choice_inst_field is None:
                return
            inst_field_name = self.choice_inst_field.name

        if sem_field_name is None and inst_field_name is None:
            return

        cmd = AnnotatePointsCmd(
            title = "Annotate points",
            sem_field_name=sem_field_name,
            inst_field_name=inst_field_name,
            sem_inds_old=None,
            inst_inds_old=None,
            sem_ind_new=sem_ind_new,
            layer_ref=weakref.ref(layer),
            ctx_ref=weakref.ref(self.ctx),
        )
        self.command_manager.do(cmd)
        self.ctx.viewer.rerender()
        self.ctx.controller.deactivate_tool()
        

        
    def key_press_hook(self, event: Event) -> None:
        super().key_press_hook(event)
        if event.key is None:
            return
        if event.key.lower() == 'escape':
            self.ctx.controller.deactivate_tool()
            
        
        
