from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
import time
import weakref

import numpy as np
from numpy.typing import NDArray

from PyQt6.QtCore import QEventLoop, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QWidget,
)

from geon.algorithms.split_between import Progress, split_between
from geon.data.pointcloud import FieldType, InstanceSegmentation
from geon.rendering.pointcloud import PointCloudLayer
from geon.util.resources import resource_path

from .base import Event, ModeTool, ToolZone
from .command_manager import Command
from .tool_context import ToolContext


@dataclass
class SplitBetweenCommitCmd(Command):
    field_name: str
    indices: NDArray[np.int64]
    old_values: NDArray[np.int32]
    new_values: NDArray[np.int32]
    layer_ref: weakref.ReferenceType[PointCloudLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]

    def _apply(self, values: NDArray[np.int32]) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        fields = layer.data.get_fields(names=self.field_name, field_type=FieldType.INSTANCE)
        if not fields or not isinstance(fields[0], InstanceSegmentation):
            return
        fields[0].data.reshape(-1)[self.indices] = values
        layer.set_active_field_name(self.field_name)
        ctx.controller.layer_data_modified.emit(layer)
        ctx.controller.scene_tree_request_change.emit()
        ctx.viewer.rerender()

    def execute(self) -> None:
        self._apply(self.new_values)

    def undo(self) -> None:
        self._apply(self.old_values)


@dataclass
class SplitBetweenTool(ModeTool):
    label: ClassVar = "split_between"
    tooltip: ClassVar = "Split between"
    icon_path: ClassVar = resource_path("annotate.png")
    shortcut: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = False
    cursor_icon_path: ClassVar = resource_path("inspect_tool.png")
    cursor_hot: ClassVar = (3, 3)
    keep_focus: ClassVar[bool] = False

    layer: PointCloudLayer = field(init=False)
    initial_field_name: str | None = field(default=None, init=False)
    field_name: str | None = field(default=None, init=False)
    condition: str = field(default="plane_distance", init=False)
    active_set: str | None = field(default=None, init=False)
    point_sets: dict[str, set[int]] = field(
        default_factory=lambda: {"work": set(), "a": set(), "b": set()}, init=False)
    _buttons: dict[str, QPushButton] = field(default_factory=dict, init=False, repr=False)
    _counters: dict[str, QLabel] = field(default_factory=dict, init=False, repr=False)
    _field_combo: QComboBox | None = field(default=None, init=False, repr=False)
    _ok_button: QPushButton | None = field(default=None, init=False, repr=False)
    _committed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        active = self.ctx.scene.active_layer
        if not isinstance(active, PointCloudLayer):
            raise RuntimeError("Split Between requires an active point cloud layer.")
        self.layer = active
        self.initial_field_name = active.active_field_name
        if isinstance(active.active_field, InstanceSegmentation):
            self.field_name = active.active_field.name

    def _field(self) -> InstanceSegmentation | None:
        if self.field_name is None:
            return None
        fields = self.layer.data.get_fields(names=self.field_name, field_type=FieldType.INSTANCE)
        return fields[0] if fields and isinstance(fields[0], InstanceSegmentation) else None

    def _visible_indices(self) -> NDArray[np.int64]:
        return np.asarray(self.layer.visible_inds, dtype=np.int64)

    def _clear_sets(self) -> None:
        for values in self.point_sets.values():
            values.clear()
        self.active_set = None
        self._refresh_ui()

    def _refresh_preview(self) -> None:
        combined = set().union(*self.point_sets.values())
        if combined:
            self.layer.set_temporary_point_opacity(
                np.fromiter(sorted(combined), dtype=np.int64), 0.5)
        else:
            self.layer.clear_temporary_point_opacity()
        self.ctx.viewer.rerender()

    def _refresh_ui(self) -> None:
        names = {"work": "Work", "a": "A", "b": "B"}
        for key, button in self._buttons.items():
            button.setText(("Done " if self.active_set == key else "Edit ") + names[key])
            button.setChecked(self.active_set == key)
        for key, counter in self._counters.items():
            counter.setText(f"{len(self.point_sets[key]):,}")
        if self._ok_button is not None:
            self._ok_button.setEnabled(
                self._field() is not None and all(bool(values) for values in self.point_sets.values()))
        self._refresh_preview()

    def _toggle_edit(self, key: str) -> None:
        self.active_set = None if self.active_set == key else key
        self._refresh_ui()

    def _choose_field(self, index: int) -> None:
        if self._field_combo is None or index < 0:
            return
        selected = self._field_combo.currentData()
        if not isinstance(selected, str) or selected == self.field_name:
            return
        if any(self.point_sets.values()):
            answer = QMessageBox.question(
                self.ctx.viewer.window(),
                "Change instance field",
                "Changing the field clears the three picked sets. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                current = self._field_combo.findData(self.field_name)
                self._field_combo.blockSignals(True)
                self._field_combo.setCurrentIndex(current)
                self._field_combo.blockSignals(False)
                return
        self.field_name = selected
        self.layer.set_active_field_name(selected)
        self._clear_sets()

    def _pick_instance(self, event: Event) -> None:
        if self.active_set is None:
            return
        field = self._field()
        result = self.ctx.viewer.pick()
        if field is None or result.layer is not self.layer or result.element_idx is None:
            return
        picked_idx = int(result.element_idx)
        labels = np.asarray(field.data, dtype=np.int32).reshape(-1)
        instance_id = int(labels[picked_idx])
        if instance_id < 0:
            QMessageBox.information(
                self.ctx.viewer.window(), "Split between", "Unassigned points cannot form a set.")
            return
        visible = self._visible_indices()
        instance_points = visible[labels[visible] == instance_id]
        selected_points = set(map(int, instance_points.tolist()))
        already_active = bool(selected_points) and selected_points.issubset(
            self.point_sets[self.active_set])
        for values in self.point_sets.values():
            values.difference_update(selected_points)
        if not already_active:
            self.point_sets[self.active_set].update(selected_points)
        if not event.shift:
            self.active_set = None
        self._refresh_ui()

    def left_button_release_hook(self, event: Event) -> None:
        self._pick_instance(event)

    def _target_ids(self, field: InstanceSegmentation) -> tuple[int, int]:
        labels = np.asarray(field.data, dtype=np.int32).reshape(-1)
        ids_a = np.unique(labels[np.fromiter(self.point_sets["a"], dtype=np.int64)])
        ids_b = np.unique(labels[np.fromiter(self.point_sets["b"], dtype=np.int64)])
        multi_a = ids_a.size > 1
        multi_b = ids_b.size > 1
        allocation = field.map_to_free(list(range(int(multi_a) + int(multi_b))))
        offset = 0
        target_a = int(ids_a[0])
        target_b = int(ids_b[0])
        if multi_a:
            target_a = allocation[offset]
            offset += 1
        if multi_b:
            target_b = allocation[offset]
        return target_a, target_b

    def _accept(self) -> None:
        field = self._field()
        if field is None or not all(self.point_sets.values()):
            return
        work = np.fromiter(sorted(self.point_sets["work"]), dtype=np.int64)
        set_a = np.fromiter(sorted(self.point_sets["a"]), dtype=np.int64)
        set_b = np.fromiter(sorted(self.point_sets["b"]), dtype=np.int64)
        progress = Progress()
        result: dict[str, object | None] = {"assignment": None, "stats": None, "error": None}

        class Worker(QThread):
            completed = pyqtSignal()

            def run(self) -> None:
                try:
                    result["assignment"], result["stats"] = split_between(
                        self_parent.layer.data.points, work, set_a, set_b,
                        self_parent.condition, progress)
                except Exception as exc:  # pragma: no cover - GUI path
                    result["error"] = str(exc)
                self.completed.emit()

        self_parent = self
        dialog = QProgressDialog("Preparing split...", "Cancel", 0, int(work.size), self.ctx.viewer.window())
        dialog.setWindowTitle("Split between")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        cancel_requested = {"value": False}

        def request_cancel() -> None:
            cancel_requested["value"] = True
            progress.request_cancel()

        dialog.canceled.connect(request_cancel)
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(100)
        started = time.perf_counter()

        def tick() -> None:
            dialog.setMaximum(max(1, progress.total()))
            dialog.setValue(min(progress.done(), max(1, progress.total())))
            dialog.setLabelText(f"{progress.stage()} | {time.perf_counter() - started:.1f}s")

        def finished() -> None:
            timer.stop()
            try:
                dialog.canceled.disconnect(request_cancel)
            except TypeError:
                pass
            dialog.close()
            loop.quit()

        worker = Worker()
        worker.completed.connect(finished)
        dialog.show()
        QApplication.processEvents()
        worker.start()
        timer.start()
        loop.exec()
        worker.wait()
        if result["error"] is not None:
            QMessageBox.critical(self.ctx.viewer.window(), "Split between failed", str(result["error"]))
            self.ctx.controller.deactivate_tool()
            return
        if cancel_requested["value"]:
            self.ctx.controller.deactivate_tool()
            return
        assignment = result["assignment"]
        if not isinstance(assignment, np.ndarray) or assignment.shape != work.shape:
            QMessageBox.critical(self.ctx.viewer.window(), "Split between failed", "Invalid native result.")
            self.ctx.controller.deactivate_tool()
            return

        target_a, target_b = self._target_ids(field)
        affected = np.unique(np.concatenate([work, set_a, set_b])).astype(np.int64)
        old_values = np.asarray(field.data, dtype=np.int32).reshape(-1)[affected].copy()
        new_by_index = dict(zip(affected.tolist(), old_values.tolist()))
        labels = np.asarray(field.data, dtype=np.int32).reshape(-1)
        if np.unique(labels[set_a]).size > 1:
            for idx in set_a: new_by_index[int(idx)] = target_a
        if np.unique(labels[set_b]).size > 1:
            for idx in set_b: new_by_index[int(idx)] = target_b
        for idx, goes_to_b in zip(work, assignment):
            new_by_index[int(idx)] = target_b if bool(goes_to_b) else target_a
        new_values = np.asarray([new_by_index[int(idx)] for idx in affected], dtype=np.int32)
        command = SplitBetweenCommitCmd(
            title="Split between instances",
            field_name=field.name,
            indices=affected,
            old_values=old_values,
            new_values=new_values,
            layer_ref=weakref.ref(self.layer),
            ctx_ref=weakref.ref(self.ctx),
        )
        self.command_manager.do(command)
        self._committed = True
        self.ctx.controller.deactivate_tool()

    def _cancel(self) -> None:
        self.ctx.controller.deactivate_tool()

    def activate(self) -> None:
        return super().activate()

    def deactivate(self) -> None:
        self.layer.clear_temporary_point_opacity()
        if not self._committed and self.initial_field_name in self.layer.data.field_names:
            self.layer.set_active_field_name(self.initial_field_name)
        self.ctx.viewer.rerender()
        return super().deactivate()

    def create_context_widget(self, parent: QWidget) -> QWidget:
        widget = QWidget(parent)
        layout = QGridLayout(widget)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(3)
        layout.addWidget(QLabel("Field", widget), 0, 0)
        self._field_combo = QComboBox(widget)
        for instance_field in self.layer.data.get_fields(field_type=FieldType.INSTANCE):
            self._field_combo.addItem(instance_field.name, instance_field.name)
        initial = self._field_combo.findData(self.field_name)
        self._field_combo.setCurrentIndex(max(0, initial))
        self._field_combo.currentIndexChanged.connect(self._choose_field)
        layout.addWidget(self._field_combo, 0, 1)
        layout.addWidget(QLabel("Condition", widget), 1, 0)
        condition_combo = QComboBox(widget)
        condition_combo.addItem("Plane distance", "plane_distance")
        condition_combo.addItem("Nearest neighbor", "nearest_neighbor")
        condition_combo.currentIndexChanged.connect(
            lambda _idx: setattr(self, "condition", str(condition_combo.currentData())))
        layout.addWidget(condition_combo, 1, 1)

        for column, (key, name) in enumerate((("work", "Work"), ("a", "A"), ("b", "B")), start=2):
            button = QPushButton(f"Edit {name}", widget)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, selected=key: self._toggle_edit(selected))
            counter = QLabel("0", widget)
            counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._buttons[key] = button
            self._counters[key] = counter
            layout.addWidget(button, 0, column)
            layout.addWidget(counter, 1, column)

        self._ok_button = QPushButton("OK", widget)
        self._ok_button.clicked.connect(self._accept)
        cancel = QPushButton("Cancel", widget)
        cancel.clicked.connect(self._cancel)
        layout.addWidget(self._ok_button, 0, 5)
        layout.addWidget(cancel, 1, 5)
        self._refresh_ui()
        return widget
