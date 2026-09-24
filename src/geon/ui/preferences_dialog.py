from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QTabWidget, QWidget, QToolButton, QScrollArea, QPushButton, QColorDialog,
    QHBoxLayout, QLabel,
)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from geon.settings import Preferences


class PreferencesDialog(QDialog):
    def __init__(self, preferences: Preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._prefs = preferences

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)
        self.sections: dict[str, tuple[QToolButton, QWidget]] = {}

        def add_tab(title: str) -> QVBoxLayout:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, title)
            return page_layout

        def add_section(tab: QVBoxLayout, title: str) -> QFormLayout:
            header = QToolButton()
            header.setText(title)
            header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            header.setArrowType(Qt.ArrowType.DownArrow)
            header.setCheckable(True)
            header.setChecked(True)
            header.setAutoRaise(True)
            content = QWidget()
            form = QFormLayout(content)
            form.setContentsMargins(18, 0, 0, 8)

            def toggle(expanded: bool) -> None:
                content.setVisible(expanded)
                header.setArrowType(
                    Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
                )

            header.toggled.connect(toggle)
            tab.addWidget(header)
            tab.addWidget(content)
            self.sections[title] = (header, content)
            return form

        general = add_tab("General")
        viewport = add_tab("Viewport")
        appearance = add_tab("Appearance")
        user_form = add_section(general, "User")
        telemetry_form = add_section(general, "Telemetry")
        camera_form = add_section(viewport, "Camera")
        viewport_form = add_section(appearance, "General viewport")
        cell_form = add_section(appearance, "Cell complex")
        point_form = add_section(appearance, "Point cloud")
        self.resize(540, 620)

        self.user_input = QLineEdit(self)
        self.user_input.setText(self._prefs.user_name)
        user_form.addRow("User name", self.user_input)

        self.telemetry_checkbox = QCheckBox(self)
        self.telemetry_checkbox.setChecked(self._prefs.enable_telemetry)
        telemetry_form.addRow("Enable telemetry", self.telemetry_checkbox)

        self.camera_sensitivity_input = QDoubleSpinBox(self)
        self.camera_sensitivity_input.setDecimals(3)
        self.camera_sensitivity_input.setRange(0.01, 1000.0)
        self.camera_sensitivity_input.setSingleStep(0.5)
        self.camera_sensitivity_input.setValue(float(self._prefs.camera_sensitivity))
        camera_form.addRow("Camera sensitivity", self.camera_sensitivity_input)

        self.cell_size_mode_combo = QComboBox(self)
        self.cell_size_mode_combo.addItem("Screen space", "screen")
        self.cell_size_mode_combo.addItem("World space", "world")
        mode_idx = self.cell_size_mode_combo.findData(self._prefs.cell_complex_size_mode)
        self.cell_size_mode_combo.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
        cell_form.addRow("Cell cube size mode", self.cell_size_mode_combo)

        self.cell_screen_size_input = QDoubleSpinBox(self)
        self.cell_screen_size_input.setDecimals(1)
        self.cell_screen_size_input.setRange(1.0, 200.0)
        self.cell_screen_size_input.setSingleStep(1.0)
        self.cell_screen_size_input.setValue(float(self._prefs.cell_complex_screen_size_px))
        cell_form.addRow("Cell cube screen size (px)", self.cell_screen_size_input)

        self.cell_world_size_input = QDoubleSpinBox(self)
        self.cell_world_size_input.setDecimals(4)
        self.cell_world_size_input.setRange(0.0001, 1_000_000.0)
        self.cell_world_size_input.setSingleStep(0.1)
        self.cell_world_size_input.setValue(float(self._prefs.cell_complex_world_size))
        cell_form.addRow("Cell cube world size", self.cell_world_size_input)

        self.cell_edge_width_input = QDoubleSpinBox(self)
        self.cell_edge_width_input.setDecimals(1)
        self.cell_edge_width_input.setRange(1.0, 20.0)
        self.cell_edge_width_input.setSingleStep(1.0)
        self.cell_edge_width_input.setValue(float(self._prefs.cell_complex_edge_width))
        cell_form.addRow("Cell edge width", self.cell_edge_width_input)

        self.cell_reference_label_text_size_input = QDoubleSpinBox(self)
        self.cell_reference_label_text_size_input.setDecimals(1)
        self.cell_reference_label_text_size_input.setRange(6.0, 72.0)
        self.cell_reference_label_text_size_input.setSingleStep(1.0)
        self.cell_reference_label_text_size_input.setValue(
            float(self._prefs.cell_complex_reference_label_text_size_px)
        )
        cell_form.addRow("Reference label text size (px)", self.cell_reference_label_text_size_input)

        color_row = QLineEdit(self)
        color_row.setText(",".join(str(int(c)) for c in self._prefs.cell_complex_default_color))
        cell_form.addRow("Cell default RGB", color_row)
        self.cell_color_text = color_row

        selection_color_row = QLineEdit(self)
        selection_color_row.setText(",".join(str(int(c)) for c in self._prefs.selection_color))
        viewport_form.addRow("Selection RGB", selection_color_row)
        self.selection_color_text = selection_color_row

        viewport_text_color_row = QLineEdit(self)
        viewport_text_color_row.setText(",".join(str(int(c)) for c in self._prefs.viewport_text_color))
        viewport_form.addRow("Viewport text RGB", viewport_text_color_row)
        self.viewport_text_color_text = viewport_text_color_row

        self._unassigned_color = QColor(*self._prefs.unassigned_point_color)
        self.unassigned_color_button = QPushButton("Choose color...", self)
        self.unassigned_color_preview = QLabel(self)
        self.unassigned_color_preview.setMinimumWidth(150)
        self.unassigned_color_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color_layout = QHBoxLayout()
        color_layout.addWidget(self.unassigned_color_preview)
        color_layout.addWidget(self.unassigned_color_button)
        point_form.addRow("Unassigned RGBA", color_layout)
        self.unassigned_color_button.clicked.connect(self._choose_unassigned_color)
        self._update_unassigned_color_preview()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_unassigned_color(self) -> None:
        color = QColorDialog.getColor(
            self._unassigned_color, self, "Unassigned point color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self._unassigned_color = color
            self._update_unassigned_color_preview()

    def _update_unassigned_color_preview(self) -> None:
        r, g, b, a = self._unassigned_color.getRgb()
        self.unassigned_color_preview.setText(f"{r}, {g}, {b}, {a}")
        foreground = "black" if self._unassigned_color.lightness() > 127 else "white"
        self.unassigned_color_preview.setStyleSheet(
            f"background-color: rgba({r}, {g}, {b}, {a}); color: {foreground};"
            "border: 1px solid gray; padding: 4px;"
        )

    def apply(self) -> None:
        self._prefs.unassigned_point_color = list(self._unassigned_color.getRgb())
        self._prefs.user_name = self.user_input.text().strip() or "Unnamed User"
        self._prefs.enable_telemetry = self.telemetry_checkbox.isChecked()
        self._prefs.camera_sensitivity = float(self.camera_sensitivity_input.value())
        mode = self.cell_size_mode_combo.currentData()
        self._prefs.cell_complex_size_mode = str(mode or "screen")
        self._prefs.cell_complex_screen_size_px = float(self.cell_screen_size_input.value())
        self._prefs.cell_complex_world_size = float(self.cell_world_size_input.value())
        self._prefs.cell_complex_edge_width = float(self.cell_edge_width_input.value())
        self._prefs.cell_complex_reference_label_text_size_px = float(
            self.cell_reference_label_text_size_input.value()
        )
        try:
            rgb = [int(part.strip()) for part in self.cell_color_text.text().split(",")]
            if len(rgb) != 3:
                raise ValueError
            self._prefs.cell_complex_default_color = [
                int(max(0, min(255, c))) for c in rgb
            ]
        except ValueError:
            self._prefs.cell_complex_default_color = [204, 204, 204]
        try:
            rgb = [int(part.strip()) for part in self.selection_color_text.text().split(",")]
            if len(rgb) != 3:
                raise ValueError
            self._prefs.selection_color = [
                int(max(0, min(255, c))) for c in rgb
            ]
        except ValueError:
            self._prefs.selection_color = [255, 128, 0]
        try:
            rgb = [int(part.strip()) for part in self.viewport_text_color_text.text().split(",")]
            if len(rgb) != 3:
                raise ValueError
            self._prefs.viewport_text_color = [
                int(max(0, min(255, c))) for c in rgb
            ]
        except ValueError:
            self._prefs.viewport_text_color = [255, 255, 255]
