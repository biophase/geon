import re
import os
import time
import math
import random
import numpy as np
import colorsys
import configparser
import json

from plyfile import PlyData
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QIcon, QCursor, QPixmap
from PyQt6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout,
                             QVBoxLayout, QMainWindow, QMessageBox, QPushButton, QSpacerItem,
                             QSizePolicy, QWidget, QFileDialog, QTabWidget, QTreeWidget, QTreeWidgetItem,
                             QComboBox, QLabel, QLineEdit, QSplitter, QMenu, QColorDialog, QHeaderView,
                             QSpinBox, QCheckBox, QFormLayout, QDoubleSpinBox)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support as ns
import vtk

import segmentation.reggrow.build.lib.reggrow as rg

# Import segmentation definitions
from annotation import IndexSegmentation, SemanticSchema, SemanticClass

# For scalar mapping using matplotlib colormaps
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from theme import set_dark_palette

# ---------------------------
# Helper functions
# ---------------------------
def create_color_icon(rgb, size=16):
    """Return a QIcon with the given RGB color (tuple with values 0-1)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    from PyQt6.QtGui import QPainter, QColor
    painter = QPainter(pixmap)
    painter.fillRect(0, 0, size, size, QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    painter.end()
    return QIcon(pixmap)

def generate_vibrant_color():
    """Generate a random vibrant color (avoid grays, blacks, whites)."""
    h = random.random()
    s = 0.9
    v = 0.9
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

def blend_colors(c1, c2, t):
    """Linearly blend two RGB colors with blend factor t (0<=t<=1)."""
    return tuple((1 - t) * a + t * b for a, b in zip(c1, c2))

# ---------------------------
# Region Growing Settings Dialog
# ---------------------------
class RegionGrowingSettingsDialog(QDialog):
    def __init__(self, parent, current_settings, current_chunk_layout):
        super().__init__(parent)
        self.setWindowTitle("Region Growing Settings")
        self.current_settings = current_settings.copy()
        self.current_chunk_layout = list(current_chunk_layout)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # Float parameters with QDoubleSpinBox
        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setRange(0.0001, 1.0)
        self.epsilon_spin.setDecimals(4)
        self.epsilon_spin.setValue(self.current_settings.get("epsilon", 0.03))
        form_layout.addRow("Epsilon:", self.epsilon_spin)
        
        self.alpha_spin = QDoubleSpinBox(self)
        self.alpha_spin.setRange(0, math.pi)
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setValue(self.current_settings.get("alpha", math.pi * 0.15))
        form_layout.addRow("Alpha (radians):", self.alpha_spin)
        
        self.min_success_ratio_spin = QDoubleSpinBox(self)
        self.min_success_ratio_spin.setRange(0.0, 1.0)
        self.min_success_ratio_spin.setDecimals(2)
        self.min_success_ratio_spin.setValue(self.current_settings.get("min_success_ratio", 0.10))
        form_layout.addRow("Min Success Ratio:", self.min_success_ratio_spin)
        
        self.max_dist_spin = QDoubleSpinBox(self)
        self.max_dist_spin.setRange(0.0, 1000.0)
        self.max_dist_spin.setDecimals(2)
        self.max_dist_spin.setValue(self.current_settings.get("max_dist_from_cent", 50.0))
        form_layout.addRow("Max Dist from Centroid:", self.max_dist_spin)
        
        self.epsilon_multiplier_spin = QDoubleSpinBox(self)
        self.epsilon_multiplier_spin.setRange(0.1, 10.0)
        self.epsilon_multiplier_spin.setDecimals(2)
        self.epsilon_multiplier_spin.setValue(self.current_settings.get("epsilon_multiplier", 3.0))
        form_layout.addRow("Epsilon Multiplier:", self.epsilon_multiplier_spin)
        
        self.epsilon_multiplier_avg_spin = QDoubleSpinBox(self)
        self.epsilon_multiplier_avg_spin.setRange(0.1, 10.0)
        self.epsilon_multiplier_avg_spin.setDecimals(2)
        self.epsilon_multiplier_avg_spin.setValue(self.current_settings.get("epsilon_multiplier_average", 1.5))
        form_layout.addRow("Epsilon Multiplier Average:", self.epsilon_multiplier_avg_spin)
        
        self.refit_multiplier_spin = QDoubleSpinBox(self)
        self.refit_multiplier_spin.setRange(0.1, 10.0)
        self.refit_multiplier_spin.setDecimals(2)
        self.refit_multiplier_spin.setValue(self.current_settings.get("refit_multiplier", 2.0))
        form_layout.addRow("Refit Multiplier:", self.refit_multiplier_spin)
        
        self.search_radius_spin = QDoubleSpinBox(self)
        self.search_radius_spin.setRange(0.0, 1000.0)
        self.search_radius_spin.setDecimals(2)
        self.search_radius_spin.setValue(self.current_settings.get("search_radius_approx", 0.0))
        form_layout.addRow("Search Radius Approx (0=exact search):", self.search_radius_spin)
        
        # Integer parameters with QSpinBox
        self.min_points_spin = QSpinBox(self)
        self.min_points_spin.setRange(1, 1000000)
        self.min_points_spin.setValue(self.current_settings.get("min_points_in_region", 80))
        form_layout.addRow("Min Points in Region:", self.min_points_spin)
        
        self.first_refit_spin = QSpinBox(self)
        self.first_refit_spin.setRange(1, 1000)
        self.first_refit_spin.setValue(self.current_settings.get("first_refit", 4))
        form_layout.addRow("First Refit:", self.first_refit_spin)
        
        self.tracker_size_spin = QSpinBox(self)
        self.tracker_size_spin.setRange(1, 1000)
        self.tracker_size_spin.setValue(self.current_settings.get("tracker_size", 50))
        form_layout.addRow("Tracker Size:", self.tracker_size_spin)
        
        # Boolean parameters with QCheckBox
        self.oriented_normals_checkbox = QCheckBox(self)
        self.oriented_normals_checkbox.setChecked(self.current_settings.get("oriented_normals", False))
        form_layout.addRow("Oriented Normals:", self.oriented_normals_checkbox)
        
        self.verbose_checkbox = QCheckBox(self)
        self.verbose_checkbox.setChecked(self.current_settings.get("verbose", True))
        form_layout.addRow("Verbose:", self.verbose_checkbox)
        
        self.perform_cca_checkbox = QCheckBox(self)
        self.perform_cca_checkbox.setChecked(self.current_settings.get("perform_cca", True))
        form_layout.addRow("Perform CCA:", self.perform_cca_checkbox)
        
        # Chunk layout: three integers
        chunk_layout = QHBoxLayout()
        self.chunk_x_spin = QSpinBox(self)
        self.chunk_x_spin.setRange(1, 100)
        self.chunk_x_spin.setValue(self.current_chunk_layout[0])
        self.chunk_y_spin = QSpinBox(self)
        self.chunk_y_spin.setRange(1, 100)
        self.chunk_y_spin.setValue(self.current_chunk_layout[1])
        self.chunk_z_spin = QSpinBox(self)
        self.chunk_z_spin.setRange(1, 100)
        self.chunk_z_spin.setValue(self.current_chunk_layout[2])
        chunk_layout.addWidget(QLabel("X:"))
        chunk_layout.addWidget(self.chunk_x_spin)
        chunk_layout.addWidget(QLabel("Y:"))
        chunk_layout.addWidget(self.chunk_y_spin)
        chunk_layout.addWidget(QLabel("Z:"))
        chunk_layout.addWidget(self.chunk_z_spin)
        form_layout.addRow("Chunk Layout:", chunk_layout)
        
        layout.addLayout(form_layout)
        
        # Load/Save JSON buttons
        buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Load JSON", self)
        self.save_button = QPushButton("Save JSON", self)
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.save_button)
        layout.addLayout(buttons_layout)
        
        self.load_button.clicked.connect(self.load_json)
        self.save_button.clicked.connect(self.save_json)
        
        # OK/Cancel dialog buttons
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)
    def get_settings(self):
        settings = {
            "epsilon": self.epsilon_spin.value(),
            "alpha": self.alpha_spin.value(),
            "min_points_in_region": self.min_points_spin.value(),
            "min_success_ratio": self.min_success_ratio_spin.value(),
            "max_dist_from_cent": self.max_dist_spin.value(),
            "epsilon_multiplier": self.epsilon_multiplier_spin.value(),
            "epsilon_multiplier_average": self.epsilon_multiplier_avg_spin.value(),
            "first_refit": self.first_refit_spin.value(),
            "refit_multiplier": self.refit_multiplier_spin.value(),
            "oriented_normals": self.oriented_normals_checkbox.isChecked(),
            "verbose": self.verbose_checkbox.isChecked(),
            "tracker_size": self.tracker_size_spin.value(),
            "search_radius_approx": self.search_radius_spin.value(),
            "perform_cca": self.perform_cca_checkbox.isChecked()
        }
        chunk_layout = (self.chunk_x_spin.value(), self.chunk_y_spin.value(), self.chunk_z_spin.value())
        return settings, chunk_layout
    def load_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Region Growing Settings", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, "r") as f:
                data = json.load(f)
            settings = data.get("settings", {})
            chunk_layout = data.get("chunk_layout", [1, 1, 1])
            self.epsilon_spin.setValue(settings.get("epsilon", 0.03))
            self.alpha_spin.setValue(settings.get("alpha", math.pi * 0.15))
            self.min_points_spin.setValue(settings.get("min_points_in_region", 80))
            self.min_success_ratio_spin.setValue(settings.get("min_success_ratio", 0.10))
            self.max_dist_spin.setValue(settings.get("max_dist_from_cent", 50.0))
            self.epsilon_multiplier_spin.setValue(settings.get("epsilon_multiplier", 3.0))
            self.epsilon_multiplier_avg_spin.setValue(settings.get("epsilon_multiplier_average", 1.5))
            self.first_refit_spin.setValue(settings.get("first_refit", 4))
            self.refit_multiplier_spin.setValue(settings.get("refit_multiplier", 2.0))
            self.oriented_normals_checkbox.setChecked(settings.get("oriented_normals", False))
            self.verbose_checkbox.setChecked(settings.get("verbose", True))
            self.tracker_size_spin.setValue(settings.get("tracker_size", 50))
            self.search_radius_spin.setValue(settings.get("search_radius_approx", 0.0))
            self.perform_cca_checkbox.setChecked(settings.get("perform_cca", True))
            self.chunk_x_spin.setValue(chunk_layout[0])
            self.chunk_y_spin.setValue(chunk_layout[1])
            self.chunk_z_spin.setValue(chunk_layout[2])
    
    def save_json(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Region Growing Settings", "", "JSON Files (*.json)")
        if file_path:
            settings, chunk_layout = self.get_settings()
            data = {"settings": settings, "chunk_layout": list(chunk_layout)}
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

# ---------------------------
# Hotkeys Settings Dialog
# ---------------------------
class HotkeysSettingsDialog(QDialog):
    def __init__(self, parent, current_hotkeys):
        super().__init__(parent)
        self.setWindowTitle("Hotkeys Settings")
        self.current_hotkeys = current_hotkeys.copy()
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Lasso Tool Hotkey:"))
        self.lasso_edit = QLineEdit(self)
        self.lasso_edit.setText(current_hotkeys.get("lasso", "t"))
        layout.addWidget(self.lasso_edit)
        
        layout.addWidget(QLabel("Wand Tool Hotkey:"))
        self.wand_edit = QLineEdit(self)
        self.wand_edit.setText(current_hotkeys.get("wand", "Control+w"))
        layout.addWidget(self.wand_edit)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_hotkeys(self):
        return {"lasso": self.lasso_edit.text(), "wand": self.wand_edit.text()}


# ---------------------------
# Custom Interactor Style
# ---------------------------
class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, viewer):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.camera = renderer.GetActiveCamera()
        self.last_click_time = 0
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("RightButtonPressEvent", self.right_button_press_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward_event)
        self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward_event)
        self.AddObserver("KeyPressEvent", self.key_press_event)

    def left_button_press_event(self, obj, event):
        if self.viewer.active_tool == "lasso":
            self.viewer.lasso_click_event(obj, event)
            return
        elif self.viewer.active_tool == "wand":
            self.viewer.wand_pick()
            self.viewer.deactivate_tool()
            return

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
        self.viewer.zoom_camera(1.1)

    def mouse_wheel_backward_event(self, obj, event):
        self.viewer.zoom_camera(1.0/1.1)

    def double_click_event(self):
        x, y = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.01)
        picker.Pick(x, y, 0, self.renderer)
        if picker.GetPointId() < 0:
            print("No valid point picked.")
            return
        picked_point = picker.GetPickPosition()
        print(f"New pivot set at: {picked_point}")
        self.viewer.set_pivot_point(picked_point)

    def key_press_event(self, obj, event):
        # Get the key and modifiers
        key = self.GetInteractor().GetKeySym().lower()
        ctrl = self.GetInteractor().GetControlKey()

        # undo
        if key == 'z' and ctrl:
            self.viewer.undo_update_active_selection()
        # Check user-defined hotkeys
        lasso_hotkey = self.viewer.hotkeys.get("lasso", "t").lower()
        wand_hotkey = self.viewer.hotkeys.get("wand", "control+w").lower()
        if key == lasso_hotkey and not ctrl:
            self.viewer.lasso_segmentation()
            return
        if wand_hotkey == "control+w" and ctrl and key == "w":
            self.viewer.activate_wand_tool()
            return

        if key == "f3":
            self.viewer.toggle_projection()
        self.OnKeyPress()


# ---------------------------
# Dialogs for new tools 
# ---------------------------
class LabelSemanticsDialog(QDialog):

    def __init__(self, parent, segmentation_list, default_segmentation=None):
        super().__init__(parent)
        self.setWindowTitle("Label Semantics")
        layout = QVBoxLayout(self)
        
        self.seg_combo = QComboBox(self)
        for seg in segmentation_list:
            self.seg_combo.addItem(seg.name, seg)
        if default_segmentation:
            index = self.seg_combo.findText(default_segmentation.name)
            if index >= 0:
                self.seg_combo.setCurrentIndex(index)
        layout.addWidget(QLabel("Select segmentation:"))
        layout.addWidget(self.seg_combo)
        
        self.sem_combo = QComboBox(self)
        current_seg = self.seg_combo.currentData()
        for sem in current_seg.semantic_schema.semantic_classes:
            self.sem_combo.addItem(f"{sem.name} (id={sem.id})", sem.id)
        layout.addWidget(QLabel("Select semantic class:"))
        layout.addWidget(self.sem_combo)
        
        self.edit_btn = QPushButton("Edit semantic schema", self)
        self.edit_btn.clicked.connect(self.edit_semantic_schema)
        layout.addWidget(self.edit_btn)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.seg_combo.currentIndexChanged.connect(self.update_semantic_combo)
    
    def update_semantic_combo(self):
        seg = self.seg_combo.currentData()
        self.sem_combo.clear()
        for sem in seg.semantic_schema.semantic_classes:
            self.sem_combo.addItem(f"{sem.name} (id={sem.id})", sem.id)
    
    def edit_semantic_schema(self):
        dlg = SemanticClassEditorDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            seg = self.seg_combo.currentData()
            new_sem = SemanticClass(id=dlg.id, name=dlg.name, color=dlg.color)
            try:
                seg.semantic_schema.add_semantic_class(new_sem)
                self.update_semantic_combo()
                QMessageBox.information(self, "Success", "New semantic class added.")
            except AssertionError as e:
                QMessageBox.warning(self, "Error", str(e))


class SemanticClassEditorDialog(QDialog):
    def __init__(self, parent=None, sem_class=None):
        super().__init__(parent)
        self.setWindowTitle("Semantic Class Editor" if sem_class else "Add New Semantic Class")
        layout = QVBoxLayout(self)
        
        self.id_label = QLabel("ID:")
        self.id_edit = QLineEdit(self)
        layout.addWidget(self.id_label)
        layout.addWidget(self.id_edit)
        
        self.name_label = QLabel("Name:")
        self.name_edit = QLineEdit(self)
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_edit)
        
        self.color_label = QLabel("Color (R,G,B 0-1):")
        self.color_edit = QLineEdit(self)
        layout.addWidget(self.color_label)
        layout.addWidget(self.color_edit)
        
        self.choose_color_btn = QPushButton("Choose Color", self)
        self.choose_color_btn.clicked.connect(self.choose_color)
        layout.addWidget(self.choose_color_btn)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        if sem_class:
            self.id_edit.setText(str(sem_class.id))
            self.name_edit.setText(sem_class.name)
            self.color_edit.setText(f"{sem_class.color[0]},{sem_class.color[1]},{sem_class.color[2]}")
        else:
            self.id_edit.setText("0")
            self.name_edit.setText("")
            self.color_edit.setText("0.0,0.0,0.0")
    
    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r = color.red() / 255.0
            g = color.green() / 255.0
            b = color.blue() / 255.0
            self.color_edit.setText(f"{r:.2f},{g:.2f},{b:.2f}")
    
    @property
    def id(self):
        try:
            return int(self.id_edit.text())
        except:
            return 0
    
    @property
    def name(self):
        return self.name_edit.text()
    
    @property
    def color(self):
        try:
            parts = self.color_edit.text().split(',')
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except:
            return (0.0, 0.0, 0.0)




# ---------------------------
# Main Viewer Class
# ---------------------------
class VTKPointCloudViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTK PLY Point Cloud Viewer")
        self.resize(1200, 800)

        # Data holders
        self.all_points = None
        self.all_colors = None
        self.poly_data = None
        self.segmentations = []
        self.fields = {}
        self.active_segmentation = None
        self.active_field = None
        self.active_tool = None
        self.active_segmentation_mode = None  # "semantic" or "instance"
        self.visibility_mask = None  # Binary mask for point visibility
        self.loaded_ply_name = None
        self.region_growing_settings = None
        self.region_growing_chunk_layout = None

        # Initialize hotkeys with defaults.
        self.hotkeys = {
            "lasso": "t",
            "wand": "Control+w"
        }

        # default region growing settings
        self.set_default_region_growing_setting()

        # ---------------------------
        # Top-Level Layout: Vertical layout with top toolbar and main horizontal area.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        v_layout = QVBoxLayout(central_widget)

        # --- Top Toolbar (Grid Layout) ---
        self.top_toolbar = QWidget()
        top_toolbar_layout = QGridLayout(self.top_toolbar)
        top_toolbar_layout.setContentsMargins(2, 2, 2, 2)
        collapse_btn = QPushButton()
        collapse_btn.setIcon(QIcon("resources/collapse.png"))
        collapse_btn.setToolTip("Collapse/Expand Sidebar")
        collapse_btn.setIconSize(QSize(28, 28))
        collapse_btn.clicked.connect(self.toggle_left_sidebar)
        top_toolbar_layout.addWidget(collapse_btn, 0, 0)
        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        top_toolbar_layout.addItem(spacer, 0, 1)
        self.top_toolbar.setFixedHeight(40)
        v_layout.addWidget(self.top_toolbar)

        # --- Main Horizontal Area using QSplitter for resizable panels ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Sidebar: QTabWidget
        self.left_sidebar = QTabWidget()
        self.left_sidebar.setTabPosition(QTabWidget.TabPosition.West)
        self.left_sidebar.setMinimumWidth(200)
        seg_tab = QWidget()
        seg_layout = QVBoxLayout(seg_tab)
        self.btn_create_seg = QPushButton("Create segmentation")
        self.btn_create_seg.clicked.connect(self.create_segmentation)
        seg_layout.addWidget(self.btn_create_seg)
        self.btn_load_seg = QPushButton("Load segmentation")
        self.btn_load_seg.clicked.connect(self.load_segmentation)
        seg_layout.addWidget(self.btn_load_seg)
        self.seg_tree = QTreeWidget()
        self.seg_tree.setColumnCount(4)
        self.seg_tree.setHeaderLabels(["Name", "Index", "Points", "Pct"])
        self.seg_tree.itemSelectionChanged.connect(self.on_seg_tree_selection_changed)
        self.seg_tree.setEditTriggers(QTreeWidget.EditTrigger.DoubleClicked)
        self.seg_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.seg_tree.customContextMenuRequested.connect(self.on_seg_tree_context_menu)
        self.seg_tree.itemChanged.connect(self.on_seg_tree_item_changed)
        # New: double click handler for selecting all points in a class.
        self.seg_tree.itemDoubleClicked.connect(self.on_seg_tree_item_double_clicked)
        seg_layout.addWidget(self.seg_tree, stretch=1)
        self.seg_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.seg_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.seg_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.seg_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.left_sidebar.addTab(seg_tab, QIcon("resources/segmentation.png"), "Segmentation")
        fields_tab = QWidget()
        fields_layout = QVBoxLayout(fields_tab)
        self.fields_tree = QTreeWidget()
        self.fields_tree.setColumnCount(2)
        self.fields_tree.setHeaderLabels(["Field", "Type"])
        self.fields_tree.itemSelectionChanged.connect(self.on_fields_tree_selection_changed)
        fields_layout.addWidget(self.fields_tree, stretch=1)
        self.fields_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.fields_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.left_sidebar.addTab(fields_tab, QIcon("resources/fields.png"), "Fields")
        splitter.addWidget(self.left_sidebar)

        # Central VTK Viewer
        self.vtkWidget = QVTKRenderWindowInteractor(central_widget)
        splitter.addWidget(self.vtkWidget)

        # Right Toolbar: using QWidget (set minimum width for resizing)
        self.right_toolbar = QWidget()
        self.right_toolbar.setMinimumWidth(120)
        right_toolbar_layout = QGridLayout(self.right_toolbar)
        right_toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Updated tool list (added hotkeys settings and hide/show points tools)
        self.tool_list_right = [
            ("resources/load.png", "Load PointCloud", self.load_ply_file),
            ("resources/settings.png", "Settings", self.settings),
            ("resources/increase_point_size.png", "Increase point size", self.increase_point_size),
            ("resources/decrease_point_size.png", "Decrease point size", self.decrease_point_size),
            ("resources/reset_view.png", "Reset View", self.reset_view),
            ("resources/lasso.png", "Lasso segmentation", self.lasso_segmentation),
            ("resources/label_semantics.png", "Label semantics", self.label_semantics),
            ("resources/label_instance.png", "Label instance", self.instance_segmentation),
            ("resources/wand.png", "Wand tool", self.activate_wand_tool),
            ("resources/deselect.png", "Deselect selection", self.deselect_selection),
            ("resources/hide_selected.png", "Hide selected points", self.hide_selected_points),
            ("resources/show_all.png", "Show all points", self.show_all_points),
            ("resources/reggrow.png", "Start regioin growing", self.region_growing),
            ("resources/reggrow_settings.png", "Region growing settings", self.change_region_growing_settings)
        ]
        row = 0
        col = 0
        for icon_path, tooltip, func in self.tool_list_right:
            btn = QPushButton()
            btn.setIcon(QIcon(icon_path))
            btn.setToolTip(tooltip)
            btn.setIconSize(QSize(36, 36))
            btn.clicked.connect(func)
            right_toolbar_layout.addWidget(btn, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1
        splitter.addWidget(self.right_toolbar)
        v_layout.addWidget(splitter)

        # ---------------------------
        # VTK Setup
        # ---------------------------
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()

        self.pivot_point = [0, 0, 0]
        self.pivot_sphere_source = None
        self.pivot_actor = None
        self.point_cloud_actor = None

        self.active_selection = None
        self.last_active_selection = None
        self.selection_actor = None
        self.selection_timer = QTimer()
        self.selection_timer.timeout.connect(self.update_selection_pulse)
        self.selection_phase = 0.0
        self.selection_base_color = (1.0, 1.0, 1.0)

        self.lasso_active = False
        self.lasso_paused = False
        self.lasso_points = []
        self.lasso_candidate_indices = np.array([], dtype=int)
        self.lasso_actor = None
        self.lasso_polygon_actor = None
        self.lasso_control_panel = None
        self.lasso_click_observer = None
        self.lasso_mouse_move_observer = None
        self.lasso_key_observer = None

        self.update_interactor()
        self.vtkWidget.GetRenderWindow().Render()

    # ---------------------------
    # Layout and Tool Activation Methods
    # ---------------------------
    def toggle_left_sidebar(self):
        if self.left_sidebar.isVisible():
            self.left_sidebar.hide()
        else:
            self.left_sidebar.show()

    def deactivate_tool(self):
        self.active_tool = None
        QApplication.restoreOverrideCursor()

    def activate_tool(self, tool_name, icon_path):
        self.deactivate_tool()
        self.active_tool = tool_name
        pixmap = QPixmap(icon_path)
        if tool_name == "wand":
            pixmap = pixmap.scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        cursor = QCursor(pixmap,0, 0)
        QApplication.setOverrideCursor(cursor)

    def activate_wand_tool(self):
        if self.active_segmentation_mode is None:
            QMessageBox.warning(self, "Wand Tool", "No semantic or instance labels are active.")
            return
        print("Wand tool activated.")
        self.activate_tool("wand", "resources/wand_cursor.png")
    
    # ---------------------------
    # VTK and Rendering Methods
    # ---------------------------
    def update_interactor(self):
        interactor_style = InteractorStyle(self.renderer, self)
        self.interactor.SetInteractorStyle(interactor_style)

    def zoom_camera(self, factor):
        camera = self.renderer.GetActiveCamera()
        camera_pos = np.array(camera.GetPosition())
        focal_point = np.array(self.pivot_point)
        direction = focal_point - camera_pos
        new_position = camera_pos + direction * (1 - factor)
        camera.SetPosition(*new_position)
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def toggle_projection(self):
        camera = self.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            print("Switching to Perspective Projection")
            camera.SetParallelProjection(False)
        else:
            print("Switching to Parallel Projection")
            camera.SetParallelProjection(True)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def set_pivot_point(self, new_pivot):
        self.pivot_point = list(new_pivot)
        self.renderer.GetActiveCamera().SetFocalPoint(*new_pivot)
        self.renderer.ResetCameraClippingRange()
        self.update_pivot_visualization()
        self.vtkWidget.GetRenderWindow().Render()

    def update_pivot_visualization(self):
        if self.pivot_sphere_source is None:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(0.05)
            sphere.SetThetaResolution(16)
            sphere.SetPhiResolution(16)
            sphere.SetCenter(*self.pivot_point)
            self.pivot_sphere_source = sphere
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            self.pivot_actor = vtk.vtkActor()
            self.pivot_actor.SetMapper(mapper)
            self.pivot_actor.GetProperty().SetColor(0.5, 0, 0.5)
            self.pivot_actor.GetProperty().SetOpacity(0.5)
            self.pivot_actor.GetProperty().LightingOff()
            self.renderer.AddActor(self.pivot_actor)
        else:
            self.pivot_sphere_source.SetCenter(*self.pivot_point)

    # ---------------------------
    # PLY Loading and Field Handling
    # ---------------------------
    def load_ply_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PLY File", "", "PLY Files (*.ply)")
        if file_path:
            print(f"Loading: {file_path}")
            ply_data = PlyData.read(file_path)
            self.display_ply(ply_data)
            self.load_fields_from_ply(ply_data)
            self.loaded_ply_name = os.path.basename(file_path)

    def display_ply(self, ply_data):
        vertex_data = ply_data['vertex']
        points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
        initial_pivot = np.mean(points, axis=0).tolist()
        self.all_points = points.copy()
        has_color = all(c in vertex_data.data.dtype.names for c in ('red', 'green', 'blue'))
        if has_color:
            colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T
        else:
            colors = np.full((len(points), 3), [255, 255, 255])
        self.all_colors = colors.copy()

        # Initialize the visibility mask (all points visible)
        self.visibility_mask = np.ones(len(points), dtype=bool)

        vtk_points = vtk.vtkPoints()
        points_vtk = ns.numpy_to_vtk(points, deep=False)
        vtk_points.SetData(points_vtk)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        # Use RGBA colors: we update later using the mask.
        self.poly_data = poly_data

        vertex = vtk.vtkVertexGlyphFilter()
        vertex.SetInputData(poly_data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex.GetOutputPort())
        mapper.SetScalarModeToUsePointData()
        mapper.Update()

        lod_points = points[::100, ...]
        lod_colors = colors[::100, ...]
        points_vtk_lod0 = ns.numpy_to_vtk(lod_points, deep=False)
        colors_vtk_lod0 = ns.numpy_to_vtk(lod_colors.astype(np.uint8), deep=False)
        vtk_points_lod0 = vtk.vtkPoints()
        vtk_points_lod0.SetData(points_vtk_lod0)
        poly_data_lod0 = vtk.vtkPolyData()
        poly_data_lod0.SetPoints(vtk_points_lod0)
        poly_data_lod0.GetPointData().SetScalars(colors_vtk_lod0)
        vertex_lod0 = vtk.vtkVertexGlyphFilter()
        vertex_lod0.SetInputData(poly_data_lod0)
        mapper_lod0 = vtk.vtkPolyDataMapper()
        mapper_lod0.SetInputConnection(vertex_lod0.GetOutputPort())
        mapper_lod0.SetScalarModeToUsePointData()
        mapper_lod0.Update()

        actor = vtk.vtkLODActor()
        actor.SetMapper(mapper)
        actor.AddLODMapper(mapper_lod0)
        actor.GetProperty().SetPointSize(2)

        self.point_cloud_actor = actor
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.set_pivot_point(initial_pivot)
        self.update_interactor()
        self.vtkWidget.GetRenderWindow().Render()
        # Apply initial colors (which will use the visibility mask)
        self.update_point_cloud_actor_colors(self.all_colors)

    def load_fields_from_ply(self, ply_data):
        vertex_names = ply_data['vertex'].data.dtype.names
        if all(c in vertex_names for c in ('red', 'green', 'blue')):
            rgb = np.vstack((ply_data['vertex']['red'],
                             ply_data['vertex']['green'],
                             ply_data['vertex']['blue'])).T.astype(np.float32) / 255.0
            self.fields["RGB"] = (rgb, "vector")
        for name in vertex_names:
            if name not in ['x','y','z','red','green','blue']:
                try:
                    values = ply_data['vertex'][name].astype(np.float32)
                    dialog = ScalarFieldMappingDialog(self, name, values)
                    if dialog.exec():
                        cmap = dialog.selected_colormap()
                    else:
                        cmap = "gray"
                    self.fields[name] = (values, "scalar", cmap)
                except Exception as e:
                    print(f"Skipping field {name}: {e}")
        self.update_fields_browser()

    def update_fields_browser(self):
        self.fields_tree.clear()
        for key, field in self.fields.items():
            if field[1] == "vector":
                item = QTreeWidgetItem([key, "Vector"])
            else:
                cmap = field[2] if len(field) > 2 else "gray"
                item = QTreeWidgetItem([key, f"Scalar ({cmap})"])
            self.fields_tree.addTopLevelItem(item)
        self.fields_tree.expandAll()
        self.fields_tree.setColumnWidth(0, int(self.fields_tree.width() * 0.6))

    # ---------------------------
    # Segmentation and Fields Browser Methods
    # ---------------------------
    def create_segmentation(self):
        if self.all_points is None:
            QMessageBox.warning(self, "Segmentation", "No point cloud loaded!")
            return
        base_name = "Segmentation"
        if self.loaded_ply_name is not None:
            base_name = self.loaded_ply_name.split('.')[0]
        existing_names = {s.name for s in self.segmentations}
        i = 1
        while True:
            new_name = f'{base_name}_{i:03d}'
            if new_name not in existing_names:
                break
            i += 1

        seg = IndexSegmentation(name = new_name, size=len(self.all_points))
        self.segmentations.append(seg)
        self.update_segmentation_browser()
        print(f"Created new segmentation: {seg.name}")

    def load_segmentation(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Segmentation", "", "Segmentation Files (*.seg)")
        if file_path:
            seg = IndexSegmentation.load(file_path)
            base_name = seg.name
            names = [s.name for s in self.segmentations]
            if base_name in names:
                count = sum(1 for n in names if n.startswith(base_name))
                seg.name = f"{base_name}({count})"
            self.segmentations.append(seg)
            self.update_segmentation_browser()

        

    def update_segmentation_browser(self):
        self.seg_tree.clear()
        for seg in self.segmentations:
            parent = QTreeWidgetItem([seg.name, "", f"{len(seg.semantic_idx)}", "100%"])
            parent.setIcon(0, QIcon("resources/segmentation.png"))
            parent.setData(0, Qt.ItemDataRole.UserRole, seg)
            stat = seg.get_statistic()
            # Add Semantic labels node
            sem_item = QTreeWidgetItem(["Semantic labels", "", "", ""])
            sem_item.setIcon(0, QIcon("resources/semantic.png"))
            for sem_class in seg.semantic_schema.semantic_classes:
                count = np.sum(seg.semantic_idx == sem_class.id) if seg.semantic_idx is not None else 0
                pct = f"{(count/len(seg.semantic_idx)*100):.1f}%" if seg.semantic_idx is not None else ""
                child = QTreeWidgetItem([sem_class.name, str(sem_class.id), str(count), pct])
                child.setIcon(0, create_color_icon(sem_class.color))
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsEditable)
                child.setData(0, Qt.ItemDataRole.UserRole, sem_class)
                sem_item.addChild(child)
            parent.addChild(sem_item)
            # Add Instance segmentation nodes: two child nodes "instance" and "stuff"
            total = len(seg.instance_idx)
            count_instance = np.sum(seg.instance_idx != -1)
            count_stuff = np.sum(seg.instance_idx == -1)
            inst_item = QTreeWidgetItem(["instance", "", str(count_instance), f"{(count_instance/total)*100:.1f}%"])
            stuff_item = QTreeWidgetItem(["stuff", "", str(count_stuff), f"{(count_stuff/total)*100:.1f}%"])
            inst_item.setIcon(0, QIcon("resources/instance.png"))
            stuff_item.setIcon(0, QIcon("resources/instance.png"))
            parent.addChild(inst_item)
            parent.addChild(stuff_item)
            self.seg_tree.addTopLevelItem(parent)
        self.seg_tree.expandAll()
        self.seg_tree.setColumnWidth(0, int(self.seg_tree.width() * 0.6))

    def on_seg_tree_selection_changed(self):
        items = self.seg_tree.selectedItems()
        if not items:
            return
        item = items[0]
        if item.text(0) == "Semantic labels":
            parent = item.parent()
            if parent is not None:
                self.active_segmentation = parent.data(0, Qt.ItemDataRole.UserRole)
                self.active_segmentation_mode = "semantic"
                self.apply_segmentation_coloring(mode="semantic")
        elif item.text(0) in ["instance", "stuff"]:
            parent = item.parent()
            if parent is not None:
                self.active_segmentation = parent.data(0, Qt.ItemDataRole.UserRole)
                self.active_segmentation_mode = "instance"
                self.apply_segmentation_coloring(mode="instance")

    def on_fields_tree_selection_changed(self):
        items = self.fields_tree.selectedItems()
        if not items:
            return
        item = items[0]
        field_name = item.text(0)
        self.active_field = field_name
        self.apply_field_coloring()

    def on_seg_tree_context_menu(self, pos):
        item = self.seg_tree.itemAt(pos)
        if item is None:
            return
        menu = QMenu()
        parent = item.parent()
        print(f'DEBUG: called context item for {item.data(0, Qt.ItemDataRole.UserRole)}')
        if parent is None: # TODO: rather weird way to test for INdexObject --> instead just test for the type of selected obj.
            # Top-level segmentation: allow adding a new semantic class
            add_action = menu.addAction("Add new semantic class")
            export_seg_action = menu.addAction("Export Segmentation")
            load_schema_action = menu.addAction("Load Semantic Schema")
            save_schema_action = menu.addAction("Save Semantic Schema")
            reindex_schema_action = menu.addAction("Reindex Semantic Schema")
            action = menu.exec(self.seg_tree.viewport().mapToGlobal(pos))
            if action == add_action:
                seg = item.data(0, Qt.ItemDataRole.UserRole)
                self.add_new_semantic_class(seg)
            elif action == load_schema_action:
                seg = item.data(0, Qt.ItemDataRole.UserRole)
                assert isinstance(seg, IndexSegmentation)
                if len(seg.semantic_schema.semantic_classes) > 1:
                    warn = QMessageBox.warning(self, "Overwrite Schema",
                        "This will overwrite the existing semantic schema. Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if warn != QMessageBox.StandardButton.Yes:
                        return
                file_path, _ = QFileDialog.getOpenFileName(self, "Load Semantic Schema", "", "JSON Files (*.json)")
                if file_path:
                    new_schema = SemanticSchema.from_json(file_path)
                    seg.replace_semantic_schema(new_schema)
                    self.update_segmentation_browser()
            elif action == export_seg_action:
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Segmentation", "", "Segmentation Files (*.seg)")
                if file_path:
                    seg = item.data(0, Qt.ItemDataRole.UserRole)
                    assert isinstance(seg, IndexSegmentation)
                    seg.save(file_path)
            elif action == save_schema_action:
                seg = item.data(0, Qt.ItemDataRole.UserRole)
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Semantic Schema", f"{seg.name}_schema.json", "JSON Files (*.json)")
                if file_path:
                    seg.semantic_schema.to_json(file_path)
                    QMessageBox.information(self, "Saved", f"Schema saved to:\n{file_path}")
            elif action == reindex_schema_action:
                seg = item.data(0, Qt.ItemDataRole.UserRole)
                assert isinstance(seg, IndexSegmentation)
                seg.semantic_schema.reindex()
                self.update_segmentation_browser()
        else:
            # For semantic or instance children, add context actions for show/hide points.
            if parent.text(0) == "Semantic labels":
                edit_action = menu.addAction("Edit semantic class")
                show_action = menu.addAction("Show points")
                hide_action = menu.addAction("Hide points")
                delete_action = menu.addAction("Delete semantic class")
                action = menu.exec(self.seg_tree.viewport().mapToGlobal(pos))
                if action == edit_action:
                    sem_class = item.data(0, Qt.ItemDataRole.UserRole)
                    self.edit_semantic_class(sem_class)
                elif action == show_action:
                    sem_class = item.data(0, Qt.ItemDataRole.UserRole)
                    if self.active_segmentation:
                        indices = np.where(self.active_segmentation.semantic_idx == sem_class.id)[0]
                        self.set_points_visibility(indices, True)
                elif action == hide_action:
                    sem_class = item.data(0, Qt.ItemDataRole.UserRole)
                    if self.active_segmentation:
                        indices = np.where(self.active_segmentation.semantic_idx == sem_class.id)[0]
                        self.set_points_visibility(indices, False)
                elif action == delete_action:
                    sem_class = item.data(0, Qt.ItemDataRole.UserRole)
                    assert isinstance(sem_class, SemanticClass)
                    sem = item.parent().parent().data(0, Qt.ItemDataRole.UserRole)
                    assert isinstance(sem, IndexSegmentation)
                    sem.remove_semantic_class(sem_class.id)
                    self.update_segmentation_browser()



            elif parent.text(0) in ["instance", "stuff"]:
                show_action = menu.addAction("Show points")
                hide_action = menu.addAction("Hide points")
                action = menu.exec(self.seg_tree.viewport().mapToGlobal(pos))
                if action == show_action:
                    if self.active_segmentation:
                        if item.text(0) == "stuff":
                            indices = np.where(self.active_segmentation.instance_idx == -1)[0]
                        else:
                            indices = np.where(self.active_segmentation.instance_idx != -1)[0]
                        self.set_points_visibility(indices, True)
                    
                elif action == hide_action:
                    if self.active_segmentation:
                        if item.text(0) == "stuff":
                            indices = np.where(self.active_segmentation.instance_idx == -1)[0]
                        else:
                            indices = np.where(self.active_segmentation.instance_idx != -1)[0]
                        self.set_points_visibility(indices, False)

    def add_new_semantic_class(self, seg):
        dialog = SemanticClassEditorDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_sem = SemanticClass(id=dialog.id, name=dialog.name, color=dialog.color)
            try:
                seg.semantic_schema.add_semantic_class(new_sem)
            except AssertionError as e:
                QMessageBox.warning(self, "Error", str(e))
            self.update_segmentation_browser()

    def edit_semantic_class(self, sem_class):
        dialog = SemanticClassEditorDialog(self, sem_class)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            sem_class.id = dialog.id
            sem_class.name = dialog.name
            sem_class.color = dialog.color
            self.update_segmentation_browser()

    def on_seg_tree_item_changed(self, item, column):
        parent = item.parent()
        if parent and parent.text(0) == "Semantic labels":
            sem_class = item.data(0, Qt.ItemDataRole.UserRole)
            if sem_class:
                if column == 0:
                    sem_class.name = item.text(0)
                elif column == 1:
                    try:
                        sem_class.id = int(item.text(1))
                    except:
                        pass

    def on_seg_tree_item_double_clicked(self, item, column):
        parent = item.parent()
        if parent is None:
            return
        if parent.text(0) == "Semantic labels":
            sem_class = item.data(0, Qt.ItemDataRole.UserRole)
            if sem_class and self.active_segmentation:
                indices = np.where(self.active_segmentation.semantic_idx == sem_class.id)[0]
                self.active_selection = indices
                self.update_selection_visualization()
        elif parent.text(0) in ["instance", "stuff"]:
            if self.active_segmentation:
                if item.text(0) == "stuff":
                    indices = np.where(self.active_segmentation.instance_idx == -1)[0]
                elif item.text(0) == "instance":
                    indices = np.where(self.active_segmentation.instance_idx != -1)[0]
                self.active_selection = indices
                self.update_selection_visualization()

    # ---------------------------
    # Coloring and Visualization Methods
    # ---------------------------
    def apply_segmentation_coloring(self, mode="semantic"):
        if self.active_segmentation is None or self.all_points is None:
            return
        new_colors = np.zeros((len(self.all_points), 3), dtype=np.uint8)
        if mode == "semantic":
            for sem_class in self.active_segmentation.semantic_schema.semantic_classes:
                mask = (self.active_segmentation.semantic_idx == sem_class.id)
                color = np.array(sem_class.color) * 255
                new_colors[mask] = color.astype(np.uint8)
        elif mode == "instance":
            unique_ids = np.unique(self.active_segmentation.instance_idx)
            color_map = {}
            for id_val in unique_ids:
                if id_val == -1:
                    color_map[id_val] = np.array([204, 204, 204])
                else:
                    color_map[id_val] = (np.array(generate_vibrant_color()) * 255).astype(np.uint8)
            for id_val, color in color_map.items():
                mask = (self.active_segmentation.instance_idx == id_val)
                new_colors[mask] = color
        self.update_point_cloud_actor_colors(new_colors)

    def apply_field_coloring(self):
        if self.active_field is None or self.all_points is None:
            return
        field = self.fields.get(self.active_field)
        if field is None:
            return
        if field[1] == "vector":
            colors = (field[0] * 255).astype(np.uint8)
        else:
            values = field[0]
            cmap = cm.get_cmap(field[2])
            norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
            mapped = cmap(norm(values))[:, :3]
            colors = (mapped * 255).astype(np.uint8)
        self.update_point_cloud_actor_colors(colors)

    def update_point_cloud_actor_colors(self, new_colors):
        if self.poly_data is None:
            return
        # Incorporate visibility: use alpha=255 for visible points, 0 for hidden.
        if self.visibility_mask is None or len(self.visibility_mask) != new_colors.shape[0]:
            self.visibility_mask = np.ones(new_colors.shape[0], dtype=bool)
        alphas = np.where(self.visibility_mask, 255, 0).astype(np.uint8)
        new_colors_rgba = np.concatenate([new_colors, alphas[:, None]], axis=1)
        vtk_colors = ns.numpy_to_vtk(new_colors_rgba.astype(np.uint8), deep=True)
        vtk_colors.SetNumberOfComponents(4)
        self.poly_data.GetPointData().SetScalars(vtk_colors)
        self.poly_data.Modified()
        self.vtkWidget.GetRenderWindow().Render()
        if new_colors.size:
            self.selection_base_color = tuple(np.mean(new_colors, axis=0)/255.0)

    # ---------------------------
    # New Tool Functions
    # ---------------------------
    def label_semantics(self):
        if self.active_selection is None or self.active_segmentation is None:
            QMessageBox.warning(self, "Label Semantics", "No active selection or segmentation available.")
            return
        dialog = LabelSemanticsDialog(self, self.segmentations, default_segmentation=self.active_segmentation)
        if dialog.exec():
            seg = dialog.seg_combo.currentData()
            semantic_id = dialog.sem_combo.currentData()
            seg.semantic_idx[self.active_selection] = semantic_id
            print(f"Labeled {len(self.active_selection)} points with semantic id {semantic_id}.")
            self.apply_segmentation_coloring(mode="semantic")
            self.deselect_selection()

    def instance_segmentation(self):
        if self.active_selection is None or self.active_segmentation is None:
            QMessageBox.warning(self, "Instance Segmentation", "No active selection or segmentation available.")
            return
        try:
            new_id = self.active_segmentation.get_next_instance_id()
            self.active_segmentation.instance_idx[self.active_selection] = new_id
            print(f"Assigned instance id {new_id} to {len(self.active_selection)} points.")
            self.apply_segmentation_coloring(mode="instance")
            self.deselect_selection()
        except Exception as e:
            QMessageBox.warning(self, "Instance Segmentation", str(e))
    def wand_pick(self):
        x, y = self.interactor.GetEventPosition()
        picker = vtk.vtkPointPicker()
        picker.Pick(x, y, 0, self.renderer)
        pt_id = picker.GetPointId()
        if pt_id < 0:
            print("No point picked with wand.")
            return
        if self.active_segmentation_mode == "semantic":
            label_value = self.active_segmentation.semantic_idx[pt_id]
            picked_indices = np.where(self.active_segmentation.semantic_idx == label_value)[0]
        elif self.active_segmentation_mode == "instance":
            label_value = self.active_segmentation.instance_idx[pt_id]
            picked_indices = np.where(self.active_segmentation.instance_idx == label_value)[0]
        else:
            print("Wand tool not active: segmentation mode not set.")
            return

        shift = self.interactor.GetShiftKey()
        ctrl = self.interactor.GetControlKey()
        if not shift and not ctrl:
            new_selection = picked_indices
        elif shift and not ctrl:
            if self.active_selection is None:
                new_selection = picked_indices
            else:
                new_selection = np.union1d(self.active_selection, picked_indices)
        elif ctrl and not shift:
            if self.active_selection is None:
                new_selection = np.array([], dtype=int)
            else:
                new_selection = np.setdiff1d(self.active_selection, picked_indices)
        elif ctrl and shift:
            if self.active_selection is None:
                new_selection = np.array([], dtype=int)
            else:
                new_selection = np.intersect1d(self.active_selection, picked_indices)
        # self.active_selection = new_selection
        self.update_active_selection(new_selection)
        print(f"Wand tool selected {len(new_selection)} points with label {label_value}.")
        self.update_selection_visualization()

    def update_active_selection(self, new_selection):
        self.last_active_selection = self.active_selection
        self.active_selection = new_selection
        
    def undo_update_active_selection(self):
        temp = self.active_selection
        self.active_selection = self.last_active_selection
        self.last_active_selection = temp
        self.update_selection_visualization()


    def deselect_selection(self):
        self.active_selection = None
        self.point_cloud_actor.GetProperty().SetOpacity(1.0)
        if self.selection_actor is not None:
            self.renderer.RemoveActor(self.selection_actor)
            self.selection_actor = None
        self.selection_timer.stop()
        print("Active selection cleared.")
        self.vtkWidget.GetRenderWindow().Render()

    # ---------------------------
    # Selection Visualization Methods
    # ---------------------------
    def update_selection_visualization(self):
        if self.active_selection is not None and self.active_selection.size > 0:
            selected_points = self.all_points[self.active_selection]
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(ns.numpy_to_vtk(selected_points, deep=True))
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(vtk_points)
            vertex = vtk.vtkVertexGlyphFilter()
            vertex.SetInputData(poly_data)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertex.GetOutputPort())
            mapper.ScalarVisibilityOff()
            if self.selection_actor is None:
                self.selection_actor = vtk.vtkActor()
                base_size = self.point_cloud_actor.GetProperty().GetPointSize()
                self.selection_actor.GetProperty().SetPointSize(base_size + 2)
                self.renderer.AddActor(self.selection_actor)
                sel_colors = self.all_colors[self.active_selection]
                self.selection_base_color = tuple(np.mean(sel_colors, axis=0)/255.0)
            self.selection_actor.SetMapper(mapper)
            if not self.selection_timer.isActive():
                self.selection_phase = 0.0
                self.selection_timer.start(50)
        else:
            if self.selection_actor is not None:
                self.renderer.RemoveActor(self.selection_actor)
                self.selection_actor = None
            self.selection_timer.stop()
        self.vtkWidget.GetRenderWindow().Render()

    def update_selection_pulse(self):
        if self.selection_actor is None:
            return
        self.selection_phase += 0.1
        t = (math.sin(self.selection_phase*4) + 1) / 2
        orange = (1.0, 0.5, 0.0)
        blended = blend_colors(self.selection_base_color, orange, t)
        self.selection_actor.GetProperty().SetColor(blended)
        self.vtkWidget.GetRenderWindow().Render()

    # ---------------------------
    # Other Tool Functions
    # ---------------------------
    def settings(self):
        dlg = HotkeysSettingsDialog(self, self.hotkeys)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_hotkeys = dlg.get_hotkeys()
            self.hotkeys = new_hotkeys
            # Save to settings.ini
            config = configparser.ConfigParser()
            config["Hotkeys"] = self.hotkeys
            with open("settings.ini", "w") as f:
                config.write(f)
            QMessageBox.information(self, "Settings", "Hotkeys saved.")

    def increase_point_size(self):
        if self.point_cloud_actor:
            current_size = self.point_cloud_actor.GetProperty().GetPointSize()
            new_size = current_size + 1
            print(f"Increasing point size: {current_size} -> {new_size}")
            self.point_cloud_actor.GetProperty().SetPointSize(new_size)
            self.vtkWidget.GetRenderWindow().Render()

    def decrease_point_size(self):
        if self.point_cloud_actor:
            current_size = self.point_cloud_actor.GetProperty().GetPointSize()
            new_size = max(1, current_size - 1)
            print(f"Decreasing point size: {current_size} -> {new_size}")
            self.point_cloud_actor.GetProperty().SetPointSize(new_size)
            self.vtkWidget.GetRenderWindow().Render()

    def reset_view(self):
        self.renderer.ResetCamera()
        print("Reset view.")
        self.vtkWidget.GetRenderWindow().Render()

    # ---------------------------
    # Lasso Segmentation Methods (with camera rotation allowed during pause)
    # ---------------------------
    def lasso_segmentation(self):
        if self.all_points is None:
            QMessageBox.warning(self, "Lasso Segmentation", "No point cloud loaded!")
            return
        print("Lasso segmentation activated.")
        self.active_tool = "lasso"
        self.lasso_active = True
        self.lasso_paused = False
        self.lasso_points = []
        # Initially, candidate indices are all visible points.
        self.lasso_candidate_indices = np.arange(len(self.all_points))
        self.create_lasso_actor()
        self.interactor.SetInteractorStyle(None)
        self.lasso_control_panel = QDialog(self)
        self.lasso_control_panel.setWindowTitle("Lasso Controls")
        self.lasso_control_panel.setWindowFlags(self.lasso_control_panel.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.lasso_control_panel.finished.connect(lambda result: self.lasso_reject())
        panel_layout = QHBoxLayout(self.lasso_control_panel)
        btn_pause = QPushButton("Pause")
        btn_pause.clicked.connect(self.lasso_toggle_pause)
        btn_escape = QPushButton("Escape")
        btn_escape.clicked.connect(self.lasso_escape)
        btn_in = QPushButton("In")
        btn_in.clicked.connect(self.lasso_in)
        btn_out = QPushButton("Out")
        btn_out.clicked.connect(self.lasso_out)
        btn_accept = QPushButton("Accept")
        btn_accept.clicked.connect(self.lasso_accept)
        btn_reject = QPushButton("Reject")
        btn_reject.clicked.connect(self.lasso_reject)
        for btn in [btn_pause, btn_escape, btn_in, btn_out, btn_accept, btn_reject]:
            panel_layout.addWidget(btn)
        self.lasso_control_panel.setLayout(panel_layout)
        pos = self.vtkWidget.mapToGlobal(self.vtkWidget.rect().topLeft())
        self.lasso_control_panel.move(pos.x() + 10, pos.y() + 10)
        self.lasso_control_panel.show()
        self.lasso_click_observer = self.interactor.AddObserver("LeftButtonPressEvent", self.lasso_click_event)
        self.lasso_mouse_move_observer = self.interactor.AddObserver("MouseMoveEvent", self.lasso_mouse_move_event)
        self.lasso_key_observer = self.interactor.AddObserver("KeyPressEvent", self.lasso_key_event)

    def lasso_click_event(self, obj, event):
        if not self.lasso_active or self.lasso_paused:
            return
        pos = self.interactor.GetEventPosition()
        self.lasso_points.append(pos)
        self.update_lasso_polygon_actor()

    def lasso_mouse_move_event(self, obj, event):
        if self.lasso_active and not self.lasso_paused and len(self.lasso_points) > 0:
            ghost = self.interactor.GetEventPosition()
            self.update_lasso_polygon_actor(ghost_point=ghost)
        return

    def update_lasso_polygon_actor(self, ghost_point=None):
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None
        points_list = list(self.lasso_points)
        if ghost_point is not None:
            points_list.append(ghost_point)
        if len(points_list) < 2:
            self.vtkWidget.GetRenderWindow().Render()
            return
        points = vtk.vtkPoints()
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(points_list))
        for i, (x, y) in enumerate(points_list):
            points.InsertNextPoint(x, y, 0)
            polyLine.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(polyData)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.5, 0)
        actor.GetProperty().SetLineWidth(4)
        self.lasso_polygon_actor = actor
        self.renderer.AddActor(actor)
        self.vtkWidget.GetRenderWindow().Render()

    def create_lasso_actor(self):
        candidate_points = self.all_points[self.lasso_candidate_indices]
        candidate_colors = self.all_colors[self.lasso_candidate_indices]
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(candidate_points, deep=True))
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        colors_vtk = ns.numpy_to_vtk(candidate_colors.astype(np.uint8), deep=True)
        poly_data.GetPointData().SetScalars(colors_vtk)
        vertex = vtk.vtkVertexGlyphFilter()
        vertex.SetInputData(poly_data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex.GetOutputPort())
        if self.lasso_actor is None:
            self.lasso_actor = vtk.vtkLODActor()
            self.lasso_actor.GetProperty().SetPointSize(self.point_cloud_actor.GetProperty().GetPointSize())
            self.renderer.AddActor(self.lasso_actor)
        self.lasso_actor.SetMapper(mapper)
        self.vtkWidget.GetRenderWindow().Render()

    def lasso_toggle_pause(self):
        self.lasso_paused = not self.lasso_paused
        if self.lasso_paused:
            # Switch to default style to allow full camera rotation, zoom, etc.
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            if self.lasso_polygon_actor is not None:
                self.renderer.RemoveActor(self.lasso_polygon_actor)
                self.lasso_polygon_actor = None
            print("Lasso segmentation paused: camera rotation enabled.")
        else:
            # Resume lasso mode by disabling default camera interactions.
            self.interactor.SetInteractorStyle(None)
            print("Lasso segmentation resumed: lasso selection active.")
        self.vtkWidget.GetRenderWindow().Render()

    def lasso_escape(self):
        print("Lasso polygon cleared (escape).")
        self.lasso_points = []
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None
        self.vtkWidget.GetRenderWindow().Render()

    # def point_in_polygon(self, x, y, poly):
    #     num = len(poly)
    #     j = num - 1
    #     c = False
    #     for i in range(num):
    #         xi, yi = poly[i]
    #         xj, yj = poly[j]
    #         if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
    #             c = not c
    #         j = i
    #     return c

    def lasso_in(self):
        if len(self.lasso_points) < 3:
            print("Lasso polygon not complete (need at least 3 points).")
            return
        poly = self.lasso_points
        candidate_indices = self.lasso_candidate_indices
        candidate_points = self.all_points[candidate_indices]
        display_coords = self.compute_display_coordinates(candidate_points)
        lasso_poly = np.array(poly)
        min_x, max_x = lasso_poly[:, 0].min(), lasso_poly[:, 0].max()
        min_y, max_y = lasso_poly[:, 1].min(), lasso_poly[:, 1].max()
        bbox_mask = (
            (display_coords[:, 0] >= min_x) & (display_coords[:, 0] <= max_x) &
            (display_coords[:, 1] >= min_y) & (display_coords[:, 1] <= max_y)
        )
        indices_in_bbox = np.nonzero(bbox_mask)[0]
        filtered_display_coords = display_coords[indices_in_bbox]
        inside_mask = self.vectorized_point_in_poly(filtered_display_coords, lasso_poly)
        # For "in", keep points inside the polygon and hide those outside.
        selected_in_bbox = indices_in_bbox[inside_mask]
        new_candidates = candidate_indices[selected_in_bbox]
        self.lasso_candidate_indices = new_candidates
        print(f"Lasso 'in': {len(self.lasso_candidate_indices)} candidates remain.")
        # Hide the points that are not in the candidate set by updating the visibility mask.
        hide_indices = np.setdiff1d(np.arange(len(self.all_points)), self.lasso_candidate_indices)
        self.set_points_visibility(hide_indices, False)
        self.create_lasso_actor()
        self.lasso_points = []
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None

    def lasso_out(self):
        if len(self.lasso_points) < 3:
            print("Lasso polygon not complete (need at least 3 points).")
            return
        poly = self.lasso_points
        candidate_indices = self.lasso_candidate_indices
        candidate_points = self.all_points[candidate_indices]
        display_coords = self.compute_display_coordinates(candidate_points)
        lasso_poly = np.array(poly)
        min_x, max_x = lasso_poly[:, 0].min(), lasso_poly[:, 0].max()
        min_y, max_y = lasso_poly[:, 1].min(), lasso_poly[:, 1].max()
        bbox_mask = (
            (display_coords[:, 0] >= min_x) & (display_coords[:, 0] <= max_x) &
            (display_coords[:, 1] >= min_y) & (display_coords[:, 1] <= max_y)
        )
        indices_in_bbox = np.nonzero(bbox_mask)[0]
        filtered_display_coords = display_coords[indices_in_bbox]
        inside_mask = self.vectorized_point_in_poly(filtered_display_coords, lasso_poly)
        # For "out", keep points outside the polygon and hide those inside.
        outside_mask = ~inside_mask
        selected_in_bbox = indices_in_bbox[outside_mask]
        new_candidates = candidate_indices[selected_in_bbox]
        self.lasso_candidate_indices = new_candidates
        print(f"Lasso 'out': {len(self.lasso_candidate_indices)} candidates remain.")
        hide_indices = np.setdiff1d(np.arange(len(self.all_points)), self.lasso_candidate_indices)
        self.set_points_visibility(hide_indices, False)
        self.create_lasso_actor()
        self.lasso_points = []
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None

    def lasso_accept(self):
        print("Lasso selection accepted.")
        new_sel = self.lasso_candidate_indices.copy()
        print("New selection:", new_sel)
        if self.active_selection is not None and self.active_selection.size > 0:
            msg = QMessageBox(self)
            msg.setWindowTitle("Combine Selections")
            msg.setText("An active selection already exists. How would you like to combine it with the new selection?")
            union_btn = msg.addButton("Union", QMessageBox.ButtonRole.AcceptRole)
            intersect_btn = msg.addButton("Intersection", QMessageBox.ButtonRole.AcceptRole)
            diff_btn = msg.addButton("Difference (new removed from old)", QMessageBox.ButtonRole.AcceptRole)
            rev_diff_btn = msg.addButton("Reverse Difference (old removed from new)", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.exec()
            if msg.clickedButton() == union_btn:
                combined = np.union1d(self.active_selection, new_sel)
            elif msg.clickedButton() == intersect_btn:
                combined = np.intersect1d(self.active_selection, new_sel)
            elif msg.clickedButton() == diff_btn:
                combined = np.setdiff1d(self.active_selection, new_sel)
            elif msg.clickedButton() == rev_diff_btn:
                combined = np.setdiff1d(new_sel, self.active_selection)
            else:
                print("Combination cancelled; keeping previous selection.")
                combined = self.active_selection
            self.active_selection = combined
        else:
            self.active_selection = new_sel
        self.exit_lasso_mode()
        self.update_selection_visualization()

    def lasso_reject(self):
        print("Lasso selection rejected.")
        self.exit_lasso_mode()

    def lasso_key_event(self, obj, event):
        key = self.interactor.GetKeySym()
        if key == "space":
            self.lasso_toggle_pause()
        elif key == "Escape":
            self.lasso_escape()
        elif key.lower() == "i":
            self.lasso_in()
        elif key.lower() == "o":
            self.lasso_out()
        elif key == "Return":
            self.lasso_accept()
        return

    def exit_lasso_mode(self):
        self.lasso_active = False
        if self.lasso_click_observer is not None:
            self.interactor.RemoveObserver(self.lasso_click_observer)
            self.lasso_click_observer = None
        if self.lasso_mouse_move_observer is not None:
            self.interactor.RemoveObserver(self.lasso_mouse_move_observer)
            self.lasso_mouse_move_observer = None
        if self.lasso_key_observer is not None:
            self.interactor.RemoveObserver(self.lasso_key_observer)
            self.lasso_key_observer = None
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None
        if self.lasso_actor is not None:
            self.renderer.RemoveActor(self.lasso_actor)
            self.lasso_actor = None
        if self.lasso_control_panel is not None:
            self.lasso_control_panel.close()
            self.lasso_control_panel = None
        # Restore the original visibility mask if it was backed up.
        if hasattr(self, 'lasso_visibility_backup'):
            self.visibility_mask = self.lasso_visibility_backup.copy()
            self.update_point_cloud_actor_colors(self.all_colors)
            del self.lasso_visibility_backup
        self.point_cloud_actor.VisibilityOn()
        self.deactivate_tool()
        self.update_interactor()
        self.vtkWidget.GetRenderWindow().Render()

    def compute_display_coordinates(self, points):
        camera = self.renderer.GetActiveCamera()
        window_size = self.vtkWidget.GetRenderWindow().GetSize()
        width, height = window_size
        clipping_range = camera.GetClippingRange()
        aspect = self.renderer.GetTiledAspectRatio()
        matrix = camera.GetCompositeProjectionTransformMatrix(aspect, clipping_range[0], clipping_range[1])
        m = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                m[i, j] = matrix.GetElement(i, j)
        N = points.shape[0]
        pts_hom = np.hstack((points, np.ones((N, 1))))
        pts_clip = pts_hom @ m.T
        pts_ndc = pts_clip[:, :3] / pts_clip[:, 3:4]
        display_x = (pts_ndc[:, 0] + 1) / 2 * width
        display_y = (pts_ndc[:, 1] + 1) / 2 * height
        return np.column_stack((display_x, display_y))

    @staticmethod
    def vectorized_point_in_poly(points, poly):
        poly = np.asarray(poly)
        x = points[:, 0]
        y = points[:, 1]
        poly_x = poly[:, 0]
        poly_y = poly[:, 1]
        poly_x_next = np.roll(poly_x, -1)
        poly_y_next = np.roll(poly_y, -1)
        cond = ((poly_y > y[:, None]) != (poly_y_next > y[:, None])) & (
            x[:, None] < (poly_x_next - poly_x) * (y[:, None] - poly_y) / (poly_y_next - poly_y + 1e-12) + poly_x
        )
        inside = np.count_nonzero(cond, axis=1) % 2 == 1
        return inside

    # ---------------------------
    # New Hide/Show Points Methods
    # ---------------------------
    def set_points_visibility(self, indices, show: bool):
        if self.all_points is None:
            return
        self.visibility_mask[indices] = show
        # Reapply current coloring (which will use the updated mask)
        if self.active_segmentation_mode == "semantic" and self.active_segmentation is not None:
            self.apply_segmentation_coloring(mode="semantic")
        elif self.active_segmentation_mode == "instance" and self.active_segmentation is not None:
            self.apply_segmentation_coloring(mode="instance")
        elif self.active_field is not None:
            self.apply_field_coloring()
        else:
            self.update_point_cloud_actor_colors(self.all_colors)

    def hide_selected_points(self):
        if self.active_selection is None:
            QMessageBox.warning(self, "Hide Points", "No points selected to hide.")
            return
        self.set_points_visibility(self.active_selection, False)
        print(f"Hid {len(self.active_selection)} points.")

    def show_all_points(self):
        if self.all_points is None:
            return
        self.visibility_mask[:] = True
        if self.active_segmentation_mode == "semantic" and self.active_segmentation is not None:
            self.apply_segmentation_coloring(mode="semantic")
        elif self.active_segmentation_mode == "instance" and self.active_segmentation is not None:
            self.apply_segmentation_coloring(mode="instance")
        elif self.active_field is not None:
            self.apply_field_coloring()
        else:
            self.update_point_cloud_actor_colors(self.all_colors)
        print("All points shown.")

    def set_default_region_growing_setting(self):
        self.region_growing_settings = dict(
            epsilon = 0.03,
            alpha = np.pi * 0.15,
            min_points_in_region = 80,
            min_success_ratio = 0.10,
            max_dist_from_cent = 50.,
            epsilon_multiplier = 3., # Value in Poux & Kobbelt paper = 3
            epsilon_multiplier_average = 1.5, # average residual should be bellow value * epsilon, otherwise stop growing
            first_refit = 4,
            refit_multiplier = 2.,
            oriented_normals = False,
            verbose = True,
            tracker_size = 50,
            search_radius_approx = 0., # accuracy of radius search; 0. = exact sarch
            perform_cca = True, # divides each region in connected components after segmentation
        )
        self.region_growing_chunk_layout = (1,1,1)
    def change_region_growing_settings(self):
        dlg = RegionGrowingSettingsDialog(self, self.region_growing_settings, self.region_growing_chunk_layout)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_settings, new_chunk_layout = dlg.get_settings()
            self.region_growing_settings = new_settings
            self.region_growing_chunk_layout = new_chunk_layout
            print("Region growing settings updated.")
    
    def region_growing(self):
        print('called region growing')
        if self.all_points is None:
            QMessageBox.warning(self, "No point cloud loaded", "You need to load a point cloud first")
            return
        
        # create segmentation object
        base_name = "Segmentation"
        if self.loaded_ply_name is not None:
            base_name = self.loaded_ply_name.split('.')[0]
        existing_names = {s.name for s in self.segmentations}
        i = 1
        while True:
            new_name = f'{base_name}_{i:03d}'
            if new_name not in existing_names:
                break
            i += 1

        seg = IndexSegmentation(name = new_name, size=len(self.all_points))
        self.segmentations.append(seg)
        print(f"Created new segmentation: {seg.name}")

        # region growing
        if self.all_points is not None:
            cla = self.region_growing_chunk_layout
            pcd = rg.PointCloud(self.all_points.astype(np.float32), None)
            rg.region_growing(pcd, seg.instance_idx,cla[0],cla[1],cla[2],**self.region_growing_settings)
        
        # save settings as segmentation metadata
        seg.metadata = seg.metadata | dict(settings = 
            dict(region_growing_settings=self.region_growing_settings, 
                 region_growing_chunk_layout = self.region_growing_chunk_layout))
        self.update_segmentation_browser()
        

if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")
    set_dark_palette(app)
    window = VTKPointCloudViewer()
    window.show()
    app.exec()
