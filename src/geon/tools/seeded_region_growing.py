from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
import weakref

import numpy as np

from PyQt6.QtCore import QEventLoop, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSpinBox,
    QWidget,
)

from geon.tools.base import Event
from geon.util.resources import resource_path

from .base import ModeTool, ToolZone
from .selection import SelectPointsCmd
from ..algorithms.region_growing import SeededGrower, estimate_parameters
from ..data.pointcloud import FieldType
from ..rendering.pointcloud import PointCloudLayer
from ..ui.seeded_region_growing_dialog import SeededRegionGrowingDialog


@dataclass
class SeededRegionGrowingTool(ModeTool):
    label: ClassVar = "live_region_growing"
    tooltip: ClassVar = "Live region growing"
    icon_path: ClassVar = resource_path("reggrow.png")
    shortcut: ClassVar = None
    ui_zones: ClassVar = {ToolZone.SIDEBAR_RIGHT_ESSENTIALS}
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = True
    cursor_icon_path: ClassVar = resource_path("reggrow.png")
    cursor_hot: ClassVar = (3, 3)
    keep_focus: ClassVar[bool] = False

    epsilon: float = 0.03
    tau: int = 80
    alpha_deg: float = 29.0
    normal_mode_value: str = "compute"
    normal_field_name_value: str | None = None
    enable_seed_gating: bool = True
    seed_min_neighbors: int = 10
    seed_planarity_min: float = 0.20
    seed_scattering_max: float = 0.35
    epsilon_multiplier: float = 3.0
    refit_multiplier: float = 2.0
    first_refit: int = 4
    max_dist_from_cent: float = 50.0
    oriented_normals: bool = False
    perform_cca: bool = True
    estimate_sample_size: int = 50_000
    estimate_seed: int = 0

    _session: SeededGrower | None = field(default=None, init=False, repr=False)
    _session_signature: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _busy: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sync_normal_defaults()

    def _sync_normal_defaults(self) -> None:
        layer = self._active_pointcloud_layer()
        if layer is None:
            return
        normal_fields = layer.data.get_fields(field_type=FieldType.NORMAL)
        if self.normal_mode_value == "use_provided" and self.normal_field_name_value:
            names = [f.name for f in normal_fields]
            if self.normal_field_name_value in names:
                return
        if normal_fields:
            self.normal_mode_value = "use_provided"
            self.normal_field_name_value = normal_fields[0].name
        else:
            self.normal_mode_value = "compute"
            self.normal_field_name_value = None

    def _active_pointcloud_layer(self) -> PointCloudLayer | None:
        layer = self.ctx.scene.active_layer
        return layer if isinstance(layer, PointCloudLayer) else None

    def _params(self) -> dict[str, object]:
        return {
            "epsilon": float(self.epsilon),
            "tau": int(self.tau),
            "alpha_deg": float(self.alpha_deg),
            "epsilon_multiplier": float(self.epsilon_multiplier),
            "refit_multiplier": float(self.refit_multiplier),
            "first_refit": int(self.first_refit),
            "max_dist_from_cent": float(self.max_dist_from_cent),
            "oriented_normals": bool(self.oriented_normals),
            "perform_cca": bool(self.perform_cca),
            "enable_seed_gating": bool(self.enable_seed_gating),
            "seed_min_neighbors": int(self.seed_min_neighbors),
            "seed_planarity_min": float(self.seed_planarity_min),
            "seed_scattering_max": float(self.seed_scattering_max),
        }

    def _invalidate_session(self) -> None:
        self._session = None
        self._session_signature = None

    def _provided_normals(self, layer: PointCloudLayer) -> np.ndarray | None:
        if self.normal_mode_value != "use_provided":
            return None
        if not self.normal_field_name_value:
            raise RuntimeError("No normals field selected for live region growing.")
        normal_fields = layer.data.get_fields(
            names=self.normal_field_name_value,
            field_type=FieldType.NORMAL,
        )
        if not normal_fields:
            raise RuntimeError(f"Normals field '{self.normal_field_name_value}' was not found.")
        normals = np.asarray(normal_fields[0].data, dtype=np.float32)
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise RuntimeError(
                f"Normals field '{self.normal_field_name_value}' must have shape (N,3)."
            )
        return normals

    def _session_sig(self, layer: PointCloudLayer) -> tuple[object, ...]:
        return (
            layer.id,
            int(layer.data.points.shape[0]),
            self.normal_mode_value,
            self.normal_field_name_value,
            round(float(self.epsilon), 8),
            int(self.tau),
            round(float(self.alpha_deg), 8),
            bool(self.enable_seed_gating),
            int(self.seed_min_neighbors),
            round(float(self.seed_planarity_min), 8),
            round(float(self.seed_scattering_max), 8),
            round(float(self.epsilon_multiplier), 8),
            round(float(self.refit_multiplier), 8),
            int(self.first_refit),
            round(float(self.max_dist_from_cent), 8),
            bool(self.oriented_normals),
            bool(self.perform_cca),
        )

    def _ensure_session(self, layer: PointCloudLayer) -> SeededGrower:
        self._sync_normal_defaults()
        sig = self._session_sig(layer)
        if self._session is not None and self._session_signature == sig:
            return self._session
        normals = self._provided_normals(layer)
        print("[seeded_region_growing] building native session...")
        self._session = SeededGrower(
            layer.data,
            normals=normals,
            normal_mode=self.normal_mode_value,
            params=self._params(),
        )
        self._session_signature = sig
        print("[seeded_region_growing] session ready")
        return self._session

    def _apply_selection(self, layer: PointCloudLayer, indices: np.ndarray) -> None:
        if indices.size == 0:
            print("[seeded_region_growing] no region accepted for the picked seed")
            return
        cmd = SelectPointsCmd(
            title="Seeded region growing selection",
            selection_new=np.asarray(indices, dtype=np.int32),
            layer_ref=weakref.ref(layer),
            ctx_ref=weakref.ref(self.ctx),
        )
        self.command_manager.do(cmd)

    def _estimate_from_active_layer(self, on_update) -> None:
        layer = self._active_pointcloud_layer()
        if layer is None:
            return

        class _EstimateWorker(QThread):
            estimated = pyqtSignal(float, int, float)
            errored = pyqtSignal(str)

            def run(self) -> None:
                try:
                    estimated = estimate_parameters(
                        layer.data,
                        sample_size=self_parent.estimate_sample_size,
                        seed=self_parent.estimate_seed,
                    )
                    self.estimated.emit(
                        float(estimated["epsilon"]),
                        int(estimated["tau"]),
                        float(estimated["alpha_deg"]),
                    )
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        self_parent = self
        progress_dialog = QProgressDialog("Estimating parameters...", "", 0, 0, self.ctx.viewer.window())
        progress_dialog.setWindowTitle("Live region growing")
        progress_dialog.setCancelButton(None)
        progress_dialog.setMinimumDuration(0)

        loop = QEventLoop(self.ctx.viewer)
        error_msg: dict[str, str | None] = {"value": None}

        def _on_estimated(eps: float, tau: int, alpha_deg: float) -> None:
            on_update(eps, tau, alpha_deg)
            progress_dialog.close()
            loop.quit()

        def _on_error(msg: str) -> None:
            error_msg["value"] = msg
            progress_dialog.close()
            loop.quit()

        worker = _EstimateWorker()
        worker.estimated.connect(_on_estimated)
        worker.errored.connect(_on_error)
        worker.start()
        progress_dialog.show()
        loop.exec()
        worker.wait()

        if error_msg["value"] is not None:
            QMessageBox.critical(self.ctx.viewer.window(), "Estimate failed", error_msg["value"])

    def _open_settings_dialog(self, sync_spins) -> None:
        layer = self._active_pointcloud_layer()
        dlg = SeededRegionGrowingDialog(layer, self._settings_dict(), parent=self.ctx.viewer.window())
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        settings = dlg.settings()
        self._apply_settings_dict(settings)
        self._invalidate_session()
        sync_spins()

    def _settings_dict(self) -> dict[str, object]:
        return {
            "normal_mode": self.normal_mode_value,
            "normal_field_name": self.normal_field_name_value,
            "epsilon": self.epsilon,
            "tau": self.tau,
            "alpha_deg": self.alpha_deg,
            "enable_seed_gating": self.enable_seed_gating,
            "seed_min_neighbors": self.seed_min_neighbors,
            "seed_planarity_min": self.seed_planarity_min,
            "seed_scattering_max": self.seed_scattering_max,
            "epsilon_multiplier": self.epsilon_multiplier,
            "refit_multiplier": self.refit_multiplier,
            "first_refit": self.first_refit,
            "max_dist_from_cent": self.max_dist_from_cent,
            "oriented_normals": self.oriented_normals,
            "perform_cca": self.perform_cca,
        }

    def _apply_settings_dict(self, settings: dict[str, object]) -> None:
        self.normal_mode_value = str(settings.get("normal_mode", self.normal_mode_value))
        normal_field_name = settings.get("normal_field_name")
        self.normal_field_name_value = (
            str(normal_field_name) if isinstance(normal_field_name, str) and normal_field_name else None
        )
        self.epsilon = float(settings.get("epsilon", self.epsilon))
        self.tau = int(settings.get("tau", self.tau))
        self.alpha_deg = float(settings.get("alpha_deg", self.alpha_deg))
        self.enable_seed_gating = bool(settings.get("enable_seed_gating", self.enable_seed_gating))
        self.seed_min_neighbors = int(settings.get("seed_min_neighbors", self.seed_min_neighbors))
        self.seed_planarity_min = float(settings.get("seed_planarity_min", self.seed_planarity_min))
        self.seed_scattering_max = float(settings.get("seed_scattering_max", self.seed_scattering_max))
        self.epsilon_multiplier = float(settings.get("epsilon_multiplier", self.epsilon_multiplier))
        self.refit_multiplier = float(settings.get("refit_multiplier", self.refit_multiplier))
        self.first_refit = int(settings.get("first_refit", self.first_refit))
        self.max_dist_from_cent = float(settings.get("max_dist_from_cent", self.max_dist_from_cent))
        self.oriented_normals = bool(settings.get("oriented_normals", self.oriented_normals))
        self.perform_cca = bool(settings.get("perform_cca", self.perform_cca))

    def left_button_press_hook(self, event: Event) -> None:
        if self._busy:
            return
        result = self.ctx.viewer.pick()
        if result.layer is None or not isinstance(result.layer, PointCloudLayer):
            return
        if result.layer.id != self.ctx.scene.active_layer_id:
            return
        seed_index = result.element_idx
        if seed_index is None:
            return

        self._busy = True
        try:
            layer = result.layer
            session = self._ensure_session(layer)
            indices, stats = session.grow(seed_index)
            if bool(stats.get("accepted", False)):
                self._apply_selection(layer, indices)
            else:
                print(f"[seeded_region_growing] seed {seed_index} produced no accepted region")
        except Exception as exc:
            QMessageBox.critical(self.ctx.viewer.window(), "Live region growing failed", str(exc))
        finally:
            self._busy = False
        super().left_button_press_hook(event)

    def activate(self) -> None:
        self._sync_normal_defaults()
        return super().activate()

    def deactivate(self) -> None:
        self._invalidate_session()
        return super().deactivate()

    def create_context_widget(self, parent: QWidget) -> QWidget | None:
        w = QWidget(parent)
        outer = QHBoxLayout(w)
        outer.setContentsMargins(2, 1, 2, 1)
        outer.setSpacing(4)

        outer.addWidget(QLabel("eps", w))
        epsilon_spin = QDoubleSpinBox(w)
        epsilon_spin.setDecimals(6)
        epsilon_spin.setRange(1e-6, 1e6)
        epsilon_spin.setSingleStep(0.01)
        epsilon_spin.setValue(float(self.epsilon))

        outer.addWidget(epsilon_spin)

        outer.addWidget(QLabel("tau", w))
        tau_spin = QSpinBox(w)
        tau_spin.setRange(3, 10_000_000)
        tau_spin.setValue(int(self.tau))
        outer.addWidget(tau_spin)

        outer.addWidget(QLabel("alpha", w))
        alpha_spin = QDoubleSpinBox(w)
        alpha_spin.setDecimals(2)
        alpha_spin.setRange(0.0, 89.9)
        alpha_spin.setSingleStep(1.0)
        alpha_spin.setValue(float(self.alpha_deg))
        outer.addWidget(alpha_spin)

        def _sync_from_state() -> None:
            epsilon_spin.setValue(float(self.epsilon))
            tau_spin.setValue(int(self.tau))
            alpha_spin.setValue(float(self.alpha_deg))

        def _on_estimate() -> None:
            self._estimate_from_active_layer(_apply_estimate)

        def _apply_estimate(eps: float, tau: int, alpha_deg: float) -> None:
            self.epsilon = float(eps)
            self.tau = int(max(1, tau))
            self.alpha_deg = float(max(0.0, min(89.9, alpha_deg)))
            self._invalidate_session()
            _sync_from_state()

        epsilon_spin.valueChanged.connect(lambda val: (setattr(self, "epsilon", float(val)), self._invalidate_session()))
        tau_spin.valueChanged.connect(lambda val: (setattr(self, "tau", int(val)), self._invalidate_session()))
        alpha_spin.valueChanged.connect(lambda val: (setattr(self, "alpha_deg", float(val)), self._invalidate_session()))

        estimate_btn = QPushButton("Estimate", w)
        estimate_btn.clicked.connect(_on_estimate)
        outer.addWidget(estimate_btn)

        settings_btn = QPushButton("Settings...", w)
        settings_btn.clicked.connect(lambda: self._open_settings_dialog(_sync_from_state))
        outer.addWidget(settings_btn)

        return w

    def key_press_hook(self, event: Event) -> None:
        super().key_press_hook(event)
        if event.key is None:
            return
        if event.key.lower() == "escape":
            self.ctx.controller.deactivate_tool()
