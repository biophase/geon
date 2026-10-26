from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout


class MeshImportDialog(QDialog):
    """Confirm geometry import without offering point-cloud field mappings."""
    def __init__(self, mesh, parent=None, *, active_document_name=None):
        super().__init__(parent)
        self.mesh = mesh
        self.setWindowTitle("Import PLY mesh")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(mesh.name, self)
        form.addRow("Mesh name:", self.name_edit)
        self.destination_combo = QComboBox(self)
        self.destination_combo.addItem("New document", False)
        if active_document_name is not None:
            self.destination_combo.addItem(f"Active document: {active_document_name}", True)
            self.destination_combo.setCurrentIndex(1)
        form.addRow("Import into:", self.destination_combo)
        form.addRow("Vertices:", QLabel(f"{len(mesh.vertices):,}", self))
        form.addRow("Faces:", QLabel(f"{len(mesh.faces):,}", self))
        layout.addLayout(form)
        layout.addWidget(QLabel("Polygon faces, including quads, are preserved. Only geometry is imported.", self))
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        self.name_edit.textChanged.connect(self._update_state)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_state()

    def _update_state(self):
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.name_edit.text().strip()))

    @property
    def append_to_active_document(self):
        return bool(self.destination_combo.currentData())

    def accept(self):
        name = self.name_edit.text().strip()
        if name:
            self.mesh.name = name
            super().accept()
