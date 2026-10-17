"""Shared semantic-schema export action for ribbons and field menus."""

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from geon.data.pointcloud import SemanticSegmentation


def export_semantic_schema(field: SemanticSegmentation, parent: QWidget) -> None:
    default_name = f"{field.schema.name}.json" if field.schema.name else "schema.json"
    path, _ = QFileDialog.getSaveFileName(
        parent, "Export schema to JSON", default_name, "JSON Files (*.json)")
    if not path:
        return
    try:
        field.schema.to_json(path)
    except Exception as exc:
        QMessageBox.critical(parent, "Export failed", str(exc))
