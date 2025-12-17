from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpinBox, QColorDialog, QLineEdit,
    QDialogButtonBox, QMessageBox
)


from geon.data.pointcloud import SemanticSchema, SemanticClass


@dataclass
class _RowMeta:
    required: bool = False


class SemanticSchemaCreationDialog(QDialog):
    COL_ID = 0
    COL_COLOR = 1
    COL_NAME = 2
    COL_BTN = 3

    def __init__(self, required_ids: Optional[List[int]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Semantic Schema")
        self.resize(820, 520)

        if required_ids is None:
            required_ids = [-1]
        self.required_ids = list(dict.fromkeys(required_ids))  # unique, keep order

        self.schema: Optional[SemanticSchema] = None
        self._row_meta: Dict[int, _RowMeta] = {}

        self._build_ui()
        self._populate_defaults()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # Left actions
        left = QVBoxLayout()
        self.btn_reindex_alpha = QPushButton("Reindex alphabetically")
        self.btn_reindex = QPushButton("Reindex")
        self.btn_up = QPushButton("Move up")
        self.btn_down = QPushButton("Move down")
        left.addWidget(self.btn_reindex_alpha)
        left.addWidget(self.btn_reindex)
        left.addSpacing(12)
        left.addWidget(self.btn_up)
        left.addWidget(self.btn_down)
        left.addStretch(1)

        self.btn_reindex_alpha.clicked.connect(self._on_reindex_alpha)
        self.btn_reindex.clicked.connect(self._on_reindex)
        self.btn_up.clicked.connect(self._on_move_up)
        self.btn_down.clicked.connect(self._on_move_down)

        # Right main: table + bottom name + ok/cancel
        right = QVBoxLayout()

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["ID", "Color", "Name", ""])
        header = self.table.horizontalHeader()
        assert isinstance(header, QHeaderView)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_ID, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_COLOR, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_BTN, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        right.addWidget(self.table, 1)

        # Bottom row: schema name input (left) + ok/cancel (right)
        bottom = QHBoxLayout()
        self.schema_name_edit = QLineEdit(self)
        self.schema_name_edit.setPlaceholderText("Schema name")
        bottom.addWidget(self.schema_name_edit, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        bottom.addWidget(self.buttons, 0)

        right.addLayout(bottom)

        # Assemble
        root.addLayout(left, 0)
        root.addLayout(right, 1)

    def _populate_defaults(self) -> None:
        # Add required rows first
        for rid in self.required_ids:
            if rid == -1:
                self._insert_class_row(rid, (128, 128, 128), "_unlabeled", required=True)
            else:
                self._insert_class_row(rid, (200, 200, 200), f"class_{rid}", required=True)

        # Add '+' row
        self._insert_add_row()

        # Default schema name
        self.schema_name_edit.setText("new_schema")

        # Make sure ids are continuous by default if required_ids were only [-1]
        self._on_reindex()

    # ------------- Row creation helpers -------------

    def _is_add_row(self, row: int) -> bool:
        return row == self.table.rowCount() - 1

    def _insert_add_row(self) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)

        for c in (self.COL_ID, self.COL_COLOR, self.COL_NAME):
            item = QTableWidgetItem("")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.table.setItem(r, c, item)

        btn = QPushButton("+", self.table)
        btn.clicked.connect(self._on_add_clicked)
        self.table.setCellWidget(r, self.COL_BTN, btn)

    def _insert_class_row(self, class_id: int, color: Tuple[int, int, int], name: str, required: bool) -> None:
        insert_at = self.table.rowCount()
        if insert_at > 0:
            # keep '+' row at the bottom
            insert_at = max(0, insert_at - 1)

        self.table.insertRow(insert_at)
        self._row_meta[insert_at] = _RowMeta(required=required)

        # ID spin
        id_spin = QSpinBox(self.table)
        id_spin.setRange(-1, 65535)
        id_spin.setValue(int(class_id))
        if required:
            id_spin.setEnabled(False)  # required ids fixed
        self.table.setCellWidget(insert_at, self.COL_ID, id_spin)

        # Color button
        color_btn = QPushButton("", self.table)
        color_btn.setFixedWidth(60)
        self._set_color_btn(color_btn, color)
        color_btn.clicked.connect(lambda _=False, r=insert_at: self._on_color_clicked(r))
        self.table.setCellWidget(insert_at, self.COL_COLOR, color_btn)

        # Name item (editable)
        name_item = QTableWidgetItem(name)
        self.table.setItem(insert_at, self.COL_NAME, name_item)

        # Remove button (or disabled)
        if required:
            rm = QPushButton("", self.table)
            rm.setEnabled(False)
        else:
            rm = QPushButton("-", self.table)
            rm.clicked.connect(lambda _=False, r=insert_at: self._on_remove_clicked(r))
        self.table.setCellWidget(insert_at, self.COL_BTN, rm)

        # Because we inserted, row indices shifted; rewire row-dependent callbacks
        self._rewire_row_callbacks()

    def _rewire_row_callbacks(self) -> None:
        """
        After inserts/removals, reconnect callbacks that capture row indices.
        This is the simplest robust approach for QTableWidget-based UIs.
        """
        # rebuild row_meta with current row indices by reading existing widgets
        new_meta: Dict[int, _RowMeta] = {}
        for r in range(self.table.rowCount()):
            if self._is_add_row(r):
                continue
            # try to preserve "required" by checking if remove button is disabled
            btn = self.table.cellWidget(r, self.COL_BTN)
            required = isinstance(btn, QPushButton) and (not btn.isEnabled())
            new_meta[r] = _RowMeta(required=required)
        self._row_meta = new_meta

        # reconnect all color/remove buttons with correct rows
        for r in range(self.table.rowCount()):
            if self._is_add_row(r):
                continue

            color_btn = self.table.cellWidget(r, self.COL_COLOR)
            if isinstance(color_btn, QPushButton):
                try:
                    color_btn.clicked.disconnect()
                except TypeError:
                    pass
                color_btn.clicked.connect(lambda _=False, rr=r: self._on_color_clicked(rr))

            btn = self.table.cellWidget(r, self.COL_BTN)
            if isinstance(btn, QPushButton) and btn.isEnabled() and btn.text() == "-":
                try:
                    btn.clicked.disconnect()
                except TypeError:
                    pass
                btn.clicked.connect(lambda _=False, rr=r: self._on_remove_clicked(rr))

    # ------------- Color helpers -------------

    def _set_color_btn(self, btn: QPushButton, rgb: Tuple[int, int, int]) -> None:
        r, g, b = rgb
        btn.setStyleSheet(
            "QPushButton {"
            f"background-color: rgb({r},{g},{b});"
            "border: 1px solid rgba(0,0,0,80);"
            "border-radius: 2px;"
            "}"
        )

    def _get_color_btn_rgb(self, btn: QPushButton) -> Tuple[int, int, int]:
        # We don’t store separately; parse from stylesheet is annoying.
        # Instead, stash rgb in Qt property:
        rgb = btn.property("rgb")
        if isinstance(rgb, tuple) and len(rgb) == 3:
            return rgb  # type: ignore
        return (128, 128, 128)

    # ------------- Actions -------------

    def _on_add_clicked(self) -> None:
        # Add a new non-required row with a placeholder id/name
        used = set(self._current_ids(include_required=True))
        # choose next available >= 0
        new_id = 0
        while new_id in used:
            new_id += 1
        self._insert_class_row(new_id, (200, 200, 200), f"class_{new_id}", required=False)
        self._on_reindex()

    def _on_remove_clicked(self, row: int) -> None:
        if row < 0 or row >= self.table.rowCount() or self._is_add_row(row):
            return
        if self._row_meta.get(row, _RowMeta()).required:
            return
        self.table.removeRow(row)
        self._rewire_row_callbacks()

    def _on_color_clicked(self, row: int) -> None:
        if self._is_add_row(row):
            return
        btn = self.table.cellWidget(row, self.COL_COLOR)
        if not isinstance(btn, QPushButton):
            return
        initial = QColor(*self._get_color_btn_rgb(btn))
        col = QColorDialog.getColor(initial, self, "Choose class color")
        if not col.isValid():
            return
        rgb = (col.red(), col.green(), col.blue())
        btn.setProperty("rgb", rgb)
        self._set_color_btn(btn, rgb)

    def _on_move_up(self) -> None:
        row = self._current_real_row()
        if row is None or row <= 0:
            return
        self._swap_rows(row, row - 1)

    def _on_move_down(self) -> None:
        row = self._current_real_row()
        if row is None:
            return
        if row >= self.table.rowCount() - 2:  # before '+' row
            return
        self._swap_rows(row, row + 1)

    def _current_real_row(self) -> Optional[int]:
        items = self.table.selectedItems()
        if not items:
            return None
        r = items[0].row()
        if self._is_add_row(r):
            return None
        return r

    def _swap_rows(self, a: int, b: int) -> None:
        if self._is_add_row(a) or self._is_add_row(b):
            return

        # swap ID spin values (and enabled state)
        a_id = self.table.cellWidget(a, self.COL_ID)
        b_id = self.table.cellWidget(b, self.COL_ID)
        if isinstance(a_id, QSpinBox) and isinstance(b_id, QSpinBox):
            av, bv = a_id.value(), b_id.value()
            ae, be = a_id.isEnabled(), b_id.isEnabled()
            a_id.setValue(bv); b_id.setValue(av)
            a_id.setEnabled(be); b_id.setEnabled(ae)

        # swap color properties/styles
        a_col = self.table.cellWidget(a, self.COL_COLOR)
        b_col = self.table.cellWidget(b, self.COL_COLOR)
        if isinstance(a_col, QPushButton) and isinstance(b_col, QPushButton):
            ar = a_col.property("rgb") or (128,128,128)
            br = b_col.property("rgb") or (128,128,128)
            a_col.setProperty("rgb", br); b_col.setProperty("rgb", ar)
            self._set_color_btn(a_col, br)
            self._set_color_btn(b_col, ar)

        # swap name text
        a_name = self.table.item(a, self.COL_NAME)
        b_name = self.table.item(b, self.COL_NAME)
        if isinstance(a_name, QTableWidgetItem) and isinstance(b_name, QTableWidgetItem):
            at, bt = a_name.text(), b_name.text()
            a_name.setText(bt); b_name.setText(at)

        # swap required meta
        am = self._row_meta.get(a, _RowMeta()).required
        bm = self._row_meta.get(b, _RowMeta()).required
        self._row_meta[a] = _RowMeta(required=bm)
        self._row_meta[b] = _RowMeta(required=am)

        # update remove buttons enabled state
        self._refresh_remove_buttons()
        self._rewire_row_callbacks()

    def _refresh_remove_buttons(self) -> None:
        for r in range(self.table.rowCount()):
            if self._is_add_row(r):
                continue
            required = self._row_meta.get(r, _RowMeta()).required
            btn = self.table.cellWidget(r, self.COL_BTN)
            if not isinstance(btn, QPushButton):
                continue
            if required:
                btn.setText("")
                btn.setEnabled(False)
            else:
                btn.setText("-")
                btn.setEnabled(True)

    def _on_reindex(self) -> None:
        """
        Renumber ids sequentially without changing row order.
        Keep -1 (if present) fixed at -1.
        Nonnegative ids become 0..N-1 in row order.
        """
        nonneg_rows: List[int] = []
        for r in range(self.table.rowCount()):
            if self._is_add_row(r):
                continue
            spin = self.table.cellWidget(r, self.COL_ID)
            if isinstance(spin, QSpinBox) and spin.value() >= 0:
                nonneg_rows.append(r)

        new_id = 0
        for r in nonneg_rows:
            spin = self.table.cellWidget(r, self.COL_ID)
            if isinstance(spin, QSpinBox) and spin.isEnabled():
                spin.setValue(new_id)
            new_id += 1

        # required ids that are nonnegative are disabled, so they don't change
        # (if you want required nonnegative ids to also be rewritten, remove isEnabled() check)

    def _on_reindex_alpha(self) -> None:
        """
        Sort rows alphabetically by name (excluding '+' row),
        then renumber sequentially (keeping -1 fixed).
        """
        rows = []
        for r in range(self.table.rowCount()):
            if self._is_add_row(r):
                continue
            name_item = self.table.item(r, self.COL_NAME)
            name = name_item.text() if name_item else ""
            rows.append((name.lower(), r))

        rows.sort(key=lambda x: x[0])

        # Snapshot current row data
        snapshot = [self._row_to_data(r) for _, r in rows]

        # Rewrite table (keep + row)
        while self.table.rowCount() > 1:
            self.table.removeRow(0)
        self._row_meta.clear()

        for data in snapshot:
            self._insert_class_row(**data)

        self._on_reindex()

    def _row_to_data(self, row: int) -> dict:
        spin = self.table.cellWidget(row, self.COL_ID)
        class_id = spin.value() if isinstance(spin, QSpinBox) else 0

        btn = self.table.cellWidget(row, self.COL_COLOR)
        rgb = btn.property("rgb") if isinstance(btn, QPushButton) else (128, 128, 128)
        if not isinstance(rgb, tuple):
            rgb = (128, 128, 128)

        name_item = self.table.item(row, self.COL_NAME)
        name = name_item.text() if name_item else ""

        required = self._row_meta.get(row, _RowMeta()).required
        return {"class_id": class_id, "color": rgb, "name": name, "required": required}

    # ------------- Validation + accept -------------

    def _current_ids(self, include_required: bool = True) -> List[int]:
        ids: List[int] = []
        for r in range(self.table.rowCount()):
            if self._is_add_row(r):
                continue
            if not include_required and self._row_meta.get(r, _RowMeta()).required:
                continue
            spin = self.table.cellWidget(r, self.COL_ID)
            if isinstance(spin, QSpinBox):
                ids.append(int(spin.value()))
        return ids

    def _validate(self) -> Optional[str]:
        # schema name
        schema_name = self.schema_name_edit.text().strip()
        if not schema_name:
            return "Schema name cannot be empty."

        # ids unique
        ids = self._current_ids(include_required=True)
        if len(ids) != len(set(ids)):
            return "Class IDs must be unique."

        # required ids present
        for rid in self.required_ids:
            if rid not in ids:
                return f"Required class id {rid} is missing."

        # names valid
        for r in range(self.table.rowCount()):
            if self._is_add_row(r):
                continue
            item = self.table.item(r, self.COL_NAME)
            name = item.text() if item else ""
            if len(name) > 256:
                return "Class name exceeds 256 characters."
            try:
                name.encode("ascii")
            except UnicodeEncodeError:
                return "Class names must be ASCII."

        # continuity rule: nonnegative ids must be 0..max with no gaps
        nonneg = sorted([i for i in ids if i >= 0])
        if nonneg:
            if nonneg[0] != 0:
                return "Nonnegative class IDs must start at 0."
            expected = list(range(0, nonneg[-1] + 1))
            if nonneg != expected:
                return "Class IDs must be continuous (no gaps). Use Reindex."

        return None

    def accept(self) -> None:
        err = self._validate()
        if err is not None:
            QMessageBox.warning(self, "Invalid schema", err)
            return

        schema_name = self.schema_name_edit.text().strip()
        schema = SemanticSchema(name=schema_name)
        schema.semantic_classes = []

        for r in range(self.table.rowCount()):
            if self._is_add_row(r):
                continue

            spin = self.table.cellWidget(r, self.COL_ID)
            cid = spin.value() if isinstance(spin, QSpinBox) else 0

            btn = self.table.cellWidget(r, self.COL_COLOR)
            rgb = btn.property("rgb") if isinstance(btn, QPushButton) else (128, 128, 128)
            if not isinstance(rgb, tuple):
                rgb = (128, 128, 128)

            item = self.table.item(r, self.COL_NAME)
            cname = item.text() if item else ""

            schema.semantic_classes.append(SemanticClass(int(cid), cname, (int(rgb[0]), int(rgb[1]), int(rgb[2]))))

        # sort by id for canonical form
        schema.semantic_classes.sort(key=lambda c: c.id)

        self.schema = schema
        super().accept()
