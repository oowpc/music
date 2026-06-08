from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.io.midi_loader import inspect_midi_tracks, load_midi
from src.ui.track_selection_dialog import TrackSelectionDialog


COLORS = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffe66d",
    "#a29bfe",
    "#fd79a8",
    "#00cec9",
    "#fab1a0",
    "#81ecec",
    "#55efc4",
    "#74b9ff",
    "#e17055",
    "#6c5ce7",
    "#00b894",
    "#e84393",
    "#0984e3",
    "#fdcb6e",
    "#636e72",
    "#d63031",
    "#2d3436",
    "#b2bec3",
]


class FilePanel(QWidget):
    """Left panel for importing, labeling, removing, and toggling curves."""

    files_loaded = Signal()
    visibility_changed = Signal()
    label_changed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.import_btn = QPushButton("+ 导入 MIDI")
        self.import_btn.clicked.connect(self._import_files)
        layout.addWidget(self.import_btn)

        self.list_widget = QListWidget()
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        self._curves = []

    def _import_files(self) -> None:
        """Open a file dialog and append successfully parsed MIDI curves."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入 MIDI 文件",
            "",
            "MIDI Files (*.mid *.midi);;All Files (*)",
        )
        if not filepaths:
            return

        for filepath in filepaths:
            curve = self._load_curve_with_options(filepath)
            if curve is None:
                continue
            curve.color = COLORS[len(self._curves) % len(COLORS)]
            self._curves.append(curve)

        self._rebuild_list()
        self.files_loaded.emit()

    def _load_curve_with_options(self, filepath: str):
        """Load one MIDI file after optional track/extraction selection."""
        track_infos = inspect_midi_tracks(filepath)
        if not track_infos:
            return None

        if len(track_infos) > 1:
            dialog = TrackSelectionDialog(filepath, track_infos, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            track_index, extraction_mode = dialog.selected_options()
        else:
            track_index, extraction_mode = None, "highest"

        return load_midi(filepath, track_index=track_index, extraction_mode=extraction_mode)

    def _rebuild_list(self) -> None:
        """Rebuild the visible list widget from the current curve list."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        for index, curve in enumerate(self._curves):
            text = curve.name
            if curve.label:
                text += f" [{curve.label}]"

            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setForeground(QColor(curve.color))
            self.list_widget.addItem(item)

        self.list_widget.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Emit visibility updates when list check states change."""
        self.visibility_changed.emit()

    def _show_context_menu(self, pos) -> None:
        """Show item context actions for label editing and removal."""
        item = self.list_widget.itemAt(pos)
        if item is None:
            return

        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return

        menu = QMenu(self)
        edit_action = QAction("编辑标签", self)
        edit_action.triggered.connect(lambda: self._edit_label(index))
        menu.addAction(edit_action)

        remove_action = QAction("移除", self)
        remove_action.triggered.connect(lambda: self._remove_curve(index))
        menu.addAction(remove_action)

        menu.exec(self.list_widget.viewport().mapToGlobal(pos))

    def _edit_label(self, index: int) -> None:
        """Prompt for and update a curve label."""
        current = self._curves[index].label or ""
        label, ok = QInputDialog.getText(self, "编辑标签", "曲风标签:", text=current)
        if not ok:
            return

        self._curves[index].label = label if label else None
        self._rebuild_list()
        self.label_changed.emit(index, label)

    def _remove_curve(self, index: int) -> None:
        """Remove a curve and notify listeners that loaded data changed."""
        self._curves.pop(index)
        self._rebuild_list()
        self.files_loaded.emit()

    def get_curves(self):
        """Return the current melody curve list."""
        return self._curves

    def get_visible_indices(self) -> list[int]:
        """Return curve indices with checked list items."""
        indices = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                indices.append(item.data(Qt.ItemDataRole.UserRole))
        return indices
