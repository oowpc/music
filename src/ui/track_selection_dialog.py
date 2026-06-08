import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


EXTRACTION_LABELS = {
    "最高音线": "highest",
    "最强力度线": "strongest",
    "全部音符": "all",
}


class TrackSelectionDialog(QDialog):
    """Dialog for choosing all tracks or one MIDI track plus extraction mode."""

    def __init__(self, filepath: str, track_infos: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择 MIDI 轨道")
        self.resize(760, 420)
        self._track_infos = track_infos

        layout = QVBoxLayout(self)

        title = QLabel(os.path.basename(filepath))
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        source_box = QGroupBox("音轨来源")
        source_layout = QHBoxLayout(source_box)
        self.combine_radio = QRadioButton("合并所有轨道")
        self.single_radio = QRadioButton("使用选中轨道")
        self.combine_radio.setChecked(True)

        self.source_group = QButtonGroup(self)
        self.source_group.addButton(self.combine_radio)
        self.source_group.addButton(self.single_radio)

        source_layout.addWidget(self.combine_radio)
        source_layout.addWidget(self.single_radio)
        source_layout.addStretch()
        layout.addWidget(source_box)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("提取方式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(EXTRACTION_LABELS))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        self.track_table = QTableWidget()
        self.track_table.setColumnCount(7)
        self.track_table.setHorizontalHeaderLabels(
            ["#", "名称", "乐器", "音符数", "音高范围", "开始", "结束"]
        )
        self.track_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.track_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.track_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.track_table.itemSelectionChanged.connect(self._on_track_selected)
        layout.addWidget(self.track_table)

        self._populate_tracks()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_tracks(self) -> None:
        """Fill the track table with track summary rows."""
        self.track_table.blockSignals(True)
        self.track_table.setRowCount(len(self._track_infos))
        for row, info in enumerate(self._track_infos):
            values = [
                str(info.index),
                info.name,
                info.instrument_name,
                str(info.note_count),
                _pitch_range_text(info),
                _seconds_text(info.start_seconds),
                _seconds_text(info.end_seconds),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if col != 1 else Qt.AlignmentFlag.AlignLeft)
                self.track_table.setItem(row, col, item)
        if self._track_infos:
            self.track_table.selectRow(0)
        self.track_table.resizeColumnsToContents()
        self.track_table.blockSignals(False)

    def _on_track_selected(self) -> None:
        """Switch to single-track mode when the user selects a table row."""
        if self.track_table.selectedItems():
            self.single_radio.setChecked(True)

    def selected_options(self) -> tuple[int | None, str]:
        """Return selected (track_index, extraction_mode)."""
        extraction_mode = EXTRACTION_LABELS[self.mode_combo.currentText()]
        if self.combine_radio.isChecked():
            return None, extraction_mode

        row = self.track_table.currentRow()
        if row < 0 and self._track_infos:
            row = 0
        if row < 0:
            return None, extraction_mode
        return self._track_infos[row].index, extraction_mode


def _pitch_range_text(info) -> str:
    """Return pitch range display text for a track info object."""
    if info.pitch_min is None or info.pitch_max is None:
        return "-"
    return f"{info.pitch_min}-{info.pitch_max}"


def _seconds_text(value: float | None) -> str:
    """Return seconds display text with two decimals or dash."""
    if value is None:
        return "-"
    return f"{value:.2f}s"
