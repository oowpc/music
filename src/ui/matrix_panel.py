import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MatrixPanel(QWidget):
    """Right-side table panel for displaying a distance matrix heatmap."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("距离矩阵")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        self._matrix = None
        self._names = []

    def set_matrix(self, matrix: np.ndarray, names: list[str]) -> None:
        """Display a distance matrix with shape (N, N) and N labels."""
        self._matrix = matrix
        self._names = names

        size = len(names)
        self.table.setRowCount(size)
        self.table.setColumnCount(size)
        self.table.setHorizontalHeaderLabels(names)
        self.table.setVerticalHeaderLabels(names)

        if size == 0:
            return

        max_distance = float(np.max(matrix)) if size > 1 else 1.0
        if max_distance <= 0:
            max_distance = 1.0

        for row in range(size):
            for col in range(size):
                value = float(matrix[row, col])
                item = QTableWidgetItem(f"{value:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                if row == col:
                    item.setBackground(QColor("#e8e8e8"))
                else:
                    ratio = max(0.0, min(1.0, value / max_distance))
                    red = int(232 * ratio + 48 * (1 - ratio))
                    green = int(80 * ratio + 166 * (1 - ratio))
                    blue = int(80 * ratio + 120 * (1 - ratio))
                    item.setBackground(QColor(red, green, blue))
                    if ratio > 0.55:
                        item.setForeground(QColor("white"))

                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()

    def get_matrix(self):
        """Return the currently displayed distance matrix or None."""
        return self._matrix
