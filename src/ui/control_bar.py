from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget


METHOD_LABELS = {
    "标准 Hausdorff": "standard",
    "Modified Hausdorff": "modified",
    "离散 Fréchet": "frechet",
}


class ControlBar(QWidget):
    """Bottom control bar for distance, normalization, compute, and export controls."""

    method_changed = Signal(str)
    normalization_changed = Signal(str)
    compute_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        layout.addWidget(QLabel("距离算法:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems(list(METHOD_LABELS))
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        layout.addWidget(self.method_combo)

        layout.addSpacing(16)
        layout.addWidget(QLabel("归一化:"))

        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["Min-Max [0,1]", "Z-Score"])
        self.norm_combo.currentTextChanged.connect(self._on_norm_changed)
        layout.addWidget(self.norm_combo)

        layout.addSpacing(16)

        self.compute_btn = QPushButton("计算距离矩阵")
        self.compute_btn.clicked.connect(self.compute_requested.emit)
        layout.addWidget(self.compute_btn)

        self.export_btn = QPushButton("导出 CSV")
        self.export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(self.export_btn)

        layout.addStretch()

        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    def _on_method_changed(self, text: str) -> None:
        """Emit the selected distance method key."""
        self.method_changed.emit(METHOD_LABELS[text])

    def _on_norm_changed(self, text: str) -> None:
        """Emit the selected normalization method key."""
        self.normalization_changed.emit("minmax" if "Min-Max" in text else "zscore")

    def get_method(self) -> str:
        """Return the selected distance method key."""
        return METHOD_LABELS[self.method_combo.currentText()]

    def get_norm(self) -> str:
        """Return the selected normalization method key."""
        return "minmax" if "Min-Max" in self.norm_combo.currentText() else "zscore"

    def set_status(self, text: str) -> None:
        """Set the status label text."""
        self.status_label.setText(text)
