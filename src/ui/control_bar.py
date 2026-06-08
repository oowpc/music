from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QWidget


METHOD_LABELS = {
    "标准 Hausdorff": "standard",
    "Modified Hausdorff": "modified",
    "离散 Fréchet": "frechet",
}


class ControlBar(QWidget):
    """Bottom control bar for distance, normalization, compute, and export controls."""

    method_changed = Signal(str)
    normalization_changed = Signal(str)
    weights_changed = Signal()
    compute_requested = Signal()
    segment_requested = Signal()
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
        layout.addWidget(QLabel("时间:"))

        self.time_weight_spin = QDoubleSpinBox()
        self._configure_weight_spin(self.time_weight_spin)
        layout.addWidget(self.time_weight_spin)

        layout.addWidget(QLabel("音高:"))

        self.pitch_weight_spin = QDoubleSpinBox()
        self._configure_weight_spin(self.pitch_weight_spin)
        layout.addWidget(self.pitch_weight_spin)

        layout.addWidget(QLabel("力度:"))

        self.velocity_weight_spin = QDoubleSpinBox()
        self._configure_weight_spin(self.velocity_weight_spin)
        layout.addWidget(self.velocity_weight_spin)

        layout.addSpacing(16)

        self.compute_btn = QPushButton("计算距离矩阵")
        self.compute_btn.clicked.connect(self.compute_requested.emit)
        layout.addWidget(self.compute_btn)

        layout.addSpacing(16)
        layout.addWidget(QLabel("窗口比例:"))

        self.segment_window_spin = QDoubleSpinBox()
        self.segment_window_spin.setRange(0.05, 1.0)
        self.segment_window_spin.setSingleStep(0.05)
        self.segment_window_spin.setDecimals(2)
        self.segment_window_spin.setValue(0.25)
        layout.addWidget(self.segment_window_spin)

        layout.addWidget(QLabel("步长比例:"))

        self.segment_step_spin = QDoubleSpinBox()
        self.segment_step_spin.setRange(0.01, 1.0)
        self.segment_step_spin.setSingleStep(0.01)
        self.segment_step_spin.setDecimals(2)
        self.segment_step_spin.setValue(0.05)
        layout.addWidget(self.segment_step_spin)

        self.segment_btn = QPushButton("最相似片段")
        self.segment_btn.clicked.connect(self.segment_requested.emit)
        layout.addWidget(self.segment_btn)

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

    def _configure_weight_spin(self, spin: QDoubleSpinBox) -> None:
        """Configure one dimension-weight spin box."""
        spin.setRange(0.0, 5.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(2)
        spin.setValue(1.0)
        spin.valueChanged.connect(lambda _value: self.weights_changed.emit())

    def get_method(self) -> str:
        """Return the selected distance method key."""
        return METHOD_LABELS[self.method_combo.currentText()]

    def get_norm(self) -> str:
        """Return the selected normalization method key."""
        return "minmax" if "Min-Max" in self.norm_combo.currentText() else "zscore"

    def get_segment_params(self) -> tuple[float, float]:
        """Return segment comparison window and step sizes in normalized time."""
        return self.segment_window_spin.value(), self.segment_step_spin.value()

    def get_dimension_weights(self) -> tuple[float, float, float]:
        """Return time, pitch, and velocity weights."""
        return (
            self.time_weight_spin.value(),
            self.pitch_weight_spin.value(),
            self.velocity_weight_spin.value(),
        )

    def set_status(self, text: str) -> None:
        """Set the status label text."""
        self.status_label.setText(text)
