import csv

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMainWindow, QSplitter, QVBoxLayout, QWidget

from src.ui.cluster_panel import ClusterPanel
from src.ui.control_bar import ControlBar
from src.ui.file_panel import FilePanel
from src.ui.gl_view import GLView
from src.ui.matrix_panel import MatrixPanel


TITLE = "音乐旋律线几何相似性分析工具"


class MainWindow(QMainWindow):
    """Main application window that wires UI panels to the analysis pipeline."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1200, 800)

        self._curves = []
        self._matrix = None
        self._cluster_result = None
        self._matrix_method = None
        self._matrix_norm = None
        self._matrix_weights = None
        self._known_curve_count = 0

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        self.file_panel = FilePanel()
        self.splitter.addWidget(self.file_panel)

        self.gl_view = GLView()
        self.splitter.addWidget(self.gl_view)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.matrix_panel = MatrixPanel()
        right_layout.addWidget(self.matrix_panel)

        self.cluster_panel = ClusterPanel()
        right_layout.addWidget(self.cluster_panel)

        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([220, 520, 460])

        self.control_bar = ControlBar()
        main_layout.addWidget(self.control_bar)

        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect child widget signals to main-window slots."""
        self.file_panel.files_loaded.connect(self._on_files_loaded)
        self.file_panel.visibility_changed.connect(self._on_visibility_changed)
        self.file_panel.label_changed.connect(self._on_label_changed)
        self.control_bar.normalization_changed.connect(self._on_normalization_changed)
        self.control_bar.weights_changed.connect(self._on_weights_changed)
        self.control_bar.compute_requested.connect(self._on_compute)
        self.control_bar.segment_requested.connect(self._on_find_best_segment)
        self.control_bar.export_requested.connect(self._on_export)

    def _on_files_loaded(self) -> None:
        """Normalize and display the current file panel curves."""
        previous_count = self._known_curve_count
        self._curves = list(self.file_panel.get_curves())
        self._run_normalization()
        self.gl_view.set_curves(self._curves)

        current_count = len(self._curves)
        self._known_curve_count = current_count
        if current_count < previous_count:
            self._invalidate_matrix()
            self.control_bar.set_status(f"已加载 {current_count} 个文件，距离矩阵已失效")
            return

        if current_count > previous_count and self._can_extend_matrix(previous_count):
            self._extend_current_matrix(previous_count)
            added_count = current_count - previous_count
            self.control_bar.set_status(f"已增量计算 {added_count} 个新增文件")
            return

        self.control_bar.set_status(f"已加载 {current_count} 个文件")

    def _on_visibility_changed(self) -> None:
        """Refresh the 3D view using checked curves only."""
        visible_indices = self.file_panel.get_visible_indices()
        visible_curves = [
            self._curves[index]
            for index in visible_indices
            if 0 <= index < len(self._curves)
        ]
        self.gl_view.set_curves(visible_curves)

    def _on_label_changed(self, index: int, label: str) -> None:
        """Update a curve label and refresh evaluation when results exist."""
        if 0 <= index < len(self._curves):
            self._curves[index].label = label if label else None
        if self._matrix is not None and self._cluster_result is not None:
            names = [curve.name for curve in self._curves]
            self.cluster_panel.set_result(self._matrix, names, self._cluster_result, self._curves)

    def _on_normalization_changed(self, normalization: str) -> None:
        """Re-normalize loaded curves and refresh the 3D view."""
        if not self._curves:
            return
        self._run_normalization()
        self._on_visibility_changed()
        self._invalidate_matrix()
        self.control_bar.set_status(f"已切换归一化: {normalization}，请重新计算距离矩阵")

    def _on_weights_changed(self) -> None:
        """Apply dimension weights and invalidate stale matrix results."""
        if not self._curves:
            return
        self._run_normalization()
        self._on_visibility_changed()
        self._invalidate_matrix()
        weights = self.control_bar.get_dimension_weights()
        self.control_bar.set_status(
            f"已更新权重 时间/音高/力度 = {weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f}"
        )

    def _run_normalization(self) -> None:
        """Normalize current curves according to the control bar selection."""
        from src.processing.normalization import normalize_minmax, normalize_zscore
        from src.processing.weights import apply_dimension_weights

        normalize = normalize_minmax if self.control_bar.get_norm() == "minmax" else normalize_zscore
        normalize(self._curves)
        apply_dimension_weights(self._curves, self.control_bar.get_dimension_weights())

    def _on_compute(self) -> None:
        """Compute distance matrix and clustering for loaded curves."""
        from src.analysis.clustering import hierarchical_clustering
        from src.analysis.distance_matrix import build_matrix

        if len(self._curves) < 2:
            self.control_bar.set_status("至少需要 2 条曲线才能计算距离矩阵")
            return

        self._run_normalization()
        self._on_visibility_changed()

        method = self.control_bar.get_method()
        self._matrix = build_matrix(self._curves, method=method)
        self._matrix_method = method
        self._matrix_norm = self.control_bar.get_norm()
        self._matrix_weights = self.control_bar.get_dimension_weights()
        names = [curve.name for curve in self._curves]
        self.matrix_panel.set_matrix(self._matrix, names)

        self._cluster_result = hierarchical_clustering(self._matrix, names)
        self.cluster_panel.set_result(self._matrix, names, self._cluster_result, self._curves)

        count = len(self._curves)
        self.control_bar.set_status(f"距离矩阵 {count}x{count} 计算完成")

    def _can_extend_matrix(self, previous_count: int) -> bool:
        """Return whether the existing matrix can be extended incrementally."""
        return (
            self._matrix is not None
            and self._matrix.shape == (previous_count, previous_count)
            and self._matrix_method == self.control_bar.get_method()
            and self._matrix_norm == self.control_bar.get_norm()
            and self._matrix_weights == self.control_bar.get_dimension_weights()
            and previous_count >= 1
        )

    def _extend_current_matrix(self, previous_count: int) -> None:
        """Extend the current matrix for newly appended curves only."""
        from src.analysis.clustering import hierarchical_clustering
        from src.analysis.distance_matrix import extend_matrix

        method = self.control_bar.get_method()
        self._matrix = extend_matrix(
            self._matrix,
            self._curves,
            previous_count=previous_count,
            method=method,
        )
        self._matrix_method = method
        self._matrix_norm = self.control_bar.get_norm()
        self._matrix_weights = self.control_bar.get_dimension_weights()

        names = [curve.name for curve in self._curves]
        self.matrix_panel.set_matrix(self._matrix, names)
        self._cluster_result = hierarchical_clustering(self._matrix, names)
        self.cluster_panel.set_result(self._matrix, names, self._cluster_result, self._curves)

    def _invalidate_matrix(self) -> None:
        """Mark cached matrix and clustering results as stale."""
        self._matrix = None
        self._cluster_result = None
        self._matrix_method = None
        self._matrix_norm = None
        self._matrix_weights = None

    def _on_find_best_segment(self) -> None:
        """Find and report the most similar segment between two visible curves."""
        from src.analysis.segment_analysis import find_best_segment_match

        visible_indices = self.file_panel.get_visible_indices()
        if len(visible_indices) < 2:
            self.control_bar.set_status("至少需要勾选 2 条曲线才能比较片段")
            return

        self._run_normalization()
        left = self._curves[visible_indices[0]]
        right = self._curves[visible_indices[1]]
        window_size, step_size = self.control_bar.get_segment_params()
        match = find_best_segment_match(
            left,
            right,
            method=self.control_bar.get_method(),
            window_size=window_size,
            step_size=step_size,
            min_points=2,
        )
        if match is None:
            self.control_bar.set_status("未找到可比较的片段")
            return

        self.control_bar.set_status(
            f"最相似片段: {left.name} [{match.left_start:.2f}-{match.left_end:.2f}] "
            f"vs {right.name} [{match.right_start:.2f}-{match.right_end:.2f}], "
            f"距离 {match.distance:.4f}"
        )

    def _on_export(self) -> None:
        """Export the current distance matrix to CSV with row and column labels."""
        if self._matrix is None:
            self.control_bar.set_status("请先计算距离矩阵")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "导出距离矩阵",
            "",
            "CSV Files (*.csv)",
        )
        if not filepath:
            return

        names = [curve.name for curve in self._curves]
        with open(filepath, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([""] + names)
            for name, row in zip(names, self._matrix):
                writer.writerow([name] + [f"{value:.6f}" for value in row])

        self.control_bar.set_status(f"已导出到 {filepath}")
