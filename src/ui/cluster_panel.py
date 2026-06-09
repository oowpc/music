import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from scipy.cluster.hierarchy import dendrogram

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    Figure = None
    FigureCanvas = None
    dendrogram = None


class ClusterPanel(QWidget):
    """Right-side tab panel for MDS, genre, KNN, confusion, and structure analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("可视化分析")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._matrix = None
        self._names = []
        self._cluster_result = {}
        self._curves = []

        if HAS_MATPLOTLIB:
            self._build_plot_tabs()
        else:
            self._build_fallback_tabs()

    def _build_plot_tabs(self) -> None:
        """Create matplotlib-backed MDS and table tabs."""
        self._mds_figure = Figure(figsize=(4, 3), dpi=100)
        self._mds_canvas = FigureCanvas(self._mds_figure)
        self._add_tab("MDS 可视化", self._mds_canvas)

        self._dendrogram_figure = Figure(figsize=(4, 3), dpi=100)
        self._dendrogram_canvas = FigureCanvas(self._dendrogram_figure)
        self._add_tab("树状图", self._dendrogram_canvas)

        self._build_analysis_tabs()

    def _build_fallback_tabs(self) -> None:
        """Create table tabs when matplotlib is unavailable."""
        unavailable = QLabel("matplotlib 不可用，图表视图不可用")
        unavailable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._add_tab("MDS 可视化", unavailable)
        self._build_analysis_tabs()

    def _build_analysis_tabs(self) -> None:
        """Create the genre matrix, KNN, and confusion-matrix tabs."""
        self.genre_table = self._create_table()
        self._add_tab("曲风距离矩阵", self.genre_table)

        self.evaluation_table = self._create_table()
        self._add_tab("定量评价", self.evaluation_table)

        knn_widget = QWidget()
        knn_layout = QVBoxLayout(knn_widget)
        knn_layout.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("K:"))
        self.knn_k_spin = QSpinBox()
        self.knn_k_spin.setRange(1, 99)
        self.knn_k_spin.setValue(5)
        self.knn_k_spin.valueChanged.connect(lambda _value: self._refresh_analysis_tables())
        controls.addWidget(self.knn_k_spin)
        controls.addStretch()
        knn_layout.addLayout(controls)

        self.knn_metrics_label = QLabel("（需要曲风标签）")
        self.knn_metrics_label.setWordWrap(True)
        knn_layout.addWidget(self.knn_metrics_label)

        self.knn_table = self._create_table()
        knn_layout.addWidget(self.knn_table)
        self._add_tab("KNN 分类", knn_widget)

        self.confusion_table = self._create_table()
        self._add_tab("混淆矩阵", self.confusion_table)

        structure_widget = QWidget()
        structure_layout = QVBoxLayout(structure_widget)
        structure_layout.setContentsMargins(4, 4, 4, 4)

        structure_controls = QHBoxLayout()
        structure_controls.addWidget(QLabel("分段:"))
        self.structure_section_spin = QSpinBox()
        self.structure_section_spin.setRange(3, 32)
        self.structure_section_spin.setValue(8)
        self.structure_section_spin.valueChanged.connect(lambda _value: self._refresh_analysis_tables())
        structure_controls.addWidget(self.structure_section_spin)
        structure_controls.addStretch()
        structure_layout.addLayout(structure_controls)

        self.structure_table = self._create_table()
        structure_layout.addWidget(self.structure_table)
        self._add_tab("结构分析", structure_widget)

        self.method_comparison_table = self._create_table()
        self._add_tab("方法对比", self.method_comparison_table)

        classification_widget = QWidget()
        classification_layout = QVBoxLayout(classification_widget)
        classification_layout.setContentsMargins(4, 4, 4, 4)

        self.query_summary_label = QLabel("（点击底部“识别曲风”选择 MIDI）")
        self.query_summary_label.setWordWrap(True)
        self.query_summary_label.setStyleSheet("font-weight: bold;")
        classification_layout.addWidget(self.query_summary_label)

        classification_layout.addWidget(QLabel("Top 最近邻"))
        self.query_neighbors_table = self._create_table()
        classification_layout.addWidget(self.query_neighbors_table)

        classification_layout.addWidget(QLabel("曲风投票"))
        self.query_votes_table = self._create_table()
        classification_layout.addWidget(self.query_votes_table)
        self._add_tab("单曲识别", classification_widget)

        similarity_widget = QWidget()
        similarity_layout = QVBoxLayout(similarity_widget)
        similarity_layout.setContentsMargins(4, 4, 4, 4)

        self.similarity_summary_label = QLabel("（点击底部“旋律检测”选择两首 MIDI）")
        self.similarity_summary_label.setWordWrap(True)
        self.similarity_summary_label.setStyleSheet("font-weight: bold;")
        similarity_layout.addWidget(self.similarity_summary_label)

        similarity_layout.addWidget(QLabel("全曲指标"))
        self.similarity_metrics_table = self._create_table()
        similarity_layout.addWidget(self.similarity_metrics_table)

        similarity_layout.addWidget(QLabel("最相似片段"))
        self.similarity_segment_table = self._create_table()
        similarity_layout.addWidget(self.similarity_segment_table)
        self._add_tab("旋律检测", similarity_widget)

    def _create_table(self) -> QTableWidget:
        """Create a read-only table with compact default behavior."""
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        return table

    def _add_tab(self, title: str, child: QWidget) -> None:
        """Add a child widget inside a zero-margin tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(child)
        self.tabs.addTab(tab, title)

    def set_result(
        self,
        matrix: np.ndarray,
        names: list[str],
        cluster_result: dict,
        curves: list,
    ) -> None:
        """Render clustering outputs for matrix shape (N, N)."""
        self._matrix = matrix
        self._names = names
        self._cluster_result = cluster_result
        self._curves = curves
        if HAS_MATPLOTLIB:
            self._draw_mds(matrix, names, cluster_result, curves)
            self._draw_dendrogram(cluster_result, names)
        self._refresh_analysis_tables()

    def _draw_mds(self, matrix, names, cluster_result, curves) -> None:
        """Draw an MDS scatter plot for a distance matrix."""
        from src.analysis.clustering import mds_reduce

        self._mds_figure.clear()
        axis = self._mds_figure.add_subplot(111)

        if matrix.shape[0] >= 2:
            coords = mds_reduce(matrix, n_components=2)
            colors = _colors_for_curves(curves, cluster_result)
            legend_labels = set()
            for index, name in enumerate(names):
                label = curves[index].label if index < len(curves) and curves[index].label else None
                legend_label = label if label and label not in legend_labels else None
                if label:
                    legend_labels.add(label)
                axis.scatter(
                    coords[index, 0],
                    coords[index, 1],
                    c=colors[index],
                    s=40,
                    edgecolors="black",
                    linewidth=0.5,
                    label=legend_label,
                )
                axis.annotate(
                    name,
                    (coords[index, 0], coords[index, 1]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    fontsize=7,
                    ha="center",
                )
            axis.set_xlabel("MDS 1")
            axis.set_ylabel("MDS 2")
            if legend_labels:
                axis.legend(fontsize=7, loc="best")
        else:
            axis.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=axis.transAxes)

        axis.set_title("MDS 降维散点图")
        self._mds_figure.tight_layout()
        self._mds_canvas.draw()

    def _draw_dendrogram(self, cluster_result, names) -> None:
        """Draw hierarchical clustering dendrogram for the current matrix."""
        self._dendrogram_figure.clear()
        axis = self._dendrogram_figure.add_subplot(111)

        linkage_matrix = cluster_result.get("linkage")
        if linkage_matrix is not None and len(names) >= 2:
            dendrogram(linkage_matrix, labels=names, ax=axis, leaf_rotation=45, leaf_font_size=7)
            axis.set_title("层次聚类树状图")
        else:
            axis.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=axis.transAxes)

        self._dendrogram_figure.tight_layout()
        self._dendrogram_canvas.draw()

    def _refresh_analysis_tables(self) -> None:
        """Refresh all label-based analysis tables from the cached result."""
        if self._matrix is None:
            return
        self._draw_genre_distance_table()
        self._draw_evaluation_table()
        self._draw_knn_tables()
        self._draw_structure_table()
        self._draw_method_comparison_table()

    def _draw_genre_distance_table(self) -> None:
        """Render average distances between genre labels."""
        from src.analysis.genre_analysis import genre_distance_matrix

        result = genre_distance_matrix(self._matrix, self._curves)
        if result is None:
            self._show_table_message(self.genre_table, "需要给曲线设置曲风标签")
            return

        labels = result.labels
        self.genre_table.setRowCount(len(labels))
        self.genre_table.setColumnCount(len(labels))
        self.genre_table.setHorizontalHeaderLabels([f"{label} ({result.counts[label]})" for label in labels])
        self.genre_table.setVerticalHeaderLabels([f"{label} ({result.counts[label]})" for label in labels])

        finite_values = result.matrix[np.isfinite(result.matrix)]
        max_distance = float(np.max(finite_values)) if len(finite_values) else 1.0
        if max_distance <= 0:
            max_distance = 1.0

        for row in range(len(labels)):
            for col in range(len(labels)):
                value = result.matrix[row, col]
                text = "-" if np.isnan(value) else f"{float(value):.4f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if np.isnan(value):
                    item.setBackground(QColor("#eeeeee"))
                elif row == col:
                    item.setBackground(QColor("#d9ead3"))
                else:
                    self._set_distance_background(item, float(value), max_distance)
                self.genre_table.setItem(row, col, item)
        self.genre_table.resizeColumnsToContents()

    def _draw_evaluation_table(self) -> None:
        """Render quantitative clustering and genre-separation metrics."""
        from src.analysis.evaluation import evaluate, evaluate_distance_matrix
        from src.analysis.genre_analysis import genre_distance_matrix

        cluster_metrics = evaluate(self._curves, self._cluster_result.get("labels", []))
        distance_metrics = evaluate_distance_matrix(self._curves, self._matrix)
        genre_result = genre_distance_matrix(self._matrix, self._curves)
        within_avg, between_avg, separation_ratio = _genre_separation_values(genre_result.matrix if genre_result else None)

        rows = [
            ("ARI", _format_optional_float(cluster_metrics.get("ari") if cluster_metrics else None)),
            ("纯度", _format_optional_float(cluster_metrics.get("purity") if cluster_metrics else None)),
            ("轮廓系数", _format_optional_float(distance_metrics.get("silhouette") if distance_metrics else None)),
            ("类内平均距离", _format_optional_float(within_avg)),
            ("类间平均距离", _format_optional_float(between_avg)),
            ("类间/类内比值", _format_optional_float(separation_ratio)),
        ]
        self.evaluation_table.setRowCount(len(rows))
        self.evaluation_table.setColumnCount(2)
        self.evaluation_table.setHorizontalHeaderLabels(["指标", "值"])
        self.evaluation_table.setVerticalHeaderLabels([str(index + 1) for index in range(len(rows))])
        for row, (metric, value) in enumerate(rows):
            for col, text in enumerate([metric, value]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.evaluation_table.setItem(row, col, item)
        self.evaluation_table.resizeColumnsToContents()

    def _draw_knn_tables(self) -> None:
        """Render KNN prediction and confusion-matrix outputs."""
        from src.analysis.evaluation import evaluate
        from src.analysis.genre_analysis import knn_genre_classification

        cluster_metrics = evaluate(self._curves, self._cluster_result.get("labels", []))
        result = knn_genre_classification(self._matrix, self._curves, k=self.knn_k_spin.value())
        if result is None:
            self.knn_metrics_label.setText("需要至少 2 个曲风、且至少 2 个带标签样本")
            self._show_table_message(self.knn_table, "无 KNN 分类结果")
            self._show_table_message(self.confusion_table, "无混淆矩阵")
            return

        metrics = result.metrics
        cluster_text = ""
        if cluster_metrics:
            cluster_text = f"  |  ARI: {cluster_metrics['ari']:.4f}  纯度: {cluster_metrics['purity']:.4f}"
        self.knn_metrics_label.setText(
            f"K={result.k}  Accuracy: {metrics['accuracy']:.4f}  "
            f"Macro-P: {metrics['macro_precision']:.4f}  "
            f"Macro-R: {metrics['macro_recall']:.4f}  "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
            f"{cluster_text}"
        )

        self._fill_knn_prediction_table(result)
        self._fill_confusion_table(result)

    def _fill_knn_prediction_table(self, result) -> None:
        """Render one row per leave-one-out KNN prediction."""
        headers = ["曲目", "真实曲风", "预测曲风", "最近邻曲风", "最近距离", "结果"]
        self.knn_table.setRowCount(len(result.predictions))
        self.knn_table.setColumnCount(len(headers))
        self.knn_table.setHorizontalHeaderLabels(headers)
        self.knn_table.setVerticalHeaderLabels([str(index + 1) for index in range(len(result.predictions))])

        names = [curve.name for curve in self._curves if curve.label]
        for row, prediction in enumerate(result.predictions):
            nearest_distance = min(prediction.neighbor_distances) if prediction.neighbor_distances else 0.0
            values = [
                names[prediction.index],
                prediction.true_label,
                prediction.predicted_label,
                ", ".join(prediction.neighbor_labels),
                f"{nearest_distance:.4f}",
                "正确" if prediction.true_label == prediction.predicted_label else "错误",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == len(values) - 1:
                    item.setBackground(QColor("#d9ead3" if value == "正确" else "#f4cccc"))
                self.knn_table.setItem(row, col, item)
        self.knn_table.resizeColumnsToContents()

    def _fill_confusion_table(self, result) -> None:
        """Render the confusion matrix with real labels as rows and predicted labels as columns."""
        labels = result.metrics["labels"]
        matrix = result.metrics["confusion_matrix"]
        self.confusion_table.setRowCount(len(labels))
        self.confusion_table.setColumnCount(len(labels))
        self.confusion_table.setHorizontalHeaderLabels([f"预测:{label}" for label in labels])
        self.confusion_table.setVerticalHeaderLabels([f"真实:{label}" for label in labels])

        max_count = int(np.max(matrix)) if matrix.size else 1
        if max_count <= 0:
            max_count = 1
        for row in range(len(labels)):
            for col in range(len(labels)):
                count = int(matrix[row, col])
                item = QTableWidgetItem(str(count))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if row == col:
                    item.setBackground(QColor("#d9ead3"))
                elif count > 0:
                    ratio = count / max_count
                    item.setBackground(QColor(255, int(235 - 80 * ratio), int(235 - 80 * ratio)))
                self.confusion_table.setItem(row, col, item)
        self.confusion_table.resizeColumnsToContents()

    def _show_table_message(self, table: QTableWidget, message: str) -> None:
        """Show a one-cell message in a table."""
        table.setRowCount(1)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["状态"])
        table.setVerticalHeaderLabels([""])
        item = QTableWidgetItem(message)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(0, 0, item)

    def _set_distance_background(self, item: QTableWidgetItem, value: float, max_distance: float) -> None:
        """Apply a green-to-red heatmap color for distance values."""
        ratio = max(0.0, min(1.0, value / max_distance))
        red = int(232 * ratio + 48 * (1 - ratio))
        green = int(80 * ratio + 166 * (1 - ratio))
        blue = int(80 * ratio + 120 * (1 - ratio))
        item.setBackground(QColor(red, green, blue))
        if ratio > 0.55:
            item.setForeground(QColor("white"))

    def _draw_structure_table(self) -> None:
        """Render repeated, varied, and macro-structure analysis for each curve."""
        from src.analysis.structure_analysis import analyze_structures

        results = analyze_structures(
            self._curves,
            section_count=self.structure_section_spin.value(),
            method="modified",
            repeat_threshold=0.08,
            variation_threshold=0.2,
            min_points=2,
        )
        if not results:
            self._show_table_message(self.structure_table, "没有足够的旋律点用于结构分析")
            return

        headers = ["曲目", "片段序列", "基础序列", "宏观结构", "重复", "变奏", "摘要"]
        self.structure_table.setRowCount(len(results))
        self.structure_table.setColumnCount(len(headers))
        self.structure_table.setHorizontalHeaderLabels(headers)
        self.structure_table.setVerticalHeaderLabels([str(index + 1) for index in range(len(results))])

        for row, result in enumerate(results):
            repeat_count = sum(1 for relation in result.relations if relation.relation == "重复")
            variation_count = sum(1 for relation in result.relations if relation.relation == "变奏")
            values = [
                result.curve_name,
                result.section_sequence,
                result.base_sequence,
                result.macro_structure,
                str(repeat_count),
                str(variation_count),
                result.summary,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 3:
                    if value == "ABA":
                        item.setBackground(QColor("#d9ead3"))
                    elif value == "重复型":
                        item.setBackground(QColor("#fff2cc"))
                self.structure_table.setItem(row, col, item)
        self.structure_table.resizeColumnsToContents()

    def _draw_method_comparison_table(self) -> None:
        """Render genre-separation comparison across distance methods."""
        from src.analysis.method_comparison import compare_genre_methods

        results = compare_genre_methods(self._curves, k=self.knn_k_spin.value())
        if not results:
            self._show_table_message(self.method_comparison_table, "需要带曲风标签的曲线")
            return

        headers = ["方法", "曲风内均值", "曲风间均值", "区分比", "KNN Accuracy", "Macro-F1", "样本/曲风"]
        self.method_comparison_table.setRowCount(len(results))
        self.method_comparison_table.setColumnCount(len(headers))
        self.method_comparison_table.setHorizontalHeaderLabels(headers)
        self.method_comparison_table.setVerticalHeaderLabels([str(index + 1) for index in range(len(results))])

        best_ratio = max(
            (result.separation_ratio for result in results if result.separation_ratio is not None),
            default=None,
        )
        best_accuracy = max(
            (result.knn_accuracy for result in results if result.knn_accuracy is not None),
            default=None,
        )
        for row, result in enumerate(results):
            values = [
                result.display_name,
                _format_optional_float(result.within_avg),
                _format_optional_float(result.between_avg),
                _format_optional_float(result.separation_ratio),
                _format_optional_float(result.knn_accuracy),
                _format_optional_float(result.macro_f1),
                f"{result.sample_count}/{result.genre_count}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 3 and best_ratio is not None and result.separation_ratio == best_ratio:
                    item.setBackground(QColor("#d9ead3"))
                if col == 4 and best_accuracy is not None and result.knn_accuracy == best_accuracy:
                    item.setBackground(QColor("#d9ead3"))
                self.method_comparison_table.setItem(row, col, item)
        self.method_comparison_table.resizeColumnsToContents()

    def set_query_classification(self, result) -> None:
        """Render one external MIDI classification result."""
        self.query_summary_label.setText(
            f"曲目: {result.query_name}  |  预测曲风: {result.predicted_label}  |  "
            f"置信度: {result.confidence:.2%}"
        )
        self._fill_query_neighbors_table(result)
        self._fill_query_votes_table(result)
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "单曲识别":
                self.tabs.setCurrentIndex(index)
                break

    def _fill_query_neighbors_table(self, result) -> None:
        """Render nearest standard-set neighbors for an external query."""
        headers = ["排名", "曲目", "曲风", "距离"]
        self.query_neighbors_table.setRowCount(len(result.neighbors))
        self.query_neighbors_table.setColumnCount(len(headers))
        self.query_neighbors_table.setHorizontalHeaderLabels(headers)
        self.query_neighbors_table.setVerticalHeaderLabels([str(index + 1) for index in range(len(result.neighbors))])

        for row, neighbor in enumerate(result.neighbors):
            values = [str(row + 1), neighbor.name, neighbor.label, f"{neighbor.distance:.4f}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if neighbor.label == result.predicted_label:
                    item.setBackground(QColor("#d9ead3"))
                self.query_neighbors_table.setItem(row, col, item)
        self.query_neighbors_table.resizeColumnsToContents()

    def _fill_query_votes_table(self, result) -> None:
        """Render vote counts and mean distances by genre."""
        labels = sorted(
            result.vote_counts,
            key=lambda label: (-result.vote_counts[label], result.mean_distances[label], label),
        )
        headers = ["曲风", "票数", "平均距离"]
        self.query_votes_table.setRowCount(len(labels))
        self.query_votes_table.setColumnCount(len(headers))
        self.query_votes_table.setHorizontalHeaderLabels(headers)
        self.query_votes_table.setVerticalHeaderLabels([str(index + 1) for index in range(len(labels))])

        for row, label in enumerate(labels):
            values = [
                label,
                str(result.vote_counts[label]),
                f"{result.mean_distances[label]:.4f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if label == result.predicted_label:
                    item.setBackground(QColor("#d9ead3"))
                self.query_votes_table.setItem(row, col, item)
        self.query_votes_table.resizeColumnsToContents()

    def set_melody_similarity(self, result) -> None:
        """Render two-song melody similarity detection results."""
        self.similarity_summary_label.setText(
            f"{result.left_name} vs {result.right_name}  |  "
            f"判定: {result.level}  |  相似度: {result.score:.1f}"
        )
        self._fill_similarity_metrics_table(result)
        self._fill_similarity_segment_table(result)
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "旋律检测":
                self.tabs.setCurrentIndex(index)
                break

    def _fill_similarity_metrics_table(self, result) -> None:
        """Render full-song distance metrics for melody detection."""
        rows = [
            ("Modified Hausdorff", f"{result.modified_distance:.4f}"),
            ("DTW", f"{result.dtw_distance:.4f}"),
            ("判定等级", result.level),
            ("相似度", f"{result.score:.1f}"),
        ]
        self.similarity_metrics_table.setRowCount(len(rows))
        self.similarity_metrics_table.setColumnCount(2)
        self.similarity_metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        self.similarity_metrics_table.setVerticalHeaderLabels([str(index + 1) for index in range(len(rows))])
        for row, (name, value) in enumerate(rows):
            for col, text in enumerate([name, value]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if name == "判定等级" and result.level != "不相似":
                    item.setBackground(QColor("#d9ead3"))
                self.similarity_metrics_table.setItem(row, col, item)
        self.similarity_metrics_table.resizeColumnsToContents()

    def _fill_similarity_segment_table(self, result) -> None:
        """Render best matching segment evidence."""
        headers = ["左曲片段", "右曲片段", "片段距离", "左点数", "右点数"]
        self.similarity_segment_table.setRowCount(1)
        self.similarity_segment_table.setColumnCount(len(headers))
        self.similarity_segment_table.setHorizontalHeaderLabels(headers)
        self.similarity_segment_table.setVerticalHeaderLabels(["1"])

        if result.best_segment is None:
            values = ["-", "-", "-", "-", "-"]
        else:
            match = result.best_segment
            values = [
                f"{match.left_start:.2f}-{match.left_end:.2f}",
                f"{match.right_start:.2f}-{match.right_end:.2f}",
                f"{match.distance:.4f}",
                str(match.left_points),
                str(match.right_points),
            ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.similarity_segment_table.setItem(0, col, item)
        self.similarity_segment_table.resizeColumnsToContents()

def _colors_for_curves(curves, cluster_result) -> list[str]:
    """Return one display color per curve using labels first, clusters second."""
    palette = [
        "#ff6b6b",
        "#4ecdc4",
        "#ffe66d",
        "#a29bfe",
        "#fd79a8",
        "#00cec9",
        "#fab1a0",
        "#81ecec",
    ]

    if any(curve.label for curve in curves):
        unique_labels = sorted({curve.label for curve in curves if curve.label})
        color_map = {label: palette[index % len(palette)] for index, label in enumerate(unique_labels)}
        return [color_map.get(curve.label, "#999999") for curve in curves]

    labels = cluster_result.get("labels", [0] * len(curves))
    return [palette[int(labels[index]) % len(palette)] for index in range(len(curves))]


def _format_optional_float(value: float | None) -> str:
    """Format optional metric values for compact tables."""
    return "-" if value is None else f"{value:.4f}"


def _genre_separation_values(matrix) -> tuple[float | None, float | None, float | None]:
    """Return diagonal mean, off-diagonal mean, and between/within ratio."""
    if matrix is None:
        return None, None, None
    values = np.asarray(matrix, dtype=np.float64)
    if values.size == 0:
        return None, None, None

    within = [float(values[index, index]) for index in range(values.shape[0]) if np.isfinite(values[index, index])]
    between = [
        float(values[row, col])
        for row in range(values.shape[0])
        for col in range(row + 1, values.shape[1])
        if np.isfinite(values[row, col])
    ]
    within_avg = float(np.mean(within)) if within else None
    between_avg = float(np.mean(between)) if between else None
    if within_avg is None or between_avg is None or within_avg <= 0:
        return within_avg, between_avg, None
    return within_avg, between_avg, between_avg / within_avg
